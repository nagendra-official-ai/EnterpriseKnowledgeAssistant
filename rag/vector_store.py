import hashlib
import os
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from utils.logger import setup_logger


class VectorStore:
    """
    Manages persistent storage and semantic retrieval
    of document chunks using ChromaDB.
    """

    def __init__(
        self,
        embedding_model: Embeddings,
        persist_directory: str | None = None,
        collection_name: str | None = None,
    ) -> None:
        load_dotenv()

        self.logger = setup_logger()

        self.persist_directory = persist_directory or os.getenv(
            "CHROMA_PERSIST_DIRECTORY",
            "vector_db",
        )

        self.collection_name = collection_name or os.getenv(
            "CHROMA_COLLECTION_NAME",
            "enterprise_knowledge",
        )

        Path(self.persist_directory).mkdir(
            parents=True,
            exist_ok=True,
        )

        self.vector_store = Chroma(
            collection_name=self.collection_name,
            embedding_function=embedding_model,
            persist_directory=self.persist_directory,
        )

        self.logger.info(
            "Chroma vector store initialized. " "Collection: %s, directory: %s",
            self.collection_name,
            self.persist_directory,
        )

    def add_documents(
        self,
        documents: List[Document],
    ) -> List[str]:
        """
        Add document chunks to the Chroma collection.

        Returns:
            IDs assigned to the stored chunks.
        """
        if not documents:
            self.logger.warning("No documents were provided to the vector store.")
            return []

        valid_documents = [
            document for document in documents if document.page_content.strip()
        ]

        if not valid_documents:
            self.logger.warning("All supplied documents contained empty text.")
            return []

        document_ids = [
            self._create_document_id(document) for document in valid_documents
        ]

        try:
            returned_ids = self.vector_store.add_documents(
                documents=valid_documents,
                ids=document_ids,
            )

            self.logger.info(
                "Added %d documents to Chroma collection '%s'.",
                len(returned_ids),
                self.collection_name,
            )

            return returned_ids

        except Exception:
            self.logger.exception("Failed to add documents to Chroma.")
            raise

    def similarity_search(
        self,
        query: str,
        result_count: int = 5,
    ) -> List[Document]:
        """
        Return documents semantically similar to a query.
        """
        if not query or not query.strip():
            raise ValueError("Search query cannot be empty.")

        if result_count <= 0:
            raise ValueError("result_count must be greater than zero.")

        try:
            results = self.vector_store.similarity_search(
                query=query,
                k=result_count,
            )

            self.logger.info(
                "Similarity search returned %d documents " "for query: %s",
                len(results),
                query,
            )

            return results

        except Exception:
            self.logger.exception("Failed to perform similarity search.")
            raise

    def similarity_search_with_score(
        self,
        query: str,
        result_count: int = 5,
    ) -> list[tuple[Document, float]]:
        """
        Return similar documents together with distance scores.

        Smaller distance generally indicates a closer match.
        """
        if not query or not query.strip():
            raise ValueError("Search query cannot be empty.")

        if result_count <= 0:
            raise ValueError("result_count must be greater than zero.")

        try:
            results = self.vector_store.similarity_search_with_score(
                query=query,
                k=result_count,
            )

            self.logger.info(
                "Similarity search with scores returned " "%d documents.",
                len(results),
            )

            return results

        except Exception:
            self.logger.exception("Failed to perform scored similarity search.")
            raise

    def get_document_count(self) -> int:
        """
        Return the number of records in the Chroma collection.
        """
        return self.vector_store._collection.count()

    def delete_collection(self) -> None:
        """
        Delete the complete Chroma collection.

        Use this before a full re-ingestion when required.
        """
        try:
            self.vector_store.delete_collection()

            self.logger.info(
                "Deleted Chroma collection: %s",
                self.collection_name,
            )

        except Exception:
            self.logger.exception("Failed to delete Chroma collection.")
            raise

    @staticmethod
    def _create_document_id(
        document: Document,
    ) -> str:
        """
        Create a deterministic ID from source, page,
        start position and content.

        The same chunk produces the same ID on repeated runs.
        """
        source = str(document.metadata.get("source", "unknown"))

        page = str(document.metadata.get("page", ""))

        start_index = str(document.metadata.get("start_index", ""))

        identity_text = f"{source}|{page}|{start_index}|" f"{document.page_content}"

        return hashlib.sha256(identity_text.encode("utf-8")).hexdigest()
