from typing import List

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever

from rag.embeddings import EmbeddingService
from rag.vector_store import VectorStore
from utils.logger import setup_logger


class RetrievalService:
    """
    Retrieves relevant document chunks from an existing
    persistent Chroma collection.
    """

    def __init__(
        self,
        result_count: int = 5,
        persist_directory: str | None = None,
        collection_name: str | None = None,
        search_type: str = "similarity",
    ) -> None:
        self.logger = setup_logger()

        self.result_count = result_count

        self.embedding_service = EmbeddingService()

        self.vector_store = VectorStore(
            embedding_model=(self.embedding_service.get_embedding_model()),
            persist_directory=persist_directory,
            collection_name=collection_name,
        )

        self.retriever: VectorStoreRetriever = self.vector_store.as_retriever(
            result_count=result_count,
            search_type=search_type,
        )

    def retrieve(
        self,
        query: str,
    ) -> List[Document]:
        """
        Retrieve document chunks relevant to a user query.
        """
        if not query or not query.strip():
            raise ValueError("Retrieval query cannot be empty.")

        if self.vector_store.get_document_count() == 0:
            raise RuntimeError(
                "The vector-store collection is empty. " "Run document ingestion first."
            )

        try:
            documents = self.retriever.invoke(query)

            self.logger.info(
                "Retrieved %d chunks for query: %s",
                len(documents),
                query,
            )

            return documents

        except Exception:
            self.logger.exception("Document retrieval failed.")
            raise

    @staticmethod
    def format_context(
        documents: List[Document],
    ) -> str:
        """
        Convert retrieved documents into one formatted context
        string suitable for an LLM prompt.
        """
        if not documents:
            return ""

        context_parts: List[str] = []

        for index, document in enumerate(
            documents,
            start=1,
        ):
            source = document.metadata.get(
                "source",
                "Unknown",
            )

            page = document.metadata.get(
                "page",
                "N/A",
            )

            content = document.page_content.strip()

            context_parts.append(
                f"[Source {index}]\n"
                f"File: {source}\n"
                f"Page: {page}\n"
                f"Content:\n{content}"
            )

        return "\n\n".join(context_parts)
