from dataclasses import dataclass
from typing import List

from langchain_core.documents import Document

from rag.document_loader import DocumentLoader
from rag.embeddings import EmbeddingService
from rag.text_splitter import TextSplitter
from rag.vector_store import VectorStore
from utils.logger import setup_logger


@dataclass
class IngestionResult:
    """
    Contains summary information about one ingestion run.
    """

    source_document_count: int
    chunk_count: int
    stored_document_count: int
    collection_document_count: int


class IngestionPipeline:
    """
    Coordinates document loading, text splitting,
    embedding generation and ChromaDB storage.
    """

    def __init__(
        self,
        source_directory: str,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        persist_directory: str | None = None,
        collection_name: str | None = None,
    ) -> None:
        self.logger = setup_logger()

        self.source_directory = source_directory

        self.document_loader = DocumentLoader()

        self.text_splitter = TextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        self.embedding_service = EmbeddingService()

        self.vector_store = VectorStore(
            embedding_model=(self.embedding_service.get_embedding_model()),
            persist_directory=persist_directory,
            collection_name=collection_name,
        )

    def run(
        self,
        rebuild_collection: bool = False,
    ) -> IngestionResult:
        """
        Execute the complete document-ingestion workflow.

        Args:
            rebuild_collection:
                Delete and recreate the collection before ingestion.

        Returns:
            Summary information about the ingestion operation.
        """
        self.logger.info(
            "Starting ingestion pipeline. Source: %s",
            self.source_directory,
        )

        if rebuild_collection:
            self._rebuild_vector_store()

        documents = self._load_documents()

        chunks = self._split_documents(documents)

        stored_ids = self.vector_store.add_documents(chunks)

        collection_count = self.vector_store.get_document_count()

        result = IngestionResult(
            source_document_count=len(documents),
            chunk_count=len(chunks),
            stored_document_count=len(stored_ids),
            collection_document_count=collection_count,
        )

        self.logger.info(
            "Ingestion completed. Documents: %d, "
            "chunks: %d, stored: %d, collection count: %d",
            result.source_document_count,
            result.chunk_count,
            result.stored_document_count,
            result.collection_document_count,
        )

        return result

    def _load_documents(self) -> List[Document]:
        """
        Load documents from the configured source directory.
        """
        documents = self.document_loader.load_documents(self.source_directory)

        if not documents:
            raise RuntimeError("No supported documents were loaded.")

        return documents

    def _split_documents(
        self,
        documents: List[Document],
    ) -> List[Document]:
        """
        Split loaded documents into searchable chunks.
        """
        chunks = self.text_splitter.split_documents(documents)

        if not chunks:
            raise RuntimeError("No chunks were generated from the documents.")

        return chunks

    def _rebuild_vector_store(self) -> None:
        """
        Delete and recreate the configured Chroma collection.
        """
        collection_name = self.vector_store.collection_name

        persist_directory = self.vector_store.persist_directory

        embedding_model = self.embedding_service.get_embedding_model()

        if self.vector_store.get_document_count() > 0:
            self.vector_store.delete_collection()

        self.vector_store = VectorStore(
            embedding_model=embedding_model,
            persist_directory=persist_directory,
            collection_name=collection_name,
        )

        self.logger.info(
            "Vector-store collection rebuilt: %s",
            collection_name,
        )
