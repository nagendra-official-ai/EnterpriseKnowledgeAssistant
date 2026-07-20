from typing import List

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever

from rag.embeddings import EmbeddingService
from rag.vector_store import VectorStore
from utils.logger import setup_logger
from dataclasses import dataclass
from typing import List

from langchain_core.documents import Document

from rag.embeddings import EmbeddingService
from rag.vector_store import (
    ScoredDocument,
    VectorStore,
)
from utils.logger import setup_logger


@dataclass
class RetrievalResult:
    """
    Contains accepted retrieval results and confidence details.
    """

    query: str
    documents: List[Document]
    scored_documents: List[ScoredDocument]
    has_relevant_context: bool
    highest_relevance_score: float


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
        minimum_relevance: float = 0.45,
    ) -> None:
        if result_count <= 0:
            raise ValueError("result_count must be greater than zero.")

        if not 0.0 <= minimum_relevance <= 1.0:
            raise ValueError("minimum_relevance must be between 0 and 1.")

        self.logger = setup_logger()

        self.result_count = result_count
        self.search_type = search_type
        self.minimum_relevance = minimum_relevance

        self.embedding_service = EmbeddingService()

        self.vector_store = VectorStore(
            embedding_model=(self.embedding_service.get_embedding_model()),
            persist_directory=persist_directory,
            collection_name=collection_name,
        )

        self.retriever = self.vector_store.as_retriever(
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

    def retrieve_with_confidence(
        self,
        query: str,
    ) -> RetrievalResult:
        """
        Retrieve a wider candidate set and retain the strongest
        unique chunks for answer generation.
        """
        if not query or not query.strip():
            raise ValueError("Retrieval query cannot be empty.")

        if self.vector_store.get_document_count() == 0:
            raise RuntimeError(
                "The vector-store collection is empty. " "Run document ingestion first."
            )

        try:
            # Retrieve more candidates than we finally return.
            candidate_count = max(
                self.result_count * 3,
                15,
            )

            scored_candidates = self.vector_store.similarity_search_with_relevance(
                query=query,
                result_count=candidate_count,
                minimum_relevance=0.0,
            )

            # Highest relevance first.
            scored_candidates.sort(
                key=lambda item: item.relevance_score,
                reverse=True,
            )

            accepted_documents: list[ScoredDocument] = []
            seen_content: set[str] = set()

            for item in scored_candidates:
                normalized_content = item.document.page_content.strip().lower()

                if not normalized_content:
                    continue

                # Remove duplicate or overlapping identical chunks.
                if normalized_content in seen_content:
                    continue

                if item.relevance_score < self.minimum_relevance:
                    continue

                seen_content.add(normalized_content)
                accepted_documents.append(item)

                if len(accepted_documents) >= self.result_count:
                    break

            documents = [item.document for item in accepted_documents]

            highest_score = (
                accepted_documents[0].relevance_score if accepted_documents else 0.0
            )

            self.logger.info(
                "Confidence retrieval completed. "
                "Query: %s, candidates: %d, accepted: %d, "
                "highest relevance: %.4f",
                query,
                len(scored_candidates),
                len(documents),
                highest_score,
            )

            # Log retrieved text temporarily for debugging.
            for index, item in enumerate(
                accepted_documents,
                start=1,
            ):
                self.logger.info(
                    "Retrieved result %d | score %.4f | " "source %s | content %.300s",
                    index,
                    item.relevance_score,
                    item.document.metadata.get(
                        "source",
                        "Unknown",
                    ),
                    item.document.page_content,
                )

            return RetrievalResult(
                query=query,
                documents=documents,
                scored_documents=accepted_documents,
                has_relevant_context=bool(documents),
                highest_relevance_score=highest_score,
            )

        except Exception:
            self.logger.exception("Confidence-based retrieval failed.")
            raise
