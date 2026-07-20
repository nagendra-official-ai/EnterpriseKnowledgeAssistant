import os
from enum import Enum
from typing import List

from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaEmbeddings

from utils.logger import setup_logger


class EmbeddingProvider(str, Enum):
    """
    Supported embedding providers.
    """

    OLLAMA = "ollama"
    HUGGINGFACE = "huggingface"


class EmbeddingService:
    """
    Creates embeddings using either Ollama or a local
    Hugging Face Sentence Transformers model.
    """

    def __init__(
        self,
        provider: str | None = None,
        model_name: str | None = None,
        base_url: str | None = None,
    ) -> None:
        load_dotenv()

        self.logger = setup_logger()

        self.provider = (
            (
                provider
                or os.getenv(
                    "EMBEDDING_PROVIDER",
                    EmbeddingProvider.OLLAMA.value,
                )
            )
            .strip()
            .lower()
        )

        self.model_name = model_name
        self.base_url = base_url

        self.embedding_model = self._create_embedding_model()

        self.logger.info(
            "Embedding service initialized. " "Provider: %s, model: %s",
            self.provider,
            self.resolved_model_name,
        )

    def _create_embedding_model(
        self,
    ) -> Embeddings:
        """
        Create the configured embedding implementation.
        """
        if self.provider == EmbeddingProvider.OLLAMA.value:
            self.resolved_model_name = self.model_name or os.getenv(
                "OLLAMA_EMBEDDING_MODEL",
                "nomic-embed-text",
            )

            resolved_base_url = self.base_url or os.getenv(
                "OLLAMA_BASE_URL",
                "http://localhost:11434",
            )

            return OllamaEmbeddings(
                model=self.resolved_model_name,
                base_url=resolved_base_url,
            )

        if self.provider == EmbeddingProvider.HUGGINGFACE.value:
            self.resolved_model_name = self.model_name or os.getenv(
                "HUGGINGFACE_EMBEDDING_MODEL",
                ("sentence-transformers/" "all-MiniLM-L6-v2"),
            )

            return HuggingFaceEmbeddings(
                model_name=self.resolved_model_name,
                model_kwargs={
                    "device": "cpu",
                },
                encode_kwargs={
                    "normalize_embeddings": True,
                },
            )

        supported = ", ".join(item.value for item in EmbeddingProvider)

        raise ValueError(
            f"Unsupported embedding provider: "
            f"{self.provider}. "
            f"Supported providers: {supported}."
        )

    def get_embedding_model(self) -> Embeddings:
        """
        Return the configured LangChain embedding object.
        """
        return self.embedding_model

    def embed_text(
        self,
        text: str,
    ) -> List[float]:
        """
        Generate an embedding for one query.
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty.")

        try:
            embedding = self.embedding_model.embed_query(text.strip())

            self.logger.info(
                "Generated query embedding with " "%d dimensions.",
                len(embedding),
            )

            return embedding

        except Exception:
            self.logger.exception("Failed to generate query embedding.")
            raise

    def embed_documents(
        self,
        texts: List[str],
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.
        """
        if not texts:
            return []

        if any(not text or not text.strip() for text in texts):
            raise ValueError("Document texts cannot contain " "empty values.")

        try:
            embeddings = self.embedding_model.embed_documents(texts)

            self.logger.info(
                "Generated embeddings for %d texts.",
                len(embeddings),
            )

            return embeddings

        except Exception:
            self.logger.exception("Failed to generate document embeddings.")
            raise
