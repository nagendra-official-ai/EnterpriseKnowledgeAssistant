import os
from typing import List

from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings

from utils.logger import setup_logger


class EmbeddingService:
    """
    Generates vector embeddings using a local Ollama embedding model.
    """

    def __init__(
        self,
        model_name: str | None = None,
        base_url: str | None = None,
    ) -> None:
        load_dotenv()

        self.logger = setup_logger()

        self.model_name = model_name or os.getenv(
            "OLLAMA_EMBEDDING_MODEL",
            "nomic-embed-text",
        )

        self.base_url = base_url or os.getenv(
            "OLLAMA_BASE_URL",
            "http://localhost:11434",
        )

        self.embedding_model = OllamaEmbeddings(
            model=self.model_name,
            base_url=self.base_url,
        )

        self.logger.info(
            "Embedding service initialized with model: %s",
            self.model_name,
        )

    def get_embedding_model(self) -> OllamaEmbeddings:
        """
        Return the configured LangChain embedding model.

        Chroma uses this object to generate embeddings when
        documents are added and queries are executed.
        """
        return self.embedding_model
    
    
    def embed_text(self, text: str) -> List[float]:
        """
        Generate an embedding for one text value.
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty.")

        try:
            embedding = self.embedding_model.embed_query(text)

            self.logger.info(
                "Generated query embedding with %d dimensions.",
                len(embedding),
            )

            return embedding

        except Exception:
            self.logger.exception("Failed to generate an embedding.")
            raise

    def embed_documents(
        self,
        texts: List[str],
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple document texts.
        """
        if not texts:
            return []

        if any(not text or not text.strip() for text in texts):
            raise ValueError("Document texts cannot contain empty values.")

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
