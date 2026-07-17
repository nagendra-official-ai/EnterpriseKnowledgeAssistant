from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from utils.logger import setup_logger


class TextSplitter:
    """
    Splits LangChain documents into smaller chunks while preserving metadata.
    """

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero.")

        if chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative.")

        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size.")

        self.logger = setup_logger()
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            add_start_index=True,
        )

    def split_documents(
        self,
        documents: List[Document],
    ) -> List[Document]:
        """
        Split documents into smaller chunks.

        Args:
            documents: Documents returned by the document loader.

        Returns:
            Chunked LangChain Document objects.
        """
        if not documents:
            self.logger.warning("No documents were provided for text splitting.")
            return []

        self.logger.info(
            "Starting text splitting for %d documents. " "Chunk size: %d, overlap: %d",
            len(documents),
            self.chunk_size,
            self.chunk_overlap,
        )

        try:
            chunks = self.text_splitter.split_documents(documents)

            for chunk_number, chunk in enumerate(chunks, start=1):
                chunk.metadata["chunk_number"] = chunk_number
                chunk.metadata["chunk_length"] = len(chunk.page_content)

            self.logger.info(
                "Text splitting completed. Created %d chunks " "from %d documents.",
                len(chunks),
                len(documents),
            )

            return chunks

        except Exception:
            self.logger.exception("An error occurred while splitting documents.")
            return []
