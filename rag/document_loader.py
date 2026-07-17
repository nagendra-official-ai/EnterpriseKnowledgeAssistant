from pathlib import Path
from typing import List

from langchain_core.documents import Document

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
)

from utils.logger import setup_logger


class DocumentLoader:
    """
    Responsible for loading enterprise documents
    from different file formats.
    """

    def __init__(self):
        self.logger = setup_logger()

        self.supported_loaders = {
            ".pdf": self.load_pdf,
            ".docx": self.load_docx,
            ".txt": self.load_text,
        }


    def load_pdf(self, file_path: str) -> List[Document]:
        """
        Load PDF documents.
        """

        try:
            self.logger.info(
                f"Loading PDF document: {file_path}"
            )

            loader = PyPDFLoader(file_path)

            documents = loader.load()

            self.logger.info(
                f"Loaded {len(documents)} pages"
            )

            return documents

        except Exception as ex:
            self.logger.error(
                f"Error loading PDF {file_path}: {ex}"
            )

            return []


    def load_docx(self, file_path: str) -> List[Document]:
        """
        Load Word documents.
        """

        try:
            self.logger.info(
                f"Loading DOCX document: {file_path}"
            )

            loader = Docx2txtLoader(file_path)

            documents = loader.load()

            return documents

        except Exception as ex:
            self.logger.error(
                f"Error loading DOCX {file_path}: {ex}"
            )

            return []


    def load_text(self, file_path: str) -> List[Document]:
        """
        Load text files.
        """

        try:
            self.logger.info(
                f"Loading TXT document: {file_path}"
            )

            loader = TextLoader(
                file_path,
                encoding="utf-8"
            )

            documents = loader.load()

            return documents

        except Exception as ex:
            self.logger.error(
                f"Error loading TXT {file_path}: {ex}"
            )

            return []


    def load_documents(
        self,
        directory_path: str
    ) -> List[Document]:
        """
        Automatically load supported documents
        from a folder.
        """

        all_documents = []

        directory = Path(directory_path)

        for file in directory.iterdir():

            extension = file.suffix.lower()
            loader = self.supported_loaders.get(extension)

            if loader:
                documents = loader(str(file))
                all_documents.extend(documents)
            else:
                self.logger.warning(f"Unsupported file skipped: {file}")

            self.logger.info(
                f"Total documents loaded: {len(all_documents)}"
        )


        return all_documents