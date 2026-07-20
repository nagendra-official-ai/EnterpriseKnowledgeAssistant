from pathlib import Path

from rag.ingestion import IngestionPipeline
from utils.logger import setup_logger


logger = setup_logger()


def ensure_knowledge_base(
    source_directory: str,
    persist_directory: str,
    collection_name: str,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> None:
    """
    Create the knowledge base when it does not already exist.
    """
    persist_path = Path(
        persist_directory
    )

    database_exists = (
        persist_path.exists()
        and any(persist_path.iterdir())
    )

    if database_exists:
        logger.info(
            "Knowledge base already exists: %s",
            persist_directory,
        )
        return

    logger.info(
        "Knowledge base not found. "
        "Starting initial ingestion."
    )

    pipeline = IngestionPipeline(
        source_directory=source_directory,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        persist_directory=persist_directory,
        collection_name=collection_name,
    )

    result = pipeline.run(
        rebuild_collection=False
    )

    logger.info(
        "Knowledge-base bootstrap completed. "
        "Documents: %d, chunks: %d",
        result.source_document_count,
        result.chunk_count,
    )