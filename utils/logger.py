import logging
from pathlib import Path


def setup_logger(name: str = "EnterpriseKnowledgeAssistant") -> logging.Logger:
    """
    Create and configure a logger for the application.
    """

    log_directory = Path("logs")
    log_directory.mkdir(exist_ok=True)

    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    file_handler = logging.FileHandler(
        log_directory / "application.log",
        encoding="utf-8"
    )
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger