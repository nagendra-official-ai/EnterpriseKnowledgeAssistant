import os

from langchain_core.language_models.chat_models import (
    BaseChatModel,
)
from langchain_ollama import ChatOllama


def create_ollama_chat_model(
    model_name: str | None = None,
    base_url: str | None = None,
    temperature: float = 0.0,
) -> BaseChatModel:
    """
    Create a locally hosted Ollama chat model.
    """
    resolved_model_name = model_name or os.getenv(
        "OLLAMA_CHAT_MODEL",
        "qwen2.5:3b",
    )

    resolved_base_url = base_url or os.getenv(
        "OLLAMA_BASE_URL",
        "http://localhost:11434",
    )

    return ChatOllama(
        model=resolved_model_name,
        base_url=resolved_base_url,
        temperature=temperature,
    )
