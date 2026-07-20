import os

from langchain_core.language_models.chat_models import (
    BaseChatModel,
)
from langchain_groq import ChatGroq


def create_groq_chat_model(
    model_name: str | None = None,
    api_key: str | None = None,
    temperature: float = 0.0,
) -> BaseChatModel:
    """
    Create a Groq-hosted chat model.
    """
    resolved_model_name = model_name or os.getenv(
        "GROQ_MODEL",
        "llama-3.3-70b-versatile",
    )

    resolved_api_key = api_key or os.getenv("GROQ_API_KEY")

    if not resolved_api_key:
        raise ValueError(
            "GROQ_API_KEY is required when " "LLM_PROVIDER is set to 'groq'."
        )

    return ChatGroq(
        model=resolved_model_name,
        api_key=resolved_api_key,
        temperature=temperature,
        max_retries=2,
    )
