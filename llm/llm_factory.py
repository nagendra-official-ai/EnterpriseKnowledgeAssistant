import os
from enum import Enum

from dotenv import load_dotenv
from langchain_core.language_models.chat_models import (
    BaseChatModel,
)

from llm.providers.groq_provider import (
    create_groq_chat_model,
)
from llm.providers.ollama_provider import (
    create_ollama_chat_model,
)
from utils.logger import setup_logger


class LLMProvider(str, Enum):
    """
    Supported chat-model providers.
    """

    OLLAMA = "ollama"
    GROQ = "groq"


class LLMFactory:
    """
    Creates a chat model based on application configuration.
    """

    def __init__(self) -> None:
        load_dotenv()
        self.logger = setup_logger()

    def create_chat_model(
        self,
        provider: str | None = None,
        model_name: str | None = None,
        temperature: float = 0.0,
        base_url: str | None = None,
        api_key: str | None = None,
    ) -> BaseChatModel:
        """
        Create the configured chat-model implementation.
        """
        resolved_provider = (
            (
                provider
                or os.getenv(
                    "LLM_PROVIDER",
                    LLMProvider.OLLAMA.value,
                )
            )
            .strip()
            .lower()
        )

        self.logger.info(
            "Creating chat model for provider: %s",
            resolved_provider,
        )

        if resolved_provider == LLMProvider.OLLAMA.value:
            return create_ollama_chat_model(
                model_name=model_name,
                base_url=base_url,
                temperature=temperature,
            )

        if resolved_provider == LLMProvider.GROQ.value:
            return create_groq_chat_model(
                model_name=model_name,
                api_key=api_key,
                temperature=temperature,
            )

        supported_providers = ", ".join(
            provider_item.value for provider_item in LLMProvider
        )

        raise ValueError(
            f"Unsupported LLM provider: "
            f"{resolved_provider}. "
            f"Supported providers: "
            f"{supported_providers}."
        )
