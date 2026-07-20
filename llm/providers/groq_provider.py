import os
import ssl

import httpx
import truststore
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_groq import ChatGroq


def create_groq_chat_model(
    model_name: str | None = None,
    api_key: str | None = None,
    temperature: float = 0.0,
) -> BaseChatModel:
    resolved_model_name = model_name or os.getenv(
        "GROQ_MODEL",
        "llama-3.3-70b-versatile",
    )

    resolved_api_key = api_key or os.getenv("GROQ_API_KEY")

    if not resolved_api_key:
        raise ValueError(
            "GROQ_API_KEY is required when " "LLM_PROVIDER is set to 'groq'."
        )

    ssl_context = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)

    http_client = httpx.Client(
        verify=ssl_context,
        timeout=60.0,
    )

    async_http_client = httpx.AsyncClient(
        verify=ssl_context,
        timeout=60.0,
    )

    return ChatGroq(
        model=resolved_model_name,
        api_key=resolved_api_key,
        temperature=temperature,
        max_retries=2,
        http_client=http_client,
        http_async_client=async_http_client,
    )
