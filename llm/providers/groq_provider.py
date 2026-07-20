import os
import ssl
import sys

import httpx
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_groq import ChatGroq


def _create_http_clients() -> tuple[
    httpx.Client | None,
    httpx.AsyncClient | None,
]:
    """
    Create custom HTTP clients only when Windows trust-store
    support is required.

    Streamlit Cloud and normal public environments use the
    default HTTPX certificate configuration.
    """
    use_system_truststore = (
        os.getenv(
            "USE_SYSTEM_TRUSTSTORE",
            "false",
        )
        .strip()
        .lower()
        == "true"
    )

    if not use_system_truststore:
        return None, None

    try:
        import truststore

    except ImportError as exception:
        raise RuntimeError(
            "USE_SYSTEM_TRUSTSTORE is enabled, but the "
            "'truststore' package is not installed."
        ) from exception

    ssl_context = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)

    return (
        httpx.Client(
            verify=ssl_context,
            timeout=60.0,
        ),
        httpx.AsyncClient(
            verify=ssl_context,
            timeout=60.0,
        ),
    )


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

    http_client, async_http_client = _create_http_clients()

    arguments = {
        "model": resolved_model_name,
        "api_key": resolved_api_key,
        "temperature": temperature,
        "max_retries": 2,
    }

    if http_client is not None:
        arguments["http_client"] = http_client
        arguments["http_async_client"] = async_http_client

    return ChatGroq(**arguments)
