import uuid
from pathlib import Path
from typing import Any

import streamlit as st

from agents.workflow import KnowledgeAssistantWorkflow
from llm.query_engine import QueryEngine, RAGResponse
from utils.logger import setup_logger
import os

from app.bootstrap import ensure_knowledge_base

logger = setup_logger()


def load_streamlit_secrets() -> None:
    """
    Copy available Streamlit secrets into environment variables.
    """
    secret_names = [
        "LLM_PROVIDER",
        "EMBEDDING_PROVIDER",
        "GROQ_API_KEY",
        "GROQ_MODEL",
        "HUGGINGFACE_EMBEDDING_MODEL",
        "CHROMA_PERSIST_DIRECTORY",
        "CHROMA_COLLECTION_NAME",
    ]

    for secret_name in secret_names:
        try:
            secret_value = st.secrets.get(secret_name)

        except FileNotFoundError:
            secret_value = None

        if secret_value is not None:
            os.environ[secret_name] = str(secret_value)


st.set_page_config(
    page_title="Enterprise Knowledge Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main() -> None:
    load_streamlit_secrets()
    initialize_session()


@st.cache_resource
def create_workflow() -> KnowledgeAssistantWorkflow:
    persist_directory = os.getenv(
        "CHROMA_PERSIST_DIRECTORY",
        "vector_db/cloud",
    )

    collection_name = os.getenv(
        "CHROMA_COLLECTION_NAME",
        "enterprise_knowledge_cloud",
    )

    ensure_knowledge_base(
        source_directory="data/sample_documents",
        persist_directory=persist_directory,
        collection_name=collection_name,
        chunk_size=500,
        chunk_overlap=100,
    )

    query_engine = QueryEngine(
        result_count=8,
        persist_directory=persist_directory,
        collection_name=collection_name,
        minimum_relevance=0.0,
        temperature=0.0,
    )

    return KnowledgeAssistantWorkflow(
        query_engine=query_engine
    )

def initialize_session() -> None:
    """
    Initialize Streamlit session values.
    """
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "show_debug" not in st.session_state:
        st.session_state.show_debug = False


def clear_chat(
    workflow: KnowledgeAssistantWorkflow,
) -> None:
    """
    Clear UI history and backend conversation memory.
    """
    session_id = st.session_state.session_id

    workflow.query_engine.clear_conversation(session_id)

    st.session_state.messages = []

    st.session_state.session_id = str(uuid.uuid4())

    logger.info("Streamlit chat session cleared.")


def add_message(
    role: str,
    content: str,
    response: RAGResponse | None = None,
) -> None:
    """
    Add one message to Streamlit session history.
    """
    message: dict[str, Any] = {
        "role": role,
        "content": content,
    }

    if response is not None:
        message["standalone_question"] = response.standalone_question

        message["highest_relevance_score"] = response.highest_relevance_score

        message["sources"] = [
            {
                "source_number": source.source_number,
                "file_path": source.file_path,
                "page": source.page,
                "content": source.content,
            }
            for source in response.sources
        ]

    st.session_state.messages.append(message)


def display_sources(
    sources: list[dict[str, Any]],
) -> None:
    """
    Display retrieved document sources.
    """
    if not sources:
        return

    with st.expander(f"View sources ({len(sources)})"):
        for source in sources:
            source_name = Path(source["file_path"]).name

            st.markdown(f"**[Source " f"{source['source_number']}] " f"{source_name}**")

            st.caption(f"Page: {source['page']}")

            st.write(source["content"])

            st.divider()


def display_message(
    message: dict[str, Any],
) -> None:
    """
    Render a stored chat message.
    """
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant":
            display_sources(message.get("sources", []))

            if st.session_state.show_debug:
                standalone_question = message.get("standalone_question")

                relevance_score = message.get("highest_relevance_score")

                if standalone_question:
                    st.caption("Retrieval question: " f"{standalone_question}")

                if relevance_score is not None:
                    st.caption("Highest retrieval relevance: " f"{relevance_score:.4f}")


def render_sidebar(
    workflow: KnowledgeAssistantWorkflow,
) -> None:
    """
    Render application information and controls.
    """
    with st.sidebar:
        st.title("Knowledge Assistant")

        st.markdown("""
            Ask questions about the enterprise documents
            stored in this demonstration knowledge base.
            """)

        st.divider()

        st.subheader("Technology")

        st.markdown("""
            - LangGraph workflow
            - Retrieval-Augmented Generation
            - ChromaDB vector store
            - Ollama local models
            - LangChain
            - Streamlit
            """)

        st.divider()

        st.toggle(
            "Show retrieval diagnostics",
            key="show_debug",
        )

        if st.button(
            "Clear conversation",
            use_container_width=True,
        ):
            clear_chat(workflow)

            st.rerun()

        st.divider()

        st.caption(
            "Answers are generated only from " "the available demonstration documents."
        )


def render_welcome_message() -> None:
    """
    Display initial instructions when the chat is empty.
    """
    if st.session_state.messages:
        return

    with st.chat_message("assistant"):
        st.markdown("""
            Hello. I am an Enterprise Knowledge Assistant.

            You can ask questions such as:

            - What information is available in the employee handbook?
            - Summarize contents of HR Policies?
            - List paid leave of employees?
            - Can unused leave be carried forward?
            
            """)


def main() -> None:
    """
    Run the Streamlit application.
    """
    initialize_session()
    load_streamlit_secrets()
    
    try:
        workflow = create_workflow()

    except Exception as exception:
        logger.exception("Failed to initialize Streamlit workflow.")

        st.error("The knowledge assistant could not be " "initialized.")

        st.exception(exception)
        st.stop()

    render_sidebar(workflow)

    st.title("Enterprise Knowledge Assistant")

    st.caption("Agentic AI · RAG · LangGraph · ChromaDB")

    for message in st.session_state.messages:
        display_message(message)

    render_welcome_message()

    user_question = st.chat_input("Ask a question about the enterprise documents")

    if not user_question:
        return

    add_message(
        role="user",
        content=user_question,
    )

    with st.chat_message("user"):
        st.markdown(user_question)

    with st.chat_message("assistant"):
        with st.spinner("Searching enterprise documents..."):
            try:
                result = workflow.invoke(
                    question=user_question,
                    session_id=(st.session_state.session_id),
                )

                response = result.get("response")

                answer = result.get(
                    "answer",
                    ("The knowledge assistant could " "not generate an answer."),
                )

                st.markdown(answer)

                if response is not None:
                    sources = [
                        {
                            "source_number": (source.source_number),
                            "file_path": (source.file_path),
                            "page": source.page,
                            "content": (source.content),
                        }
                        for source in response.sources
                    ]

                    display_sources(sources)

                    if st.session_state.show_debug:
                        st.caption(
                            "Retrieval question: " f"{response.standalone_question}"
                        )

                        st.caption(
                            "Highest retrieval relevance: "
                            f"{response.highest_relevance_score:.4f}"
                        )

                    add_message(
                        role="assistant",
                        content=answer,
                        response=response,
                    )

                else:
                    error = result.get("error")

                    if error:
                        logger.error(
                            "Workflow error: %s",
                            error,
                        )

                    add_message(
                        role="assistant",
                        content=answer,
                    )

            except Exception as exception:
                logger.exception("Streamlit question processing failed.")

                st.error("The knowledge assistant could not process " "the request.")

                st.exception(exception)

                add_message(
                    role="assistant",
                    content=(
                        "The knowledge assistant could not " "process the request."
                    ),
                )


if __name__ == "__main__":
    main()
