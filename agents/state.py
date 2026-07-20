from typing import List, TypedDict

from langchain_core.documents import Document

from llm.query_engine import RAGResponse


class KnowledgeAssistantState(TypedDict, total=False):
    """
    Shared state passed between LangGraph nodes.
    """

    question: str
    session_id: str
    standalone_question: str

    retrieved_documents: List[Document]
    highest_relevance_score: float
    has_relevant_context: bool

    answer: str
    response: RAGResponse
    error: str
