from langgraph.graph import END, START, StateGraph

from agents.state import KnowledgeAssistantState
from llm.query_engine import QueryEngine
from utils.logger import setup_logger


class KnowledgeAssistantWorkflow:
    """
    LangGraph workflow for conversational,
    retrieval-grounded enterprise question answering.
    """

    def __init__(
        self,
        query_engine: QueryEngine,
    ) -> None:
        self.logger = setup_logger()
        self.query_engine = query_engine

        self.graph = self._build_graph()

    def _build_graph(self):
        """
        Build and compile the LangGraph workflow.
        """
        workflow = StateGraph(KnowledgeAssistantState)

        workflow.add_node(
            "process_question",
            self._process_question,
        )

        workflow.add_node(
            "handle_error",
            self._handle_error,
        )

        workflow.add_edge(
            START,
            "process_question",
        )

        workflow.add_conditional_edges(
            "process_question",
            self._route_after_processing,
            {
                "success": END,
                "error": "handle_error",
            },
        )

        workflow.add_edge(
            "handle_error",
            END,
        )

        return workflow.compile()

    def invoke(
        self,
        question: str,
        session_id: str = "default",
    ) -> KnowledgeAssistantState:
        """
        Execute the compiled LangGraph workflow.
        """
        if not question or not question.strip():
            raise ValueError("Question cannot be empty.")

        if not session_id or not session_id.strip():
            raise ValueError("session_id cannot be empty.")

        initial_state: KnowledgeAssistantState = {
            "question": question.strip(),
            "session_id": session_id.strip(),
        }

        result = self.graph.invoke(initial_state)

        return result

    def _process_question(
        self,
        state: KnowledgeAssistantState,
    ) -> KnowledgeAssistantState:
        """
        Run the existing conversational RAG query engine.
        """
        try:
            response = self.query_engine.ask(
                question=state["question"],
                session_id=state["session_id"],
            )

            return {
                **state,
                "standalone_question": (response.standalone_question),
                "answer": response.answer,
                "response": response,
                "highest_relevance_score": (response.highest_relevance_score),
                "has_relevant_context": bool(response.sources),
            }

        except Exception as exception:
            self.logger.exception("Knowledge-assistant workflow failed.")

            return {
                **state,
                "error": str(exception),
            }

    @staticmethod
    def _route_after_processing(
        state: KnowledgeAssistantState,
    ) -> str:
        """
        Select the next graph path.
        """
        if state.get("error"):
            return "error"

        return "success"

    @staticmethod
    def _handle_error(
        state: KnowledgeAssistantState,
    ) -> KnowledgeAssistantState:
        """
        Produce a safe user-facing error response.
        """
        return {
            **state,
            "answer": (
                "The knowledge assistant could not process "
                "the request. Please try again."
            ),
        }
