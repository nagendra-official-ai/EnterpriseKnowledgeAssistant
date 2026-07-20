import os
from dataclasses import dataclass
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from llm.llm_factory import LLMFactory

from rag.retriever import RetrievalService
from llm.conversation_memory import ConversationMemory
from utils.logger import setup_logger


@dataclass
class SourceReference:
    """
    Represents one source used to generate a RAG answer.
    """

    source_number: int
    file_path: str
    page: int | str
    content: str


@dataclass
class RAGResponse:
    """
    Contains the generated answer and supporting sources.
    """

    question: str
    standalone_question: str
    answer: str
    sources: List[SourceReference]
    highest_relevance_score: float


class QueryEngine:
    """
    Coordinates retrieval and grounded answer generation
    using an existing Chroma knowledge base and ChatOllama.
    """

    def __init__(
        self,
        result_count: int = 5,
        persist_directory: str | None = None,
        collection_name: str | None = None,
        llm_provider: str | None = None,
        chat_model_name: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        prompt_path: str = "prompts/retrieval_prompt.txt",
        rewrite_prompt_path: str = ("prompts/query_rewrite_prompt.txt"),
        temperature: float = 0.0,
        max_history_messages: int = 10,
        minimum_relevance: float = 0.45,
    ) -> None:
        self.logger = setup_logger()

        # self.chat_model_name = chat_model_name or os.getenv(
        #     "OLLAMA_CHAT_MODEL",
        #     "qwen2.5:3b",
        # )

        # self.base_url = base_url or os.getenv(
        #     "OLLAMA_BASE_URL",
        #     "http://localhost:11434",
        # )

        self.prompt_template = self._load_prompt(
            prompt_path,
            required_placeholders={
                "{context}",
                "{question}",
                "{chat_history}",
            },
        )

        self.rewrite_prompt_template = self._load_prompt(
            rewrite_prompt_path,
            required_placeholders={
                "{chat_history}",
                "{question}",
            },
        )

        self.conversation_memory = ConversationMemory(max_messages=max_history_messages)

        self.retrieval_service = RetrievalService(
            result_count=result_count,
            persist_directory=persist_directory,
            collection_name=collection_name,
            minimum_relevance=minimum_relevance,
        )

        self.llm_factory = LLMFactory()

        self.chat_model = self.llm_factory.create_chat_model(
            provider=llm_provider,
            model_name=chat_model_name,
            temperature=temperature,
            base_url=base_url,
            api_key=api_key,
        )

        resolved_provider = llm_provider or os.getenv(
            "LLM_PROVIDER",
            "ollama",
        )

        self.logger.info(
            "Query engine initialized. " "Provider: %s, retrieved chunks: %d",
            resolved_provider,
            result_count,
        )

    def _rewrite_question(
        self,
        question: str,
        chat_history: str,
    ) -> str:
        """
        Rewrite a conversational follow-up into a
        standalone retrieval question.
        """
        if chat_history == "No previous conversation.":
            return question

        rewrite_prompt = self.rewrite_prompt_template.format(
            chat_history=chat_history,
            question=question,
        )

        try:
            response = self.chat_model.invoke(
                [
                    SystemMessage(
                        content=(
                            "Rewrite questions for document "
                            "retrieval. Do not answer them."
                        )
                    ),
                    HumanMessage(content=rewrite_prompt),
                ]
            )

            standalone_question = self._extract_response_text(response.content)

            self.logger.info(
                "Rewritten retrieval question: %s",
                standalone_question,
            )

            return standalone_question

        except Exception:
            self.logger.exception(
                "Question rewriting failed. " "Using the original question."
            )

            return question

    def get_conversation_history(
        self,
        session_id: str = "default",
    ) -> str:
        """
        Return formatted conversation history.
        """
        return self.conversation_memory.format_history(session_id)

    def clear_conversation(
        self,
        session_id: str = "default",
    ) -> None:
        """
        Clear conversation history for one session.
        """
        self.conversation_memory.clear(session_id)

    def ask(
        self,
        question: str,
        session_id: str = "default",
    ) -> RAGResponse:
        """
        Process a conversational question using retrieval,
        chat history and grounded answer generation.
        """
        if not question or not question.strip():
            raise ValueError("Question cannot be empty.")

        if not session_id or not session_id.strip():
            raise ValueError("session_id cannot be empty.")

        cleaned_question = question.strip()

        self.logger.info(
            "Processing question for session '%s': %s",
            session_id,
            cleaned_question,
        )

        chat_history = self.conversation_memory.format_history(session_id)

        standalone_question = self._rewrite_question(
            question=cleaned_question,
            chat_history=chat_history,
        )

        retrieval_result = self.retrieval_service.retrieve_with_confidence(
            standalone_question
        )

        retrieved_documents = retrieval_result.documents

        if not retrieval_result.has_relevant_context:
            answer = (
                "I could not find this information "
                "in the available enterprise documents."
            )

            self.conversation_memory.add_exchange(
                session_id=session_id,
                user_message=cleaned_question,
                assistant_message=answer,
            )

            return RAGResponse(
                question=cleaned_question,
                standalone_question=standalone_question,
                answer=answer,
                sources=[],
                highest_relevance_score=(retrieval_result.highest_relevance_score),
            )

        context = self.retrieval_service.format_context(retrieved_documents)

        prompt = self.prompt_template.format(
            chat_history=chat_history,
            context=context,
            question=cleaned_question,
        )

        try:
            response = self.chat_model.invoke(
                [
                    SystemMessage(
                        content=(
                            "You answer questions only from "
                            "the supplied enterprise documents."
                        )
                    ),
                    HumanMessage(content=prompt),
                ]
            )

            answer = self._extract_response_text(response.content)

            sources = self._create_source_references(retrieved_documents)

            self.conversation_memory.add_exchange(
                session_id=session_id,
                user_message=cleaned_question,
                assistant_message=answer,
            )

            self.logger.info(
                "Generated conversational RAG answer " "using %d sources.",
                len(sources),
            )

            return RAGResponse(
                question=cleaned_question,
                standalone_question=standalone_question,
                answer=answer,
                sources=sources,
                highest_relevance_score=(retrieval_result.highest_relevance_score),
            )

        except Exception:
            self.logger.exception("Failed to generate the RAG answer.")
            raise

    @staticmethod
    def _load_prompt(
        prompt_path: str,
        required_placeholders: set[str],
    ) -> str:
        """
        Load and validate a prompt template.
        """
        path = Path(prompt_path)

        if not path.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

        prompt = path.read_text(encoding="utf-8").strip()

        if not prompt:
            raise ValueError(f"Prompt template is empty: {prompt_path}")

        missing_placeholders = [
            placeholder
            for placeholder in required_placeholders
            if placeholder not in prompt
        ]

        if missing_placeholders:
            raise ValueError(
                "Prompt template is missing placeholders: "
                + ", ".join(missing_placeholders)
            )

        return prompt

    @staticmethod
    def _extract_response_text(
        response_content: object,
    ) -> str:
        """
        Convert the chat-model response content into text.
        """
        if isinstance(response_content, str):
            answer = response_content.strip()

        elif isinstance(response_content, list):
            text_parts: List[str] = []

            for item in response_content:
                if isinstance(item, str):
                    text_parts.append(item)

                elif isinstance(item, dict):
                    text = item.get("text")

                    if text:
                        text_parts.append(str(text))

            answer = "\n".join(text_parts).strip()

        else:
            answer = str(response_content).strip()

        if not answer:
            raise RuntimeError("The chat model returned an empty answer.")

        return answer

    @staticmethod
    def _create_source_references(
        documents: List[Document],
    ) -> List[SourceReference]:
        """
        Convert retrieved documents into source references.
        """
        sources: List[SourceReference] = []

        for source_number, document in enumerate(
            documents,
            start=1,
        ):
            sources.append(
                SourceReference(
                    source_number=source_number,
                    file_path=str(
                        document.metadata.get(
                            "source",
                            "Unknown",
                        )
                    ),
                    page=document.metadata.get(
                        "page",
                        "N/A",
                    ),
                    content=document.page_content.strip(),
                )
            )

        return sources
