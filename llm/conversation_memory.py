from typing import Dict, List

from langchain_core.chat_history import (
    InMemoryChatMessageHistory,
)
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
)

from utils.logger import setup_logger


class ConversationMemory:
    """
    Manages in-memory chat histories for multiple sessions.

    Each session ID receives an independent conversation history.
    """

    def __init__(
        self,
        max_messages: int = 10,
    ) -> None:
        if max_messages <= 0:
            raise ValueError("max_messages must be greater than zero.")

        self.logger = setup_logger()
        self.max_messages = max_messages

        self._sessions: Dict[
            str,
            InMemoryChatMessageHistory,
        ] = {}

    def get_history(
        self,
        session_id: str,
    ) -> InMemoryChatMessageHistory:
        """
        Return the history for a session, creating it when needed.
        """
        self._validate_session_id(session_id)

        if session_id not in self._sessions:
            self._sessions[session_id] = InMemoryChatMessageHistory()

            self.logger.info(
                "Created conversation session: %s",
                session_id,
            )

        return self._sessions[session_id]

    def add_exchange(
        self,
        session_id: str,
        user_message: str,
        assistant_message: str,
    ) -> None:
        """
        Add one user/assistant exchange to a session.
        """
        if not user_message or not user_message.strip():
            raise ValueError("User message cannot be empty.")

        if not assistant_message or not assistant_message.strip():
            raise ValueError("Assistant message cannot be empty.")

        history = self.get_history(session_id)

        history.add_messages(
            [
                HumanMessage(content=user_message.strip()),
                AIMessage(content=assistant_message.strip()),
            ]
        )

        self._trim_history(history)

        self.logger.info(
            "Added conversation exchange to session: %s",
            session_id,
        )

    def get_messages(
        self,
        session_id: str,
    ) -> List[BaseMessage]:
        """
        Return a copy of the session's messages.
        """
        history = self.get_history(session_id)

        return list(history.messages)

    def format_history(
        self,
        session_id: str,
    ) -> str:
        """
        Convert the session history into readable prompt text.
        """
        messages = self.get_messages(session_id)

        if not messages:
            return "No previous conversation."

        formatted_messages: List[str] = []

        for message in messages:
            if isinstance(message, HumanMessage):
                role = "User"

            elif isinstance(message, AIMessage):
                role = "Assistant"

            else:
                role = "System"

            formatted_messages.append(f"{role}: {message.content}")

        return "\n".join(formatted_messages)

    def clear(
        self,
        session_id: str,
    ) -> None:
        """
        Clear the conversation history for one session.
        """
        history = self.get_history(session_id)

        history.clear()

        self.logger.info(
            "Cleared conversation session: %s",
            session_id,
        )

    def session_exists(
        self,
        session_id: str,
    ) -> bool:
        """
        Return whether a session already exists.
        """
        return session_id in self._sessions

    def _trim_history(
        self,
        history: InMemoryChatMessageHistory,
    ) -> None:
        """
        Retain only the most recent configured messages.
        """
        if len(history.messages) <= self.max_messages:
            return

        history.messages = history.messages[-self.max_messages :]

    @staticmethod
    def _validate_session_id(
        session_id: str,
    ) -> None:
        if not session_id or not session_id.strip():
            raise ValueError("session_id cannot be empty.")
