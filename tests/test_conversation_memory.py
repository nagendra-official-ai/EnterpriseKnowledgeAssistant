from llm.query_engine import QueryEngine

def main() -> None:
    query_engine = QueryEngine(
        result_count=5,
        persist_directory="vector_db/knowledge",
        collection_name="enterprise_knowledge",
        temperature=0.0,
    )

    session_id = "conversation-test"

    query_engine.clear_conversation(session_id)

    first_response = query_engine.ask(
        question=("What is the annual leave policy " "for employees?"),
        session_id=session_id,
    )

    print("\nFirst question")
    print("-" * 70)
    print(first_response.question)

    print("\nFirst standalone question")
    print("-" * 70)
    print(first_response.standalone_question)

    print("\nFirst answer")
    print("-" * 70)
    print(first_response.answer)

    second_response = query_engine.ask(
        question="Can it be carried forward?",
        session_id=session_id,
    )

    print("\nFollow-up question")
    print("-" * 70)
    print(second_response.question)

    print("\nRewritten standalone question")
    print("-" * 70)
    print(second_response.standalone_question)

    print("\nFollow-up answer")
    print("-" * 70)
    print(second_response.answer)

    print("\nConversation history")
    print("-" * 70)
    print(query_engine.get_conversation_history(session_id))

    assert first_response.answer
    assert second_response.answer

    rewritten_question = (
        second_response.standalone_question
        .strip()
        .lower()
    )

    assert rewritten_question
    assert (
        rewritten_question
        != second_response.question.lower()
    )

    assert any(
        keyword in rewritten_question
        for keyword in [
            "leave",
            "carry",
            "carried",
            "unused",
        ]
    )

    history = (
        query_engine.get_conversation_history(
            session_id
        )
    )

    assert first_response.question in history
    assert second_response.question in history


if __name__ == "__main__":
    main()
