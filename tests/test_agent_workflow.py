from agents.workflow import (
    KnowledgeAssistantWorkflow,
)
from llm.query_engine import QueryEngine


def main() -> None:
    query_engine = QueryEngine(
        result_count=5,
        persist_directory="vector_db/knowledge",
        collection_name="enterprise_knowledge",
        minimum_relevance=0.45,
        temperature=0.0,
    )

    workflow = KnowledgeAssistantWorkflow(query_engine=query_engine)

    result = workflow.invoke(
        question=("How many annual leave days " "are employees entitled to?"),
        session_id="agent-workflow-test",
    )

    print("\nQuestion")
    print("-" * 70)
    print(result["question"])

    print("\nStandalone question")
    print("-" * 70)
    print(result.get("standalone_question"))

    print("\nAnswer")
    print("-" * 70)
    print(result.get("answer"))

    print("\nRelevant context")
    print("-" * 70)
    print(result.get("has_relevant_context"))

    print("\nHighest relevance")
    print("-" * 70)
    print(result.get("highest_relevance_score"))

    assert result.get("answer")
    assert not result.get("error")
    assert result.get("response") is not None


if __name__ == "__main__":
    main()
