from rag.retriever import RetrievalService


def main() -> None:
    retrieval_service = RetrievalService(
        result_count=5,
        persist_directory="vector_db/huggingface",
        collection_name="enterprise_knowledge_hf",
        minimum_relevance=0.0,
    )

    query = "What information is available " "in the employee handbook?"

    result = retrieval_service.retrieve_with_confidence(query)

    print("\nQuery")
    print("-" * 60)
    print(query)

    print("\nRetrieved documents")
    print("-" * 60)

    for index, item in enumerate(
        result.scored_documents,
        start=1,
    ):
        print(f"\nResult {index}")
        print(
            "Score:",
            item.relevance_score,
        )
        print(
            "Source:",
            item.document.metadata.get("source"),
        )
        print(item.document.page_content[:300])

    assert result.documents
    assert result.has_relevant_context


if __name__ == "__main__":
    main()
