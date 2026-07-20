from rag.retriever import RetrievalService


def print_result(
    title: str,
    result,
) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

    print("Query:", result.query)
    print(
        "Relevant context:",
        result.has_relevant_context,
    )
    print(
        "Highest relevance:",
        f"{result.highest_relevance_score:.4f}",
    )
    print(
        "Accepted document count:",
        len(result.documents),
    )

    for index, scored_item in enumerate(
        result.scored_documents,
        start=1,
    ):
        document = scored_item.document

        print("\n" + "-" * 70)
        print(f"Result: {index}")
        print(
            "Relevance score:",
            f"{scored_item.relevance_score:.4f}",
        )
        print(
            "Source:",
            document.metadata.get(
                "source",
                "Unknown",
            ),
        )
        print(
            "Page:",
            document.metadata.get(
                "page",
                "N/A",
            ),
        )
        print(document.page_content[:300])


def main() -> None:
    retrieval_service = RetrievalService(
        result_count=5,
        persist_directory="vector_db/knowledge",
        collection_name="enterprise_knowledge",
        minimum_relevance=0.45,
    )

    supported_result = retrieval_service.retrieve_with_confidence(
        "How many annual leave days " "are employees entitled to?"
    )

    unsupported_result = retrieval_service.retrieve_with_confidence(
        "What is the policy for employees " "travelling to Mars?"
    )

    print_result(
        "Supported question",
        supported_result,
    )

    print_result(
        "Unsupported question",
        unsupported_result,
    )

    assert supported_result.has_relevant_context
    assert supported_result.documents
    assert supported_result.highest_relevance_score >= 0.45

    assert 0.0 <= unsupported_result.highest_relevance_score <= 1.0


if __name__ == "__main__":
    main()
