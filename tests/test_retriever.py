from rag.retriever import RetrievalService


def main() -> None:
    retrieval_service = RetrievalService(
        result_count=5,
        persist_directory="vector_db/knowledge",
        collection_name="enterprise_knowledge",
    )

    query = "How many annual leave days " "are employees entitled to?"

    documents = retrieval_service.retrieve(query)

    context = retrieval_service.format_context(documents)

    print("\nQuery")
    print("-" * 70)
    print(query)

    print("\nRetrieved context")
    print("-" * 70)
    print(context)

    assert documents
    assert len(documents) <= 5
    assert context
    assert "Source 1" in context


if __name__ == "__main__":
    main()
