from llm.query_engine import QueryEngine


def main() -> None:
    query_engine = QueryEngine(
        result_count=5,
        persist_directory="vector_db/knowledge",
        collection_name="enterprise_knowledge",
        temperature=0.0,
    )

    question = "How many annual leave days " "are employees entitled to?"

    response = query_engine.ask(question)

    print("\nQuestion")
    print("-" * 70)
    print(response.question)

    print("\nAnswer")
    print("-" * 70)
    print(response.answer)

    print("\nRetrieved sources")
    print("-" * 70)

    for source in response.sources:
        print(f"\n[Source {source.source_number}]")
        print(f"File: {source.file_path}")
        print(f"Page: {source.page}")
        print(
            "Content:",
            source.content[:300],
        )

    assert response.question
    assert response.answer
    assert response.sources
    assert len(response.sources) <= 5


if __name__ == "__main__":
    main()
