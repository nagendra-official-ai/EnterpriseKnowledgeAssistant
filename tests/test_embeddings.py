from rag.embeddings import EmbeddingService


def main() -> None:
    service = EmbeddingService()

    query_embedding = service.embed_text("How many annual leave days are available?")

    document_embeddings = service.embed_documents(
        [
            "Employees receive 20 days of annual leave.",
            "The company provides medical insurance.",
        ]
    )

    print(f"Query vector dimensions: {len(query_embedding)}")
    print(
        "Document vectors generated:",
        len(document_embeddings),
    )
    print(
        "First document dimensions:",
        len(document_embeddings[0]),
    )
    print(
        "First ten vector values:",
        query_embedding[:10],
    )

    assert query_embedding
    assert len(document_embeddings) == 2
    assert len(query_embedding) == len(document_embeddings[0])
    assert all(isinstance(value, float) for value in query_embedding)


if __name__ == "__main__":
    main()
