from rag.document_loader import DocumentLoader
from rag.embeddings import EmbeddingService
from rag.text_splitter import TextSplitter
from rag.vector_store import VectorStore


def main() -> None:
    document_loader = DocumentLoader()

    documents = document_loader.load_documents("data/sample_documents")

    text_splitter = TextSplitter(
        chunk_size=500,
        chunk_overlap=100,
    )

    chunks = text_splitter.split_documents(documents)

    embedding_service = EmbeddingService()

    vector_store = VectorStore(
        embedding_model=(embedding_service.get_embedding_model()),
        persist_directory="vector_db/test",
        collection_name="test_enterprise_knowledge",
    )

    print(
        "Existing vector-store count:",
        vector_store.get_document_count(),
    )

    stored_ids = vector_store.add_documents(chunks)

    print(
        "Documents submitted:",
        len(chunks),
    )

    print(
        "IDs returned:",
        len(stored_ids),
    )

    print(
        "Vector-store count:",
        vector_store.get_document_count(),
    )

    query = "How many annual leave days are available?"

    results = vector_store.similarity_search_with_score(
        query=query,
        result_count=5,
    )

    print("\nSearch query:")
    print(query)

    print("\nRetrieved results:")

    for position, (document, score) in enumerate(
        results,
        start=1,
    ):
        print("\n" + "=" * 70)
        print(f"Result: {position}")
        print(f"Distance score: {score}")
        print(f"Metadata: {document.metadata}")
        print("-" * 70)
        print(document.page_content[:500])

    assert documents
    assert chunks
    assert stored_ids
    assert len(stored_ids) == len(chunks)
    assert vector_store.get_document_count() > 0
    assert results
    assert all(result.page_content.strip() for result, _ in results)


if __name__ == "__main__":
    main()
