from rag.ingestion import IngestionPipeline


def main() -> None:
    pipeline = IngestionPipeline(
        source_directory="data/sample_documents",
        chunk_size=500,
        chunk_overlap=100,
        persist_directory="vector_db/huggingface",
        collection_name="enterprise_knowledge_hf",
    )

    result = pipeline.run(rebuild_collection=True)

    print("\nCloud-compatible knowledge base created")
    print("-" * 60)
    print(
        "Source documents:",
        result.source_document_count,
    )
    print(
        "Generated chunks:",
        result.chunk_count,
    )
    print(
        "Stored documents:",
        result.stored_document_count,
    )
    print(
        "Collection count:",
        result.collection_document_count,
    )


if __name__ == "__main__":
    main()
