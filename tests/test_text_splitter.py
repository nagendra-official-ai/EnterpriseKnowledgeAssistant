from rag.document_loader import DocumentLoader
from rag.text_splitter import TextSplitter
from collections import Counter

def main() -> None:
    document_loader = DocumentLoader()

    documents = document_loader.load_documents(
        "data/sample_documents"
    )

    text_splitter = TextSplitter(
        chunk_size=500,
        chunk_overlap=100,
    )

    chunks = text_splitter.split_documents(documents)

    chunk_lengths = [len(chunk.page_content) for chunk in chunks]

    print("\nChunk statistics")
    print("-" * 50)
    print(f"Total source characters: {sum(len(doc.page_content) for doc in documents)}")
    print(f"Total chunk characters:  {sum(chunk_lengths)}")
    print(f"Minimum chunk length:    {min(chunk_lengths)}")
    print(f"Maximum chunk length:    {max(chunk_lengths)}")
    print(f"Average chunk length:    {sum(chunk_lengths) / len(chunk_lengths):.2f}")

    source_counts = Counter(
        chunk.metadata.get("source", "Unknown")
        for chunk in chunks
    )

    print("\nChunks by source")
    print("-" * 50)

    for source, count in source_counts.items():
        print(f"{source}: {count}")

    print(f"\nOriginal documents: {len(documents)}")
    print(f"Generated chunks: {len(chunks)}")

    assert documents, "No source documents were loaded."
    assert chunks, "No chunks were generated."
    assert len(chunks) >= len(documents)
    assert all(chunk.page_content.strip() for chunk in chunks)
    assert all("source" in chunk.metadata for chunk in chunks)
    assert all("chunk_number" in chunk.metadata for chunk in chunks)
    assert all("chunk_length" in chunk.metadata for chunk in chunks)
    assert max(chunk_lengths) <= 500
    assert all(length > 0 for length in chunk_lengths)

    print("\nFirst five chunks:")

    for chunk in chunks[:5]:
        print("\n" + "=" * 70)
        print("Metadata:", chunk.metadata)
        print("Content length:", len(chunk.page_content))
        print("-" * 70)
        print(chunk.page_content)


if __name__ == "__main__":
    main()