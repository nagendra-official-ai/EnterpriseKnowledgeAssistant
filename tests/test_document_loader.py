from rag.document_loader import DocumentLoader


def main():

    loader = DocumentLoader()

    documents = loader.load_documents(
        "data/sample_documents"
    )


    print(
        f"Total documents: {len(documents)}"
    )


    for doc in documents:

        print("-------------------")

        print(
            doc.metadata
        )

        print(
            doc.page_content[:200]
        )


if __name__ == "__main__":

    main()