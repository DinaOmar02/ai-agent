import chromadb
from sentence_transformers import SentenceTransformer


CHROMA_DIR = "data/chroma"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


class Retriever:

    def __init__(self,collection_name: str):

        # Load embedding model
        self.model = SentenceTransformer(
            EMBEDDING_MODEL
        )

        # Connect to ChromaDB
        self.client = chromadb.PersistentClient(
            path=CHROMA_DIR
        )

        # Select collection dynamically
        # The domain provides the collection name
        self.collection = self.client.get_collection(
            name=collection_name
        )

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        month: str | None = None,
        document_type: str | None = None
    ):
        """
        Retrieve relevant chunks from the selected
        domain knowledge base.

        Optional filters:
        - month
        - document_type
        """

        # Convert query to embedding
        query_embedding = self.model.encode(
            query
        ).tolist()

        # Build query parameters
        query_kwargs = {
            "query_embeddings": [
                query_embedding
            ],
            "n_results": top_k,
        }

        # Build metadata filters
        filters = []

        if month:

            filters.append(
                {
                    "month": month
                }
            )

        if document_type:

            filters.append(
                {
                    "document_type": document_type
                }
            )

        # Apply filters
        if len(filters) == 1:

            query_kwargs["where"] = filters[0]

        elif len(filters) > 1:

            query_kwargs["where"] = {
                "$and": filters
            }

        # Search ChromaDB
        results = self.collection.query(
            **query_kwargs
        )

        # Format retrieved chunks
        retrieved_chunks = []

        for i, document in enumerate(
            results["documents"][0]
        ):

            metadata = results["metadatas"][0][i]

            retrieved_chunks.append(
                {
                    "text": document,
                    "source": metadata["source"],
                    "page": metadata["page"],
                    "document_type": metadata.get(
                        "document_type"
                    ),
                    "month": metadata.get(
                        "month"
                    ),
                    "year": metadata.get(
                        "year"
                    ),
                    "distance": results[
                        "distances"
                    ][0][i],
                }
            )

        return retrieved_chunks


def main():

    retriever = Retriever(
        collection_name="hotel_knowledge_base"
    )

    query = (
        "What improvements should the hotel "
        "make for AC Noise, Wi-Fi, and Check-in?"
    )

    results = retriever.retrieve(
        query,
        top_k=5,
        document_type="improvement"
    )

    print("\n" + "=" * 80)

    print("QUERY:")
    print(query)

    print("\n" + "=" * 80)

    print("RETRIEVED CONTEXT:")

    for i, result in enumerate(results):

        print("\n" + "-" * 80)

        print(f"Result #{i + 1}")

        print(f"Source: {result['source']}")

        print(f"Page: {result['page']}")

        print(f"Document Type: " f"{result['document_type']}")

        print(f"Month: {result['month']}")

        print(f"Year: {result['year']}")

        print(f"Distance: {result['distance']}")

        print("\nText:")

        print(result["text"])

if __name__ == "__main__":
    main()