from pathlib import Path
import json

import chromadb
from sentence_transformers import SentenceTransformer


# Paths
CHUNKS_FILE = Path("data/processed/chunks.json")
CHROMA_DIR = Path("data/chroma")

# ChromaDB collection
COLLECTION_NAME = "hotel_knowledge_base"

# Embedding model
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def load_chunks() -> list[dict]:
    """Load chunks from the JSON file."""

    with open(CHUNKS_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


def main():
    # 1. Load chunks
    chunks = load_chunks()

    print(f"Loaded {len(chunks)} chunks.")

    # 2. Load embedding model
    print("Loading embedding model...")

    model = SentenceTransformer(EMBEDDING_MODEL)

    print("Embedding model loaded.")
    print(f"Embedding dimension: {model.get_embedding_dimension()}")

    # 3. Create persistent ChromaDB client
    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR)
    )

    # 4. Create or get collection
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME
    )

    # 5. Prepare data
    documents = [
        chunk["text"]
        for chunk in chunks
    ]

    ids = [
        chunk["chunk_id"]
        for chunk in chunks
    ]

    metadatas = [
    {
        "source": chunk["source"],
        "page": chunk["page"],
        "document_type": chunk["document_type"],
        "month": chunk["month"] or "",
        "year": chunk["year"] or 0,
    }
    for chunk in chunks
    ]

    # 6. Generate embeddings
    print("Generating embeddings...")

    embeddings = model.encode(
        documents,
        show_progress_bar=True,
    )

    print("Embeddings generated.")

    # 7. Store documents + embeddings + metadata in ChromaDB
    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings.tolist(),
        metadatas=metadatas,
    )

    print(f"Stored {len(chunks)} chunks in ChromaDB.")
    print(f"Collection: {COLLECTION_NAME}")


if __name__ == "__main__":
    main()