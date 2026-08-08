from pathlib import Path
import json

from pypdf import PdfReader


DATA_DIR = Path("data/microsoft")
OUTPUT_DIR = Path("data/processed")
OUTPUT_FILE = OUTPUT_DIR / "chunks.json"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def extract_pages(pdf_path: Path) -> list[dict]:
    """Extract text from each page of a PDF."""
    reader = PdfReader(pdf_path)

    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        if text.strip():
            pages.append(
                {
                    "page": page_number,
                    "text": text,
                }
            )

    return pages


def create_chunks(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """Split text into overlapping chunks without cutting words."""
    chunks = []

    start = 0

    while start < len(text):
        end = start + chunk_size

        if end < len(text):
            while end > start and not text[end].isspace():
                end -= 1

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start = end - overlap

        if start <= 0:
            start = end

    return chunks


def process_pdf(pdf_path: Path) -> list[dict]:
    """Extract pages and create chunks with metadata."""
    pages = extract_pages(pdf_path)

    chunks = []

    for page in pages:
        page_chunks = create_chunks(page["text"])

        for chunk_number, chunk_text in enumerate(page_chunks, start=1):
            chunks.append(
                {
                    "chunk_id": (
                        f"{pdf_path.stem}_"
                        f"p{page['page']}_"
                        f"c{chunk_number}"
                    ),
                    "text": chunk_text,
                    "source": pdf_path.name,
                    "page": page["page"],
                }
            )

    return chunks


def save_chunks(chunks: list[dict]) -> None:
    """Save all chunks to a JSON file."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        json.dump(chunks, file, ensure_ascii=False, indent=2)

    print(f"\nSaved {len(chunks)} chunks to:")
    print(OUTPUT_FILE)


def main():
    pdf_files = list(DATA_DIR.glob("*.pdf"))

    if not pdf_files:
        print("No PDF files found.")
        return

    all_chunks = []

    for pdf_path in pdf_files:
        chunks = process_pdf(pdf_path)

        print("=" * 80)
        print(f"File: {pdf_path.name}")
        print(f"Number of chunks: {len(chunks)}")

        all_chunks.extend(chunks)

    save_chunks(all_chunks)


if __name__ == "__main__":
    main()