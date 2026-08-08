from pathlib import Path

from pypdf import PdfReader


DATA_DIR = Path("data/microsoft")


def extract_pdf_pages(pdf_path: Path) -> list[dict]:
    """Extract text from each PDF page while keeping page number."""
    reader = PdfReader(pdf_path)

    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        pages.append(
            {
                "page": page_number,
                "text": text,
            }
        )

    return pages


def main():
    pdf_files = list(DATA_DIR.glob("*.pdf"))

    if not pdf_files:
        print("No PDF files found.")
        return

    for pdf_path in pdf_files:
        pages = extract_pdf_pages(pdf_path)

        print("=" * 80)
        print(f"File: {pdf_path.name}")
        print(f"Number of pages: {len(pages)}")

        for page in pages[:2]:
            print(f"\n--- Page {page['page']} ---")
            print(page["text"][:1000])


if __name__ == "__main__":
    main()