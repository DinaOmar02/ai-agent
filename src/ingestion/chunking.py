from pathlib import Path
import re
import json

from pypdf import PdfReader


DATA_DIR = Path("data/hotel")
OUTPUT_DIR = Path("data/processed")
OUTPUT_FILE = OUTPUT_DIR / "chunks.json"


def extract_pages(pdf_path: Path) -> list[dict]:
    """Extract text from each PDF page."""

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


def detect_document_type(filename: str) -> str:
    """Identify the type of hotel document."""

    filename = filename.lower()

    if "review" in filename:
        return "review"

    if "room" in filename:
        return "room"

    if "policy" in filename:
        return "policy"

    if "improvement" in filename or "guideline" in filename:
        return "improvement"

    return "unknown"


def detect_month(filename: str) -> str | None:
    """Extract review month from filename."""

    filename = filename.lower()

    months = [
        "january",
        "february",
        "march",
        "april",
        "may",
        "june",
        "july",
        "august",
        "september",
        "october",
        "november",
        "december",
    ]

    for month in months:
        if month in filename:
            return month.capitalize()

    return None


def split_by_sections(text: str, document_type: str) -> list[str]:
    """
    Split a document according to its natural structure.

    Review:
        Review 1, Review 2, ...

    Room:
        Standard Room, Deluxe Room, ...

    Policy:
        Check-in and Check-out, Dining Services, ...
    """

    text = text.replace("\r\n", "\n")
    text = re.sub(r"[ \t]+", " ", text)

    if document_type == "review":

        pattern = r"(?=(?:Review\s+\d+\s*[–-]))"

    elif document_type == "room":

        pattern = r"(?=(?:Standard Room|Deluxe Room|Executive Suite|Family Suite))"

    elif document_type == "policy":

        pattern = (
            r"(?=(?:Check-in and Check-out:|"
            r"Dining Services:|"
            r"General Hotel Rules and Facilities:|"
            r"Emergency and Medical Procedures:))"
        )

    else:
        return [text.strip()]

    sections = re.split(pattern, text, flags=re.IGNORECASE)

    sections = [
        section.strip()
        for section in sections
        if section.strip()
    ]

    return sections


def process_pdf(pdf_path: Path) -> list[dict]:
    """Extract and create structure-aware chunks."""

    document_type = detect_document_type(pdf_path.name)
    month = detect_month(pdf_path.name)

    pages = extract_pages(pdf_path)

    chunks = []

    for page in pages:

        sections = split_by_sections(
            page["text"],
            document_type
        )

        for chunk_number, section in enumerate(sections, start=1):

            chunks.append(
                {
                    "chunk_id": (
                        f"{pdf_path.stem}_"
                        f"p{page['page']}_"
                        f"c{chunk_number}"
                    ),

                    "text": section,

                    "source": pdf_path.name,

                    "page": page["page"],

                    "document_type": document_type,

                    "month": month,

                    "year": 2024 if month else None,
                }
            )

    return chunks


def save_chunks(chunks: list[dict]) -> None:
    """Save chunks to JSON."""

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            chunks,
            file,
            ensure_ascii=False,
            indent=2
        )

    print(f"\nSaved {len(chunks)} chunks to:")
    print(OUTPUT_FILE)


def main():

    pdf_files = list(
        DATA_DIR.glob("*.pdf")
    )

    if not pdf_files:

        print("No PDF files found.")

        return

    all_chunks = []

    for pdf_path in pdf_files:

        chunks = process_pdf(pdf_path)

        print("=" * 80)
        print(f"File: {pdf_path.name}")
        print(f"Document type: {detect_document_type(pdf_path.name)}")
        print(f"Month: {detect_month(pdf_path.name)}")
        print(f"Number of chunks: {len(chunks)}")

        all_chunks.extend(chunks)

    save_chunks(all_chunks)


if __name__ == "__main__":
    main()