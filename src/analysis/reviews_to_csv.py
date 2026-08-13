from pathlib import Path
import json
import csv
import re
import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


CHUNKS_FILE = Path("data/processed/chunks.json")
OUTPUT_DIR = Path("data/processed")
OUTPUT_FILE = OUTPUT_DIR / "reviews.csv"

load_dotenv()

llm = ChatOpenAI(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
    temperature=0,
)


def extract_json(text: str):
    """
    Extract JSON object from LLM response.
    """

    match = re.search(r"\{.*\}", text, re.DOTALL)

    if not match:
        raise ValueError(
            f"Could not find JSON in response:\n{text}"
        )

    return json.loads(match.group())


def structure_review(review_text: str) -> dict:

    prompt = f"""
You are extracting structured information from a hotel guest review.

Return ONLY valid JSON.

Use exactly this structure:

{{
    "room": null,
    "room_type": null,
    "rating": null,
    "issue_categories": [],
    "sentiment": "Positive"
}}

Rules:

1. ROOM
- Extract the room number only if explicitly mentioned.
- Do not guess the room number.

2. ROOM TYPE
- Extract the room type only if explicitly mentioned.
- Possible values:
  "Standard Room"
  "Deluxe Room"
  "Executive Suite"
  "Family Suite"
  "Suite"
- If not explicitly mentioned, return null.
- Do not infer room type from the room number.

3. RATING
- Extract the numerical rating only if explicitly stated.
- Otherwise return null.

4. ISSUE CATEGORIES

- Extract all distinct complaints explicitly described in the review.
- Each complaint should be assigned the most specific applicable category.
- Do NOT assign multiple categories to the same complaint unless the review clearly describes two separate problems.
- Do NOT use a broader category together with a more specific category for the same complaint.
- Do not infer a separate complaint from the location, department, or staff involved in an existing complaint.
- Every issue category must be directly supported by a distinct complaint in the review text.


Wi-Fi
AC Noise
Air Conditioning
Luggage Delay
Housekeeping
Breakfast
Pool Towels
Check-in
Check-out
Front Desk
Noise
Room Cleanliness
Bathroom
Elevator
Staff
Location
Room Service
Other

Important distinction:

- "Check-in was slow."
  → ["Check-in"]

- "The lobby service during check-in was very poor."
  → ["Check-in"]

- "The front desk staff were rude."
  → ["Front Desk"]

- "Check-in was slow and the front desk staff were rude."
  → ["Check-in", "Front Desk"]

- "The guest waited 20 minutes for luggage and complained about check-in."
  → ["Luggage Delay", "Check-in"]

If the review contains multiple complaints, include ALL relevant categories.

5. SENTIMENT
Classify the overall review as:
- Positive
- Negative
- Mixed

Use "Mixed" when the review contains both positive and negative feedback.

6. IMPORTANT
- Do not invent information.
- Do not infer information that is not explicitly stated.
- Return [] for issue_categories if there are no complaints.

Guest Review:

{review_text}
"""

    response = llm.invoke(prompt)

    return extract_json(response.content)


def main():


    with open(
        CHUNKS_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        chunks = json.load(f)


    review_chunks = []

    for chunk in chunks:

        if chunk.get("document_type") != "review":
            continue

        text = chunk.get("text", "").strip()

        # Ignore empty chunks
        if not text:
            continue

        # Ignore document title/header chunks
        if not re.match(
            r"^Review\s+\d+\s*[–-]",
            text,
            re.IGNORECASE
        ):
            print(
                f"Skipping non-review chunk: "
                f"{chunk.get('chunk_id')}"
            )
            continue

        review_chunks.append(chunk)

    print(
        f"\nFound {len(review_chunks)} actual review chunks."
    )


    rows = []

    for index, chunk in enumerate(
        review_chunks,
        start=1
    ):

        print(
            f"\nProcessing review "
            f"{index}/{len(review_chunks)}"
        )

        structured = structure_review(
            chunk["text"]
        )


        issue_categories = structured.get(
            "issue_categories",
            []
        )

        if not isinstance(issue_categories, list):
            issue_categories = [issue_categories]

        issue_categories = [
            str(issue).strip()
            for issue in issue_categories
            if str(issue).strip()
        ]

        # Remove duplicates while preserving order
        issue_categories = list(
            dict.fromkeys(issue_categories)
        )


        row = {
            "review_id": index,
            "month": chunk.get("month"),
            "year": chunk.get("year"),
            "room": structured.get("room"),
            "room_type": structured.get("room_type"),
            "rating": structured.get("rating"),
            "issue_categories": ";".join(
                issue_categories
            ),
            "sentiment": structured.get(
                "sentiment"
            ),
            "review_text": chunk["text"],
        }

        rows.append(row)


    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    fieldnames = [
        "review_id",
        "month",
        "year",
        "room",
        "room_type",
        "rating",
        "issue_categories",
        "sentiment",
        "review_text",
    ]

    with open(
        OUTPUT_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(rows)

    print("\n" + "=" * 80)
    print(
        f"Saved {len(rows)} reviews."
    )
    print(
        f"CSV: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()