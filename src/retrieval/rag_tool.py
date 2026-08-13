from enum import Enum
from typing import Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from src.retrieval.search import Retriever
from src.retrieval.query_parser import extract_month


# Canonical document types
class DocumentType(str, Enum):
    POLICY = "policy"
    ROOM = "room"
    IMPROVEMENT = "improvement"
    REVIEW = "review"


# Tool input schema
class SearchHotelKnowledgeInput(BaseModel):

    query: str = Field(
        description=(
            "Semantic search query for the domain knowledge base. "
            "Include the relevant issue, topic, or information."
        )
    )

    month: Optional[str] = Field(
        default=None,
        description=(
            "Optional month or date period mentioned by the user. "
            "Examples: August, July 2024. "
            "Leave empty if no month was requested."
        )
    )

    document_type: Optional[DocumentType] = Field(
        default=None,
        description=(
            "Optional document type. "
            "For hospitality use 'review' for guest reviews, "
            "'improvement' for hotel improvement guidelines, "
            "'policy' for hotel policies, "
            "and 'room' for room information."
        )
    )


# ============================================================
# Tool Factory
# ============================================================
#
# Each domain creates its own RAG tool and provides the
# corresponding Chroma collection.
#
# Example:
# create_search_tool("hotel_knowledge_base")
# create_search_tool("finance_knowledge_base")
# ============================================================

def create_search_tool(collection_name: str):

    # Create a Retriever configured for this domain's
    # Chroma collection.
    retriever = Retriever(
        collection_name=collection_name
    )

    # Create the actual LangChain tool.
    # The collection is already fixed inside this tool.
    # The LLM does NOT choose the collection name.

    @tool(
        args_schema=SearchHotelKnowledgeInput,
    )
    def search_domain_knowledge(
        query: str,
        month: Optional[str] = None,
        document_type: Optional[DocumentType] = None,
    ) -> str:
        """
        Search the selected domain's knowledge base using
        semantic search with optional filters.
        """

        # Normalize month

        month_filter = (
            extract_month(month)
            if month
            else None
        )

        # Convert Enum to canonical database value

        document_type_filter = (
            document_type.value
            if document_type is not None
            else None
        )


        # Retrieve relevant documents
        # The Retriever is already connected to the correct
        # domain-specific Chroma collection.

        results = retriever.retrieve(
            query=query,
            top_k=5,
            month=month_filter,
            document_type=document_type_filter,
        )

        # Handle empty retrieval
        if not results:
            return (
                "No relevant information was found "
                "in the domain knowledge base."
            )

        # Format retrieved documents for the LLM

        formatted_results = []

        for result in results:

            formatted_results.append(
                f"Source: {result['source']}\n"
                f"Page: {result['page']}\n"
                f"Document Type: {result['document_type']}\n"
                f"Month: {result['month']}\n"
                f"Text:\n{result['text']}"
            )

        return "\n\n---\n\n".join(
            formatted_results
        )

    return search_domain_knowledge

