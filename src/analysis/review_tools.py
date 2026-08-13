from enum import Enum
from typing import Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from src.analysis.review_analyzer import (
    get_top_issues,
    get_issue_frequency,
    get_monthly_issue_comparison,
    get_top_issues_by_month,
    get_rating_analysis,
    get_rating_distribution,
    get_sentiment_distribution,
    get_issue_sentiment,
    get_room_complaints,
    get_room_type_analysis,
    get_review_count,
    get_complaint_rate,
)


class AnalysisType(str, Enum):

    TOP_ISSUES = "top_issues"
    ISSUE_FREQUENCY = "issue_frequency"
    MONTHLY_COMPARISON = "monthly_comparison"
    TOP_ISSUES_BY_MONTH = "top_issues_by_month"

    RATING = "rating"
    RATING_DISTRIBUTION = "rating_distribution"

    SENTIMENT = "sentiment"
    ISSUE_SENTIMENT = "issue_sentiment"

    ROOM_COMPLAINTS = "room_complaints"
    ROOM_TYPE = "room_type"

    REVIEW_COUNT = "review_count"
    COMPLAINT_RATE = "complaint_rate"


# Tool Input Schema
class AnalyzeReviewsInput(BaseModel):

    analysis_type: AnalysisType = Field(
        description=(
            "Type of review analysis to perform. "
            "Use 'top_issues' for top complaints in one month. "
            "Use 'issue_frequency' to count a specific issue. "
            "Use 'monthly_comparison' to compare an issue across months. "
            "Use 'top_issues_by_month' to find top issues for each month. "
            "Use 'rating' for average/min/max rating. "
            "Use 'rating_distribution' for rating counts. "
            "Use 'sentiment' for overall sentiment distribution. "
            "Use 'issue_sentiment' for sentiment of a specific issue. "
            "Use 'room_complaints' to find rooms with most complaints. "
            "Use 'room_type' to compare complaints by room type. "
            "Use 'review_count' to count reviews. "
            "Use 'complaint_rate' to calculate percentage of reviews "
            "with complaints."
        )
    )

    month: Optional[str] = Field(
        default=None,
        description=(
            "Optional month to analyze, such as August or July 2024."
        )
    )

    months: Optional[list[str]] = Field(
        default=None,
        description=(
            "Optional list of months for top_issues_by_month analysis."
        )
    )

    issue: Optional[str] = Field(
        default=None,
        description=(
            "Specific complaint category such as Wi-Fi, AC Noise, "
            "or Check-in."
        )
    )

    top_n: Optional[int] = Field(
        default=None,
        description=(
            "Number of top complaint categories to return."
        )
    )



# Review Analysis Tool
@tool(args_schema=AnalyzeReviewsInput)
def analyze_reviews(
    analysis_type: AnalysisType,
    month: Optional[str] = None,
    months: Optional[list[str]] = None,
    issue: Optional[str] = None,
    top_n: Optional[int] = None,
) -> str:
    """
    Analyze structured hotel guest reviews.

    This tool performs calculations over the review dataset,
    including complaint analysis, sentiment analysis,
    rating analysis, room analysis, review counts,
    and complaint rates.
    """


    # 1. Top Issues
    if analysis_type == AnalysisType.TOP_ISSUES:

        if not month:
            return (
                "A month is required for "
                "top_issues analysis."
            )

        if top_n is None:
            top_n = 3

        result = get_top_issues(
            month=month,
            top_n=top_n,
        )

        return str(result)

    # 2. Issue Frequency
    if analysis_type == AnalysisType.ISSUE_FREQUENCY:

        if not issue:
            return (
                "An issue is required for "
                "issue_frequency analysis."
            )

        result = get_issue_frequency(
            issue=issue,
            month=month,
        )

        return str(result)

    # 3. Monthly Issue Comparison
    if analysis_type == AnalysisType.MONTHLY_COMPARISON:

        if not issue:
            return (
                "An issue is required for "
                "monthly_comparison analysis."
            )

        result = get_monthly_issue_comparison(
            issue=issue,
        )

        return str(result)

    # 4. Top Issues By Month
    if analysis_type == AnalysisType.TOP_ISSUES_BY_MONTH:

        if top_n is None:
            top_n = 3

        result = get_top_issues_by_month(
            months=months,
            top_n=top_n,
        )

        return str(result)

    # 5. Rating Analysis
    if analysis_type == AnalysisType.RATING:

        result = get_rating_analysis(
            month=month,
        )

        return str(result)

    # 6. Rating Distribution
    if analysis_type == AnalysisType.RATING_DISTRIBUTION:

        result = get_rating_distribution(
            month=month,
        )

        return str(result)

    # 7. Sentiment Distribution
    if analysis_type == AnalysisType.SENTIMENT:

        result = get_sentiment_distribution(
            month=month,
        )

        return str(result)


    # 8. Issue + Sentiment
    if analysis_type == AnalysisType.ISSUE_SENTIMENT:

        if not issue:
            return (
                "An issue is required for "
                "issue_sentiment analysis."
            )

        result = get_issue_sentiment(
            issue=issue,
            month=month,
        )

        return str(result)

    # 9. Room Complaints
    if analysis_type == AnalysisType.ROOM_COMPLAINTS:

        result = get_room_complaints(
            month=month,
        )

        return str(result)

    # 10. Room Type Analysis
    if analysis_type == AnalysisType.ROOM_TYPE:

        result = get_room_type_analysis(
            month=month,
        )

        return str(result)

    # 11. Review Count
    if analysis_type == AnalysisType.REVIEW_COUNT:

        result = get_review_count(
            month=month,
        )

        return str(result)

    # 12. Complaint Rate
    if analysis_type == AnalysisType.COMPLAINT_RATE:

        result = get_complaint_rate(
            month=month,
        )

        return str(result)

    # Unsupported Analysis
    return "Unsupported analysis type."


if __name__ == "__main__":

    result = analyze_reviews.invoke(
        {
            "analysis_type": "top_issues",
            "month": "August",
            "top_n": 3,
        }
    )

    print("\n" + "=" * 80)
    print("REVIEW ANALYSIS RESULT:")
    print(result)

