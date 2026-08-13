from pathlib import Path
import pandas as pd


CSV_FILE = Path("data/processed/reviews.csv")

# Load reviews
def load_reviews():
    """
    Load the structured guest reviews from CSV.
    """

    return pd.read_csv(CSV_FILE)

# 1. Top Issues
def get_top_issues(
    month: str,
    top_n: int = 3
):
    """
    Return the most common guest complaint categories
    for a specific month.
    """

    df = load_reviews()

    # Filter by month
    df = df[
        df["month"].fillna("").str.lower()
        == month.lower()
    ]

    # Remove reviews with no issues
    df = df[
        df["issue_categories"].notna()
        & (df["issue_categories"].str.strip() != "")
    ]

    # Split multiple issues in the same review
    issues = (
        df["issue_categories"]
        .str.split(";")
        .explode()
        .str.strip()
    )

    # Count issues
    issue_counts = (
        issues
        .value_counts()
        .head(top_n)
    )

    return [
        {
            "issue": issue,
            "count": int(count)
        }
        for issue, count in issue_counts.items()
    ]


# 2. Issue Frequency
def get_issue_frequency(
    issue: str,
    month: str | None = None
):
    """
    Count how many reviews mention a specific issue.

    If month is provided, only reviews from that month
    are analyzed.
    """

    df = load_reviews()

    # Optional month filter
    if month:
        df = df[
            df["month"].fillna("").str.lower()
            == month.lower()
        ]

    # Remove empty issue categories
    df = df[
        df["issue_categories"].notna()
        & (df["issue_categories"].str.strip() != "")
    ]

    # Split multiple issues
    issues = (
        df["issue_categories"]
        .str.split(";")
        .explode()
        .str.strip()
    )

    # Count the requested issue
    count = (
        issues.str.lower()
        == issue.lower()
    ).sum()

    return {
        "issue": issue,
        "month": month,
        "count": int(count)
    }


# 3. Monthly Issue Comparison
def get_monthly_issue_comparison(
    issue: str
):
    """
    Compare the frequency of a specific issue
    across all available months.
    """

    df = load_reviews()

    # Remove empty issue categories
    df = df[
        df["issue_categories"].notna()
        & (df["issue_categories"].str.strip() != "")
    ]

    # Split multiple issues
    exploded = (
        df.assign(
            issue_category=df["issue_categories"]
            .str.split(";")
        )
        .explode("issue_category")
    )

    exploded["issue_category"] = (
        exploded["issue_category"]
        .str.strip()
    )

    # Filter requested issue
    issue_rows = exploded[
        exploded["issue_category"].str.lower()
        == issue.lower()
    ]

    # Count by month
    monthly_counts = (
        issue_rows["month"]
        .fillna("Unknown")
        .value_counts()
    )

    return [
        {
            "month": month,
            "count": int(count)
        }
        for month, count in monthly_counts.items()
    ]


# 4. Sentiment Distribution
def get_sentiment_distribution(
    month: str | None = None
):
    """
    Return the distribution of Positive, Negative,
    and Mixed reviews.

    If month is provided, only reviews from that month
    are analyzed.
    """

    df = load_reviews()

    # Optional month filter
    if month:
        df = df[
            df["month"].fillna("").str.lower()
            == month.lower()
        ]

    # Remove empty sentiments
    df = df[
        df["sentiment"].notna()
        & (df["sentiment"].str.strip() != "")
    ]

    sentiment_counts = (
        df["sentiment"]
        .str.strip()
        .value_counts()
    )

    return {
        sentiment: int(count)
        for sentiment, count
        in sentiment_counts.items()
    }


# 5. Rating Analysis
def get_rating_analysis(
    month: str | None = None
):
    """
    Analyze guest ratings.

    Returns:
    - average rating
    - number of rated reviews
    - minimum rating
    - maximum rating
    """

    df = load_reviews()

    # Optional month filter
    if month:
        df = df[
            df["month"].fillna("").str.lower()
            == month.lower()
        ]

    # Convert rating to numeric
    ratings = pd.to_numeric(
        df["rating"],
        errors="coerce"
    ).dropna()

    # No ratings available
    if ratings.empty:
        return {
            "month": month,
            "average_rating": None,
            "rated_reviews": 0,
            "minimum_rating": None,
            "maximum_rating": None,
        }

    return {
        "month": month,
        "average_rating": round(
            float(ratings.mean()),
            2
        ),
        "rated_reviews": int(
            ratings.count()
        ),
        "minimum_rating": float(
            ratings.min()
        ),
        "maximum_rating": float(
            ratings.max()
        ),
    }


# Helper: Filter by month
def _filter_by_month(df, month=None):
    """
    Filter reviews by month if a month is provided.
    """

    if month:
        df = df[
            df["month"].fillna("").str.lower().str.strip()
            == month.lower().strip()
        ]

    return df


# Helper: Explode issue categories
def _explode_issues(df):
    """
    Split multiple issue categories in each review
    into separate rows.
    """

    df = df[
        df["issue_categories"].notna()
        & (df["issue_categories"].str.strip() != "")
    ].copy()

    if df.empty:
        return df

    df["issue_category"] = (
        df["issue_categories"]
        .str.split(";")
    )

    df = df.explode("issue_category")

    df["issue_category"] = (
        df["issue_category"]
        .str.strip()
    )

    return df


# 6. Top Issues By Month
def get_top_issues_by_month(
    months: list[str] | None = None,
    top_n: int = 3
):
    """
    Return the most common complaint categories
    for each month.

    If months is provided, only those months are analyzed.
    """

    df = load_reviews()

    # Optional month filter
    if months:
        normalized_months = [
            month.lower().strip()
            for month in months
        ]

        df = df[
            df["month"]
            .fillna("")
            .str.lower()
            .str.strip()
            .isin(normalized_months)
        ]

    exploded = _explode_issues(df)

    if exploded.empty:
        return {}

    results = {}

    for month, month_df in exploded.groupby("month"):

        issue_counts = (
            month_df["issue_category"]
            .value_counts()
            .head(top_n)
        )

        results[month] = [
            {
                "issue": issue,
                "count": int(count)
            }
            for issue, count in issue_counts.items()
        ]

    return results


# 7. Rating Distribution
def get_rating_distribution(
    month: str | None = None
):
    """
    Return the number of reviews for each rating.
    """

    df = load_reviews()

    df = _filter_by_month(df, month)

    ratings = pd.to_numeric(
        df["rating"],
        errors="coerce"
    ).dropna()

    if ratings.empty:
        return {
            "month": month,
            "distribution": {}
        }

    distribution = (
        ratings
        .value_counts()
        .sort_index()
    )

    return {
        "month": month,
        "distribution": {
            str(rating): int(count)
            for rating, count
            in distribution.items()
        }
    }


# 8. Issue + Sentiment
def get_issue_sentiment(
    issue: str,
    month: str | None = None
):
    """
    Return the sentiment distribution for reviews
    mentioning a specific issue.
    """

    df = load_reviews()

    df = _filter_by_month(df, month)

    exploded = _explode_issues(df)

    if exploded.empty:
        return {
            "issue": issue,
            "month": month,
            "sentiment_distribution": {}
        }

    # Filter requested issue
    issue_df = exploded[
        exploded["issue_category"]
        .str.lower()
        .str.strip()
        == issue.lower().strip()
    ]

    if issue_df.empty:
        return {
            "issue": issue,
            "month": month,
            "sentiment_distribution": {}
        }

    # Remove empty sentiments
    issue_df = issue_df[
        issue_df["sentiment"].notna()
        & (issue_df["sentiment"].str.strip() != "")
    ]

    sentiment_counts = (
        issue_df["sentiment"]
        .str.strip()
        .value_counts()
    )

    return {
        "issue": issue,
        "month": month,
        "sentiment_distribution": {
            sentiment: int(count)
            for sentiment, count
            in sentiment_counts.items()
        }
    }


# 9. Room Complaints
def get_room_complaints(
    month: str | None = None
):
    """
    Return the rooms with the highest number of
    complaint issues.
    """

    df = load_reviews()

    df = _filter_by_month(df, month)

    exploded = _explode_issues(df)

    if exploded.empty:
        return []

    # Count complaints per room.
    #
    # Assumes the CSV contains a "room" column.
    room_counts = (
        exploded["room"]
        .dropna()
        .astype(str)
        .str.strip()
        .value_counts()
    )

    return [
        {
            "room": room,
            "complaint_count": int(count)
        }
        for room, count in room_counts.items()
    ]


# 10. Room Type Analysis
def get_room_type_analysis(
    month: str | None = None
):
    """
    Return the number of complaints for each room type.

    Assumes the CSV contains a "room_type" column.
    """

    df = load_reviews()

    df = _filter_by_month(df, month)

    exploded = _explode_issues(df)

    if exploded.empty:
        return []

    room_type_counts = (
        exploded["room_type"]
        .dropna()
        .astype(str)
        .str.strip()
        .value_counts()
    )

    return [
        {
            "room_type": room_type,
            "complaint_count": int(count)
        }
        for room_type, count
        in room_type_counts.items()
    ]


# 11. Review Count
def get_review_count(
    month: str | None = None
):
    """
    Return the number of reviews.
    """

    df = load_reviews()

    df = _filter_by_month(df, month)

    return {
        "month": month,
        "review_count": int(len(df))
    }


# 12. Complaint Rate
def get_complaint_rate(
    month: str | None = None
):
    """
    Return the percentage of reviews that contain
    at least one complaint issue.

    A review is considered a complaint review if
    issue_categories is not empty.
    """

    df = load_reviews()

    df = _filter_by_month(df, month)

    total_reviews = len(df)

    if total_reviews == 0:
        return {
            "month": month,
            "total_reviews": 0,
            "complaint_reviews": 0,
            "complaint_rate": 0.0
        }

    complaint_reviews = (
        df["issue_categories"].notna()
        & (df["issue_categories"].str.strip() != "")
    ).sum()

    complaint_rate = (
        complaint_reviews / total_reviews
    ) * 100

    return {
        "month": month,
        "total_reviews": int(total_reviews),
        "complaint_reviews": int(complaint_reviews),
        "complaint_rate": round(
            float(complaint_rate),
            2
        )
    }


def main():

    print("\n" + "=" * 80)
    print("1. TOP ISSUES")
    print("=" * 80)

    print(
        get_top_issues(
            month="August",
            top_n=3
        )
    )


    print("\n" + "=" * 80)
    print("2. ISSUE FREQUENCY")
    print("=" * 80)

    print(
        get_issue_frequency(
            issue="Wi-Fi",
            month="August"
        )
    )


    print("\n" + "=" * 80)
    print("3. MONTHLY ISSUE COMPARISON")
    print("=" * 80)

    print(
        get_monthly_issue_comparison(
            issue="Wi-Fi"
        )
    )


    print("\n" + "=" * 80)
    print("4. SENTIMENT DISTRIBUTION")
    print("=" * 80)

    print(
        get_sentiment_distribution(
            month="August"
        )
    )


    print("\n" + "=" * 80)
    print("5. RATING ANALYSIS")
    print("=" * 80)

    print(
        get_rating_analysis(
            month="August"
        )
    )


if __name__ == "__main__":
    main()