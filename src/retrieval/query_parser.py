import re


MONTHS = {
    "january": "January",
    "february": "February",
    "march": "March",
    "april": "April",
    "may": "May",
    "june": "June",
    "july": "July",
    "august": "August",
    "september": "September",
    "october": "October",
    "november": "November",
    "december": "December",
}


def extract_month(query: str) -> str | None:
    query_lower = query.lower()

    for month_name, month_value in MONTHS.items():

        if re.search(rf"\b{month_name}\b", query_lower):
            return month_value

    return None


if __name__ == "__main__":

    queries = [
        "What problems did guests report in August?",
        "What happened with Wi-Fi?",
        "What were the complaints in July?",
    ]

    for query in queries:
        print(query)
        print("Month:", extract_month(query))
        print("-" * 50)