import pandas as pd
import re

FILE = "data/cleaned_tickets.csv"


def normalize(text):
    """Convert text into clean lowercase words."""
    if pd.isna(text):
        return ""

    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def search_tickets(query):
    """Find tickets relevant to the customer's problem."""

    df = pd.read_csv(FILE)

    query = normalize(query)

    if not query:
        return pd.DataFrame()

    # Split query into individual words
    query_words = set(query.split())

    # Columns that are useful for searching
    search_columns = [
        "Ticket Subject",
        "Ticket Description",
        "Resolution"
    ]

    # Keep only columns that actually exist
    search_columns = [
        column for column in search_columns
        if column in df.columns
    ]

    scores = []

    for index, row in df.iterrows():

        score = 0

        subject = normalize(row.get("Ticket Subject", ""))
        description = normalize(row.get("Ticket Description", ""))
        resolution = normalize(row.get("Resolution", ""))

        # Search in subject
        subject_words = set(subject.split())

        # Search in description
        description_words = set(description.split())

        # Search in resolution
        resolution_words = set(resolution.split())

        # Exact query match gets higher score
        if query in subject:
            score += 10

        if query in description:
            score += 5

        if query in resolution:
            score += 3

        # Individual word matching
        score += len(query_words.intersection(subject_words)) * 4
        score += len(query_words.intersection(description_words)) * 2
        score += len(query_words.intersection(resolution_words))

        scores.append(score)

    df["_score"] = scores

    # Remove tickets with no relevance
    results = df[df["_score"] > 0].copy()

    # Highest relevance first
    results = results.sort_values(
        by="_score",
        ascending=False
    )

    return results.head(10)


def generate_response(query, results):
    """Generate a customer support response using the best matching ticket."""

    if results.empty:
        return (
            "I’m sorry, but I could not find a similar resolved issue "
            "in our support history.\n\n"
            "Please provide more details about the problem so that "
            "we can investigate it further."
        )

    # Prefer closed/resolved tickets
    if "Ticket Status" in results.columns:

        resolved = results[
            results["Ticket Status"]
            .fillna("")
            .astype(str)
            .str.lower()
            .isin(["closed", "resolved"])
        ]

        if not resolved.empty:
            results = resolved

    # Pick the most relevant ticket
    ticket = results.iloc[0]

    subject = str(
        ticket.get("Ticket Subject", "")
    ).strip()

    description = str(
        ticket.get("Ticket Description", "")
    ).strip()

    resolution = str(
        ticket.get("Resolution", "")
    ).strip()

    # Handle missing values
    if subject.lower() == "nan":
        subject = "Similar customer issue"

    if description.lower() == "nan":
        description = ""

    if (
        not resolution
        or resolution.lower() in ["nan", "none", "null"]
    ):
        resolution = (
            "Please contact customer support with your ticket "
            "details so that the issue can be investigated."
        )

    response = (
        f"I understand that you are facing an issue related to "
        f'"{query}".\n\n'
        f"I found a similar support case in our previous tickets.\n\n"
        f"Similar issue:\n"
        f"{subject}\n\n"
        f"Recommended resolution:\n"
        f"{resolution}\n\n"
        f"If this does not resolve your issue, please provide "
        f"additional details about the problem and we can "
        f"investigate it further."
    )

    return response


def print_ticket(ticket):
    """Print information about a matching ticket."""

    print("\n------------------------------")
    print("TICKET ID:", ticket.get("Ticket ID", "N/A"))
    print("SUBJECT:", ticket.get("Ticket Subject", "N/A"))
    print("STATUS:", ticket.get("Ticket Status", "N/A"))
    print("PRIORITY:", ticket.get("Ticket Priority", "N/A"))
    print("RESOLUTION:", ticket.get("Resolution", "N/A"))
    print("------------------------------")


def main():

    query = input("Customer problem: ").strip()

    if not query:
        print("Please enter a customer problem.")
        return

    results = search_tickets(query)

    print(f"\nFound {len(results)} matching tickets.")

    # Show top matching tickets
    if not results.empty:

        print("\n===== TOP MATCHING TICKETS =====")

        for _, ticket in results.head(3).iterrows():
            print_ticket(ticket)

    # Generate support response
    response = generate_response(
        query,
        results
    )

    print("\n========== AI SUPPORT RESPONSE ==========")
    print(response)
    print("=========================================")


if __name__ == "__main__":
    main()