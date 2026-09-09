import pandas as pd

FILE = "data/cleaned_tickets.csv"


def main():
    df = pd.read_csv(FILE)

    print("\n=== SUPPORT INSIGHTS ===")

    # Most common ticket subject
    print("\nMost common issues:")
    print(df["Ticket Subject"].value_counts().head(10))

    # Status
    print("\nTicket status:")
    print(df["Ticket Status"].value_counts())

    # Priority
    print("\nTicket priority:")
    print(df["Ticket Priority"].value_counts())

    # Customer satisfaction
    print("\nCustomer satisfaction:")
    print(df["Customer Satisfaction Rating"].value_counts().sort_index())

    # Missing values
    print("\nMissing values:")
    print(df.isnull().sum())


if __name__ == "__main__":
    main()