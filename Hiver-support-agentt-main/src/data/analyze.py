import pandas as pd

FILE = "data/cleaned_tickets.csv"


def main():
    df = pd.read_csv(FILE)

    print("\n=== DATASET INFO ===")
    print("Total tickets:", len(df))

    print("\n=== TICKET STATUS ===")
    print(df["Ticket Status"].value_counts())

    print("\n=== TICKET PRIORITY ===")
    print(df["Ticket Priority"].value_counts())

    print("\n=== TICKET TYPE ===")
    print(df["Ticket Type"].value_counts())

    print("\n=== TICKET CHANNEL ===")
    print(df["Ticket Channel"].value_counts())

    print("\n=== TOP TICKET SUBJECTS ===")
    print(df["Ticket Subject"].value_counts().head(10))


if __name__ == "__main__":
    main()