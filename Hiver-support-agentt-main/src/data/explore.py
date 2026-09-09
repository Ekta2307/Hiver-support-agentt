import pandas as pd

FILE_PATH = "data/raw/customer_support_tickets.csv"


def main():
    print("Loading dataset...")

    df = pd.read_csv(FILE_PATH)

    print("\nDataset loaded successfully!")

    print("Rows:", len(df))
    print("Columns:", len(df.columns))

    print("\nColumn names:")
    for column in df.columns:
        print("-", column)

    print("\nFirst 5 rows:")
    print(df.head())


if __name__ == "__main__":
    main()