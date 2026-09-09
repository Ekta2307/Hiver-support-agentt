import pandas as pd


INPUT_FILE = "data/raw/customer_support_tickets.csv"
OUTPUT_FILE = "data/cleaned_tickets.csv"


def main():
    df = pd.read_csv(INPUT_FILE)

    print("Original shape:", df.shape)

    # Remove completely empty rows
    df = df.dropna(how="all")

    # Remove duplicate tickets
    df = df.drop_duplicates(subset=["Ticket ID"])

    # Remove extra spaces from column names
    df.columns = df.columns.str.strip()

    # Clean string columns
    for column in df.select_dtypes(include="object").columns:
        df[column] = df[column].astype(str).str.strip()

    # Convert date columns
    df["Date of Purchase"] = pd.to_datetime(
        df["Date of Purchase"], errors="coerce"
    )

    df["First Response Time"] = pd.to_datetime(
        df["First Response Time"], errors="coerce"
    )

    df["Time to Resolution"] = pd.to_datetime(
        df["Time to Resolution"], errors="coerce"
    )

    # Save cleaned data
    df.to_csv(OUTPUT_FILE, index=False)

    print("Cleaned shape:", df.shape)
    print("Cleaned file saved to:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
    