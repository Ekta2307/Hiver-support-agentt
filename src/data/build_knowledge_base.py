"""
Extracts and indexes high-quality historical customer service conversations for @AmazonHelp
from the Twitter Customer Support dataset (twcs.csv).
Outputs a compact, fast-loading knowledge base of historical customer-agent resolution pairs.
"""

import os
import re
import sys
import json
import logging
from pathlib import Path
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CACHE_SNAPSHOT_PATH = r"C:\Users\Administrator\.cache\huggingface\hub\datasets--SunidhiSriram--twcs\snapshots\b03fa0a7d338f210f7bae9ff9fdbe0f36d62a8b4\twcs.csv"
OUTPUT_KB_PATH = Path("data/amazon_knowledge_base.jsonl")


def find_twcs_path(explicit_path=None):
    if explicit_path and os.path.exists(explicit_path):
        return explicit_path
    if os.path.exists(CACHE_SNAPSHOT_PATH):
        return CACHE_SNAPSHOT_PATH
    
    # Check default huggingface cache directory dynamically
    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    hub_dir = Path(hf_home) / "hub"
    for p in hub_dir.glob("**/twcs.csv"):
        if p.is_file():
            return str(p)
            
    # Try downloading if huggingface_hub is installed
    try:
        from huggingface_hub import hf_hub_download
        logger.info("twcs.csv not found locally. Downloading from Hugging Face...")
        downloaded = hf_hub_download(repo_id="SunidhiSriram/twcs", filename="twcs.csv", repo_type="dataset")
        return downloaded
    except Exception as e:
        logger.error(f"Failed to auto-download twcs.csv: {e}")
        raise FileNotFoundError("Could not find or download twcs.csv.")


def is_english(text):
    if not isinstance(text, str) or len(text.strip()) < 15:
        return False
    # Check ASCII printable ratio to filter out non-Latin / non-English tweets
    ascii_count = sum(1 for c in text if ord(c) < 128)
    if (ascii_count / len(text)) < 0.85:
        return False
    
    # Common non-English stop word check
    text_lower = text.lower()
    foreign_markers = [" und ", " für ", " bitte ", " para ", " por ", " qué ", " vous ", " avec ", " hola ", " danke "]
    if any(marker in text_lower for marker in foreign_markers):
        return False
    return True


def clean_tweet_text(text):
    if not isinstance(text, str):
        return ""
    # Remove usernames like @115712, @AmazonHelp
    text = re.sub(r"@[A-Za-z0-9_]+", "", text)
    # Remove URLs for clean matching, or normalize them
    text = re.sub(r"https?://\S+", "", text)
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_amazon_conversations(csv_path, max_rows=600000, target_kb_size=5000):
    logger.info(f"Reading {csv_path} (scanning up to {max_rows} rows)...")
    chunks = []
    total_scanned = 0

    for chunk in pd.read_csv(csv_path, chunksize=100000, low_memory=False):
        # Filter rows belonging to Amazon or customer inbound
        subset = chunk[(chunk["author_id"] == "AmazonHelp") | (chunk["inbound"] == True)]
        chunks.append(subset)
        total_scanned += len(chunk)
        logger.info(f"Scanned {total_scanned} rows... accumulated {sum(len(c) for c in chunks)} candidate records")
        if total_scanned >= max_rows:
            break

    df = pd.concat(chunks, ignore_index=True)
    logger.info(f"Total candidate rows loaded: {len(df)}")

    # Isolate AmazonHelp replies and customer inbound tweets
    amazon_replies = df[df["author_id"] == "AmazonHelp"].copy()
    customer_tweets = df[df["inbound"] == True].set_index("tweet_id")

    logger.info(f"AmazonHelp replies: {len(amazon_replies)}, Customer inbound tweets: {len(customer_tweets)}")

    # Merge customer tweet with Amazon reply
    merged = amazon_replies.merge(
        customer_tweets[["text", "author_id", "created_at"]],
        left_on="in_response_to_tweet_id",
        right_index=True,
        suffixes=("_reply", "_customer")
    )
    logger.info(f"Merged conversation pairs: {len(merged)}")

    # Filter for English
    merged = merged[merged["text_customer"].apply(is_english) & merged["text_reply"].apply(is_english)].copy()
    logger.info(f"English pairs: {len(merged)}")

    # Clean text
    merged["clean_customer"] = merged["text_customer"].apply(clean_tweet_text)
    merged["clean_reply"] = merged["text_reply"].apply(clean_tweet_text)

    # Filter out empty or too-short tweets
    merged = merged[(merged["clean_customer"].str.len() >= 20) & (merged["clean_reply"].str.len() >= 20)]
    # Drop duplicates
    merged = merged.drop_duplicates(subset=["clean_customer"])

    # Sample top historical resolution pairs for knowledge base
    kb_subset = merged.head(target_kb_size)
    logger.info(f"Extracted {len(kb_subset)} high-quality knowledge base exemplar pairs.")

    records = []
    for _, row in kb_subset.iterrows():
        records.append({
            "pair_id": int(row["tweet_id"]),
            "customer_tweet_id": int(row["in_response_to_tweet_id"]) if pd.notna(row["in_response_to_tweet_id"]) else None,
            "customer_text": row["text_customer"].strip(),
            "customer_text_clean": row["clean_customer"],
            "reply_text": row["text_reply"].strip(),
            "reply_text_clean": row["clean_reply"],
            "created_at": str(row.get("created_at_customer", ""))
        })

    return records


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Build AmazonHelp Historical Knowledge Base")
    parser.add_argument("--csv", type=str, default=None, help="Path to twcs.csv")
    parser.add_argument("--size", type=int, default=5000, help="Target KB size")
    args = parser.parse_args()

    csv_path = find_twcs_path(args.csv)
    logger.info(f"Using dataset from: {csv_path}")

    kb_records = extract_amazon_conversations(csv_path, target_kb_size=args.size)

    OUTPUT_KB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_KB_PATH, "w", encoding="utf-8") as f:
        for r in kb_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    logger.info(f"Successfully saved {len(kb_records)} knowledge base records to {OUTPUT_KB_PATH}")


if __name__ == "__main__":
    main()
