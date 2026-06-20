"""
prepare_data.py
Cleans the Kaggle SQLiV3.csv into a tidy training file.

The raw file is messy: some queries contain commas, which pushed their
label into extra 'Unnamed' columns. This script recovers the correct
label, drops blanks and duplicates, and saves data/clean_dataset.csv
with two clean columns: query, label.

Run from your project root:  python prepare_data.py
"""

import numpy as np
import pandas as pd

RAW = "data/SQLiV3.csv"
OUT = "data/clean_dataset.csv"


def pick_label(row):
    """The real label is 0 or 1. Because of the comma mess it might sit in
    'Label', 'Unnamed: 2', or 'Unnamed: 3' — take the first valid 0/1."""
    for col in ["Label", "Unnamed: 2", "Unnamed: 3"]:
        if col in row:
            s = str(row[col]).strip()
            if s in ("0", "0.0"):
                return 0
            if s in ("1", "1.0"):
                return 1
    return np.nan


def main():
    df = pd.read_csv(RAW, encoding="utf-8", on_bad_lines="skip")
    print("Loaded raw file:", df.shape)

    # Recover the correct label
    df["label"] = df.apply(pick_label, axis=1)

    # Keep only the query text and the clean label
    clean = df[["Sentence", "label"]].copy()
    clean.columns = ["query", "label"]

    # Drop rows with no valid label or empty query
    clean = clean.dropna(subset=["label", "query"])
    clean["label"] = clean["label"].astype(int)
    clean["query"] = clean["query"].astype(str).str.strip()
    clean = clean[clean["query"] != ""]

    # Remove duplicate queries
    before = len(clean)
    clean = clean.drop_duplicates(subset=["query"]).reset_index(drop=True)
    print(f"Removed {before - len(clean)} duplicate rows")

    # Save
    clean.to_csv(OUT, index=False, encoding="utf-8")

    print("\nSaved clean dataset to:", OUT)
    print("Final shape:", clean.shape)
    print("\nLabel balance:")
    print(clean["label"].value_counts().rename({0: "safe (0)", 1: "attack (1)"}))
    print("\nExample attacks:")
    for q in clean[clean["label"] == 1]["query"].head(3):
        print("  ", q[:70])
    print("Example safe inputs:")
    for q in clean[clean["label"] == 0]["query"].head(3):
        print("  ", q[:70])


if __name__ == "__main__":
    main()
