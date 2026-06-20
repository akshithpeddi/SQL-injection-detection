"""
explore_data.py
Quick look at the SQL injection datasets to see what's inside each file:
how many rows, what the columns are, a few example rows, and the
balance of safe vs attack labels.
Run from your project root:  python explore_data.py
"""

import os
import pandas as pd

DATA_DIR = "data"
FILES = ["sqli.csv", "sqliv2.csv", "SQLiV3.csv"]


def load_csv(path):
    """Try a few encodings, since these files are known to be messy."""
    for enc in ["utf-8", "utf-16", "latin-1"]:
        try:
            df = pd.read_csv(path, encoding=enc, on_bad_lines="skip")
            print(f"  loaded with encoding: {enc}")
            return df
        except Exception as e:
            print(f"  failed with {enc}: {str(e)[:70]}")
    return None


for f in FILES:
    path = os.path.join(DATA_DIR, f)
    print("=" * 60)
    print("FILE:", f)

    if not os.path.exists(path):
        print("  NOT FOUND at", path)
        continue

    df = load_csv(path)
    if df is None:
        print("  could not read this file")
        continue

    print("  rows, columns:", df.shape)
    print("  column names:", list(df.columns))
    print("  --- first 3 rows ---")
    print(df.head(3).to_string(max_colwidth=60))

    # Look for a label column and show the safe/attack split
    for col in df.columns:
        if str(col).strip().lower() in ("label", "class", "target"):
            print(f"  --- label counts in '{col}' ---")
            print(df[col].value_counts(dropna=False).to_string())

print("=" * 60)
print("Done.")
