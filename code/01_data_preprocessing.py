"""
Preprocess text data: clean, split, and save processed CSVs.
Input: data/sample/feedback_sample.csv with columns: text,label
Outputs: data/processed/train.csv, data/processed/test.csv
"""
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from code00_helpers import clean_text  # using inline fallback below if not created
from 00_config import DATA_PROCESSED, ROOT

SRC = ROOT / "data" / "sample" / "feedback_sample.csv"
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

def _clean_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.replace(r"\s+", " ", regex=True).str.lower()

def main(test_size=0.2, random_state=42):
    df = pd.read_csv(SRC)
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("Expected columns: text,label")
    df["text"] = _clean_series(df["text"])
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state, stratify=df["label"])
    train_df.to_csv(DATA_PROCESSED / "train.csv", index=False)
    test_df.to_csv(DATA_PROCESSED / "test.csv", index=False)
    print("Saved:", DATA_PROCESSED / "train.csv", DATA_PROCESSED / "test.csv")

if __name__ == "__main__":
    main()

