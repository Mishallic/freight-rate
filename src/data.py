"""Load the raw freight data."""

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_raw():
    """Return (train_test, validation) DataFrames with `date` parsed."""
    train_test = pd.read_csv(DATA_DIR / "train_test.csv", parse_dates=["date"])
    validation = pd.read_csv(DATA_DIR / "validation.csv", parse_dates=["date"])
    return train_test, validation


if __name__ == "__main__":
    train_test, validation = load_raw()
    for name, df in [("train_test", train_test), ("validation", validation)]:
        print(f"{name}: {len(df):,} rows, {df['date'].min().date()} to {df['date'].max().date()}")
