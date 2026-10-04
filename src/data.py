"""Load the raw freight data and flag loads whose posted rate looks corrupted."""

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Distance bands in miles. A band includes its lower edge: 150 miles falls in "150-300".
DISTANCE_BANDS = [0, 150, 300, 600, 1000, 2000, np.inf]
BAND_LABELS = ["0-150", "150-300", "300-600", "600-1000", "1000-2000", "2000+"]

# A rate per mile below RATIO_LOW or above RATIO_HIGH times the median for its
# equipment and distance band is treated as corrupted. Normal loads sit at about 0.8-1.3.
RATIO_LOW, RATIO_HIGH = 0.6, 1.8


def load_raw():
    """Return (train_test, validation) DataFrames with `date` parsed."""
    train_test = pd.read_csv(DATA_DIR / "train_test.csv", parse_dates=["date"])
    validation = pd.read_csv(DATA_DIR / "validation.csv", parse_dates=["date"])
    return train_test, validation


def distance_band(distance):
    return pd.cut(distance, DISTANCE_BANDS, labels=BAND_LABELS, right=False)


def rate_ratio(df):
    """Rate per mile divided by the median rate per mile of loads with the same
    equipment and distance band. About 1.0 for a normal load."""
    rate_per_mile = df["posted_rate"] / df["distance"]
    groups = [df["equipment"], distance_band(df["distance"])]
    typical = rate_per_mile.groupby(groups, observed=True).transform("median")
    return rate_per_mile / typical


def is_corrupted(df):
    """True for loads whose rate is far from the typical rate for their equipment and distance."""
    ratio = rate_ratio(df)
    return (ratio < RATIO_LOW) | (ratio > RATIO_HIGH)


if __name__ == "__main__":
    train_test, validation = load_raw()
    for name, df in [("train_test", train_test), ("validation", validation)]:
        print(f"{name}: {len(df):,} rows, {df['date'].min().date()} to {df['date'].max().date()}")
