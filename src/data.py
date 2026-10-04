"""Load and clean the freight data.

prepare() cleans a file the same way for training and for prediction, so the two can
never differ. drop_corrupted() removes loads with a corrupted rate; use it for training only.

quote_signal stays in the data but is NOT a model feature. In most labelled months it is
a near-copy of the rate per mile (mirrored in Apr, May, Jul and Oct), but in August and
in the whole validation period (Nov-Dec) it carries no rate information. Trained on
Jan-Jul and tested on August, a model that uses it has 4.65% clean MAPE against 1.82%
without it (see compare.py).
"""

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

WEIGHT_CAP = 47_500  # lbs; many loads sit exactly here, so their real weight was clipped

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


def weight_medians(df):
    """Median weight per equipment type, after abs(). Compute it from training rows only."""
    return df["weight"].abs().groupby(df["equipment"]).median()


def prepare(df, medians):
    """Clean one file for training or prediction. Never drops rows.

    `medians` is the output of weight_medians() on the training rows.
    """
    df = df.copy()  # leave the caller's DataFrame unchanged
    df["weight"] = df["weight"].abs()  # negative weights are sign flips
    df["weight_capped"] = df["weight"] == WEIGHT_CAP
    df["weight_missing"] = df["weight"].isna()
    df["weight"] = df["weight"].fillna(df["equipment"].map(medians))
    # market_index is one value per day plus small noise, so the day's mean is a good fill.
    day_mean = df.groupby("date")["market_index"].transform("mean")
    df["market_index"] = df["market_index"].fillna(day_mean)
    return df


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


def drop_corrupted(df):
    """Remove loads with a corrupted rate. Training only: prediction rows are never dropped."""
    return df[~is_corrupted(df)]


if __name__ == "__main__":
    # Before/after summary of the cleaning, with weight medians from all of train_test.
    train_test, validation = load_raw()
    medians = weight_medians(train_test)
    print("weight medians used to fill (from train_test):", medians.to_dict())
    for name, raw in [("train_test", train_test), ("validation", validation)]:
        clean = prepare(raw, medians)
        print(f"\n{name}: {len(raw):,} rows in, {raw['date'].min().date()} to {raw['date'].max().date()}")
        print(f"  weight: {(raw['weight'] < 0).sum()} negative made positive, "
              f"{clean['weight_capped'].sum():,} flagged as capped at {WEIGHT_CAP:,}, "
              f"{clean['weight_missing'].sum()} missing filled with the equipment median")
        print(f"  market_index: {raw['market_index'].isna().sum()} missing filled with the day's mean")
        print(f"  rows out of prepare(): {len(clean):,}; missing values left: {clean.isna().sum().sum()}")
    training = drop_corrupted(prepare(train_test, medians))
    print(f"\ntraining rows: {len(train_test):,} in, {len(train_test) - len(training)} dropped as corrupted, "
          f"{len(training):,} left")
