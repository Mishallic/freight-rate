"""Time-based validation: two folds, the metrics, and a baseline model.

Run from the repo root:  python src/evaluate.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

from data import distance_band, drop_corrupted, is_corrupted, load_raw, prepare, weight_medians

REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"

# Each fold trains on the months before its test months and predicts the next two months,
# like the real task (train Jan-Oct, predict Nov-Dec). All dates are in 2025.
FOLDS = {
    "A": {"train": [1, 2, 3, 4, 5, 6], "test": [7, 8]},
    "B": {"train": [1, 2, 3, 4, 5, 6, 7, 8], "test": [9, 10]},
}


def split(df, fold):
    """Return the (training, test) rows of one fold, picked by month."""
    month = df["date"].dt.month
    return df[month.isin(FOLDS[fold]["train"])], df[month.isin(FOLDS[fold]["test"])]


def metrics(y_true, y_pred):
    """MAE and RMSE in dollars, MAPE in percent."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    error = y_pred - y_true
    return {
        "MAE": np.mean(np.abs(error)),
        "MAPE": 100 * np.mean(np.abs(error) / y_true),
        "RMSE": np.sqrt(np.mean(error ** 2)),
    }


def baseline_key(df):
    """Distance band and equipment as one label, e.g. "300-600 / Dry Van"."""
    return distance_band(df["distance"]).astype(str) + " / " + df["equipment"]


def fit_baseline(train):
    """Median rate per mile for each distance band and equipment type."""
    rate_per_mile = train["posted_rate"] / train["distance"]
    return rate_per_mile.groupby(baseline_key(train)).median()


def predict_baseline(model, df):
    """Predicted rate = median rate per mile for the load's band and equipment, times distance."""
    return baseline_key(df).map(model) * df["distance"]


def evaluate_fold(train_test, fold):
    train, test = split(train_test, fold)
    medians = weight_medians(train)  # from the training months only
    train = drop_corrupted(prepare(train, medians))
    test = prepare(test, medians)
    predicted = predict_baseline(fit_baseline(train), test)
    clean = ~is_corrupted(test)
    return [
        {"fold": fold, "rows": "all", "test_rows": len(test),
         **metrics(test["posted_rate"], predicted)},
        {"fold": fold, "rows": "clean", "test_rows": clean.sum(),
         **metrics(test.loc[clean, "posted_rate"], predicted[clean])},
    ]


def main():
    train_test, _ = load_raw()
    results = []
    for fold in FOLDS:
        results += evaluate_fold(train_test, fold)
    table = pd.DataFrame(results).set_index(["fold", "rows"]).round(2)
    print(table.to_string())
    table.to_csv(REPORTS_DIR / "metrics_baseline.csv")
    print("saved", REPORTS_DIR / "metrics_baseline.csv")


if __name__ == "__main__":
    main()
