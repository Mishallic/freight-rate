"""Tests for the validation folds and the metrics in src/evaluate.py."""

import pytest

from data import load_raw
from evaluate import FOLDS, metrics, split


def test_folds_do_not_overlap_and_test_comes_after_training():
    train_test, _ = load_raw()
    for fold in FOLDS:
        train, test = split(train_test, fold)
        assert set(train["load_id"]).isdisjoint(test["load_id"])
        assert train["date"].max() < test["date"].min()
        # Like the real task: two test months, starting the day after training ends.
        assert (test["date"].min() - train["date"].max()).days == 1
        assert test["date"].dt.month.nunique() == 2


def test_metrics():
    result = metrics([100, 200], [110, 180])
    assert result["MAE"] == pytest.approx(15)  # (10 + 20) / 2
    assert result["MAPE"] == pytest.approx(10)  # (10% + 10%) / 2
    assert result["RMSE"] == pytest.approx(250 ** 0.5)  # sqrt((10^2 + 20^2) / 2)
