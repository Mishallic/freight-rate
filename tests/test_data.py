"""One small test per cleaning rule in src/data.py."""

import numpy as np
import pandas as pd
import pytest

from data import drop_corrupted, is_corrupted, load_raw, prepare, weight_medians

# Weight medians as they would come from training data. The Reefer value differs from
# the Reefer weight in small_loads(), so a test can tell which one was used.
MEDIANS = pd.Series({"Dry Van": 30_000.0, "Reefer": 25_000.0, "Flatbed": 35_000.0})


def small_loads():
    """Five loads on two days, with one example of each weight and market_index problem."""
    return pd.DataFrame({
        "load_id": ["A", "B", "C", "D", "E"],
        "equipment": ["Dry Van", "Dry Van", "Reefer", "Reefer", "Flatbed"],
        "weight": [-20_000.0, 47_500.0, np.nan, 40_000.0, -47_500.0],
        "date": pd.to_datetime(["2025-01-01", "2025-01-01", "2025-01-01", "2025-01-02", "2025-01-02"]),
        "market_index": [1.0, 1.2, np.nan, 0.9, np.nan],
    })


def test_negative_weight_becomes_positive():
    loads = small_loads()
    out = prepare(loads, MEDIANS)
    assert out["weight"].tolist() == [20_000, 47_500, 25_000, 40_000, 47_500]
    assert loads.loc[0, "weight"] == -20_000  # the input is left unchanged


def test_cap_flag_is_set_after_abs():
    out = prepare(small_loads(), MEDIANS)
    # B is at the cap; E was -47,500, so it is at the cap after abs().
    assert out["weight_capped"].tolist() == [False, True, False, False, True]


def test_missing_weight_filled_with_training_median():
    out = prepare(small_loads(), MEDIANS)
    assert out["weight_missing"].tolist() == [False, False, True, False, False]
    # C is a Reefer: it gets the training median (25,000), not the other Reefer's 40,000.
    assert out.loc[2, "weight"] == 25_000


def test_missing_market_index_filled_with_day_mean():
    out = prepare(small_loads(), MEDIANS)
    assert out.loc[2, "market_index"] == pytest.approx(1.1)  # mean of 1.0 and 1.2 on Jan 1
    assert out.loc[4, "market_index"] == pytest.approx(0.9)  # only value on Jan 2


def test_corrupted_rate_cutoffs():
    # Nine Dry Van loads of 500 miles. The median rate per mile is 2.00, so the last four
    # loads have ratios 0.59, 0.61, 1.79 and 1.81: just outside, inside, inside, outside.
    rates_per_mile = [2.0, 2.0, 2.0, 2.0, 2.0, 1.18, 1.22, 3.58, 3.62]
    loads = pd.DataFrame({
        "equipment": "Dry Van",
        "distance": 500.0,
        "posted_rate": [r * 500 for r in rates_per_mile],
    })
    assert is_corrupted(loads).tolist() == [False] * 5 + [True, False, False, True]
    assert len(drop_corrupted(loads)) == 7


def test_prepare_never_drops_rows():
    train_test, _ = load_raw()
    out = prepare(train_test, weight_medians(train_test))
    assert out["load_id"].tolist() == train_test["load_id"].tolist()
    assert is_corrupted(out).sum() == 677  # corrupted rates are still there


def test_prepare_works_on_validation():
    train_test, validation = load_raw()
    out = prepare(validation, weight_medians(train_test))
    assert "posted_rate" not in out.columns
    assert len(out) == len(validation)
    assert out[["weight", "market_index"]].notna().all().all()
