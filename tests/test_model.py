"""Tests for the model features in src/model.py."""

import pandas as pd

from model import features


def test_quarter_features():
    dates = pd.to_datetime(["2025-01-01", "2025-03-31", "2025-07-01", "2025-12-31"])
    loads = pd.DataFrame({
        "date": dates, "distance": 500.0, "weight": 30_000.0, "weight_capped": False,
        "equipment": "Dry Van", "pickup_lat": 35.0, "pickup_lon": -90.0,
        "delivery_lat": 36.0, "delivery_lon": -91.0,
    })
    out = features(loads, "V2")
    assert out["month_in_quarter"].tolist() == [1, 3, 1, 3]
    assert out["day_of_quarter"].tolist() == [1, 90, 1, 92]
    assert out["day_of_week"].tolist() == [2, 0, 1, 2]  # Wed, Mon, Tue, Wed
