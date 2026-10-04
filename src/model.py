"""LightGBM model on log(posted_rate), and the features it uses.

No city names (8 validation cities never appear in training) and no quote_signal
(no rate information in Nov-Dec, see data.py).
"""

import lightgbm as lgb
import numpy as np
import pandas as pd

# Feature variants. V1 to V3 each add features to the one before; V1+ramp is V1 plus a
# quarter-end ramp, testing whether rates climbing through quarter-end months can be learned.
V0 = ["distance", "weight", "weight_capped", "equipment",
      "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"]
VARIANTS = {
    "V0": V0,
    "V1": V0 + ["day_of_week"],
    "V2": V0 + ["day_of_week", "month_in_quarter", "day_of_quarter"],
    "V3": V0 + ["day_of_week", "month_in_quarter", "day_of_quarter", "market_index"],
    "V1+ramp": V0 + ["day_of_week", "quarter_end_ramp"],
}

# The final model. V1 had the lowest clean MAPE on both validation folds, won on most
# test days, and the weekday effect it adds is real (about +/-1% in the raw data).
# Time-decay weights (half-life 30, 60 or 90 days) did not beat equal weights on both
# folds by 0.1 points, so all training loads count the same.
CHOSEN_VARIANT = "V1"
HALF_LIFE = None

# Fixed categories, so equipment is encoded the same way in training and prediction.
EQUIPMENT_TYPES = pd.CategoricalDtype(["Dry Van", "Reefer", "Flatbed"])

PARAMS = {
    "n_estimators": 600,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 50,
    "random_state": 0,
    "verbose": -1,  # no training log
}


def features(df, variant):
    """Model inputs for one variant. The only place features are built, for training and prediction."""
    x = df.copy()
    x["equipment"] = x["equipment"].astype(EQUIPMENT_TYPES)
    quarter_start = x["date"].dt.to_period("Q").dt.start_time  # first day of the date's quarter
    x["day_of_week"] = x["date"].dt.dayofweek  # 0 = Monday
    x["month_in_quarter"] = (x["date"].dt.month - 1) % 3 + 1  # 1, 2 or 3
    x["day_of_quarter"] = (x["date"] - quarter_start).dt.days + 1  # 1 to 92
    quarter_end_month = x["date"].dt.month.isin([3, 6, 9, 12])
    x["quarter_end_ramp"] = x["date"].dt.day.where(quarter_end_month, 0)  # day of month, else 0
    return x[VARIANTS[variant]]


def decay_weights(dates, half_life):
    """Sample weight 1 on the last training day, halving every `half_life` days further back."""
    age_days = (dates.max() - dates).dt.days
    return 0.5 ** (age_days / half_life)


def train_model(train, variant, half_life=None, **settings):
    """Fit LightGBM on log(posted_rate).

    half_life (days): weight recent loads more; None = all loads count the same.
    settings: override PARAMS, to compare a few settings.
    """
    weights = decay_weights(train["date"], half_life) if half_life else None
    model = lgb.LGBMRegressor(**{**PARAMS, **settings})
    model.fit(features(train, variant), np.log(train["posted_rate"]), sample_weight=weights)
    return model


def predict(model, df, variant):
    """Predicted posted_rate in dollars: exp() of the model's log prediction."""
    return pd.Series(np.exp(model.predict(features(df, variant))), index=df.index)


def importance(model):
    """Share of the total gain per feature, in %."""
    gain = pd.Series(model.booster_.feature_importance(importance_type="gain"), index=model.feature_name_)
    return 100 * gain / gain.sum()
