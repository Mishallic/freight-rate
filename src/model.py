"""LightGBM model on log(posted_rate), and the features it uses.

No city names (8 validation cities never appear in training) and no quote_signal
(no rate information in Nov-Dec, see data.py).
"""

import lightgbm as lgb
import numpy as np
import pandas as pd

# Feature variants. Each one adds features to the one before.
V0 = ["distance", "weight", "weight_capped", "equipment",
      "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"]
VARIANTS = {
    "V0": V0,
    "V1": V0 + ["day_of_week"],
    "V2": V0 + ["day_of_week", "month_in_quarter", "day_of_quarter"],
    "V3": V0 + ["day_of_week", "month_in_quarter", "day_of_quarter", "market_index"],
}

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
    return x[VARIANTS[variant]]


def train_model(train, variant, **settings):
    """Fit LightGBM on log(posted_rate). `settings` override PARAMS, to compare a few settings."""
    model = lgb.LGBMRegressor(**{**PARAMS, **settings})
    model.fit(features(train, variant), np.log(train["posted_rate"]))
    return model


def predict(model, df, variant):
    """Predicted posted_rate in dollars: exp() of the model's log prediction."""
    return pd.Series(np.exp(model.predict(features(df, variant))), index=df.index)


def importance(model):
    """Share of the total gain per feature, in %."""
    gain = pd.Series(model.booster_.feature_importance(importance_type="gain"), index=model.feature_name_)
    return 100 * gain / gain.sum()
