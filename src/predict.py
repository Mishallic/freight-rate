"""Train the final model on all of Jan-Oct, write both prediction files, run the scorer.

Run from the repo root:  python src/predict.py
"""

import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

from data import DATA_DIR, WEIGHT_CAP, drop_corrupted, load_raw, prepare, weight_medians
from evaluate import baseline_key, fit_baseline
from model import CHOSEN_VARIANT, HALF_LIFE, predict, train_model

ROOT = Path(__file__).resolve().parent.parent
PREDICTIONS_FILE = "validation_predictions.csv"           # relative to ROOT
DECEMBER_FILE = "outputs/december_predictions.csv"        # relative to ROOT


def city_coordinates(df):
    """One (lat, lon) row per city, from the pickups and deliveries in df."""
    pickups = df[["pickup", "pickup_lat", "pickup_lon"]].set_axis(["city", "lat", "lon"], axis=1)
    deliveries = df[["delivery", "delivery_lat", "delivery_lon"]].set_axis(["city", "lat", "lon"], axis=1)
    return pd.concat([pickups, deliveries]).drop_duplicates("city").set_index("city")


def predict_december(model, coords):
    """Fill predicted_rate for the 31 December chart rows, keeping their 7 columns."""
    december = pd.read_csv(DATA_DIR / "december_chart_inputs.csv")  # dates stay text, as in the file
    # The rows only give lane, distance, equipment, weight and date. Add what the model needs.
    # No prepare() needed: the weight is a clean 32,000 lbs and the model doesn't use market_index.
    rows = december.assign(
        date=pd.to_datetime(december["date"]),
        pickup_lat=december["pickup"].map(coords["lat"]),
        pickup_lon=december["pickup"].map(coords["lon"]),
        delivery_lat=december["delivery"].map(coords["lat"]),
        delivery_lon=december["delivery"].map(coords["lon"]),
        weight_capped=december["weight"] == WEIGHT_CAP,  # False: 32,000 is below the cap
    )
    december["predicted_rate"] = predict(model, rows, CHOSEN_VARIANT).round(2)
    return december


def sanity_checks(train, val, predicted, december):
    rpm_train = train["posted_rate"] / train["distance"]
    rpm_pred = predicted / val["distance"]
    typical = fit_baseline(train)  # median rate per mile per distance band and equipment

    print("\n== Median rate per mile: training (clean Jan-Oct) vs validation predictions ==")
    by_group = pd.DataFrame({"training": typical, "predicted": rpm_pred.groupby(baseline_key(val)).median()})
    by_group["predicted / training"] = by_group["predicted"] / by_group["training"]
    print(by_group.round(3).to_string())

    print("\n== Validation rows with a city never seen in training ==")
    known = set(train["pickup"]) | set(train["delivery"])
    new_city = ~val["pickup"].isin(known) | ~val["delivery"].isin(known)
    ratio = rpm_pred / baseline_key(val).map(typical)  # vs similar training loads
    summary = ratio.groupby(new_city.map({True: "new city", False: "known cities"})).describe(
        percentiles=[0.05, 0.5, 0.95])
    print("predicted rate per mile / training median for the same band and equipment:")
    print(summary[["count", "5%", "50%", "95%"]].round(3).to_string())

    print("\n== December chart (Lexington -> Fort Wayne, 360 mi, Dry Van, 32,000 lbs) ==")
    rate = december["predicted_rate"]
    print(f"predicted: min ${rate.min():.2f}, max ${rate.max():.2f}, mean ${rate.mean():.2f}, "
          f"spread ${rate.max() - rate.min():.2f} ({(rate.max() - rate.min()) / rate.mean():.1%} of the mean)")
    # Training loads for comparison, scaled to 360 miles (distance varies from load to load).
    dry_van = train[train["equipment"] == "Dry Van"]
    lane = dry_van[(dry_van["pickup"] == "Lexington") & (dry_van["delivery"] == "Fort Wayne")]
    similar = dry_van[dry_van["distance"].between(340, 380)]
    for name, loads in [("same lane, Dry Van", lane), ("any Dry Van load of 340-380 mi", similar)]:
        at_360 = loads["posted_rate"] / loads["distance"] * 360
        recent = at_360[loads["date"].dt.month >= 9]
        print(f"{name}: {len(loads)} loads, rate at 360 mi median ${at_360.median():.2f} "
              f"(10-90%: ${at_360.quantile(0.1):.2f} to ${at_360.quantile(0.9):.2f}); "
              f"Sep-Oct median ${recent.median():.2f} ({len(recent)} loads)")


def main():
    train_test, validation = load_raw()

    # Final model: all of Jan-Oct, cleaned exactly as in validation.
    medians = weight_medians(train_test)
    train = drop_corrupted(prepare(train_test, medians))
    model = train_model(train, CHOSEN_VARIANT, half_life=HALF_LIFE)
    print(f"trained {CHOSEN_VARIANT} on {len(train):,} loads (Jan-Oct, corrupted rates dropped)")

    # Validation predictions, in the template's load_id order.
    val = prepare(validation, medians)
    predicted = predict(model, val, CHOSEN_VARIANT)
    output = pd.read_csv(DATA_DIR / "validation_predictions_template.csv")[["load_id"]]
    output["predicted_rate"] = output["load_id"].map(dict(zip(val["load_id"], predicted.round(2))))
    assert len(output) == 12_000 and output["predicted_rate"].gt(0).all()  # also fails on any NaN
    output.to_csv(ROOT / PREDICTIONS_FILE, index=False)
    print("wrote", PREDICTIONS_FILE)

    december = predict_december(model, city_coordinates(train_test))
    (ROOT / DECEMBER_FILE).parent.mkdir(exist_ok=True)
    december.to_csv(ROOT / DECEMBER_FILE, index=False)
    print("wrote", DECEMBER_FILE)

    # The provided scorer checks both files and draws the December chart.
    sys.stdout.flush()  # so the lines above print before the scorer's output
    subprocess.run([sys.executable, "provided/score.py", "--predictions", PREDICTIONS_FILE,
                    "--december-predictions", DECEMBER_FILE], cwd=ROOT, check=True)
    shutil.copy(ROOT / "scorer_results" / "candidate_december.png",
                ROOT / "reports" / "figures" / "6_candidate_december.png")
    print("copied the scorer's chart to reports/figures/6_candidate_december.png")

    sanity_checks(train, val, predicted, december)


if __name__ == "__main__":
    main()
