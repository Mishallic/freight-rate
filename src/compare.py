"""Compare the baseline and the LightGBM feature variants on both folds.

Run from the repo root:  python src/compare.py
"""

import pandas as pd

from data import is_corrupted, load_raw
from evaluate import FOLDS, REPORTS_DIR, fit_baseline, predict_baseline, prepare_fold, score
from model import importance, predict, train_model
from style import GRAY, INK, plt

FIG_DIR = REPORTS_DIR / "figures"

FOLD_TITLES = {"A": "Fold A: train Jan-Jun, test Jul-Aug", "B": "Fold B: train Jan-Aug, test Sep-Oct"}

# The feature variants (model.py) that compete for the final model.
CANDIDATES = ["V0", "V1", "V2", "V3", "V1+ramp"]

# A few fixed LightGBM settings, tried on V0 only. This is not a search: the variants
# below all use the default settings (PARAMS in model.py).
SETTINGS = {
    "default": {},
    "l1_loss": {"objective": "l1"},
    "smaller_trees": {"num_leaves": 15, "min_child_samples": 200},
}

# Time-decay sample weights tried on V1: half-life in days, counted back from the last
# training day. Adopted only if one beats no weights on BOTH folds by >= 0.1 points.
HALF_LIVES = [30, 60, 90]

# quote_signal check. August is the only labelled month where quote_signal carries no rate
# information, like November-December, so train on Jan-Jul and test on August. Fold B tests
# on Sep-Oct, where quote_signal does carry the rate, for contrast.
QUOTE_SIGNAL_TESTS = {"August": {"train": [1, 2, 3, 4, 5, 6, 7], "test": [8]}, "B": FOLDS["B"]}


def signed_error_by_month(test, predicted):
    """Mean of (predicted / actual - 1) in % per test month, on clean test rows."""
    clean = ~is_corrupted(test)
    error = 100 * (predicted[clean] / test.loc[clean, "posted_rate"] - 1)
    return error.groupby(test.loc[clean, "date"].dt.month).mean()


def daily_error(test, predicted):
    """Mean absolute % error per test day, on clean test rows."""
    clean = ~is_corrupted(test)
    actual = test.loc[clean, "posted_rate"]
    return (100 * (predicted[clean] - actual).abs() / actual).groupby(test.loc[clean, "date"]).mean()


def plot_comparison(clean_mape):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, (fold, title) in zip(axes, FOLD_TITLES.items()):
        bars = ax.bar(clean_mape.index, clean_mape[fold], width=0.6, color=GRAY)
        ax.bar_label(bars, fmt="%.2f", padding=3, color=INK)
        ax.grid(axis="x", visible=False)
        ax.set_title(title, loc="left")
    axes[0].set_ylabel("MAPE on clean test rows (%)")
    fig.suptitle("Baseline and LightGBM feature variants (lower is better)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "5_model_comparison.png", dpi=150)
    plt.close(fig)
    print("saved", FIG_DIR / "5_model_comparison.png")


def main():
    train_test, _ = load_raw()
    folds = {name: prepare_fold(train_test, fold) for name, fold in FOLDS.items()}  # name -> (train, test)

    print("== LightGBM settings, tried on V0 only ==")
    results = []
    for name, settings in SETTINGS.items():
        for fold, (train, test) in folds.items():
            model = train_model(train, "V0", **settings)
            results += score(name, fold, test, predict(model, test, "V0"))
    table = pd.DataFrame(results).set_index(["model", "fold", "rows"]).round(2)
    print(table.to_string())
    table.to_csv(REPORTS_DIR / "metrics_settings_v0.csv")

    print("\n== Baseline and feature variants, default settings ==")
    results, predictions, models = [], {}, {}
    for fold, (train, test) in folds.items():
        predictions["baseline", fold] = predict_baseline(fit_baseline(train), test)
        results += score("baseline", fold, test, predictions["baseline", fold])
        for variant in CANDIDATES:
            models[variant, fold] = train_model(train, variant)
            predictions[variant, fold] = predict(models[variant, fold], test, variant)
            results += score(variant, fold, test, predictions[variant, fold])
        for half_life in HALF_LIVES:
            model = train_model(train, "V1", half_life=half_life)
            results += score(f"V1_hl{half_life}", fold, test, predict(model, test, "V1"))
    results = pd.DataFrame(results)
    table = results.set_index(["model", "fold", "rows"]).round(2)
    print(table.to_string())
    table.to_csv(REPORTS_DIR / "metrics_models.csv")

    clean_mape = results[results["rows"] == "clean"].pivot(index="model", columns="fold", values="MAPE")
    print("\n== Time-decay weights on V1: clean MAPE gain over no weights, points ==")
    decay = [f"V1_hl{half_life}" for half_life in HALF_LIVES]
    print((clean_mape.loc["V1"] - clean_mape.loc[decay]).round(2).to_string())

    clean_mape = clean_mape.loc[["baseline", *CANDIDATES]]
    best = clean_mape.drop("baseline").mean(axis=1).idxmin()
    print(f"\nbest variant (lowest clean MAPE averaged over both folds): {best}")

    print("\n== V1 vs V0 (the same features without day of week), per test day ==")
    for fold, (_, test) in folds.items():
        won = daily_error(test, predictions["V1", fold]) < daily_error(test, predictions["V0", fold])
        print(f"fold {fold}: V1 has the lower error on {won.mean():.0%} of {len(won)} test days")

    print(f"\n== Feature importance of {best}, % of total gain ==")
    gains = pd.DataFrame({f"fold {fold}": importance(models[best, fold]) for fold in FOLDS})
    print(gains.sort_values("fold B", ascending=False).round(1).to_string())

    print("\n== Mean signed error by test month, clean rows (%, negative = under-predicted) ==")
    by_month = pd.DataFrame({
        name: pd.concat([signed_error_by_month(folds[fold][1], predictions[name, fold]) for fold in FOLDS])
        for name in ["baseline", *CANDIDATES]
    })
    print(by_month.round(2).to_string())

    print("\n== quote_signal check: V1 with and without quote_signal ==")
    results = []
    for fold, months in QUOTE_SIGNAL_TESTS.items():
        train, test = prepare_fold(train_test, months)
        for variant in ["V1", "V1+quote_signal"]:
            model = train_model(train, variant)
            results += score(variant, fold, test, predict(model, test, variant))
    table = pd.DataFrame(results).set_index(["model", "fold", "rows"]).round(2)
    print(table.to_string())
    table.to_csv(REPORTS_DIR / "metrics_quote_signal.csv")

    plot_comparison(clean_mape)


if __name__ == "__main__":
    main()
