"""Compare the baseline and the LightGBM feature variants on both folds.

Run from the repo root:  python src/compare.py
"""

import pandas as pd

from data import is_corrupted, load_raw
from evaluate import FOLDS, REPORTS_DIR, fit_baseline, predict_baseline, prepare_fold, score
from model import VARIANTS, importance, predict, train_model
from style import GRAY, INK, plt

FIG_DIR = REPORTS_DIR / "figures"

FOLD_TITLES = {"A": "Fold A: train Jan-Jun, test Jul-Aug", "B": "Fold B: train Jan-Aug, test Sep-Oct"}

# A few fixed LightGBM settings, tried on V0 only. This is not a search: the variants
# below all use the default settings (PARAMS in model.py).
SETTINGS = {
    "default": {},
    "l1_loss": {"objective": "l1"},
    "smaller_trees": {"num_leaves": 15, "min_child_samples": 200},
}


def signed_error_by_month(test, predicted):
    """Mean of (predicted / actual - 1) in % per test month, on clean test rows."""
    clean = ~is_corrupted(test)
    error = 100 * (predicted[clean] / test.loc[clean, "posted_rate"] - 1)
    return error.groupby(test.loc[clean, "date"].dt.month).mean()


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
    folds = {fold: prepare_fold(train_test, fold) for fold in FOLDS}  # fold -> (train, test)

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
        for variant in VARIANTS:
            models[variant, fold] = train_model(train, variant)
            predictions[variant, fold] = predict(models[variant, fold], test, variant)
            results += score(variant, fold, test, predictions[variant, fold])
    results = pd.DataFrame(results)
    table = results.set_index(["model", "fold", "rows"]).round(2)
    print(table.to_string())
    table.to_csv(REPORTS_DIR / "metrics_models.csv")

    clean_mape = results[results["rows"] == "clean"].pivot(index="model", columns="fold", values="MAPE")
    clean_mape = clean_mape.loc[["baseline", *VARIANTS]]
    best = clean_mape.drop("baseline").mean(axis=1).idxmin()
    print(f"\nbest variant (lowest clean MAPE averaged over both folds): {best}")

    print(f"\n== Feature importance of {best}, % of total gain ==")
    gains = pd.DataFrame({f"fold {fold}": importance(models[best, fold]) for fold in FOLDS})
    print(gains.sort_values("fold B", ascending=False).round(1).to_string())

    print("\n== Mean signed error by test month, clean rows (%, negative = under-predicted) ==")
    by_month = pd.DataFrame({
        name: pd.concat([signed_error_by_month(folds[fold][1], predictions[name, fold]) for fold in FOLDS])
        for name in ["baseline", best]
    })
    print(by_month.round(2).to_string())

    plot_comparison(clean_mape)


if __name__ == "__main__":
    main()
