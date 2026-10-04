"""Explore the raw data: print key facts and save four charts to reports/figures/.

Run from the repo root:  python src/explore.py
"""

import calendar
from pathlib import Path

import numpy as np
import pandas as pd

from data import RATIO_HIGH, RATIO_LOW, WEIGHT_CAP, distance_band, is_corrupted, load_raw, rate_ratio
from style import EQUIPMENT_COLORS, GRAY, INK, LIGHT_GRAY, RED, plt

FIG_DIR = Path(__file__).resolve().parent.parent / "reports" / "figures"

WEIGHT_FLOOR = 5_000


def haversine_miles(lat1, lon1, lat2, lon2):
    """Straight-line (great-circle) distance in miles between two points."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * 3958.8 * np.arcsin(np.sqrt(a))  # 3958.8 = Earth radius in miles


def print_facts(train_test, validation):
    files = {"train_test": train_test, "validation": validation}
    both = pd.concat([train_test, validation], ignore_index=True)
    rpm = train_test["posted_rate"] / train_test["distance"]

    print("== Rows and dates ==")
    for name, df in files.items():
        print(f"{name}: {len(df):,} rows, {df['date'].min():%Y-%m-%d} to {df['date'].max():%Y-%m-%d}, "
              f"{df['date'].nunique()} distinct days")
    print("columns only in train_test:", sorted(set(train_test.columns) - set(validation.columns)))
    print("equipment:", train_test["equipment"].value_counts().to_dict())

    print("\n== Integrity ==")
    for name, df in files.items():
        print(f"{name}: duplicate load_ids {df['load_id'].duplicated().sum()}, "
              f"duplicate rows ignoring load_id {df.drop(columns='load_id').duplicated().sum()}, "
              f"pickup == delivery {(df['pickup'] == df['delivery']).sum()}")
    print("load_ids in both files:", len(set(train_test["load_id"]) & set(validation["load_id"])))

    print("\n== Cities ==")
    # Every (city, lat, lon) combination seen as a pickup or a delivery, in either file.
    stops = pd.concat([
        both[["pickup", "pickup_lat", "pickup_lon"]].set_axis(["city", "lat", "lon"], axis=1),
        both[["delivery", "delivery_lat", "delivery_lon"]].set_axis(["city", "lat", "lon"], axis=1),
    ]).drop_duplicates()
    print(f"{stops['city'].nunique()} cities; cities with more than one coordinate pair: "
          f"{(stops['city'].value_counts() > 1).sum()}")
    # Consistent per city, but not real geography (real longitudes: Tampa -82.5, Baltimore -76.6).
    coords = stops.set_index("city")
    for city in ["Tampa", "Baltimore"]:
        print(f"{city}: lat {coords.loc[city, 'lat']:.1f}, lon {coords.loc[city, 'lon']:.1f}")
    train_cities = set(train_test["pickup"]) | set(train_test["delivery"])
    val_cities = set(validation["pickup"]) | set(validation["delivery"])
    new_cities = sorted(val_cities - train_cities)
    touches_new = validation["pickup"].isin(new_cities) | validation["delivery"].isin(new_cities)
    print(f"cities only in validation ({len(new_cities)}): {', '.join(new_cities)}")
    print(f"validation rows touching them: {touches_new.sum():,} ({touches_new.mean():.1%})")
    print("cities only in train_test:", len(train_cities - val_cities))

    print("\n== Missing values ==")
    for name, df in files.items():
        missing = df.isna().sum()
        print(f"{name}: {missing[missing > 0].to_dict()}")

    print("\n== Corrupted rates (train_test) ==")
    print(f"rate per mile: median {rpm.median():.2f}, min {rpm.min():.2f}, max {rpm.max():.2f}")
    print("median rate per mile by equipment and distance band (miles):")
    band = distance_band(train_test["distance"])
    typical = rpm.groupby([train_test["equipment"], band], observed=True).median()
    print(typical.unstack().round(2).to_string())
    ratio = rate_ratio(train_test)
    low, high = ratio < RATIO_LOW, ratio > RATIO_HIGH
    kept = ~(low | high)
    print(f"ratio = rate per mile / that median; flagged if below {RATIO_LOW} or above {RATIO_HIGH}")
    print(f"kept: {kept.sum():,} rows, ratio {ratio[kept].min():.2f} to {ratio[kept].max():.2f}")
    print(f"flagged low: {low.sum()} rows, ratio {ratio[low].min():.2f} to {ratio[low].max():.2f}")
    print(f"flagged high: {high.sum()} rows, ratio {ratio[high].min():.2f} to {ratio[high].max():.2f}")
    print(f"flagged total: {(~kept).sum()} rows ({(~kept).mean():.1%})")
    print(f"flagged rows at $4-5 per mile: {(~kept & rpm.between(4, 5)).sum()}; flagged rows that a "
          f"flat $1.20-$5.00 per mile rule would keep: {(~kept & rpm.between(1.2, 5)).sum()}")

    print("\n== Weight (lbs), per file: raw values, then after abs() ==")
    for name, df in files.items():
        raw, fixed = df["weight"], df["weight"].abs()
        print(f"{name} raw: negative {(raw < 0).sum()} (lowest {raw.min():,.0f}), "
              f"missing {raw.isna().sum()}, exactly {WEIGHT_CAP:,}: {(raw == WEIGHT_CAP).sum():,}, "
              f"exactly {WEIGHT_FLOOR:,}: {(raw == WEIGHT_FLOOR).sum()}")
        print(f"{name} after abs(): range {fixed.min():,.0f} to {fixed.max():,.0f}, "
              f"exactly {WEIGHT_CAP:,}: {(fixed == WEIGHT_CAP).sum():,}, "
              f"exactly {WEIGHT_FLOOR:,}: {(fixed == WEIGHT_FLOOR).sum()}")

    print("\n== market_index ==")
    day_std = both.groupby("date")["market_index"].std()
    print(f"spread of values within one day (std): mean {day_std.mean():.3f}")
    by_month = pd.DataFrame({
        "market_index (mean)": both.groupby(both["date"].dt.month)["market_index"].mean(),
        "rate per mile (median)": rpm.groupby(train_test["date"].dt.month).median(),
    }).set_axis(list(calendar.month_abbr)[1:])
    print(by_month.round(3).T.to_string())
    by_weekday = pd.DataFrame({
        "market_index (mean)": both.groupby(both["date"].dt.dayofweek)["market_index"].mean(),
        "rate per mile (median)": rpm.groupby(train_test["date"].dt.dayofweek).median(),
    }).set_axis(list(calendar.day_abbr))
    print(by_weekday.round(3).T.to_string())

    print("\n== Distance ==")
    straight = haversine_miles(train_test["pickup_lat"], train_test["pickup_lon"],
                               train_test["delivery_lat"], train_test["delivery_lon"])
    detour = train_test["distance"] / straight
    print(f"distance / straight-line distance: median {detour.median():.2f}, "
          f"middle 90% {detour.quantile(0.05):.2f} to {detour.quantile(0.95):.2f}")
    shortest = both["distance"].min()
    print(f"shortest distance {shortest:.0f} mi; rows at exactly that value: "
          f"train_test {(train_test['distance'] == shortest).sum()}, "
          f"validation {(validation['distance'] == shortest).sum()}")

    print("\n== quote_signal ==")
    # Pooled over all months this looks U-shaped. The per-month table below shows why:
    # within a month the link is linear, but its sign depends on the month.
    decile = pd.qcut(train_test["quote_signal"], 10, labels=range(1, 11))
    print("median rate per mile by quote_signal decile (1 = lowest):",
          rpm.groupby(decile, observed=True).median().round(2).tolist())
    # Correlation per month. With the rate: kept train_test rows (validation has no rates).
    # With distance: both files, so November and December can be compared with labelled months.
    clean = train_test[kept]
    by_month = pd.DataFrame({
        "corr with rate per mile": clean["quote_signal"].groupby(clean["date"].dt.month)
                                   .corr(clean["posted_rate"] / clean["distance"]),
        "corr with distance": both["quote_signal"].groupby(both["date"].dt.month).corr(both["distance"]),
    }).set_axis(list(calendar.month_abbr)[1:])
    print(by_month.round(2).T.to_string())


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG_DIR / name, dpi=150)
    plt.close(fig)
    print("saved", FIG_DIR / name)


def plot_rate_vs_distance(train_test):
    flagged = is_corrupted(train_test)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharex=True, sharey=True)
    for ax, (equipment, color) in zip(axes, EQUIPMENT_COLORS.items()):
        rows = train_test["equipment"] == equipment
        kept, bad = train_test[rows & ~flagged], train_test[rows & flagged]
        ax.scatter(kept["distance"], kept["posted_rate"], s=3, alpha=0.3, color=color, linewidths=0)
        ax.scatter(bad["distance"], bad["posted_rate"], s=6, color=RED, linewidths=0,
                   label=f"flagged: {len(bad)} rows")
        median_rpm = (train_test.loc[rows, "posted_rate"] / train_test.loc[rows, "distance"]).median()
        ax.set_title(f"{equipment}: {rows.sum():,} loads, median ${median_rpm:.2f}/mile", loc="left")
        ax.set_xlabel("Distance (miles)")
        ax.legend(loc="upper left")
    axes[0].set_ylabel("Posted rate (USD)")
    axes[0].yaxis.set_major_formatter("${x:,.0f}")
    fig.suptitle("Posted rate vs distance by equipment (train_test)")
    save(fig, "1_rate_vs_distance.png")


def plot_rate_ratio(train_test):
    ratio = rate_ratio(train_test)
    flagged = is_corrupted(train_test)
    bins = np.arange(0, ratio.max() + 0.02, 0.02)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.hist(ratio[~flagged], bins=bins, color=GRAY, label=f"kept: {(~flagged).sum():,} rows")
    ax.hist(ratio[flagged], bins=bins, color=RED, label=f"flagged: {flagged.sum():,} rows")
    for cutoff in (RATIO_LOW, RATIO_HIGH):
        ax.axvline(cutoff, color=INK, linewidth=1)
        # x in data units, y as a fraction of the axes height (1.02 = just above the plot).
        ax.text(cutoff, 1.02, f"{cutoff}", transform=ax.get_xaxis_transform(), ha="center", color=INK)
    ax.set_yscale("log")  # log scale, so the flagged rows are visible next to the bulk
    ax.set_xlabel("Rate per mile / median rate per mile for the same equipment and distance band")
    ax.set_ylabel("Rows per 0.02 bin (log scale)")
    ax.legend(loc="upper right")
    ax.set_title("Rate per mile relative to similar loads, train_test", loc="left",
                 pad=22)  # room for the cutoff labels
    save(fig, "2_rate_ratio_outliers.png")


def plot_market_index_and_rate(train_test, validation):
    both = pd.concat([train_test, validation], ignore_index=True)
    panels = {
        "Daily mean market_index (train_test + validation)":
            both.groupby("date")["market_index"].mean(),
        "Daily median rate per mile, USD (train_test only, validation has no rates)":
            (train_test["posted_rate"] / train_test["distance"]).groupby(train_test["date"]).median(),
    }
    validation_start = validation["date"].min()
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    for ax, (title, daily) in zip(axes, panels.items()):
        ax.plot(daily.index, daily, color=LIGHT_GRAY, linewidth=1, label="daily")
        ax.plot(daily.index, daily.rolling(7, center=True).mean(), color=GRAY, linewidth=2,
                label="7-day mean")
        # Same relative range in both panels: equal heights mean equal % changes.
        ax.set_ylim(daily.mean() * 0.65, daily.mean() * 1.4)
        ax.axvline(validation_start, color=INK, linewidth=1)
        # x is a date, y a fraction of the axes height (0.97 = just below the top).
        ax.text(validation_start, 0.97, "  validation", transform=ax.get_xaxis_transform(),
                va="top", color=INK)
        ax.set_title(title, loc="left")
    axes[0].legend(loc="upper left")
    fig.suptitle("Both panels use the same relative y-range (65% to 140% of the series mean)",
                 color=INK, fontsize=10)
    save(fig, "3_market_index_vs_rate_per_mile.png")


def plot_weight(train_test):
    weight = train_test["weight"].dropna()
    bins = np.arange(-48_000, 48_001, 500)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.hist(weight[weight > 0], bins=bins, color=GRAY, label=f"positive: {(weight > 0).sum():,} rows")
    ax.hist(weight[weight < 0], bins=bins, color=RED,
            label=f"negative (sign flipped, fixed with abs()): {(weight < 0).sum():,} rows")
    for value, label in [(WEIGHT_CAP, "cap"), (WEIGHT_FLOOR, "floor")]:
        rows = (weight == value).sum()
        ax.annotate(f"{rows:,} rows at exactly {value:,} ({label})", xy=(value, rows),
                    xytext=(-6, 6), textcoords="offset points", ha="right", va="bottom", color=INK)
    ax.set_yscale("log")  # log scale, so the few negative values are visible
    ax.set_ylim(top=5_000)  # headroom for the cap label
    ax.xaxis.set_major_formatter("{x:,.0f}")
    ax.set_xlabel("Weight (lbs)")
    ax.set_ylabel("Rows per 500 lbs bin (log scale)")
    ax.legend(loc="upper left")
    ax.set_title(f"Weight, train_test raw values ({train_test['weight'].isna().sum()} missing not shown)",
                 loc="left")
    save(fig, "4_weight_distribution.png")


def main():
    train_test, validation = load_raw()
    print_facts(train_test, validation)
    print()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plot_rate_vs_distance(train_test)
    plot_rate_ratio(train_test)
    plot_market_index_and_rate(train_test, validation)
    plot_weight(train_test)


if __name__ == "__main__":
    main()
