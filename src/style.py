"""Shared chart style: one meaning per color, the same in every chart."""

import matplotlib

matplotlib.use("Agg")  # only save files, never open a window
import matplotlib.pyplot as plt

EQUIPMENT_COLORS = {"Dry Van": "#2a78d6", "Reefer": "#1baf7a", "Flatbed": "#4a3aa7"}
GRAY = "#898781"        # ordinary rows and values
LIGHT_GRAY = "#c3c2b7"  # daily values behind a 7-day mean
RED = "#d03b3b"         # rows treated as wrong (dropped or fixed)
INK = "#52514e"         # text and reference lines

plt.rcParams.update({
    "axes.edgecolor": "#c3c2b7",
    "axes.labelcolor": INK,
    "axes.titlesize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": "#e1e0d9",
    "grid.linewidth": 0.8,
    "xtick.color": "#898781",
    "ytick.color": "#898781",
    "legend.frameon": False,
})
