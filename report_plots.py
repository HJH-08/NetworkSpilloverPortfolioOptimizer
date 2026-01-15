"""
report_plots.py

Quick validation + report-ready plots for Phase 2 outputs.

Plots:
1) Total Connectedness Index (TCI) over time (from rolling spillovers cache)
2) Optional: Heatmap snapshot of spillover matrix W_t at a chosen date

Usage:
    python report_plots.py

Outputs:
    results/plots/tci_<RUN_TAG>.png
    results/plots/heatmap_W_<RUN_TAG>_<DATE>.png
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------
# Config (robust)
# -------------------------

def _cfg(name: str, default):
    try:
        import config
        return getattr(config, name, default)
    except Exception:
        return default


RUN_TAG = _cfg("RUN_TAG", "run")

WINDOW = _cfg("WINDOW", 250)
REBALANCE_EVERY_N_DAYS = _cfg("REBALANCE_EVERY_N_DAYS", 20)
FEVD_HORIZON = _cfg("FEVD_HORIZON", 10)

RESULTS_DIR = Path(_cfg("RESULTS_DIR", "results"))
CACHE_DIR = Path(_cfg("CACHE_DIR", RESULTS_DIR / "cache"))
PLOTS_DIR = Path(_cfg("PLOTS_DIR", RESULTS_DIR / "plots"))

PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def _load_npz(path: Path) -> Dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=False)
    return {k: data[k] for k in data.files}


def _cache_path() -> Path:
    # Must match naming in rolling_spillovers.py
    return CACHE_DIR / f"spillovers_{RUN_TAG}_win{WINDOW}_step{REBALANCE_EVERY_N_DAYS}_H{FEVD_HORIZON}.npz"


# -------------------------
# Plots
# -------------------------

def plot_tci(dates: pd.DatetimeIndex, tci: pd.Series) -> Path:
    """
    Plot Total Connectedness Index over time.
    """
    out_path = PLOTS_DIR / f"tci_{RUN_TAG}.png"

    plt.figure()
    plt.plot(dates, tci.values)
    plt.title("Total Connectedness Index (TCI) Over Time")
    plt.xlabel("Date")
    plt.ylabel("TCI (%)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    return out_path


def plot_W_heatmap(
    dates: pd.DatetimeIndex,
    assets: list[str],
    W_stack: np.ndarray,
    *,
    snapshot_date: str | None = None,
) -> Path:
    """
    Heatmap of spillover matrix W_t at a chosen date.

    If snapshot_date is None:
        pick the middle date (nice default)
    If snapshot_date is provided:
        choose closest available evaluation date in dates.
    """
    if snapshot_date is None:
        idx = len(dates) // 2
    else:
        target = pd.to_datetime(snapshot_date)
        idx = int(np.argmin(np.abs((dates - target).days)))

    date_used = dates[idx].strftime("%Y-%m-%d")
    W = W_stack[idx]  # (N, N)

    out_path = PLOTS_DIR / f"heatmap_W_{RUN_TAG}_{date_used}.png"

    plt.figure()
    im = plt.imshow(W, aspect="auto")
    plt.title(f"Spillover Matrix W (FEVD, %), {date_used}")
    plt.xticks(range(len(assets)), assets, rotation=45, ha="right")
    plt.yticks(range(len(assets)), assets)
    plt.colorbar(im, label="Contribution (%)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    return out_path


# -------------------------
# Main
# -------------------------

if __name__ == "__main__":
    cache_path = _cache_path()
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Could not find cache at: {cache_path}\n"
            "Run rolling_spillovers.py first, or check RUN_TAG/WINDOW/STEP/H config."
        )

    blob = _load_npz(cache_path)

    dates = pd.to_datetime(blob["dates"].astype("datetime64[ns]"))
    assets = blob["assets"].astype(str).tolist()

    W_stack = blob["W_stack"]
    tci = pd.Series(blob["tci"], index=dates, name="TCI")

    print("[plots] Loaded cache:", cache_path)
    print("[plots] dates:", dates.min().date(), "->", dates.max().date(), "| points:", len(dates))
    print("[plots] assets:", assets)
    print("[plots] W_stack:", W_stack.shape)

    p1 = plot_tci(dates, tci)
    print("[plots] Saved:", p1)

    # Choose a nice “stress-ish” default snapshot (you can change this anytime)
    # e.g. "2020-03-31" for COVID period; script will pick closest eval date.
    p2 = plot_W_heatmap(dates, assets, W_stack, snapshot_date="2020-03-31")
    print("[plots] Saved:", p2)

    # Quick sanity print: biggest TCI dates
    top = tci.sort_values(ascending=False).head(5)
    print("\n[plots] Top 5 TCI dates:")
    print(top)
