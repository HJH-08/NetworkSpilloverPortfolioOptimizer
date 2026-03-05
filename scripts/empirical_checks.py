"""
empirical_checks.py

Manual (visual) empirical validation helpers.
Currently includes TCI plot with shaded stress windows.

Usage:
    python scripts/empirical_checks.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Ensure repo root on path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import CACHE_DIR, PLOTS_DIR, REPORTS_DIR, STRESS_PERIODS, RUN_TAG


def _pick_latest_npz() -> str:
    files = [os.path.join(CACHE_DIR, f) for f in os.listdir(CACHE_DIR) if f.endswith(".npz")]
    if not files:
        raise FileNotFoundError(f"No .npz files in CACHE_DIR={CACHE_DIR}")
    files.sort(key=lambda p: os.path.getmtime(p))
    return files[-1]


def plot_tci_with_stress_windows() -> Path:
    npz_path = _pick_latest_npz()
    blob = np.load(npz_path, allow_pickle=False)

    dates = pd.to_datetime(blob["dates"].astype("datetime64[ns]"))
    tci = pd.Series(blob["tci"], index=dates, name="TCI")

    out_path = PLOTS_DIR / f"tci_stress_{RUN_TAG}.png"

    plt.figure()
    plt.plot(tci.index, tci.values, label="TCI")

    # Shade only windows that overlap the data range, and avoid duplicate legend labels
    x_min, x_max = tci.index.min(), tci.index.max()
    used_labels = set()
    for name, start, end in STRESS_PERIODS:
        start_ts = pd.to_datetime(start)
        end_ts = pd.to_datetime(end)
        if end_ts < x_min or start_ts > x_max:
            continue
        label = name if name not in used_labels else None
        used_labels.add(name)
        plt.axvspan(start_ts, end_ts, alpha=0.2, label=label)

    plt.title("Total Connectedness Index (TCI) with Stress Windows")
    plt.xlabel("Date")
    plt.ylabel("TCI (%)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    return out_path


def _banking_window() -> tuple[pd.Timestamp, pd.Timestamp]:
    for name, start, end in STRESS_PERIODS:
        if "bank" in name.lower():
            return pd.to_datetime(start), pd.to_datetime(end)
    # Fallback to commonly used window
    return pd.to_datetime("2023-03-06"), pd.to_datetime("2023-03-31")


def plot_xlf_allocation_banking_stress() -> Path:
    """
    Minimal bar chart of average XLF weights across strategies
    during the 2023 banking stress window.
    """
    start_ts, end_ts = _banking_window()

    files = {
        "EqualWeight": "weights_equalweight.csv",
        "MeanVar": "weights_meanvar.csv",
        "MinVar": "weights_minvar.csv",
        "Spillover": "weights_spillover.csv",
    }

    values = []
    labels = []

    for label, fname in files.items():
        path = REPORTS_DIR / fname
        if not path.exists():
            raise FileNotFoundError(f"Missing weights file: {path}")
        df = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
        if "XLF" not in df.columns:
            raise ValueError(f"XLF not found in {path.name}")
        window = df.loc[(df.index >= start_ts) & (df.index <= end_ts)]
        if window.empty:
            raise ValueError(f"No weights in banking window for {path.name}")
        values.append(float(window["XLF"].mean()))
        labels.append(label)

    out_path = PLOTS_DIR / f"figure_6_7_xlf_allocation_2023_banking_{RUN_TAG}.png"

    plt.figure(figsize=(5.5, 3.2))
    bars = plt.bar(labels, values, color=["#4C78A8", "#F58518", "#54A24B", "#E45756"])
    plt.title("XLF Allocation During 2023 Banking Stress")
    plt.ylabel("Average Weight")
    plt.ylim(0.0, 0.12)
    for b, v in zip(bars, values):
        plt.text(
            b.get_x() + b.get_width() / 2,
            v + 0.002,
            f"{v:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    # Ensure near-zero bars are still visible
    plt.axhline(0, color="black", linewidth=0.6)
    plt.grid(axis="y", alpha=0.25, linewidth=0.7)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    return out_path


def main() -> None:
    out = plot_tci_with_stress_windows()
    print("[empirical] Saved:", out)
    out2 = plot_xlf_allocation_banking_stress()
    print("[empirical] Saved:", out2)


if __name__ == "__main__":
    main()
