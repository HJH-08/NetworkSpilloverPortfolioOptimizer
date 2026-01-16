"""
network_metrics.py

Compute per-date network metrics from rolling spillover matrices W_t.

Given W_t where:
    W[i, j] = % contribution of shocks in j to variance of i
(i receives from j)

We compute, per date t:
- from_others[i] = sum_{j!=i} W[i,j]  (row sum, since row is receiver)
- to_others[j]   = sum_{i!=j} W[i,j]  (column sum, since column is transmitter)
- net[j]         = to_others[j] - from_others[j]

Outputs:
- results/network_metrics_<RUN_TAG>.csv  (long/tidy format for easy plotting)
- results/network_metrics_wide_<RUN_TAG>.csv (wide format, convenient)
- results/plots/top_transmitters_<RUN_TAG>.png
- results/plots/top_receivers_<RUN_TAG>.png
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -------------------------
# Config helpers
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

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def _cache_path() -> Path:
    """
    Must match your rolling_spillover cache naming.
    If your existing name is different, just edit this to match.
    """
    # Common minimal name (recommended)
    candidate = CACHE_DIR / f"spillovers_{RUN_TAG}_win{WINDOW}_step{REBALANCE_EVERY_N_DAYS}_H{FEVD_HORIZON}.npz"
    if candidate.exists():
        return candidate

    # Your actual observed filename in terminal logs:
    # spillovers_v1_win250_reb20_fevd10_win250_step20_H10.npz
    candidate2 = CACHE_DIR / f"spillovers_{RUN_TAG}_win{WINDOW}_reb{REBALANCE_EVERY_N_DAYS}_fevd{FEVD_HORIZON}_win{WINDOW}_step{REBALANCE_EVERY_N_DAYS}_H{FEVD_HORIZON}.npz"
    if candidate2.exists():
        return candidate2

    # Fallback: try to find any spillovers_<RUN_TAG>*.npz
    matches = sorted(CACHE_DIR.glob(f"spillovers_{RUN_TAG}*.npz"))
    if matches:
        return matches[-1]

    raise FileNotFoundError(
        "Could not locate spillover cache .npz in CACHE_DIR.\n"
        f"Looked for:\n- {candidate}\n- {candidate2}\n- spillovers_{RUN_TAG}*.npz\n"
        f"CACHE_DIR={CACHE_DIR}"
    )


def _load_npz(path: Path) -> Dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=False)
    return {k: data[k] for k in data.files}


# -------------------------
# Metrics
# -------------------------

def compute_metrics_for_W(W: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    W: (N,N), ideally diagonal already 0.

    Returns:
    - to_others:   (N,) transmitter strength per node (column sums)
    - from_others: (N,) receiver strength per node (row sums)
    - net:         (N,) net transmitter (to - from)
    """
    # Defensive: ensure float
    W = np.asarray(W, dtype=float)

    # Remove diagonal effect if not already
    W2 = W.copy()
    np.fill_diagonal(W2, 0.0)

    from_others = W2.sum(axis=1)  # row sum: how much i receives from others
    to_others = W2.sum(axis=0)    # col sum: how much j transmits to others
    net = to_others - from_others
    return to_others, from_others, net


def compute_metrics_over_time(
    dates: pd.DatetimeIndex,
    assets: list[str],
    W_stack: np.ndarray,
) -> Dict[str, pd.DataFrame]:
    """
    Build wide DataFrames indexed by date with columns = assets:
    - to_others_df
    - from_others_df
    - net_df
    """
    T, N, N2 = W_stack.shape
    if N != len(assets) or N2 != len(assets):
        raise ValueError("W_stack shape does not match assets length.")

    to_rows = []
    from_rows = []
    net_rows = []

    for t in range(T):
        to_, from_, net_ = compute_metrics_for_W(W_stack[t])
        to_rows.append(to_)
        from_rows.append(from_)
        net_rows.append(net_)

    to_df = pd.DataFrame(to_rows, index=dates, columns=assets)
    from_df = pd.DataFrame(from_rows, index=dates, columns=assets)
    net_df = pd.DataFrame(net_rows, index=dates, columns=assets)

    return {"to_others": to_df, "from_others": from_df, "net": net_df}


def to_long(df: pd.DataFrame, value_name: str) -> pd.DataFrame:
    """
    Wide (date x asset) -> long tidy.
    """
    out = df.copy()
    out.index.name = "date"
    return out.reset_index().melt(id_vars="date", var_name="asset", value_name=value_name)


# -------------------------
# Plots
# -------------------------

def plot_top_k_time_series(
    df: pd.DataFrame,
    *,
    title: str,
    ylabel: str,
    out_path: Path,
    k: int = 3,
) -> Path:
    """
    Plot top-k assets by average value over time (lines on one chart).
    """
    avg = df.mean(axis=0).sort_values(ascending=False)
    top_assets = avg.head(k).index.tolist()

    plt.figure()
    for a in top_assets:
        plt.plot(df.index, df[a].values, label=a)

    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    return out_path


# -------------------------
# Main (self-test)
# -------------------------

if __name__ == "__main__":
    cache_path = _cache_path()
    blob = _load_npz(cache_path)

    dates = pd.to_datetime(blob["dates"].astype("datetime64[ns]"))
    assets = blob["assets"].astype(str).tolist()
    W_stack = blob["W_stack"]

    print("[network_metrics] Loaded:", cache_path)
    print("[network_metrics] dates:", dates.min().date(), "->", dates.max().date(), "| points:", len(dates))
    print("[network_metrics] assets:", assets)
    print("[network_metrics] W_stack:", W_stack.shape)

    metrics = compute_metrics_over_time(dates, assets, W_stack)
    to_df = metrics["to_others"]
    from_df = metrics["from_others"]
    net_df = metrics["net"]

    # Save wide
    wide_path = RESULTS_DIR / f"network_metrics_wide_{RUN_TAG}.csv"
    wide = pd.concat(
        {
            "to_others": to_df,
            "from_others": from_df,
            "net": net_df,
        },
        axis=1,
    )
    wide.to_csv(wide_path)
    print("[network_metrics] Saved wide CSV:", wide_path)

    # Save tidy/long
    long = to_long(to_df, "to_others").merge(
        to_long(from_df, "from_others"),
        on=["date", "asset"],
        how="inner",
    ).merge(
        to_long(net_df, "net"),
        on=["date", "asset"],
        how="inner",
    )
    long_path = RESULTS_DIR / f"network_metrics_{RUN_TAG}.csv"
    long.to_csv(long_path, index=False)
    print("[network_metrics] Saved long CSV:", long_path)

    # Quick sanity prints
    # Who are the biggest transmitters on the peak TCI date?
    # (Your rolling saved TCI too, but we can just pick a high-vol date by max average net)
    peak_date = net_df.abs().sum(axis=1).idxmax()
    print("\n[network_metrics] Example date (largest overall net magnitude):", peak_date.date())

    top_tx = to_df.loc[peak_date].sort_values(ascending=False).head(3)
    top_rx = from_df.loc[peak_date].sort_values(ascending=False).head(3)
    print("[network_metrics] Top 3 transmitters (to_others) that day:")
    print(top_tx)
    print("[network_metrics] Top 3 receivers (from_others) that day:")
    print(top_rx)

    # Plots
    p_tx = plot_top_k_time_series(
        to_df,
        title="Top Transmitters Over Time (to_others)",
        ylabel="to_others (%)",
        out_path=PLOTS_DIR / f"top_transmitters_{RUN_TAG}.png",
        k=3,
    )
    print("\n[network_metrics] Saved:", p_tx)

    p_rx = plot_top_k_time_series(
        from_df,
        title="Top Receivers Over Time (from_others)",
        ylabel="from_others (%)",
        out_path=PLOTS_DIR / f"top_receivers_{RUN_TAG}.png",
        k=3,
    )
    print("[network_metrics] Saved:", p_rx)
