"""
crisis_eval.py

Crisis / stress-window evaluation for your strategies.

Reads:
- equity_curves.csv produced by run_experiment.py (columns: EqualWeight, MinVar, Spillover)

Computes, for each window:
- Cumulative return
- Annualized volatility
- Sharpe (simple, using config risk-free)
- Max drawdown (within the window)

Usage:
    python crisis_eval.py

Outputs:
- prints tables
- saves CSVs to cache/results/
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from config import CACHE_DIR, ANNUALIZATION_FACTOR, RISK_FREE_RATE_ANNUAL


# -------------------------
# Config: stress windows
# -------------------------
# You can edit these dates freely.
WINDOWS: List[Tuple[str, str, str]] = [
    ("COVID Crash", "2020-02-19", "2020-04-30"),
    ("2022 Tightening", "2022-01-03", "2022-10-14"),
    ("2023 Banking Stress", "2023-03-06", "2023-03-31"),
]


# -------------------------
# Metrics helpers
# -------------------------

def _max_drawdown_from_equity(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min())


def _annualized_vol(returns: pd.Series, annualization: int = ANNUALIZATION_FACTOR) -> float:
    r = returns.dropna()
    if len(r) < 2:
        return float("nan")
    return float(r.std(ddof=1) * np.sqrt(annualization))


def _sharpe(
    returns: pd.Series,
    *,
    annualization: int = ANNUALIZATION_FACTOR,
    rf_annual: float = RISK_FREE_RATE_ANNUAL,
) -> float:
    r = returns.dropna()
    if len(r) < 2:
        return float("nan")
    rf_daily = (1.0 + rf_annual) ** (1.0 / annualization) - 1.0
    excess = r - rf_daily
    sd = excess.std(ddof=1)
    if sd <= 0:
        return float("nan")
    return float(excess.mean() / sd * np.sqrt(annualization))


def _cum_return(equity: pd.Series) -> float:
    equity = equity.dropna()
    if equity.empty:
        return float("nan")
    return float(equity.iloc[-1] / equity.iloc[0] - 1.0)


def compute_window_metrics(equity: pd.Series) -> Dict[str, float]:
    equity = equity.dropna()
    rets = equity.pct_change().dropna()

    return {
        "CumReturn(%)": 100.0 * _cum_return(equity),
        "AnnVol(%)": 100.0 * _annualized_vol(rets),
        "Sharpe": _sharpe(rets),
        "MaxDD(%)": 100.0 * _max_drawdown_from_equity(equity),
    }


# -------------------------
# Main
# -------------------------

def main() -> None:
    # Locate the equity file (prefer cache/results, but accept local file if you run elsewhere)
    candidate_paths = [
        os.path.join(CACHE_DIR, "results", "equity_curves.csv"),
        os.path.join(os.getcwd(), "equity_curves.csv"),
    ]
    path = next((p for p in candidate_paths if os.path.exists(p)), None)
    if path is None:
        raise FileNotFoundError("Could not find equity_curves.csv. Run run_experiment.py first.")

    eq = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    print("[crisis_eval] loaded:", path)
    print("[crisis_eval] equity shape:", eq.shape, "columns:", list(eq.columns))
    print("[crisis_eval] date range:", eq.index.min().date(), "->", eq.index.max().date())

    out_dir = os.path.join(CACHE_DIR, "results")
    os.makedirs(out_dir, exist_ok=True)

    # Full-sample metrics (useful baseline)
    full_table = {col: compute_window_metrics(eq[col]) for col in eq.columns}
    full_df = pd.DataFrame(full_table).T
    print("\n=== Full sample ===")
    print(full_df.round(3))
    full_df.to_csv(os.path.join(out_dir, "metrics_full_sample.csv"))

    # Windowed tables
    for name, start, end in WINDOWS:
        start_ts = pd.to_datetime(start)
        end_ts = pd.to_datetime(end)

        sliced = eq.loc[(eq.index >= start_ts) & (eq.index <= end_ts)]
        if sliced.empty:
            print(f"\n=== {name} ({start} -> {end}) ===")
            print("No data in this window (check dates).")
            continue

        # ---- FIX: rebase equity at window start ----
        sliced = sliced / sliced.iloc[0]

        table = {col: compute_window_metrics(sliced[col]) for col in sliced.columns}

        df = pd.DataFrame(table).T

        print(f"\n=== {name} ({start} -> {end}) ===")
        print(df.round(3))

        safe_name = name.lower().replace(" ", "_").replace("/", "_")
        df.to_csv(os.path.join(out_dir, f"metrics_{safe_name}.csv"))

    print("\n[saved] metrics_*.csv in", out_dir)
    print("Tip: If Spillover only wins in stress windows (lower MaxDD), that supports your thesis.")


if __name__ == "__main__":
    main()
