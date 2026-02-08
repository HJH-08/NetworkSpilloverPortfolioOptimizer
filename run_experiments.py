"""
run_experiment.py

One-command experiment runner:
- loads returns
- computes weights for:
    1) Equal-weight
    2) Min-variance (covariance-only)
    3) Spillover-aware
- backtests each with the SAME engine + assumptions
- prints a comparison table
- saves outputs to cache/

Usage:
    python run_experiment.py

Tip:
- Make sure you’ve refreshed caches if needed (prices + spillovers).
"""

from __future__ import annotations

import os
from dataclasses import asdict
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from config import CACHE_DIR, REPORTS_DIR, TCOST_BPS, WINDOW, STEP, WEIGHT_BOUNDS
from returns import get_returns_bundle
from benchmarks import equal_weight_over_time, min_variance_over_time
from spillover_aware_optimizer import OptConfig
from rebalance_engine import compute_weights_over_time
from backtest import run_backtest, compute_metrics


def _pick_spillover_npz() -> str:
    """
    Pick the most recently modified .npz in CACHE_DIR.
    (Better than lexicographic ordering.)
    """
    files = [os.path.join(CACHE_DIR, f) for f in os.listdir(CACHE_DIR) if f.endswith(".npz")]
    if not files:
        raise RuntimeError(f"No .npz spillover files found in CACHE_DIR={CACHE_DIR}")
    files.sort(key=lambda p: os.path.getmtime(p))
    return files[-1]


def _load_spillover_dates(npz_path: str) -> pd.DatetimeIndex:
    data = np.load(npz_path, allow_pickle=True)
    if "dates" not in data.files:
        raise RuntimeError("Spillover npz missing 'dates' field; cannot align rebalance dates.")
    return pd.to_datetime(data["dates"])


def _align_weights_to_common_assets(
    w: pd.DataFrame, assets: pd.Index
) -> pd.DataFrame:
    """
    Ensure all strategies have identical asset columns (same order).
    Any missing columns get filled with 0 and renormalized.
    """
    w2 = w.copy()
    # Add missing columns
    for a in assets:
        if a not in w2.columns:
            w2[a] = 0.0
    # Drop extra columns
    w2 = w2[list(assets)]
    # Clean dust + renormalize each row
    w2 = w2.where(w2.abs() >= 1e-10, 0.0)
    s = w2.sum(axis=1)
    if (s <= 0).any():
        bad = s[s <= 0].index[:5]
        raise ValueError(f"Found zero/negative weight row sums in weights: {list(bad)}")
    w2 = w2.div(s, axis=0)
    return w2


def _run_strategy(
    name: str,
    weights_df: pd.DataFrame,
    returns_df: pd.DataFrame,
    *,
    tcost_bps: float,
) -> Tuple[pd.Series, Dict[str, float]]:
    """
    Backtest + metrics for one strategy.
    """
    res = run_backtest(weights_df=weights_df, returns_df=returns_df, tcost_bps=tcost_bps)
    met = compute_metrics(res.portfolio_returns, turnover=res.turnover)
    return res.equity_curve, met


def main() -> None:
    # ----------------------------
    # Load returns (daily)
    # ----------------------------
    bundle = get_returns_bundle(use_cache=True)
    rets = bundle.returns.dropna(how="any")  # keep it strict for fair comparison
    assets = rets.columns
    print("[run] returns shape:", rets.shape, "assets:", list(assets))

    # Use spillover cache dates so ALL strategies share identical rebalance dates
    npz_path = _pick_spillover_npz()
    print("[run] Using spillover npz:", npz_path)
    rebal_dates = _load_spillover_dates(npz_path).intersection(rets.index)
    if len(rebal_dates) == 0:
        raise RuntimeError("No spillover rebalance dates overlap returns index.")
    print("[run] #rebalance dates:", len(rebal_dates), "first/last:", rebal_dates[0].date(), rebal_dates[-1].date())

    # ----------------------------
    # 1) Equal weight
    # ----------------------------
    w_max = WEIGHT_BOUNDS[1]
    ew_w = equal_weight_over_time(rebal_dates, assets, w_max=w_max)

    # ----------------------------
    # 2) Min variance (cov-only)
    # ----------------------------
    mv_cfg = OptConfig(lam=0.0, w_max=w_max, long_only=True, fully_invested=True)
    mv_w = min_variance_over_time(rets, rebal_dates, window=WINDOW, opt_cfg=mv_cfg)

    # ----------------------------
    # 3) Spillover aware
    # ----------------------------
    sp_cfg = OptConfig(lam=0.5, w_max=w_max, long_only=True, fully_invested=True)
    sp_w = compute_weights_over_time(
        spillover_npz_path=npz_path,
        model="spillover_aware",
        score_method="to_others",
        opt_cfg=sp_cfg,
        use_cache_prices=True,
    )

    # ----------------------------
    # Align all weight panels to identical assets
    # ----------------------------
    ew_w = _align_weights_to_common_assets(ew_w, assets)
    mv_w = _align_weights_to_common_assets(mv_w, assets)
    sp_w = _align_weights_to_common_assets(sp_w, assets)

    # ----------------------------
    # Backtest each
    # ----------------------------
    results = {}
    equity = {}

    for name, wdf in [("EqualWeight", ew_w), ("MinVar", mv_w), ("Spillover", sp_w)]:
        eq, met = _run_strategy(name, wdf, rets, tcost_bps=TCOST_BPS)
        results[name] = met
        equity[name] = eq
        print(f"[done] {name}:", met)

    # ----------------------------
    # Comparison table
    # ----------------------------
    table = pd.DataFrame(results).T
    # Make it prettier
    table["ann_return"] = table["ann_return"] * 100.0
    table["ann_vol"] = table["ann_vol"] * 100.0
    table["max_drawdown"] = table["max_drawdown"] * 100.0
    table["avg_turnover"] = table["avg_turnover"] * 100.0
    table = table[["ann_return", "ann_vol", "sharpe", "max_drawdown", "avg_turnover"]]
    table = table.rename(
        columns={
            "ann_return": "AnnReturn(%)",
            "ann_vol": "AnnVol(%)",
            "sharpe": "Sharpe",
            "max_drawdown": "MaxDD(%)",
            "avg_turnover": "AvgTurnover(%)",
        }
    )

    print("\n=== Comparison (same data / same costs / same constraints) ===")
    print(table.round(3))

    # ----------------------------
    # Save outputs
    # ----------------------------
    out_dir = REPORTS_DIR
    os.makedirs(out_dir, exist_ok=True)

    table_path = os.path.join(out_dir, "comparison_table.csv")
    table.to_csv(table_path)

    equity_df = pd.DataFrame(equity)
    equity_path = os.path.join(out_dir, "equity_curves.csv")
    equity_df.to_csv(equity_path)

    ew_w.to_csv(os.path.join(out_dir, "weights_equalweight.csv"))
    mv_w.to_csv(os.path.join(out_dir, "weights_minvar.csv"))
    sp_w.to_csv(os.path.join(out_dir, "weights_spillover.csv"))

    print("\n[saved]")
    print(" -", table_path)
    print(" -", equity_path)
    print(" - weights_*.csv in", out_dir)


if __name__ == "__main__":
    main()
