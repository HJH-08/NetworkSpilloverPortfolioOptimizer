"""
backtest.py

Backtest engine for the spillover-aware portfolio framework.

Inputs:
- weights_df: target weights at rebalance dates (index=rebalance dates, columns=assets)
- returns_df: daily returns (index=days, columns=assets), same assets (or superset)

Key conventions (IMPORTANT):
- weights_df.loc[t] is the target portfolio set *at the close of date t*
  using information up to and including t (no leakage).
- Therefore, those weights are applied to returns on the *next trading day*.

This module handles:
- weight drift between rebalances
- transaction costs (bps * turnover)
- output series for evaluation and plotting

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from config import (
    TCOST_BPS,
    ANNUALIZATION_FACTOR,
    RISK_FREE_RATE_ANNUAL,
)


# -------------------------
# Results container
# -------------------------

@dataclass(frozen=True)
class BacktestResult:
    portfolio_returns: pd.Series     # daily returns
    equity_curve: pd.Series          # cumulative growth from 1.0
    turnover: pd.Series              # per-rebalance turnover (NaN on non-rebalance days)
    tcosts: pd.Series                # transaction cost (as return drag) on rebalance days
    weights_drifted: pd.DataFrame    # daily drifted weights (optional but useful)


# -------------------------
# Helpers
# -------------------------

def _clean_and_align(
    weights_df: pd.DataFrame,
    returns_df: pd.DataFrame,
    *,
    tiny: float = 1e-10,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Align assets and dates, clean tiny weights, and ensure weights sum to 1 on rebalance dates.
    """
    if weights_df.empty:
        raise ValueError("weights_df is empty.")

    # Common assets
    common_assets = [c for c in weights_df.columns if c in returns_df.columns]
    if len(common_assets) < 2:
        raise ValueError("Too few common assets between weights_df and returns_df.")

    w = weights_df[common_assets].copy().sort_index()
    r = returns_df[common_assets].copy().sort_index()

    # Drop any rebalance dates not in returns index (should be rare)
    w = w.loc[w.index.intersection(r.index)]

    # Clean tiny numerical dust and renormalize each row
    w = w.where(w.abs() >= tiny, 0.0)
    row_sums = w.sum(axis=1)

    # If you ever get 0 sum (shouldn't), that's a failure
    if (row_sums <= 0).any():
        bad = row_sums[row_sums <= 0].index[:5]
        raise ValueError(f"Found non-positive row sum in weights on dates: {list(bad)}")

    w = w.div(row_sums, axis=0)
    return w, r


def _max_drawdown(equity: pd.Series) -> float:
    """
    Max drawdown as a negative number (e.g., -0.35 means -35% peak-to-trough).
    """
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min())


def compute_metrics(
    portfolio_returns: pd.Series,
    *,
    annualization: int = ANNUALIZATION_FACTOR,
    rf_annual: float = RISK_FREE_RATE_ANNUAL,
    turnover: Optional[pd.Series] = None,
) -> Dict[str, float]:
    """
    Simple, report-friendly metrics.
    """
    pr = portfolio_returns.dropna()
    if pr.empty:
        raise ValueError("portfolio_returns is empty after dropping NaNs.")

    # Annualized return (geometric)
    equity = (1.0 + pr).cumprod()
    years = len(pr) / annualization
    ann_return = float(equity.iloc[-1] ** (1.0 / years) - 1.0) if years > 0 else np.nan

    # Annualized volatility
    ann_vol = float(pr.std(ddof=1) * np.sqrt(annualization))

    # Sharpe (simple)
    rf_daily = (1.0 + rf_annual) ** (1.0 / annualization) - 1.0
    excess = pr - rf_daily
    sharpe = float(excess.mean() / excess.std(ddof=1) * np.sqrt(annualization)) if excess.std(ddof=1) > 0 else np.nan

    # Max drawdown
    mdd = _max_drawdown(equity)

    out = {
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": mdd,
    }

    if turnover is not None:
        t = turnover.dropna()
        out["avg_turnover"] = float(t.mean()) if len(t) else np.nan

    return out


# -------------------------
# Core backtest
# -------------------------

def run_backtest(
    *,
    weights_df: pd.DataFrame,
    returns_df: pd.DataFrame,
    tcost_bps: float = TCOST_BPS,
    store_daily_weights: bool = True,
) -> BacktestResult:
    """
    Backtest using drifted weights and discrete rebalancing.

    Transaction cost model (simple, standard):
    - turnover_oneway = 0.5 * sum_i |w_target_i - w_current_i|
    - cost_return_drag = (tcost_bps / 10000) * turnover_oneway
    - apply on rebalance day as a negative return

    Weight timing:
    - if target weights are set at date t close,
      they apply to returns on date t+1.
    Implementation:
    - we shift target weights forward by 1 day on the returns index.
    """
    w_targets, rets = _clean_and_align(weights_df, returns_df)

    # Shift targets forward by 1 trading day (apply weights on next day)
    w_targets_shifted = w_targets.reindex(rets.index).ffill()
    w_targets_shifted = w_targets_shifted.shift(1)  # apply starting next day
    w_targets_shifted = w_targets_shifted.dropna(how="all")

    # Limit returns to the period where we have weights
    rets = rets.loc[w_targets_shifted.index]

    assets = list(rets.columns)

    # State: current weights at start of each day (after any rebalance at prior close)
    w_current = w_targets_shifted.iloc[0].values.astype(float)

    port_rets = []
    dates = []
    turnover_series = pd.Series(index=rets.index, dtype=float)
    tcost_series = pd.Series(index=rets.index, dtype=float)

    weights_daily = [] if store_daily_weights else None

    for dt in rets.index:
        r_vec = rets.loc[dt].values.astype(float)

        # 1) Portfolio return from current weights
        pr = float(np.dot(w_current, r_vec))

        # 2) Drift weights due to returns
        #    w_i <- w_i*(1+r_i) / (1+portfolio_return)
        denom = 1.0 + pr
        if denom <= 0:
            # Extremely rare (portfolio return <= -100%), but guard anyway
            denom = 1e-12
        w_drift = w_current * (1.0 + r_vec) / denom

        # 3) If dt is a rebalance day in the *shifted target* schedule, trade to target
        w_target_today = w_targets_shifted.loc[dt].values.astype(float)

        if not np.allclose(w_target_today, w_current, atol=1e-12, rtol=0.0):
            # turnover (one-way)
            turnover = 0.5 * float(np.abs(w_target_today - w_drift).sum())
            cost = (tcost_bps / 10000.0) * turnover

            turnover_series.loc[dt] = turnover
            tcost_series.loc[dt] = cost

            # Apply cost as return drag today
            pr -= cost

            # After paying cost, set holdings to target (at end of rebalance)
            w_current = w_target_today
        else:
            # no rebalance action; keep drifted weights
            w_current = w_drift

        if store_daily_weights:
            weights_daily.append(w_current.copy())

        port_rets.append(pr)
        dates.append(dt)

    portfolio_returns = pd.Series(port_rets, index=pd.to_datetime(dates), name="portfolio_return")
    equity = (1.0 + portfolio_returns).cumprod()
    equity.name = "equity"

    if store_daily_weights:
        weights_drifted = pd.DataFrame(weights_daily, index=pd.to_datetime(dates), columns=assets)
    else:
        weights_drifted = pd.DataFrame(index=pd.to_datetime(dates), columns=assets)

    return BacktestResult(
        portfolio_returns=portfolio_returns,
        equity_curve=equity,
        turnover=turnover_series,
        tcosts=tcost_series,
        weights_drifted=weights_drifted,
    )


# -------------------------
# Self-test
# -------------------------

if __name__ == "__main__":
    from returns import get_returns_bundle
    from rebalance_engine import compute_weights_over_time
    from spillover_aware_optimizer import OptConfig
    import os
    from config import CACHE_DIR

    # Load returns
    bundle = get_returns_bundle(use_cache=True)
    rets = bundle.returns

    # Load a spillover npz (last in cache folder)
    cache_files = [f for f in os.listdir(CACHE_DIR) if f.endswith(".npz")]
    if not cache_files:
        raise RuntimeError(f"No .npz files found in CACHE_DIR={CACHE_DIR}")
    npz_path = os.path.join(CACHE_DIR, cache_files[-1])
    print("[backtest] Using spillover file:", npz_path)

    # Compute weights (spillover-aware example)
    cfg = OptConfig(lam=0.5, w_max=0.20, long_only=True, fully_invested=True)
    w_df = compute_weights_over_time(
        spillover_npz_path=npz_path,
        model="spillover_aware",
        score_method="to_others",
        opt_cfg=cfg,
        use_cache_prices=True,
    )

    # Run backtest
    res = run_backtest(weights_df=w_df, returns_df=rets, tcost_bps=TCOST_BPS)

    print("\nMetrics:")
    print(compute_metrics(res.portfolio_returns, turnover=res.turnover))

    print("\nEquity head/tail:")
    print(res.equity_curve.head())
    print(res.equity_curve.tail())

    print("\nMax drawdown:", _max_drawdown(res.equity_curve))
