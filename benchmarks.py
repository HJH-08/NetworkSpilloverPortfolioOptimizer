"""
benchmarks.py

Benchmark portfolio weight generators for fair comparison against your spillover-aware strategy.

Goal:
- produce target weights on the SAME rebalance dates
- using the SAME constraints (long-only, fully invested, w_max, etc.)

Benchmarks included:
1) Equal-weight
2) Minimum-variance (covariance-only) using the same CVX optimizer as spillover_aware_optimizer.py

Typical usage (in run_experiment.py):
- load returns (daily)
- get rebalance dates (or reuse spillover cache dates)
- compute weights for:
    - equal-weight
    - min-variance
    - spillover-aware (already in rebalance_engine.py)
- then pass each weights_df into backtest.run_backtest(...)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from config import WINDOW, STEP
from spillover_aware_optimizer import OptConfig, optimize_min_variance


# -------------------------
# Rebalance schedule helpers
# -------------------------

def compute_rebalance_dates(
    returns_index: pd.DatetimeIndex,
    *,
    window: int = WINDOW,
    step: int = STEP,
) -> pd.DatetimeIndex:
    """
    Produce rebalance dates aligned to the returns index.

    Convention (matches your project):
    - you need `window` days of history up to and including date t
    - so the first eligible t is returns_index[window-1]
    - then every `step` trading days thereafter
    """
    idx = pd.DatetimeIndex(returns_index).sort_values()
    if len(idx) < window:
        raise ValueError(f"Not enough data: len(index)={len(idx)} < window={window}")

    start_pos = window - 1
    positions = range(start_pos, len(idx), step)
    return idx[list(positions)]


# -------------------------
# Equal-weight benchmark
# -------------------------

def equal_weight_over_time(
    rebalance_dates: pd.DatetimeIndex,
    assets: Sequence[str],
    *,
    w_max: Optional[float] = None,
) -> pd.DataFrame:
    """
    Equal-weight benchmark: w_i = 1/N at every rebalance date.

    If w_max is provided and 1/N > w_max, we raise an error because
    equal-weight would violate your constraint. (You can then choose a higher cap.)
    """
    assets = list(assets)
    N = len(assets)
    if N < 1:
        raise ValueError("assets is empty")

    w = 1.0 / N
    if w_max is not None and w > w_max + 1e-12:
        raise ValueError(
            f"Equal-weight violates w_max: 1/N={w:.4f} > w_max={w_max:.4f}. "
            f"Either increase w_max or reduce N."
        )

    W = np.full((len(rebalance_dates), N), w, dtype=float)
    return pd.DataFrame(W, index=pd.to_datetime(rebalance_dates), columns=assets)


# -------------------------
# Min-variance benchmark
# -------------------------

def min_variance_over_time(
    returns_df: pd.DataFrame,
    rebalance_dates: pd.DatetimeIndex,
    *,
    window: int = WINDOW,
    opt_cfg: OptConfig = OptConfig(lam=0.0),
) -> pd.DataFrame:
    """
    Minimum-variance benchmark (covariance-only).

    For each rebalance date t:
    - take the past `window` daily returns up to t
    - compute covariance
    - solve min w'Σw (+ tiny gamma*||w||^2) with the same constraints

    Note:
    - We do NOT use spillover scores here.
    - We reuse the same OptConfig structure so constraints are apples-to-apples.
    """
    rets = returns_df.copy().sort_index()
    rebal = pd.DatetimeIndex(rebalance_dates).sort_values()

    # Only keep rebalance dates that exist in returns index
    rebal = rebal.intersection(rets.index)

    assets = list(rets.columns)
    if len(assets) < 2:
        raise ValueError("Need at least 2 assets for min-variance")

    W_rows: List[np.ndarray] = []
    W_dates: List[pd.Timestamp] = []

    w_prev: Optional[np.ndarray] = None

    for t in rebal:
        end_loc = rets.index.get_loc(t)
        start_loc = end_loc - (window - 1)
        if start_loc < 0:
            continue  # not enough history

        window_rets = rets.iloc[start_loc : end_loc + 1]
        if window_rets.isna().any().any():
            continue  # skip if missing data

        Sigma = window_rets.cov().values

        res = optimize_min_variance(Sigma, cfg=opt_cfg, w_prev=w_prev)
        w = np.asarray(res.w, dtype=float).reshape(-1)

        # Clean numerical dust and renormalize (keeps things tidy)
        w[w < 1e-10] = 0.0
        s = w.sum()
        if s <= 0:
            continue
        w = w / s

        W_rows.append(w)
        W_dates.append(t)
        w_prev = w

    if not W_rows:
        raise RuntimeError("No min-variance weights computed. Check window / date alignment.")

    return pd.DataFrame(W_rows, index=pd.to_datetime(W_dates), columns=assets)


# -------------------------
# Tiny self-test
# -------------------------

if __name__ == "__main__":
    from returns import get_returns_bundle

    bundle = get_returns_bundle(use_cache=True)
    rets = bundle.returns

    rebal_dates = compute_rebalance_dates(rets.index, window=WINDOW, step=STEP)

    # Equal weight
    ew = equal_weight_over_time(rebal_dates, rets.columns, w_max=0.25)
    print("[equal] shape:", ew.shape, "row sum head:", ew.sum(axis=1).head().tolist())

    # Min var
    cfg = OptConfig(lam=0.0, w_max=0.25, long_only=True, fully_invested=True)
    mv = min_variance_over_time(rets, rebal_dates, window=WINDOW, opt_cfg=cfg)
    print("[minvar] shape:", mv.shape, "row sum head:", mv.sum(axis=1).head().tolist())
    print("[minvar] first weights:\n", mv.iloc[0].sort_values(ascending=False).head(8))
