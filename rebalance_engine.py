"""
rebalance_engine.py

Compute portfolio weights over time using:
- rolling covariance from returns windows
- rolling spillover metrics from cached FEVD results
- optimizer (min-var or spillover-aware)

Outputs:
- weights DataFrame indexed by rebalance date, columns=assets

No backtest returns here yet (that's backtest.py).
This module just produces weights in a leak-free way.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from config import (
    WINDOW,
    REBALANCE_EVERY_N_DAYS,
    RUN_TAG,
    CACHE_DIR,
)


from returns import get_returns_bundle
from spillover_aware_optimizer import (
    OptConfig,
    optimize_min_variance,
    optimize_spillover_aware,
)


# -------------------------
# Loading spillover cache
# -------------------------

@dataclass(frozen=True)
class SpilloverCache:
    dates: pd.DatetimeIndex          # length T
    assets: list[str]                # length N
    W_stack: np.ndarray              # shape (T, N, N)
    TCI: pd.Series                   # length T
    to_others: pd.DataFrame          # (T, N)
    from_others: pd.DataFrame        # (T, N)
    net: pd.DataFrame                # (T, N)


def load_spillover_cache(npz_path: str) -> SpilloverCache:
    """
    Loads the .npz you saved in rolling_spillover.py.

    Expect keys like:
    - dates (as ISO strings)
    - assets
    - W_stack
    - TCI
    - to_others / from_others / net (T x N)
    """
    data = np.load(npz_path, allow_pickle=True)

    dates = pd.to_datetime(data["dates"])
    assets = list(data["assets"].tolist())
    W_stack = data["W_stack"]
    TCI = pd.Series(data["tci"], index=dates, name="TCI")


    to_others = pd.DataFrame(data["to_others"], index=dates, columns=assets)
    from_others = pd.DataFrame(data["from_others"], index=dates, columns=assets)
    net = pd.DataFrame(data["net"], index=dates, columns=assets)

    return SpilloverCache(
        dates=dates,
        assets=assets,
        W_stack=W_stack,
        TCI=TCI,
        to_others=to_others,
        from_others=from_others,
        net=net,
    )


# -------------------------
# Core engine
# -------------------------

def _cov_from_window(window_rets: pd.DataFrame) -> np.ndarray:
    """
    Compute sample covariance. Returns NxN numpy array.
    """
    # ddof=1 by default; that's fine for sample covariance
    Sigma = window_rets.cov().values
    return Sigma


def _systemic_score(
    cache: SpilloverCache,
    date: pd.Timestamp,
    method: str = "to_others",
) -> np.ndarray:
    """
    Choose spillover-based systemic score s_t (length N).

    Methods:
    - 'to_others': use directional "to" (transmitters)
    - 'net_pos': use max(net, 0) (only penalize net transmitters)
    """
    if method == "to_others":
        s = cache.to_others.loc[date].values
    elif method == "net_pos":
        s = np.maximum(cache.net.loc[date].values, 0.0)
    else:
        raise ValueError("method must be 'to_others' or 'net_pos'")
    return s.astype(float)


def compute_weights_over_time(
    *,
    spillover_npz_path: str,
    model: str = "spillover_aware",          # 'min_var' or 'spillover_aware'
    score_method: str = "to_others",         # 'to_others' or 'net_pos'
    opt_cfg: OptConfig = OptConfig(),
    use_cache_prices: bool = True,
) -> pd.DataFrame:
    """
    Returns:
      weights_df: index=rebalance dates (spillover dates), columns=assets

    The rebalance dates are the same as the spillover cache dates,
    which are typically every STEP business days after the first window.
    """
    # Load returns
    bundle = get_returns_bundle(use_cache=use_cache_prices)
    rets = bundle.returns  # daily returns, index = business days, columns = assets

    # Load spillovers
    cache = load_spillover_cache(spillover_npz_path)

    # Align assets: intersection, in cache order
    common = [a for a in cache.assets if a in rets.columns]
    if len(common) < 2:
        raise RuntimeError("Too few common assets between returns and spillover cache.")

    rets = rets[common]
    assets = common

    # Weights storage
    W_rows = []
    W_dates = []

    w_prev = None

    for t in cache.dates:
        # Only compute weights if we have enough returns history up to t
        if t not in rets.index:
            # spillover dates should normally be in returns index; skip if not
            continue

        end_loc = rets.index.get_loc(t)
        start_loc = end_loc - (WINDOW - 1)


        if start_loc < 0:
            # not enough history yet
            continue

        window = rets.iloc[start_loc : end_loc + 1]

        # Safety: if any NaNs, skip (ideally none after your cleaning)
        if window.isna().any().any():
            # you could also choose window.dropna(), but that changes window length
            continue

        Sigma = _cov_from_window(window)

        if model == "min_var":
            res = optimize_min_variance(Sigma, cfg=opt_cfg, w_prev=w_prev)
        elif model == "spillover_aware":
            s = _systemic_score(cache, t, method=score_method)
            # Align s to the common assets ordering
            # cache assets -> common assets mapping
            s_full = pd.Series(s, index=cache.assets)
            s = s_full.loc[assets].values
            res = optimize_spillover_aware(Sigma, s, cfg=opt_cfg, w_prev=w_prev, penalty="linear")
        else:
            raise ValueError("model must be 'min_var' or 'spillover_aware'.")

        w = res.w
        W_rows.append(w)
        W_dates.append(t)
        active = (w > 1e-6).sum()
        print("Active assets:", active)
        top = pd.Series(w, index=assets).sort_values(ascending=False).head(5)
        print(top)


        w_prev = w

    weights_df = pd.DataFrame(W_rows, index=pd.to_datetime(W_dates), columns=assets)
    return weights_df


# -------------------------
# Self-test
# -------------------------

if __name__ == "__main__":
    # Update this path to match your saved file name.
    # Example from your logs:
    # /Users/.../cache/spillovers_v1_win250_reb20_fevd10_win250_step20_H10.npz
    import os


    # A simple heuristic to find an npz in your cache folder
    cache_files = [f for f in os.listdir(CACHE_DIR) if f.endswith(".npz")]
    if not cache_files:
        raise RuntimeError(f"No .npz files found in CACHE_DIR={CACHE_DIR}")

    npz_path = os.path.join(CACHE_DIR, cache_files[-1])
    print("[rebalance] Using:", npz_path)

    cfg = OptConfig(lam=0.5, w_max=0.20, long_only=True, fully_invested=True)
    w_df = compute_weights_over_time(
        spillover_npz_path=npz_path,
        model="spillover_aware",
        score_method="to_others",
        opt_cfg=cfg,
        use_cache_prices=True,
    )

    print("Weights shape:", w_df.shape)
    print("Dates:", w_df.index.min().date(), "->", w_df.index.max().date())
    print("Head:\n", w_df.head())
    print("Row sums (should be 1):\n", w_df.sum(axis=1).head())

    

