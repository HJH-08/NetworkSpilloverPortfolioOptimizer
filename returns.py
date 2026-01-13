"""
returns.py

Construct returns from the cleaned price panel.

Responsibilities:
- Ensure columns are clean (handle possible LSEG MultiIndex-like columns)
- Compute log or simple returns
- Optional winsorization (outlier clipping)
- Provide chronological train/valid/test splits (no leakage)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from config import (
    RETURN_TYPE,
    WINSORIZE,
    WINSOR_Q,
    START_DATE,
    END_DATE,
    TRAIN_END,
    VALID_END,
    TEST_END,
    RUN_TAG,
)
from data_loader import get_price_panel


# -------------------------
# Helpers
# -------------------------

def _flatten_columns(cols) -> pd.Index:
    """
    LSEG sometimes yields columns that display like a MultiIndex with a field name level
    (e.g., TRDPRC_1 over tickers). Depending on how it comes through pandas + casting,
    we defensively flatten.

    Strategy:
    - If true MultiIndex: drop the level that looks like a constant field name
    - Else: if column labels are tuples/strings containing 'TRDPRC_1', strip to last token
    """
    if isinstance(cols, pd.MultiIndex):
        # If one level is constant or looks like a field name, keep the instrument level.
        # Common patterns: (Field, Instrument) or (Instrument, Field)
        lvl0 = cols.get_level_values(0)
        lvl1 = cols.get_level_values(1)

        # If level 0 is mostly identical (field), use level 1
        if lvl0.nunique() == 1 or any(str(x).upper().startswith("TRDPRC") for x in lvl0.unique()):
            return pd.Index(lvl1.astype(str))

        # Else if level 1 is field-like, use level 0
        if lvl1.nunique() == 1 or any(str(x).upper().startswith("TRDPRC") for x in lvl1.unique()):
            return pd.Index(lvl0.astype(str))

        # Fallback: join tuples
        return pd.Index(["|".join(map(str, t)) for t in cols])

    # Not a MultiIndex: try to clean stringified tuples
    clean = []
    for c in cols:
        s = str(c)
        # Common case: "('TRDPRC_1', 'XLB')" -> take last token XLB
        if "TRDPRC" in s and "," in s:
            # crude but effective parsing
            parts = s.replace("(", "").replace(")", "").replace("'", "").replace('"', "").split(",")
            last = parts[-1].strip()
            clean.append(last)
        else:
            clean.append(s)
    return pd.Index(clean)


def _winsorize_df(df: pd.DataFrame, q_low: float, q_high: float) -> pd.DataFrame:
    """
    Column-wise winsorization: clip each asset's returns to its own quantiles.
    Keeps time-series characteristics per asset.
    """
    lo = df.quantile(q_low)
    hi = df.quantile(q_high)
    return df.clip(lower=lo, upper=hi, axis=1)


def _chronological_splits(
    df: pd.DataFrame,
    train_end: str,
    valid_end: str,
    test_end: str,
) -> Dict[str, pd.DataFrame]:
    """
    Splits by date (inclusive end dates), assuming df has DatetimeIndex.
    """
    train = df.loc[:pd.to_datetime(train_end)]
    valid = df.loc[pd.to_datetime(train_end) + pd.Timedelta(days=1): pd.to_datetime(valid_end)]
    test = df.loc[pd.to_datetime(valid_end) + pd.Timedelta(days=1): pd.to_datetime(test_end)]
    return {"train": train, "valid": valid, "test": test}


# -------------------------
# Public API
# -------------------------

@dataclass(frozen=True)
class ReturnsBundle:
    prices: pd.DataFrame
    returns: pd.DataFrame
    splits: Dict[str, pd.DataFrame]


def get_returns_bundle(
    *,
    use_cache: bool = True,
    return_type: str = RETURN_TYPE,
    winsorize: bool = WINSORIZE,
    winsor_q: Tuple[float, float] = WINSOR_Q,
) -> ReturnsBundle:
    """
    Main entry point for Phase 1.5:
    - Load clean prices via data_loader
    - Normalize columns
    - Compute returns
    - Optional winsorization
    - Chronological train/valid/test splits
    """
    prices = get_price_panel(use_cache=use_cache)

    # Normalize columns (critical for LSEG field-level outputs)
    prices = prices.copy()
    prices.columns = _flatten_columns(prices.columns)
    prices = prices.sort_index().sort_index(axis=1)

    # Basic sanity
    if (prices <= 0).any().any():
        raise ValueError("Prices contain non-positive values; cannot compute log returns safely.")

    # Compute returns
    if return_type == "log":
        rets = np.log(prices).diff()
    elif return_type == "simple":
        rets = prices.pct_change()
    else:
        raise ValueError(f"Unknown RETURN_TYPE={return_type}. Use 'log' or 'simple'.")

    rets = rets.dropna(axis=0, how="any")  # after Phase 1 cleaning, this should be safe

    # Optional winsorization
    if winsorize:
        ql, qh = winsor_q
        if not (0.0 < ql < qh < 1.0):
            raise ValueError(f"Invalid winsor quantiles: {winsor_q}")
        rets = _winsorize_df(rets, ql, qh)

    # Splits
    splits = _chronological_splits(rets, TRAIN_END, VALID_END, TEST_END)

    # Final sanity: each split non-empty (or at least train should be)
    if len(splits["train"]) == 0:
        raise RuntimeError("Train split is empty. Check TRAIN_END and your MIN_START_DATE/date range.")
    if len(splits["valid"]) == 0:
        print("[returns] Warning: valid split is empty. (This may be ok early, but check dates.)")
    if len(splits["test"]) == 0:
        print("[returns] Warning: test split is empty. (This may be ok early, but check dates.)")

    return ReturnsBundle(prices=prices, returns=rets, splits=splits)


# -------------------------
# Self-test
# -------------------------

if __name__ == "__main__":
    bundle = get_returns_bundle(use_cache=False)

    prices = bundle.prices
    rets = bundle.returns
    splits = bundle.splits

    print("Prices:", prices.shape, "from", prices.index.min().date(), "to", prices.index.max().date())
    print("Returns:", rets.shape, "from", rets.index.min().date(), "to", rets.index.max().date())
    print("Assets:", list(rets.columns))

    for k, v in splits.items():
        if len(v) > 0:
            print(f"{k}: {v.shape} ({v.index.min().date()} -> {v.index.max().date()})")
        else:
            print(f"{k}: EMPTY")

    # Quick magnitude check
    desc = rets.describe().T[["mean", "std", "min", "max"]]
    print("\nReturn summary (per asset):")
    print(desc)
