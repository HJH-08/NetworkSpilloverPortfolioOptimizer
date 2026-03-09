"""
data_loader.py

Fetch/load price history, align dates into a clean price panel, and handle missing data.

Design goals:
- One function that returns a clean DataFrame: index = DatetimeIndex, columns = assets, values = prices
- Supports:
  (1) Load from cached CSV (fast + reproducible)
  (2) Fetch from LSEG Data Library (Desktop session), then cache to CSV
- Missing data policy is controlled by config.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence, Tuple

import pandas as pd

from config import (
    DATA_CACHE_DIR,
    REPORTS_DIR,
    DATA_PROVIDER,
    UNIVERSE,
    UNIVERSE_IS_RIC,
    START_DATE,
    END_DATE,
    PRICE_FIELD,
    MISSING_POLICY,
    FFILL_LIMIT,
    RUN_TAG,
    MIN_ASSET_COVERAGE,
    MIN_START_DATE,
    RETURN_TYPE,
)


# -------------------------
# Utilities
# -------------------------

def _ensure_datetime_index(df: pd.DataFrame, date_col: str = "Date") -> pd.DataFrame:
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], utc=False)
        df = df.set_index(date_col)
    else:
        # If we have an unnamed index column from CSV, promote it
        unnamed_cols = [c for c in df.columns if str(c).lower().startswith("unnamed")]
        if len(unnamed_cols) == 1:
            c0 = unnamed_cols[0]
            df[c0] = pd.to_datetime(df[c0], utc=False)
            df = df.set_index(c0)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=False)
    df = df.sort_index()
    return df


def _pivot_long_to_wide(
    df: pd.DataFrame,
    date_col_candidates: Tuple[str, ...] = ("Date", "DATE", "date", "TIMESTAMP", "timestamp"),
    asset_col_candidates: Tuple[str, ...] = ("Instrument", "RIC", "ric", "Ticker", "ticker", "universe"),
    value_col_candidates: Tuple[str, ...] = (PRICE_FIELD, "Price", "PRICE", "value", "VALUE"),
) -> pd.DataFrame:
    """
    Try to coerce a typical LSEG-ish long format into a wide panel:
      rows: dates
      cols: instruments
      vals: price
    """
    cols = set(df.columns)

    date_col = next((c for c in date_col_candidates if c in cols), None)
    asset_col = next((c for c in asset_col_candidates if c in cols), None)
    value_col = next((c for c in value_col_candidates if c in cols), None)

    if date_col is None or asset_col is None:
        # Already wide?
        return df

    if value_col is None:
        # If PRICE_FIELD not present, fall back to the last column (common in some exports)
        value_col = df.columns[-1]

    out = df[[date_col, asset_col, value_col]].copy()
    out[date_col] = pd.to_datetime(out[date_col], utc=False)
    out = out.pivot(index=date_col, columns=asset_col, values=value_col)
    out.index.name = "Date"
    return out.sort_index()


def _apply_missing_policy(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Apply missing data policy from config.py:
    - "drop_any": drop any date where at least 1 asset is missing
    - "ffill_then_drop": forward fill up to FFILL_LIMIT, then drop remaining NaNs
    """
    if MISSING_POLICY == "drop_any":
        return prices.dropna(axis=0, how="any")

    if MISSING_POLICY == "ffill_then_drop":
        filled = prices.ffill(limit=FFILL_LIMIT)
        return filled.dropna(axis=0, how="any")

    raise ValueError(f"Unknown MISSING_POLICY: {MISSING_POLICY}")


def _normalize_price_columns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize price-panel columns to stable string asset identifiers.
    Handles common LSEG MultiIndex layouts like (Field, Instrument).
    """
    out = prices.copy()
    if isinstance(out.columns, pd.MultiIndex):
        lvl0 = out.columns.get_level_values(0).astype(str)
        lvl1 = out.columns.get_level_values(1).astype(str)

        if lvl0.nunique() == 1 or any(x.upper().startswith("TRDPRC") for x in lvl0.unique()):
            out.columns = lvl1
        elif lvl1.nunique() == 1 or any(x.upper().startswith("TRDPRC") for x in lvl1.unique()):
            out.columns = lvl0
        else:
            out.columns = pd.Index(["|".join(map(str, t)) for t in out.columns])

    out.columns = out.columns.astype(str)
    return out


def _maybe_convert_to_ric(universe: Sequence[str]) -> Sequence[str]:
    """
    If your universe is given as tickers (e.g., XLK) but LSEG expects RICs,
    do the mapping here. Leave as-is by default.

    IMPORTANT: RIC formats for ETFs can vary by venue/data entitlement.
    If you know the exact RICs, set UNIVERSE_IS_RIC=True in config.py and
    put the RICs directly in UNIVERSE.
    """
    if UNIVERSE_IS_RIC:
        return universe
    # Default: pass through unchanged
    return universe


# -------------------------
# CSV loading / caching
# -------------------------

def load_prices_from_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = _pivot_long_to_wide(df)
    df = _ensure_datetime_index(df, date_col="Date" if "Date" in df.columns else "Date")
    # If the CSV came out wide but has an unnamed index column
    if df.columns.dtype == "object" and any(str(c).startswith("Unnamed") for c in df.columns):
        df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")]
    return df.sort_index()


def save_prices_to_csv(prices: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = prices.copy()
    out.index.name = "Date"
    out.to_csv(path)


# -------------------------
# LSEG fetch (Desktop session)
# -------------------------

def fetch_prices_lseg_desktop(
    universe: Sequence[str],
    start_date: str,
    end_date: str,
    price_field: str,
) -> pd.DataFrame:
    """
    Fetch historical daily prices from LSEG Data Library (Desktop Workspace).

    IMPORTANT:
    - A plain get_data(universe, fields=[...]) often returns a snapshot.
    - For a time series, we must request a *timeseries* / *history* endpoint.

    This function tries common LSEG patterns:
    1) ld.get_history(...)
    2) ld.get_data(..., parameters={SDate, EDate, Frq})
    """
    import lseg.data as ld
    import pandas as pd

    ld.open_session("desktop.workspace")
    try:
        # --- Attempt 1: get_history (most explicit if available) ---
        if hasattr(ld, "get_history"):
            df = ld.get_history(
                universe=list(universe),
                fields=[price_field],
                start=start_date,
                end=end_date,
                interval="daily",
            )
            if isinstance(df, pd.DataFrame) and not df.empty:
                wide = _pivot_long_to_wide(df)
                wide = _ensure_datetime_index(wide, date_col="Date")
                return wide.loc[start_date:end_date].sort_index()

        # --- Attempt 2: get_data with time-series parameters (environment dependent) ---
        df = ld.get_data(
            universe=list(universe),
            fields=[price_field],
            parameters={
                "SDate": start_date,
                "EDate": end_date,
                "Frq": "D",
            },
        )

        if not isinstance(df, pd.DataFrame) or df.empty:
            raise RuntimeError("LSEG returned empty result for historical request.")

        wide = _pivot_long_to_wide(df)

        # If still no date column/index → we got a snapshot again
        if not isinstance(wide.index, pd.DatetimeIndex) and "Date" not in wide.columns:
            raise RuntimeError(
                "Still receiving SNAPSHOT data (no Date column) when requesting history. "
                "Your LSEG library/environment likely needs a different historical API/field. "
                "Inspect LSEG docs for your entitlement and switch the historical call accordingly."
            )

        wide = _ensure_datetime_index(wide, date_col="Date")
        return wide.loc[start_date:end_date].sort_index()

    finally:
        ld.close_session()



# -------------------------
# Public API
# -------------------------

def get_price_panel(
    *,
    use_cache: bool = True,
    cache_name: Optional[str] = None,
    universe: Sequence[str] = UNIVERSE,
    start_date: str = START_DATE,
    end_date: str = END_DATE,
    price_field: str = PRICE_FIELD,
    return_metadata: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, Any]]:
    """
    Main entry point:
      - loads cached CSV if available (and use_cache=True)
      - otherwise fetches from provider, then caches

    Returns:
      - cleaned prices with aligned dates and missing-data policy applied
      - if return_metadata=True, returns (prices, provenance_metadata)
    """
    cache_name = cache_name or f"prices_{RUN_TAG}.csv"
    cache_path = DATA_CACHE_DIR / cache_name
    source_mode = "cached_csv"
    source_desc = str(cache_path.resolve())
    requested_universe = [str(x) for x in universe]

    if use_cache and cache_path.exists():
        prices = load_prices_from_csv(cache_path)
    else:
        if DATA_PROVIDER != "lseg":
            raise ValueError(f"Unsupported DATA_PROVIDER={DATA_PROVIDER}. Only 'lseg' is implemented here.")

        uni = _maybe_convert_to_ric(universe)
        prices = fetch_prices_lseg_desktop(uni, start_date, end_date, price_field)
        save_prices_to_csv(prices, cache_path)
        source_mode = "lseg_fetch_then_cached_csv"
        source_desc = f"lseg.data history request cached at {cache_path.resolve()}"

    prices = _normalize_price_columns(prices)
    prices = prices.sort_index().sort_index(axis=1)

    raw_assets = [str(c) for c in prices.columns]
    raw_n_dates = int(len(prices))
    raw_n_assets = int(prices.shape[1])

    # -------------------------
    # Asset coverage filter (prevents drop_any nuking everything)
    # -------------------------
    min_start_date_applied = None
    if "MIN_START_DATE" in globals() and MIN_START_DATE:
        min_start_date_applied = str(pd.to_datetime(MIN_START_DATE).date())
        prices = prices.loc[pd.to_datetime(MIN_START_DATE):]

    # Drop assets that are mostly missing (bad identifiers or short history)
    coverage = prices.notna().mean(axis=0)
    keep_cols = coverage[coverage >= MIN_ASSET_COVERAGE].index.tolist()
    dropped = [c for c in prices.columns if c not in keep_cols]
    dropped_assets = [
        {
            "asset": str(a),
            "reason": "coverage_below_threshold",
            "coverage": float(coverage.loc[a]),
            "coverage_threshold": float(MIN_ASSET_COVERAGE),
        }
        for a in dropped
    ]

    if len(keep_cols) == 0:
        raise RuntimeError(
            "All assets fail MIN_ASSET_COVERAGE. "
            "Likely wrong identifiers (RICs) or too-long date range."
        )

    if dropped:
        print(f"[data_loader] Dropping {len(dropped)} assets for low coverage: {dropped}")
        dropped_path = REPORTS_DIR / f"dropped_assets_{RUN_TAG}.txt"
        with open(dropped_path, "w") as f:
            f.write("Dropped assets (low coverage):\n")
            for d in dropped_assets:
                f.write(f"{d['asset']} (coverage={d['coverage']:.4f}, threshold={d['coverage_threshold']:.2f})\n")


    prices = prices[keep_cols]
    post_coverage_n_dates = int(len(prices))
    post_coverage_n_assets = int(prices.shape[1])

    # Apply missing policy (alignment + cleaning)
    prices = _apply_missing_policy(prices)

    prices = prices.infer_objects(copy=False)
    prices = prices.apply(pd.to_numeric, errors="coerce")
    prices = prices.dropna(axis=0, how="any")

    # Final sanity checks
    if prices.isna().any().any():
        raise RuntimeError("Price panel still contains NaNs after missing-data policy. Check config.")
    if len(prices) == 0:
        raise RuntimeError("No rows left after filtering/cleaning. Check date range or missing policy.")

    provenance = {
        "run_tag": RUN_TAG,
        "provider_used": DATA_PROVIDER,
        "raw_source_used": source_desc,
        "source_mode": source_mode,
        "cache_file": str(cache_path.resolve()),
        "price_field_used": price_field,
        "universe_requested": requested_universe,
        "universe_requested_count": len(requested_universe),
        "universe_raw_columns": raw_assets,
        "universe_raw_columns_count": raw_n_assets,
        "assets_retained": [str(c) for c in prices.columns],
        "assets_retained_count": int(prices.shape[1]),
        "assets_dropped": dropped_assets,
        "start_date_requested": str(pd.to_datetime(start_date).date()),
        "end_date_requested": str(pd.to_datetime(end_date).date()),
        "start_date_actual": str(prices.index.min().date()),
        "end_date_actual": str(prices.index.max().date()),
        "min_start_date_applied": min_start_date_applied,
        "price_rows_raw": raw_n_dates,
        "price_rows_post_coverage": post_coverage_n_dates,
        "price_rows_final": int(len(prices)),
        "price_observations_raw": raw_n_dates * raw_n_assets,
        "price_observations_post_coverage": post_coverage_n_dates * post_coverage_n_assets,
        "price_observations_final": int(prices.size),
        "missing_data_policy": MISSING_POLICY,
        "ffill_limit": int(FFILL_LIMIT) if MISSING_POLICY == "ffill_then_drop" else None,
        "min_asset_coverage": float(MIN_ASSET_COVERAGE),
        "return_type": RETURN_TYPE,
    }

    if return_metadata:
        return prices, provenance
    return prices


if __name__ == "__main__":
    prices = get_price_panel(use_cache=False, universe=("META.O",))
    print(prices.head())
    print(prices.tail())
    print("Shape:", prices.shape)
