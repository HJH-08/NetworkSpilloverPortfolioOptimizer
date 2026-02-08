"""
rolling_spillovers.py

Phase 2C: Rolling spillover estimation over time.

For each evaluation date t (typically on a rebalance schedule):
- Take the last WINDOW returns (t-WINDOW+1 ... t)
- Fit VAR (with auto lag selection)
- Compute generalized FEVD spillover matrix W_t (network adjacency)
- Store:
    - W_t for each t (as np.ndarray)
    - total connectedness TCI_t
    - directional to/from/net series

Outputs are cached to disk for reproducibility and speed.

Typical usage:
    python rolling_spillovers.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from returns import get_returns_bundle
from var_model import fit_var, VarFitError
from fevd import compute_gfevd

# -------------------------
# Config helpers (robust to missing config keys)
# -------------------------

def _cfg(name: str, default):
    try:
        import config
        return getattr(config, name, default)
    except Exception:
        return default


WINDOW = _cfg("WINDOW", 250)
REBALANCE_EVERY_N_DAYS = _cfg("REBALANCE_EVERY_N_DAYS", 20)
FEVD_HORIZON = _cfg("FEVD_HORIZON", 10)
RUN_TAG = _cfg("RUN_TAG", "run")

# directories
RESULTS_DIR = Path(_cfg("RESULTS_DIR", "results"))
REPORTS_DIR = Path(_cfg("REPORTS_DIR", RESULTS_DIR / "reports"))
CACHE_DIR = Path(_cfg("CACHE_DIR", RESULTS_DIR / "cache"))
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Behavior knobs
MAX_FAILS_IN_ROW = _cfg("MAX_FAILS_IN_ROW", 50)  # stop if something is badly broken
VERBOSE = _cfg("VERBOSE_ROLLING", True)

# If you want to estimate only on train/valid/test separately, you can adjust later
SPLIT_TO_USE = _cfg("ROLLING_SPLIT", "returns")  # "train" | "valid" | "test" | "returns"


# -------------------------
# Data structures
# -------------------------

@dataclass(frozen=True)
class RollingSpillovers:
    dates: pd.DatetimeIndex          # evaluation dates
    assets: List[str]               # asset names (columns)
    W_stack: np.ndarray             # shape (T_eval, N, N)
    tci: pd.Series                  # total connectedness over time
    to_others: pd.DataFrame         # shape (T_eval, N)
    from_others: pd.DataFrame       # shape (T_eval, N)
    net: pd.DataFrame               # shape (T_eval, N)
    meta: Dict[str, object]         # parameters, counts, etc.


# -------------------------
# Core logic
# -------------------------

def _evaluation_indices(n_rows: int, window: int, step: int) -> np.ndarray:
    """
    Return integer indices in [window-1 .. n_rows-1] spaced by 'step'.
    These indices indicate the END of each rolling window.
    """
    start = window - 1
    if n_rows <= start:
        return np.array([], dtype=int)
    return np.arange(start, n_rows, step, dtype=int)


def _save_npz(path: Path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def _load_npz(path: Path) -> Dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=False)
    return {k: data[k] for k in data.files}


def run_rolling_spillovers(
    *,
    use_cache: bool = True,
    force_recompute: bool = False,
) -> RollingSpillovers:
    """
    Main entry point.
    Computes rolling spillovers and returns a structured object with all outputs.
    """

    cache_path = CACHE_DIR / f"spillovers_{RUN_TAG}_win{WINDOW}_step{REBALANCE_EVERY_N_DAYS}_H{FEVD_HORIZON}.npz"
    summary_csv = REPORTS_DIR / f"spillovers_summary_{RUN_TAG}.csv"

    if use_cache and (not force_recompute) and cache_path.exists():
        if VERBOSE:
            print(f"[rolling] Loading cached spillovers: {cache_path}")
        blob = _load_npz(cache_path)

        dates = pd.to_datetime(blob["dates"].astype("datetime64[ns]"))
        assets = [s for s in blob["assets"].astype(str).tolist()]
        W_stack = blob["W_stack"]
        tci = pd.Series(blob["tci"], index=dates, name="TCI")
        to_others = pd.DataFrame(blob["to_others"], index=dates, columns=assets)
        from_others = pd.DataFrame(blob["from_others"], index=dates, columns=assets)
        net = pd.DataFrame(blob["net"], index=dates, columns=assets)

        meta = {
            "WINDOW": WINDOW,
            "STEP": REBALANCE_EVERY_N_DAYS,
            "H": FEVD_HORIZON,
            "RUN_TAG": RUN_TAG,
            "loaded_from_cache": True,
        }
        return RollingSpillovers(
            dates=dates,
            assets=assets,
            W_stack=W_stack,
            tci=tci,
            to_others=to_others,
            from_others=from_others,
            net=net,
            meta=meta,
        )

    # Load returns
    bundle = get_returns_bundle(use_cache=use_cache)

    if SPLIT_TO_USE == "train":
        rets = bundle.splits["train"]
    elif SPLIT_TO_USE == "valid":
        rets = bundle.splits["valid"]
    elif SPLIT_TO_USE == "test":
        rets = bundle.splits["test"]
    else:
        rets = bundle.returns

    rets = rets.dropna(axis=0, how="any")
    assets = list(rets.columns)
    n_rows, n_assets = rets.shape

    eval_idx = _evaluation_indices(n_rows, WINDOW, REBALANCE_EVERY_N_DAYS)
    if len(eval_idx) == 0:
        raise RuntimeError(f"[rolling] Not enough data rows ({n_rows}) for WINDOW={WINDOW}.")

    # Pre-allocate lists (we’ll stack later)
    dates_out: List[pd.Timestamp] = []
    W_list: List[np.ndarray] = []
    tci_list: List[float] = []
    to_list: List[np.ndarray] = []
    from_list: List[np.ndarray] = []
    net_list: List[np.ndarray] = []

    fails_in_row = 0
    n_fail = 0

    if VERBOSE:
        print(f"[rolling] Computing spillovers:")
        print(f"          rows={n_rows}, assets={n_assets}, WINDOW={WINDOW}, STEP={REBALANCE_EVERY_N_DAYS}, H={FEVD_HORIZON}")
        print(f"          evaluation points={len(eval_idx)}")

    for k, end_i in enumerate(eval_idx):
        window = rets.iloc[end_i - WINDOW + 1 : end_i + 1]
        date_t = rets.index[end_i]

        try:
            var_out = fit_var(window, lag=None)  # auto lag selection inside
            sp = compute_gfevd(var_out.model_result, H=FEVD_HORIZON, scale_to_100=True)

            # Store
            dates_out.append(date_t)
            W_list.append(sp.W.copy())  # NxN
            tci_list.append(float(sp.total_connectedness))
            to_list.append(sp.to_others.values.astype(float))
            from_list.append(sp.from_others.values.astype(float))
            net_list.append(sp.net.values.astype(float))

            fails_in_row = 0

        except VarFitError as e:
            n_fail += 1
            fails_in_row += 1
            if VERBOSE:
                print(f"[rolling] SKIP {date_t.date()} (VAR issue): {e}")
        except Exception as e:
            n_fail += 1
            fails_in_row += 1
            if VERBOSE:
                print(f"[rolling] SKIP {date_t.date()} (other issue): {e}")

        if fails_in_row >= MAX_FAILS_IN_ROW:
            raise RuntimeError(
                f"[rolling] Too many consecutive failures ({fails_in_row}). "
                "Check identifiers, missingness, or VAR stability."
            )

        # Light progress print
        if VERBOSE and (k % 25 == 0 or k == len(eval_idx) - 1):
            print(f"[rolling] progress {k+1}/{len(eval_idx)} | kept={len(dates_out)} | skipped={n_fail}")

    if len(dates_out) == 0:
        raise RuntimeError("[rolling] All windows failed. Check VAR stability and data integrity.")

    dates = pd.DatetimeIndex(dates_out)
    W_stack = np.stack(W_list, axis=0)  # (T_eval, N, N)

    tci = pd.Series(tci_list, index=dates, name="TCI")
    to_others = pd.DataFrame(np.vstack(to_list), index=dates, columns=assets)
    from_others = pd.DataFrame(np.vstack(from_list), index=dates, columns=assets)
    net = pd.DataFrame(np.vstack(net_list), index=dates, columns=assets)

    # Save cache
    if VERBOSE:
        print(f"[rolling] Saving cache: {cache_path}")
    _save_npz(
        cache_path,
        dates=dates.values.astype("datetime64[ns]"),
        assets=np.array(assets, dtype="U"),
        W_stack=W_stack,
        tci=tci.values.astype(float),
        to_others=to_others.values.astype(float),
        from_others=from_others.values.astype(float),
        net=net.values.astype(float),
    )

    # Save a friendly CSV summary (TCI + average transmitters/receivers)
    summary = pd.DataFrame(index=dates)
    summary["TCI"] = tci
    # system-wide rankings per date can be computed later; here store for quick plotting
    summary.to_csv(summary_csv)
    if VERBOSE:
        print(f"[rolling] Saved summary CSV: {summary_csv}")

    meta = {
        "WINDOW": WINDOW,
        "STEP": REBALANCE_EVERY_N_DAYS,
        "H": FEVD_HORIZON,
        "RUN_TAG": RUN_TAG,
        "n_eval_points": int(len(eval_idx)),
        "n_kept": int(len(dates)),
        "n_skipped": int(n_fail),
        "split_used": SPLIT_TO_USE,
        "loaded_from_cache": False,
    }

    return RollingSpillovers(
        dates=dates,
        assets=assets,
        W_stack=W_stack,
        tci=tci,
        to_others=to_others,
        from_others=from_others,
        net=net,
        meta=meta,
    )


# -------------------------
# Self-test / CLI
# -------------------------

if __name__ == "__main__":
    out = run_rolling_spillovers(use_cache=True, force_recompute=False)

    print("\n[rolling] DONE")
    print("Kept points:", out.meta["n_kept"], "Skipped:", out.meta["n_skipped"])
    print("W_stack shape:", out.W_stack.shape, "(T_eval, N, N)")
    print("TCI head:")
    print(out.tci.head())
    print("TCI tail:")
    print(out.tci.tail())
