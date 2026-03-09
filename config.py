"""
Single source of truth for the spillover-aware portfolio optimization framework.
Keep *all* experiment knobs here so runs are reproducible and comparable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Sequence, Tuple, Optional
import numpy as np


# -------------------------
# Project / Reproducibility
# -------------------------

PROJECT_NAME: str = "spillover_aware_portfolio_opt"
RUN_TAG: str = "v1_win250_reb20_fevd10"

SEED: int = 42
np.random.seed(SEED)

ROOT_DIR: Path = Path(__file__).resolve().parent

# Raw data cache (e.g., downloaded price CSVs)
DATA_CACHE_DIR: Path = ROOT_DIR / "cache"

# All experiment outputs (plots, reports, model caches)
RESULTS_DIR: Path = ROOT_DIR / "results"
PLOTS_DIR: Path = RESULTS_DIR / "plots"
REPORTS_DIR: Path = RESULTS_DIR / "reports"
CACHE_DIR: Path = RESULTS_DIR / "cache"

for _p in (DATA_CACHE_DIR, RESULTS_DIR, PLOTS_DIR, REPORTS_DIR, CACHE_DIR):
    _p.mkdir(parents=True, exist_ok=True)


# -------------------------
# Data universe / provider
# -------------------------

DATA_PROVIDER: Literal["lseg"] = "lseg"

# Prefer sector ETFs as your core universe. Keep this list stable per experiment.
# Note: RICs vary by venue; adjust to the exact RICs you want to use.
# (These are common US ETF tickers; verify RICs in your environment.)
UNIVERSE: Tuple[str, ...] = (
    "XLF",  # Financials
    "XLK",  # Technology
    "XLE",  # Energy
    "XLV",  # Health Care
    "XLY",  # Consumer Discretionary
    "XLP",  # Consumer Staples
    "XLI",  # Industrials
    "XLB",  # Materials
    "XLU",  # Utilities
    "XLRE", # Real Estate
    "XLC",  # Communication Services
)

# If your LSEG setup expects RICs for ETFs (often "XLK.P" etc.), convert in data_loader.py.
UNIVERSE_IS_RIC: bool = False

# Date bounds for the full sample (ISO-8601). Keep these explicit (no "today") for reproducibility.
START_DATE: str = "2010-01-01"
END_DATE: str = "2025-12-31"

# Which price field to use from the data source. For total-return style prices, you may prefer adjusted.
PRICE_FIELD: str = "TRDPRC_1"  # LSEG example snapshot field; for history you may use a timeseries field in data_loader.py

# Frequency for returns construction
FREQ: Literal["1D"] = "1D"


# -------------------------
# Returns / cleaning choices
# -------------------------

RETURN_TYPE: Literal["log", "simple"] = "log"

# Missing data policy on aligned price panel
# - "drop_any": keep only dates where all assets have prices
# - "ffill_then_drop": forward-fill within a limit, then drop remaining NaNs
MISSING_POLICY: Literal["drop_any", "ffill_then_drop"] = "drop_any"
FFILL_LIMIT: int = 3  # used only if MISSING_POLICY == "ffill_then_drop"


MIN_ASSET_COVERAGE = 0.90  # keep assets with >=90% coverage in sample
MIN_START_DATE = "2015-01-01"  # optional: force a later common start

# Optional outlier robustness (keep off initially for a clean baseline)
WINSORIZE: bool = False
WINSOR_Q: Tuple[float, float] = (0.01, 0.99)


# -------------------------
# Rolling windows / splits
# -------------------------

# Rolling estimation window for VAR / spillovers (trading days)
WINDOW: int = 250

# How often you recompute spillovers/weights (in trading days)
# Common: 20 (monthly-ish) or 5 (weekly-ish)
STEP: int = 20
REBALANCE_EVERY_N_DAYS: int = 20  # keep same as STEP at the start

# Minimum observations required before first portfolio is formed
MIN_OBS: int = WINDOW

# Simple chronological splits for tuning λ and model choices (no leakage)
# You can tune λ on train/valid, then hold out test.
TRAIN_END: str = "2016-12-31"
VALID_END: str = "2019-12-31"
TEST_END: str = END_DATE


# -------------------------
# VAR / FEVD (used in Phase 2)
# -------------------------

VAR_LAG_MAX: int = 10
VAR_LAG_CRITERION: Literal["aic", "bic"] = "bic"

# FEVD horizon (H-step ahead) and method
FEVD_HORIZON: int = 10
FEVD_TYPE: Literal["generalized", "orthogonalized"] = "generalized"  # generalized is order-invariant

# Stability / numerical safety
STABILITY_CHECK: bool = True
RIDGE_EPS: float = 1e-8  # tiny diagonal jitter for near-singular matrices (if needed)


# -------------------------
# Network construction / metrics (Phase 3)
# -------------------------

ZERO_DIAGONAL: bool = True
# (Network thresholding removed for now; spillover matrices are used directly.)


# -------------------------
# Portfolio constraints / trading assumptions (Phase 4/5)
# -------------------------

ALLOW_SHORT: bool = False
WEIGHT_BOUNDS: Tuple[float, float] = (0.0, 0.25)  # cap concentration
SUM_TO_ONE: bool = True

# Transaction costs (bps per unit turnover). Keep explicit.
TCOST_BPS: float = 10.0  # 10 bps as a conservative starting point

# Optional turnover constraint (None disables)
TURNOVER_LIMIT: Optional[float] = None  # e.g. 0.30 means max 30% turnover per rebalance


# -------------------------
# Spillover-aware optimization knobs
# -------------------------

# Baseline spillover penalty used in the main experiment pipeline.
DEFAULT_LAMBDA: float = 0.5

# Candidate lambdas for tuning on validation set.
# Include 0 for "no spillover penalty" to compare apples-to-apples.
LAMBDA_GRID: Tuple[float, ...] = (0.0, 0.1, 0.25, 0.5, 1.0)

# Optimization form for the spillover term:
# - "linear": lambda * s' w
# - "quadratic": lambda * sum_i s_i * w_i^2
SPILLOVER_OBJECTIVE_FORM: Literal["linear", "quadratic"] = "linear"

# Define what the penalty uses (you’ll implement later):
# - "out_spill": penalize holdings in transmitters
# - "centrality": penalize holdings proportional to a centrality score
SPILLOVER_PENALTY_MODE: Literal["out_spill", "centrality"] = "out_spill"


# -------------------------
# Backtest / evaluation (Phase 5)
# -------------------------

ANNUALIZATION_FACTOR: int = 252
RISK_FREE_RATE_ANNUAL: float = 0.0  # set nonzero if you want Sharpe excess returns

# Stress-period slices for crisis analysis (edit to match what you evaluate)
# These are just defaults; update as needed.
STRESS_PERIODS: Tuple[Tuple[str, str, str], ...] = (
    ("GFC_aftershock", "2011-07-01", "2011-12-31"),
    ("Euro_crisis", "2012-05-01", "2012-08-31"),
    ("Volmageddon", "2018-02-01", "2018-02-28"),
    ("Q4_2018_selloff", "2018-10-01", "2018-12-24"),
    ("COVID_crash", "2020-02-15", "2020-04-30"),
    ("Inflation_shock", "2022-01-01", "2022-10-31"),
    ("Banking_stress", "2023-03-06", "2023-03-31"),
)

CALM_PERIODS: Tuple[Tuple[str, str, str], ...] = (
    ("Low_vol_2017", "2017-01-01", "2017-12-31"),
    ("Post_COVID_calm", "2021-04-01", "2021-12-31"),
)


# -------------------------
# Convenience dataclass (optional)
# -------------------------

@dataclass(frozen=True)
class Config:
    project_name: str = PROJECT_NAME
    run_tag: str = RUN_TAG
    seed: int = SEED

    universe: Sequence[str] = field(default_factory=lambda: list(UNIVERSE))
    universe_is_ric: bool = UNIVERSE_IS_RIC
    start_date: str = START_DATE
    end_date: str = END_DATE
    price_field: str = PRICE_FIELD

    return_type: str = RETURN_TYPE
    missing_policy: str = MISSING_POLICY

    window: int = WINDOW
    step: int = STEP
    rebalance_every_n_days: int = REBALANCE_EVERY_N_DAYS

    var_lag_max: int = VAR_LAG_MAX
    var_lag_criterion: str = VAR_LAG_CRITERION
    fevd_horizon: int = FEVD_HORIZON
    fevd_type: str = FEVD_TYPE

    allow_short: bool = ALLOW_SHORT
    weight_bounds: Tuple[float, float] = WEIGHT_BOUNDS
    tcost_bps: float = TCOST_BPS
    turnover_limit: Optional[float] = TURNOVER_LIMIT

    objective_form: str = SPILLOVER_OBJECTIVE_FORM
    default_lambda: float = DEFAULT_LAMBDA
    lambda_grid: Tuple[float, ...] = LAMBDA_GRID
    penalty_mode: str = SPILLOVER_PENALTY_MODE


CFG = Config()


def config_dict() -> dict:
    """
    Return a stable dictionary of config values for reproducibility.
    Includes all UPPERCASE module-level constants.
    """
    out = {}
    for k, v in globals().items():
        if not k.isupper():
            continue
        # Skip modules / classes
        if callable(v):
            continue
        if isinstance(v, Config):
            continue
        try:
            out[k] = v
        except Exception:
            pass
    return out


def save_config(path: Path) -> None:
    """
    Save config dictionary to JSON for run manifests.
    """
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    data = config_dict()
    # Convert Paths to strings for JSON
    for k, v in list(data.items()):
        if isinstance(v, Path):
            data[k] = str(v)
    with path.open("w") as f:
        json.dump(data, f, indent=2, sort_keys=True)
