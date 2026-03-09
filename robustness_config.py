"""
robustness_config.py

Central settings for one-factor-at-a-time robustness runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from config import DEFAULT_LAMBDA, LAMBDA_GRID


@dataclass(frozen=True)
class CrisisWindow:
    key: str
    label: str
    start: str
    end: str


@dataclass(frozen=True)
class RobustnessConfig:
    # Baseline setup
    baseline_window_length: int = 250
    baseline_rebalance_days: int = 20
    baseline_var_lag_rule: str = "aic"  # "aic" or "bic"
    baseline_var_lag_fixed_p: Optional[int] = None
    baseline_fevd_horizon: int = 10
    lambda_star: float = DEFAULT_LAMBDA

    # One-factor-at-a-time axes
    axis_window_values: Tuple[int, ...] = (200, 250, 300)
    axis_rebalance_values: Tuple[int, ...] = (10, 20, 60)
    axis_lag_rules: Tuple[str, ...] = ("aic", "bic")
    axis_fevd_horizons: Tuple[int, ...] = (10, 20)

    # Separate lambda sweep at baseline settings
    lambda_values: Tuple[float, ...] = LAMBDA_GRID

    # Core crises (fixed across all runs)
    crises: Tuple[CrisisWindow, ...] = (
        CrisisWindow("covid_crash", "COVID Crash", "2020-02-19", "2020-05-29"),
        CrisisWindow("tightening_2022", "2022 Tightening", "2022-01-03", "2022-10-14"),
        CrisisWindow("banking_stress_2023", "2023 Banking Stress", "2023-03-08", "2023-05-01"),
    )


ROBUSTNESS_CFG = RobustnessConfig()
