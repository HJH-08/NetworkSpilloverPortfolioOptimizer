"""
run_robustness.py

One-factor-at-a-time robustness runner for spillover-aware backtests.

Examples:
    python run_robustness.py
    python run_robustness.py --axis window
    python run_robustness.py --section lambda_sensitivity
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

# Keep matplotlib cache writable/reproducible in constrained environments.
_MPLCONFIGDIR_DEFAULT = Path(__file__).resolve().parent / "results" / "reports" / ".mplconfig"
_MPLCONFIGDIR_DEFAULT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR_DEFAULT))

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

from backtest import compute_metrics, run_backtest
from benchmarks import equal_weight_over_time, mean_variance_over_time, min_variance_over_time
from config import ANNUALIZATION_FACTOR, REPORTS_DIR, RUN_TAG, TCOST_BPS, VAR_LAG_MAX, WEIGHT_BOUNDS
from rebalance_engine import compute_weights_over_time
from returns import get_returns_bundle
from rolling_spillover import run_rolling_spillovers
from robustness_config import ROBUSTNESS_CFG, CrisisWindow, RobustnessConfig
from spillover_aware_optimizer import OptConfig


@dataclass(frozen=True)
class RunSpec:
    section: str
    axis_name: str
    axis_value: str
    window_length: int
    rebalance_days: int
    var_lag_rule: str
    var_lag_fixed_p: Optional[int]
    fevd_horizon: int
    lambda_value: float


STRATEGY_ORDER = ["EqualWeight", "MeanVar", "MinVar", "Spillover"]
STRATEGY_COLORS = {
    "EqualWeight": "#7F7F7F",  # grey
    "MeanVar": "#1F77B4",      # blue
    "MinVar": "#FF7F0E",       # orange
    "Spillover": "#D62728",    # red
}


def _git_commit_short() -> Optional[str]:
    try:
        out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
        return out or None
    except Exception:
        return None


def _safe_token(x: object) -> str:
    if isinstance(x, float):
        return f"{x:g}".replace(".", "p")
    return str(x).replace(" ", "_").replace("/", "_").replace(".", "p")


def _lag_token(rule: str, fixed_p: Optional[int]) -> str:
    if fixed_p is not None:
        return f"p{fixed_p}"
    return rule.lower()


def _build_baseline(cfg: RobustnessConfig) -> RunSpec:
    return RunSpec(
        section="axis_robustness",
        axis_name="baseline",
        axis_value="baseline",
        window_length=cfg.baseline_window_length,
        rebalance_days=cfg.baseline_rebalance_days,
        var_lag_rule=cfg.baseline_var_lag_rule.lower(),
        var_lag_fixed_p=cfg.baseline_var_lag_fixed_p,
        fevd_horizon=cfg.baseline_fevd_horizon,
        lambda_value=cfg.lambda_star,
    )


def _build_axis_runs(cfg: RobustnessConfig, axis_filter: str) -> List[RunSpec]:
    baseline = _build_baseline(cfg)
    runs: List[RunSpec] = [baseline]

    if axis_filter in {"all", "window"}:
        for v in cfg.axis_window_values:
            if v == cfg.baseline_window_length:
                continue
            runs.append(
                RunSpec(
                    section="axis_robustness",
                    axis_name="window_length",
                    axis_value=str(v),
                    window_length=int(v),
                    rebalance_days=baseline.rebalance_days,
                    var_lag_rule=baseline.var_lag_rule,
                    var_lag_fixed_p=baseline.var_lag_fixed_p,
                    fevd_horizon=baseline.fevd_horizon,
                    lambda_value=baseline.lambda_value,
                )
            )

    if axis_filter in {"all", "rebalance"}:
        for v in cfg.axis_rebalance_values:
            if v == cfg.baseline_rebalance_days:
                continue
            runs.append(
                RunSpec(
                    section="axis_robustness",
                    axis_name="rebalance_days",
                    axis_value=str(v),
                    window_length=baseline.window_length,
                    rebalance_days=int(v),
                    var_lag_rule=baseline.var_lag_rule,
                    var_lag_fixed_p=baseline.var_lag_fixed_p,
                    fevd_horizon=baseline.fevd_horizon,
                    lambda_value=baseline.lambda_value,
                )
            )

    if axis_filter in {"all", "lag"}:
        for v in cfg.axis_lag_rules:
            v_l = v.lower()
            if baseline.var_lag_fixed_p is None and v_l == baseline.var_lag_rule:
                continue
            runs.append(
                RunSpec(
                    section="axis_robustness",
                    axis_name="var_lag_rule",
                    axis_value=v_l,
                    window_length=baseline.window_length,
                    rebalance_days=baseline.rebalance_days,
                    var_lag_rule=v_l,
                    var_lag_fixed_p=baseline.var_lag_fixed_p,
                    fevd_horizon=baseline.fevd_horizon,
                    lambda_value=baseline.lambda_value,
                )
            )

    if axis_filter in {"all", "horizon"}:
        for v in cfg.axis_fevd_horizons:
            if v == cfg.baseline_fevd_horizon:
                continue
            runs.append(
                RunSpec(
                    section="axis_robustness",
                    axis_name="fevd_horizon",
                    axis_value=str(v),
                    window_length=baseline.window_length,
                    rebalance_days=baseline.rebalance_days,
                    var_lag_rule=baseline.var_lag_rule,
                    var_lag_fixed_p=baseline.var_lag_fixed_p,
                    fevd_horizon=int(v),
                    lambda_value=baseline.lambda_value,
                )
            )

    return runs


def _build_lambda_runs(cfg: RobustnessConfig) -> List[RunSpec]:
    baseline = _build_baseline(cfg)
    runs: List[RunSpec] = []
    for lam in cfg.lambda_values:
        runs.append(
            RunSpec(
                section="lambda_sensitivity",
                axis_name="lambda",
                axis_value=f"{lam:g}",
                window_length=baseline.window_length,
                rebalance_days=baseline.rebalance_days,
                var_lag_rule=baseline.var_lag_rule,
                var_lag_fixed_p=baseline.var_lag_fixed_p,
                fevd_horizon=baseline.fevd_horizon,
                lambda_value=float(lam),
            )
        )
    return runs


def _max_drawdown_from_equity(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min())


def _window_maxdd(equity: pd.Series, start: str, end: str) -> float:
    start_ts = pd.to_datetime(start)
    end_ts = pd.to_datetime(end)
    sliced = equity.loc[(equity.index >= start_ts) & (equity.index <= end_ts)].dropna()
    if sliced.empty:
        return float("nan")
    rebased = sliced / sliced.iloc[0]
    return _max_drawdown_from_equity(rebased)


def _run_id(spec: RunSpec) -> str:
    lag = _lag_token(spec.var_lag_rule, spec.var_lag_fixed_p)
    return (
        f"{spec.section}_{spec.axis_name}_{_safe_token(spec.axis_value)}"
        f"_win{spec.window_length}_reb{spec.rebalance_days}_lag{lag}"
        f"_h{spec.fevd_horizon}_lam{_safe_token(spec.lambda_value)}"
    )


def _strategy_metrics(
    *,
    strategy_key: str,
    weights_df: pd.DataFrame,
    returns_df: pd.DataFrame,
    crises: List[CrisisWindow],
    rebalance_days: int,
) -> Dict[str, float]:
    bt = run_backtest(weights_df=weights_df, returns_df=returns_df, tcost_bps=TCOST_BPS, store_daily_weights=False)
    met = compute_metrics(bt.portfolio_returns, turnover=bt.turnover)

    turnover_avg = float(bt.turnover.dropna().mean()) if len(bt.turnover.dropna()) else np.nan
    turnover_annualized = (
        float(turnover_avg * (ANNUALIZATION_FACTOR / rebalance_days))
        if np.isfinite(turnover_avg)
        else np.nan
    )
    cost_drag_annualized = float(bt.tcosts.fillna(0.0).mean() * ANNUALIZATION_FACTOR)

    crisis_vals: List[float] = []
    out: Dict[str, float] = {
        f"{strategy_key}_full_sharpe": float(met["sharpe"]),
        f"{strategy_key}_full_maxdd": float(met["max_drawdown"]),
        f"{strategy_key}_turnover_avg": turnover_avg,
        f"{strategy_key}_turnover_annualized": turnover_annualized,
        f"{strategy_key}_cost_drag_annualized": cost_drag_annualized,
    }
    for c in crises:
        mdd = _window_maxdd(bt.equity_curve, c.start, c.end)
        out[f"{strategy_key}_maxdd_{c.key}"] = mdd
        crisis_vals.append(mdd)

    out[f"{strategy_key}_avg_crisis_maxdd"] = float(np.nanmean(np.asarray(crisis_vals, dtype=float)))
    return out


def _run_single(
    spec: RunSpec,
    *,
    returns_df: pd.DataFrame,
    crises: List[CrisisWindow],
    force_recompute: bool,
) -> Dict[str, object]:
    run_id = _run_id(spec)
    lag_token = _lag_token(spec.var_lag_rule, spec.var_lag_fixed_p)
    run_tag = f"{RUN_TAG}_{spec.section}_{spec.axis_name}_{_safe_token(spec.axis_value)}"

    rolling = run_rolling_spillovers(
        use_cache=True,
        force_recompute=force_recompute,
        window=spec.window_length,
        step=spec.rebalance_days,
        fevd_horizon=spec.fevd_horizon,
        run_tag=run_tag,
        var_lag_rule=spec.var_lag_rule,
        var_lag_fixed_p=spec.var_lag_fixed_p,
        var_lag_max=VAR_LAG_MAX,
    )
    npz_path = str(rolling.meta["cache_path"])
    rebal_dates = rolling.dates.intersection(returns_df.index)
    if len(rebal_dates) == 0:
        raise RuntimeError(f"[robustness] {run_id}: no overlapping rebalance dates.")

    w_max = WEIGHT_BOUNDS[1]
    base_cfg = OptConfig(lam=0.0, w_max=w_max, long_only=True, fully_invested=True)
    sp_cfg = OptConfig(lam=spec.lambda_value, w_max=w_max, long_only=True, fully_invested=True)

    min_w = min_variance_over_time(returns_df, rebal_dates, window=spec.window_length, opt_cfg=base_cfg)
    sp_w = compute_weights_over_time(
        spillover_npz_path=npz_path,
        model="spillover_aware",
        score_method="to_others",
        opt_cfg=sp_cfg,
        use_cache_prices=True,
        window=spec.window_length,
        returns_df=returns_df,
        verbose=False,
    )

    strategy_weights: Dict[str, pd.DataFrame] = {
        "minvar": min_w,
        "spillover": sp_w,
    }
    # Full 4-strategy comparisons are exported only for window/rebalance axes.
    if spec.section == "axis_robustness" and spec.axis_name in {"baseline", "window_length", "rebalance_days"}:
        strategy_weights["equalweight"] = equal_weight_over_time(rebal_dates, returns_df.columns, w_max=w_max)
        strategy_weights["meanvar"] = mean_variance_over_time(
            returns_df,
            rebal_dates,
            window=spec.window_length,
            opt_cfg=base_cfg,
            risk_aversion=1.0,
        )

    out: Dict[str, object] = {
        "run_id": run_id,
        "section": spec.section,
        "axis_name": spec.axis_name,
        "axis_value": spec.axis_value,
        "window_length": spec.window_length,
        "rebalance_days": spec.rebalance_days,
        "var_lag_rule": spec.var_lag_rule,
        "var_lag_fixed_p": spec.var_lag_fixed_p,
        "lag_token": lag_token,
        "fevd_horizon": spec.fevd_horizon,
        "lambda": spec.lambda_value,
    }

    for strategy_key, weights_df in strategy_weights.items():
        out.update(
            _strategy_metrics(
                strategy_key=strategy_key,
                weights_df=weights_df,
                returns_df=returns_df,
                crises=crises,
                rebalance_days=spec.rebalance_days,
            )
        )

    # Backward-compatible aliases used by existing analysis scripts/tables.
    out["spillover_sharpe"] = out["spillover_full_sharpe"]
    out["avg_crisis_maxdd"] = out["spillover_avg_crisis_maxdd"]

    return out


def _export_tables(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    axis_df = df.loc[df["section"] == "axis_robustness"].copy()

    win_df = axis_df.loc[axis_df["axis_name"].isin(["baseline", "window_length"])].copy()
    if not win_df.empty:
        t = (
            win_df.sort_values("window_length")
            .drop_duplicates(subset=["window_length"], keep="first")
            .loc[
                :,
                [
                    "window_length",
                    "equalweight_full_maxdd",
                    "meanvar_full_maxdd",
                    "spillover_full_maxdd",
                    "minvar_full_maxdd",
                    "equalweight_maxdd_covid_crash",
                    "meanvar_maxdd_covid_crash",
                    "spillover_maxdd_covid_crash",
                    "minvar_maxdd_covid_crash",
                ],
            ]
            .rename(
                columns={
                    "window_length": "Window",
                    "equalweight_full_maxdd": "EqualWeight Full MaxDD",
                    "meanvar_full_maxdd": "MeanVar Full MaxDD",
                    "spillover_full_maxdd": "Spillover Full MaxDD",
                    "minvar_full_maxdd": "MinVar Full MaxDD",
                    "equalweight_maxdd_covid_crash": "EqualWeight COVID MaxDD",
                    "meanvar_maxdd_covid_crash": "MeanVar COVID MaxDD",
                    "spillover_maxdd_covid_crash": "Spillover COVID MaxDD",
                    "minvar_maxdd_covid_crash": "MinVar COVID MaxDD",
                }
            )
        )
        t.to_csv(out_dir / "table_window_sensitivity.csv", index=False)

    reb_df = axis_df.loc[axis_df["axis_name"].isin(["baseline", "rebalance_days"])].copy()
    if not reb_df.empty:
        t = (
            reb_df.sort_values("rebalance_days")
            .drop_duplicates(subset=["rebalance_days"], keep="first")
            .loc[
                :,
                [
                    "rebalance_days",
                    "equalweight_avg_crisis_maxdd",
                    "meanvar_avg_crisis_maxdd",
                    "minvar_avg_crisis_maxdd",
                    "spillover_avg_crisis_maxdd",
                    "equalweight_turnover_annualized",
                    "meanvar_turnover_annualized",
                    "minvar_turnover_annualized",
                    "spillover_turnover_annualized",
                ],
            ]
            .rename(
                columns={
                    "rebalance_days": "Rebalance",
                    "equalweight_avg_crisis_maxdd": "EqualWeight Avg Crisis MaxDD",
                    "meanvar_avg_crisis_maxdd": "MeanVar Avg Crisis MaxDD",
                    "minvar_avg_crisis_maxdd": "MinVar Avg Crisis MaxDD",
                    "spillover_avg_crisis_maxdd": "Spillover Avg Crisis MaxDD",
                    "equalweight_turnover_annualized": "EqualWeight Annualised turnover",
                    "meanvar_turnover_annualized": "MeanVar Annualised turnover",
                    "minvar_turnover_annualized": "MinVar Annualised turnover",
                    "spillover_turnover_annualized": "Spillover Annualised turnover",
                }
            )
        )
        t.to_csv(out_dir / "table_rebalance_sensitivity.csv", index=False)

    lag_df = axis_df.loc[axis_df["axis_name"].isin(["baseline", "var_lag_rule"])].copy()
    if not lag_df.empty:
        t = (
            lag_df.sort_values("var_lag_rule")
            .drop_duplicates(subset=["lag_token"], keep="first")
            .assign(lag_rule_display=lambda x: x["var_lag_rule"].str.upper())
            .loc[
                :,
                [
                    "lag_rule_display",
                    "spillover_avg_crisis_maxdd",
                    "minvar_avg_crisis_maxdd",
                    "spillover_full_sharpe",
                    "minvar_full_sharpe",
                ],
            ]
            .rename(
                columns={
                    "lag_rule_display": "Lag rule",
                    "spillover_avg_crisis_maxdd": "Spillover MaxDD (avg crises)",
                    "minvar_avg_crisis_maxdd": "MinVar MaxDD (avg crises)",
                    "spillover_full_sharpe": "Spillover Sharpe",
                    "minvar_full_sharpe": "MinVar Sharpe",
                }
            )
        )
        t["Notes"] = ""
        t.to_csv(out_dir / "table_lag_sensitivity.csv", index=False)

    h_df = axis_df.loc[axis_df["axis_name"].isin(["baseline", "fevd_horizon"])].copy()
    if not h_df.empty:
        t = (
            h_df.sort_values("fevd_horizon")
            .drop_duplicates(subset=["fevd_horizon"], keep="first")
            .loc[
                :,
                [
                    "fevd_horizon",
                    "spillover_avg_crisis_maxdd",
                    "minvar_avg_crisis_maxdd",
                    "spillover_full_sharpe",
                    "minvar_full_sharpe",
                ],
            ]
            .rename(
                columns={
                    "fevd_horizon": "FEVD H",
                    "spillover_avg_crisis_maxdd": "Spillover MaxDD (avg crises)",
                    "minvar_avg_crisis_maxdd": "MinVar MaxDD (avg crises)",
                    "spillover_full_sharpe": "Spillover Sharpe",
                    "minvar_full_sharpe": "MinVar Sharpe",
                }
            )
        )
        t.to_csv(out_dir / "table_horizon_sensitivity.csv", index=False)

    lam_df = df.loc[df["section"] == "lambda_sensitivity"].copy()
    if not lam_df.empty:
        t = (
            lam_df.sort_values("lambda")
            .drop_duplicates(subset=["lambda"], keep="first")
            .loc[
                :,
                [
                    "lambda",
                    "spillover_full_sharpe",
                    "minvar_full_sharpe",
                    "spillover_avg_crisis_maxdd",
                    "minvar_avg_crisis_maxdd",
                ],
            ]
            .rename(
                columns={
                    "lambda": "lambda",
                    "spillover_full_sharpe": "Spillover Full-sample Sharpe",
                    "minvar_full_sharpe": "MinVar Full-sample Sharpe",
                    "spillover_avg_crisis_maxdd": "Spillover Avg crisis MaxDD",
                    "minvar_avg_crisis_maxdd": "MinVar Avg crisis MaxDD",
                }
            )
        )
        t.to_csv(out_dir / "table_lambda_sensitivity.csv", index=False)


def _export_figures(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    lam_df = (
        df.loc[df["section"] == "lambda_sensitivity", ["lambda", "spillover_avg_crisis_maxdd"]]
        .dropna()
        .drop_duplicates(subset=["lambda"], keep="first")
        .sort_values("lambda")
    )
    if not lam_df.empty:
        plt.figure(figsize=(5.6, 3.4))
        plt.plot(
            lam_df["lambda"].values,
            lam_df["spillover_avg_crisis_maxdd"].values,
            marker="o",
            linewidth=1.8,
            color="#2A6F97",
        )
        plt.title("Sensitivity of Crisis Drawdown to Spillover Penalty (λ)")
        plt.xlabel("λ (spillover penalty)")
        plt.ylabel("Average crisis maximum drawdown")
        plt.grid(True, alpha=0.25, linewidth=0.7)
        plt.tight_layout()
        plt.savefig(out_dir / "figure_lambda_sensitivity_avg_crisis_maxdd.png", dpi=220)
        plt.close()

    reb_df = (
        df.loc[
            (df["section"] == "axis_robustness") & (df["axis_name"].isin(["baseline", "rebalance_days"])),
            ["rebalance_days", "spillover_turnover_annualized", "spillover_avg_crisis_maxdd"],
        ]
        .dropna()
        .drop_duplicates(subset=["rebalance_days"], keep="first")
        .sort_values("rebalance_days")
    )
    if not reb_df.empty:
        plt.figure(figsize=(5.6, 3.4))
        plt.plot(
            reb_df["rebalance_days"].values,
            reb_df["spillover_turnover_annualized"].values,
            marker="o",
            linewidth=1.8,
            color="#D95F02",
        )
        for _, row in reb_df.iterrows():
            plt.text(
                float(row["rebalance_days"]),
                float(row["spillover_turnover_annualized"]),
                f"MaxDD {float(row['spillover_avg_crisis_maxdd']):.3f}",
                fontsize=7,
                ha="center",
                va="bottom",
            )
        plt.title("Rebalancing Frequency and Trading Turnover")
        plt.xlabel("Rebalance interval (trading days)")
        plt.ylabel("Annualised turnover")
        plt.grid(True, alpha=0.25, linewidth=0.7)
        plt.tight_layout()
        plt.savefig(out_dir / "figure_rebalance_vs_turnover.png", dpi=220)
        plt.close()


def _compute_baseline_equity_curves(
    *,
    returns_df: pd.DataFrame,
    cfg: RobustnessConfig,
    force_recompute: bool,
) -> Dict[str, pd.Series]:
    baseline = _build_baseline(cfg)
    baseline_tag = f"{RUN_TAG}_axis_robustness_baseline_baseline"
    rolling = run_rolling_spillovers(
        use_cache=True,
        force_recompute=force_recompute,
        window=baseline.window_length,
        step=baseline.rebalance_days,
        fevd_horizon=baseline.fevd_horizon,
        run_tag=baseline_tag,
        var_lag_rule=baseline.var_lag_rule,
        var_lag_fixed_p=baseline.var_lag_fixed_p,
        var_lag_max=VAR_LAG_MAX,
    )
    rebal_dates = rolling.dates.intersection(returns_df.index)
    if len(rebal_dates) == 0:
        raise RuntimeError("No overlapping baseline rebalance dates for crisis equity figures.")

    w_max = WEIGHT_BOUNDS[1]
    base_cfg = OptConfig(lam=0.0, w_max=w_max, long_only=True, fully_invested=True)
    sp_cfg = OptConfig(lam=baseline.lambda_value, w_max=w_max, long_only=True, fully_invested=True)

    npz_path = str(rolling.meta["cache_path"])
    weights = {
        "EqualWeight": equal_weight_over_time(rebal_dates, returns_df.columns, w_max=w_max),
        "MeanVar": mean_variance_over_time(
            returns_df,
            rebal_dates,
            window=baseline.window_length,
            opt_cfg=base_cfg,
            risk_aversion=1.0,
        ),
        "MinVar": min_variance_over_time(returns_df, rebal_dates, window=baseline.window_length, opt_cfg=base_cfg),
        "Spillover": compute_weights_over_time(
            spillover_npz_path=npz_path,
            model="spillover_aware",
            score_method="to_others",
            opt_cfg=sp_cfg,
            use_cache_prices=True,
            window=baseline.window_length,
            returns_df=returns_df,
            verbose=False,
        ),
    }

    curves: Dict[str, pd.Series] = {}
    for strategy, w in weights.items():
        bt = run_backtest(weights_df=w, returns_df=returns_df, tcost_bps=TCOST_BPS, store_daily_weights=False)
        curves[strategy] = bt.equity_curve.sort_index()
    return curves


def _export_crisis_equity_figures(
    *,
    equity_curves: Dict[str, pd.Series],
    crises: List[CrisisWindow],
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for crisis in crises:
        start_ts = pd.to_datetime(crisis.start)
        end_ts = pd.to_datetime(crisis.end)
        rebased: Dict[str, pd.Series] = {}

        for strategy in STRATEGY_ORDER:
            s = equity_curves[strategy].loc[
                (equity_curves[strategy].index >= start_ts) & (equity_curves[strategy].index <= end_ts)
            ].dropna()
            if s.empty:
                continue
            rebased[strategy] = s / s.iloc[0]

        if not rebased:
            continue

        month_count = (end_ts.year - start_ts.year) * 12 + (end_ts.month - start_ts.month) + 1
        month_interval = max(1, int(np.ceil(month_count / 6)))
        year_label = str(start_ts.year) if start_ts.year == end_ts.year else f"{start_ts.year}-{end_ts.year}"

        plt.figure(figsize=(6.4, 3.8))
        for strategy in STRATEGY_ORDER:
            if strategy not in rebased:
                continue
            s = rebased[strategy]
            plt.plot(
                s.index,
                s.values,
                label=strategy,
                color=STRATEGY_COLORS[strategy],
                linewidth=1.2,
            )

        ax = plt.gca()
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=month_interval))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))

        plt.title(f"{crisis.label} ({year_label}): Rebased Equity Curves")
        plt.xlabel("Date")
        plt.ylabel("Portfolio value (rebased = 1.0)")
        plt.grid(True, alpha=0.25, linewidth=0.7)
        plt.legend(frameon=True, framealpha=0.95)
        plt.tight_layout()
        plt.savefig(out_dir / f"figure_crisis_equity_{crisis.key}.png", dpi=220)
        plt.close()


def _save_run_manifest(path: Path, *, args: argparse.Namespace, run_specs: List[RunSpec], cfg: RobustnessConfig) -> None:
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit_short": _git_commit_short(),
        "args": vars(args),
        "config": {
            "baseline_window_length": cfg.baseline_window_length,
            "baseline_rebalance_days": cfg.baseline_rebalance_days,
            "baseline_var_lag_rule": cfg.baseline_var_lag_rule,
            "baseline_var_lag_fixed_p": cfg.baseline_var_lag_fixed_p,
            "baseline_fevd_horizon": cfg.baseline_fevd_horizon,
            "lambda_star": cfg.lambda_star,
            "axis_window_values": list(cfg.axis_window_values),
            "axis_rebalance_values": list(cfg.axis_rebalance_values),
            "axis_lag_rules": list(cfg.axis_lag_rules),
            "axis_fevd_horizons": list(cfg.axis_fevd_horizons),
            "lambda_values": list(cfg.lambda_values),
            "crises": [
                {
                    "key": c.key,
                    "label": c.label,
                    "start": c.start,
                    "end": c.end,
                }
                for c in cfg.crises
            ],
        },
        "selected_runs": [
            {
                "section": s.section,
                "axis_name": s.axis_name,
                "axis_value": s.axis_value,
                "window_length": s.window_length,
                "rebalance_days": s.rebalance_days,
                "var_lag_rule": s.var_lag_rule,
                "var_lag_fixed_p": s.var_lag_fixed_p,
                "fevd_horizon": s.fevd_horizon,
                "lambda_value": s.lambda_value,
            }
            for s in run_specs
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run robustness grid and export sensitivity tables.")
    parser.add_argument(
        "--section",
        choices=["all", "axis_robustness", "lambda_sensitivity"],
        default="axis_robustness",
        help="Subset to run. If --section=all and --axis!=all, only axis robustness is run.",
    )
    parser.add_argument(
        "--axis",
        choices=["all", "window", "rebalance", "lag", "horizon"],
        default="all",
        help="Axis subset for axis robustness.",
    )
    parser.add_argument(
        "--force-recompute",
        action="store_true",
        help="Force recomputation instead of using cached spillovers.",
    )
    parser.add_argument(
        "--results-csv",
        default=str(REPORTS_DIR / "robustness_runs.csv"),
        help="Output CSV path for per-run metrics.",
    )
    parser.add_argument(
        "--tables-dir",
        default=str(REPORTS_DIR / "robustness_tables"),
        help="Directory to write thesis-ready sensitivity tables.",
    )
    parser.add_argument(
        "--figures-dir",
        default=str(REPORTS_DIR / "robustness_figures"),
        help="Directory to write robustness figures.",
    )
    args = parser.parse_args()

    cfg = ROBUSTNESS_CFG
    crises = list(cfg.crises)

    run_specs: List[RunSpec] = []
    include_axis = args.section in {"all", "axis_robustness"}
    include_lambda = args.section in {"all", "lambda_sensitivity"}

    # If user asks for a specific axis with section=all, interpret that as axis-only run.
    if args.section == "all" and args.axis != "all":
        include_lambda = False

    if include_axis:
        run_specs.extend(_build_axis_runs(cfg, axis_filter=args.axis))
    if include_lambda:
        run_specs.extend(_build_lambda_runs(cfg))

    if not run_specs:
        raise RuntimeError("No runs selected. Check --section and --axis.")

    bundle = get_returns_bundle(use_cache=True)
    returns_df = bundle.returns.dropna(how="any").sort_index()

    rows: List[Dict[str, object]] = []
    n_total = len(run_specs)
    for i, spec in enumerate(run_specs, start=1):
        print(
            f"[robustness] {i}/{n_total} "
            f"section={spec.section} axis={spec.axis_name} value={spec.axis_value} "
            f"win={spec.window_length} reb={spec.rebalance_days} lag={_lag_token(spec.var_lag_rule, spec.var_lag_fixed_p)} "
            f"H={spec.fevd_horizon} lam={spec.lambda_value:g}"
        )
        row = _run_single(
            spec,
            returns_df=returns_df,
            crises=crises,
            force_recompute=args.force_recompute,
        )
        rows.append(row)

    out_df = pd.DataFrame(rows)
    out_df = out_df.sort_values(["section", "axis_name", "window_length", "rebalance_days", "fevd_horizon", "lambda"])

    results_csv = Path(args.results_csv)
    results_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(results_csv, index=False)
    print(f"[robustness] saved run rows: {results_csv}")

    tables_dir = Path(args.tables_dir)
    _export_tables(out_df, tables_dir)
    print(f"[robustness] saved tables in: {tables_dir}")

    figures_dir = Path(args.figures_dir)
    _export_figures(out_df, figures_dir)
    baseline_curves = _compute_baseline_equity_curves(
        returns_df=returns_df,
        cfg=cfg,
        force_recompute=args.force_recompute,
    )
    _export_crisis_equity_figures(
        equity_curves=baseline_curves,
        crises=list(cfg.crises),
        out_dir=figures_dir,
    )
    print(f"[robustness] saved figures in: {figures_dir}")

    manifest_path = Path(args.results_csv).parent / "robustness_run_manifest.json"
    _save_run_manifest(manifest_path, args=args, run_specs=run_specs, cfg=cfg)
    print(f"[robustness] saved manifest: {manifest_path}")


if __name__ == "__main__":
    main()
