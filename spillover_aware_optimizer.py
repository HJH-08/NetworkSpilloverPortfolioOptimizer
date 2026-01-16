"""
spillover_aware_optimizer.py

Single-date optimizer.

We solve a convex portfolio optimization problem that combines:
- classical variance (w' Σ w)
- a spillover-aware penalty (lambda * s' w), where s is a systemic-risk score per asset

Key ideas:
- Σ is symmetric (covariance risk)
- s is directional/systemic (from spillover network metrics)
- λ controls trade-off: diversification/variance vs avoiding systemic transmitters

This module is intentionally "one-shot": it optimizes weights for one rebalance date.
Later, rebalance_engine.py will call this repeatedly over time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

# Optional dependency: cvxpy makes this clean and robust.
# Install: pip install cvxpy
import cvxpy as cp


@dataclass(frozen=True)
class OptConfig:
    # Risk trade-off
    lam: float = 0.5  # spillover penalty strength λ
    gamma: float = 1e-3  # diversification regularizer (L2). Higher => more spread weights

    # Constraints
    long_only: bool = True
    w_min: float = 0.0
    w_max: float = 0.25  # e.g., max 25% per sector
    fully_invested: bool = True

    # Numerical stability
    ridge_eps: float = 1e-8

    # Optional turnover control vs previous weights
    turnover_limit: Optional[float] = None  # e.g. 0.30 means sum |w - w_prev| <= 0.30


@dataclass(frozen=True)
class OptResult:
    w: np.ndarray
    objective_value: float
    status: str


def _check_shapes(Sigma: np.ndarray, s: np.ndarray) -> int:
    Sigma = np.asarray(Sigma, dtype=float)
    s = np.asarray(s, dtype=float).reshape(-1)

    if Sigma.ndim != 2 or Sigma.shape[0] != Sigma.shape[1]:
        raise ValueError("Sigma must be square (N x N).")
    N = Sigma.shape[0]
    if s.shape[0] != N:
        raise ValueError(f"s must have length N={N}. Got {s.shape[0]}.")
    return N


def _make_psd(Sigma: np.ndarray, eps: float) -> np.ndarray:
    """
    Ensure covariance matrix is numerically PSD-ish by adding small ridge.
    """
    Sigma = np.asarray(Sigma, dtype=float)
    return Sigma + eps * np.eye(Sigma.shape[0])


def optimize_min_variance(
    Sigma: np.ndarray,
    *,
    cfg: OptConfig = OptConfig(),
    w_prev: Optional[np.ndarray] = None,
) -> OptResult:
    """
    Baseline: min w' Σ w
    """
    Sigma = _make_psd(Sigma, cfg.ridge_eps)
    N = Sigma.shape[0]

    w = cp.Variable(N)

    obj = cp.Minimize(cp.quad_form(w, Sigma) + cfg.gamma * cp.sum_squares(w))

    cons = []

    if cfg.fully_invested:
        cons.append(cp.sum(w) == 1.0)

    if cfg.long_only:
        cons.append(w >= cfg.w_min)
        cons.append(w <= cfg.w_max)

    # Turnover constraint if desired
    if cfg.turnover_limit is not None:
        if w_prev is None:
            raise ValueError("turnover_limit set but w_prev is None.")
        w_prev = np.asarray(w_prev, dtype=float).reshape(-1)
        cons.append(cp.norm1(w - w_prev) <= cfg.turnover_limit)

    prob = cp.Problem(obj, cons)
    prob.solve(solver=cp.ECOS, verbose=False)

    if w.value is None:
        raise RuntimeError(f"Optimization failed. Status={prob.status}")

    return OptResult(w=np.asarray(w.value).reshape(-1), objective_value=float(prob.value), status=prob.status)


def optimize_spillover_aware(
    Sigma: np.ndarray,
    s: np.ndarray,
    *,
    cfg: OptConfig = OptConfig(),
    w_prev: Optional[np.ndarray] = None,
    normalize_s: bool = True,
    penalty: str = "linear",
) -> OptResult:
    """
    Spillover-aware objective:

        minimize    w' Σ w   +   λ * Penalty(w, s)

    Penalty options:
    - "linear":     s' w               (avoid high-s assets)
    - "quadratic":  sum_i s_i * w_i^2  (more aggressively punishes concentration in high-s assets)

    Notes:
    - s should be non-negative (e.g. to_others, or max(net,0)).
    - normalize_s rescales s to mean=1 to make λ easier to interpret.
    """
    Sigma = np.asarray(Sigma, dtype=float)
    s = np.asarray(s, dtype=float).reshape(-1)

    N = _check_shapes(Sigma, s)
    Sigma = _make_psd(Sigma, cfg.ridge_eps)

    # Make sure s is usable as a penalty weight
    s = np.maximum(s, 0.0)
    if normalize_s:
        m = float(np.mean(s))
        if m > 0:
            s = s / m  # mean ~ 1

    w = cp.Variable(N)

    risk_term = cp.quad_form(w, Sigma)

    if penalty == "linear":
        spill_term = s @ w
    elif penalty == "quadratic":
        spill_term = cp.sum(cp.multiply(s, cp.square(w)))
    else:
        raise ValueError("penalty must be 'linear' or 'quadratic'.")

    obj = cp.Minimize(risk_term + cfg.lam * spill_term + cfg.gamma * cp.sum_squares(w))


    cons = []
    if cfg.fully_invested:
        cons.append(cp.sum(w) == 1.0)

    if cfg.long_only:
        cons.append(w >= cfg.w_min)
        cons.append(w <= cfg.w_max)

    if cfg.turnover_limit is not None:
        if w_prev is None:
            raise ValueError("turnover_limit set but w_prev is None.")
        w_prev = np.asarray(w_prev, dtype=float).reshape(-1)
        cons.append(cp.norm1(w - w_prev) <= cfg.turnover_limit)

    prob = cp.Problem(obj, cons)
    prob.solve(solver=cp.ECOS, verbose=False)

    if w.value is None:
        raise RuntimeError(f"Optimization failed. Status={prob.status}")

    return OptResult(w=np.asarray(w.value).reshape(-1), objective_value=float(prob.value), status=prob.status)


# -------------------------
# Tiny self-test example
# -------------------------
if __name__ == "__main__":
    np.random.seed(0)
    N = 5

    # Fake covariance (PSD)
    A = np.random.randn(N, N)
    Sigma = A.T @ A

    # Fake spillover systemic scores (higher = more "dangerous transmitter")
    s = np.array([1.0, 3.0, 0.5, 2.0, 4.0])

    cfg0 = OptConfig(lam=0.0, w_max=0.6)
    cfg1 = OptConfig(lam=1.0, w_max=0.6)

    base = optimize_min_variance(Sigma, cfg=cfg0)
    aware = optimize_spillover_aware(Sigma, s, cfg=cfg1, penalty="linear")

    print("Base weights:", np.round(base.w, 4))
    print("Aware weights:", np.round(aware.w, 4))
    print("s:", s)
