"""
var_model.py

Phase 2A: VAR estimation (per rolling window)

What this module does:
- Takes a window of returns (T x N)
- Selects an appropriate lag length (AIC/BIC) up to VAR_LAG_MAX
- Fits a VAR(p) model
- Checks stability (all roots outside unit circle)
- Returns a clean "result object" you can feed into FEVD later

This is intentionally "window-local" so rolling_spillovers.py can call it repeatedly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Literal, Tuple

import numpy as np
import pandas as pd

from config import VAR_LAG_MAX, VAR_LAG_CRITERION, STABILITY_CHECK, RIDGE_EPS

# statsmodels is the standard library for VAR in Python
from statsmodels.tsa.api import VAR


@dataclass(frozen=True)
class VarFitResult:
    lag: int
    model_result: object  # statsmodels VARResults
    is_stable: bool
    roots: np.ndarray
    sigma_u: np.ndarray  # residual covariance (N x N)
    aic: float
    bic: float


class VarFitError(RuntimeError):
    """Raised when VAR fitting fails for a given window."""


def _ensure_2d_float(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    # enforce numeric
    for c in x.columns:
        x[c] = pd.to_numeric(x[c], errors="coerce")
    if x.isna().any().any():
        raise VarFitError("Returns window contains NaNs after coercion; VAR cannot be fit.")
    return x.astype(float)


def select_var_lag(
    window_rets: pd.DataFrame,
    *,
    maxlags: int = VAR_LAG_MAX,
    criterion: Literal["aic", "bic"] = VAR_LAG_CRITERION,
) -> int:
    """
    Choose p for VAR(p) using AIC or BIC over [1..maxlags].

    Returns an integer lag >= 1.
    """
    x = _ensure_2d_float(window_rets)

    # statsmodels can pick lag order for VAR
    model = VAR(x)
    order = model.select_order(maxlags=maxlags)

    # order.aic/bic can be int or None depending on data adequacy
    chosen = getattr(order, criterion)
    if chosen is None or (isinstance(chosen, float) and np.isnan(chosen)):
        # fallback: choose 1 if selection failed
        return 1

    # Sometimes select_order returns numpy int
    p = int(chosen)
    return max(1, p)


def fit_var(
    window_rets: pd.DataFrame,
    *,
    lag: Optional[int] = None,
    maxlags: int = VAR_LAG_MAX,
    criterion: Literal["aic", "bic"] = VAR_LAG_CRITERION,
    stability_check: bool = STABILITY_CHECK,
    ridge_eps: float = RIDGE_EPS,
    trend: Literal["n", "c", "ct", "ctt"] = "c",
) -> VarFitResult:
    """
    Fit VAR(p) on a single rolling window.

    Parameters
    - lag: if None, auto-select using criterion.
    - trend: 'c' includes constant. (Common for return VARs.)

    Returns VarFitResult for downstream FEVD.
    """
    x = _ensure_2d_float(window_rets)

    # Make sure we have enough observations for the chosen lag
    # Rough heuristic: need T >> N*lag; if too tight, selection can misbehave.
    if lag is None:
        lag = select_var_lag(x, maxlags=maxlags, criterion=criterion)

    if len(x) <= lag + 5:
        raise VarFitError(f"Not enough observations ({len(x)}) to fit VAR({lag}).")

    model = VAR(x)

    # Fitting can fail due to near-singularity / collinearity in some windows.
    # statsmodels doesn't expose ridge directly for VAR, but we can slightly jitter data
    # if it is perfectly collinear (rare). Keep this conservative.
    try:
        res = model.fit(lag, trend=trend)
    except Exception as e:
        # Conservative "last resort": tiny noise jitter (very small; does not change economics meaningfully)
        x_j = x + np.random.default_rng(0).normal(scale=ridge_eps, size=x.shape)
        try:
            res = VAR(x_j).fit(lag, trend=trend)
        except Exception as e2:
            raise VarFitError(f"VAR fit failed for lag={lag}. Original: {e}. After jitter: {e2}") from e2

    # Stability check: VAR is stable if all roots lie outside unit circle.
    # statsmodels: res.roots are roots of companion matrix; stable if abs(roots) > 1
    roots = np.asarray(res.roots)
    is_stable = bool(np.all(np.abs(roots) > 1.0)) if roots.size else True

    if stability_check and not is_stable:
        # You can choose to raise, warn, or accept unstable fits.
        # For your project, it's usually better to raise and let the rolling module decide how to handle it.
        raise VarFitError(f"Unstable VAR({lag}) window: some roots inside unit circle.")

    # Residual covariance matrix (Sigma_u)
    sigma_u = np.asarray(res.sigma_u)

    # Information criteria from the fitted model (for logging)
    aic = float(res.aic)
    bic = float(res.bic)

    return VarFitResult(
        lag=int(lag),
        model_result=res,
        is_stable=is_stable,
        roots=roots,
        sigma_u=sigma_u,
        aic=aic,
        bic=bic,
    )


if __name__ == "__main__":
    # Quick self-test using your existing returns pipeline
    from returns import get_returns_bundle
    from config import WINDOW

    bundle = get_returns_bundle(use_cache=True)
    rets = bundle.returns

    # Take one window at the start of train period
    w = rets.iloc[:WINDOW]
    out = fit_var(w, lag=None)

    print("VAR fit OK")
    print("Chosen lag:", out.lag)
    print("Stable:", out.is_stable)
    print("Sigma_u shape:", out.sigma_u.shape)
    print("AIC:", out.aic, "BIC:", out.bic)
