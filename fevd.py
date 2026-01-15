"""
fevd.py

Phase 2B: Generalized FEVD (GFEVD) + Diebold–Yilmaz spillover matrix.

Inputs:
- A fitted statsmodels VARResults object (from var_model.fit_var)

Outputs:
- W: (N x N) spillover matrix, where W[i, j] = contribution of shock in j to forecast error variance of i
- Optional connectedness summaries (total, directional)

Notes:
- Uses generalized FEVD (Pesaran & Shin), so results are invariant to variable ordering.
- Normalizes rows to sum to 1 (or 100%) so each row is a variance decomposition.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from config import FEVD_HORIZON, ZERO_DIAGONAL, RIDGE_EPS


@dataclass(frozen=True)
class SpilloverResult:
    W: np.ndarray                 # NxN spillover matrix (rows sum to 1)
    W_df: pd.DataFrame            # same as DataFrame with labels
    total_connectedness: float    # scalar in [0,1] (or [0,100] if you scale)
    to_others: pd.Series          # directional "to" (outgoing) per node
    from_others: pd.Series        # directional "from" (incoming) per node
    net: pd.Series                # net transmitter (+) vs receiver (-)


def _ma_representation(var_res, H: int) -> np.ndarray:
    """
    Get moving-average coefficient matrices Phi_h for h=0..H-1.
    Returns: array of shape (H, N, N)
    """
    # statsmodels provides ma_rep which returns Phi_0..Phi_{H-1}
    # Phi_0 should be identity.
    Phi = np.asarray(var_res.ma_rep(H))
    return Phi


def compute_gfevd(
    var_res,
    *,
    H: int = FEVD_HORIZON,
    zero_diagonal: bool = ZERO_DIAGONAL,
    eps: float = RIDGE_EPS,
    scale_to_100: bool = False,
) -> SpilloverResult:
    """
    Compute generalized FEVD spillover matrix.

    W_full[i,j] = contribution of shocks in j to the H-step forecast error variance of i.
    W_full rows sum to 1 (includes diagonal "own" contribution).

    For network usage, we often want no self-loops, so we also produce W_net
    which is W_full with diagonal set to 0 (but NOT renormalized).
    Connectedness summaries are computed from W_full (Diebold–Yilmaz style).
    """

    # Dimensions / labels
    names = list(getattr(var_res.model, "endog_names", None) or range(var_res.neqs))
    N = var_res.neqs

    # Sigma_u = covariance matrix of residuals (N x N)
    Sigma = np.asarray(var_res.sigma_u, dtype=float)
    Sigma = Sigma + eps * np.eye(N)

    # MA coefficients Phi_h, shape (H, N, N)
    Phi = _ma_representation(var_res, H)

    Sigma_diag = np.diag(Sigma)
    if np.any(Sigma_diag <= 0):
        raise RuntimeError("Non-positive residual variances encountered in Sigma_u.")

    numer = np.zeros((N, N), dtype=float)
    denom = np.zeros(N, dtype=float)

    for h in range(H):
        A = Phi[h] @ Sigma  # (N x N)
        numer += (A ** 2) / Sigma_diag[None, :]  # divide each column j by sigma_jj

        B = Phi[h] @ Sigma @ Phi[h].T
        denom += np.diag(B)

    denom = np.where(denom <= 0, eps, denom)
    Theta = numer / denom[:, None]  # (N x N)

    # Row-normalize once (standard DY normalization): rows sum to 1
    row_sums = Theta.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums <= 0, eps, row_sums)
    W_full = Theta / row_sums  # includes diagonal

    # Connectedness measures computed from W_full (NOT from diagonal-zero renormalized)
    off_diag_sum = W_full.sum() - np.trace(W_full)
    total_connectedness = off_diag_sum / N  # in [0,1]

    from_others = W_full.sum(axis=1) - np.diag(W_full)  # = 1 - diag for each row
    to_others = W_full.sum(axis=0) - np.diag(W_full)    # column spillovers to others
    net = to_others - from_others

    # Network adjacency version (diagonal removed, but not renormalized)
    W_net = W_full.copy()
    if zero_diagonal:
        np.fill_diagonal(W_net, 0.0)

    # Scale if desired
    if scale_to_100:
        W_net = 100.0 * W_net
        total_connectedness *= 100.0
        from_others *= 100.0
        to_others *= 100.0
        net *= 100.0

    W_df = pd.DataFrame(W_net, index=names, columns=names)

    return SpilloverResult(
        W=W_net,
        W_df=W_df,
        total_connectedness=float(total_connectedness),
        to_others=pd.Series(to_others, index=names, name="to_others"),
        from_others=pd.Series(from_others, index=names, name="from_others"),
        net=pd.Series(net, index=names, name="net"),
    )



if __name__ == "__main__":
    # Self-test: fit VAR on one window then compute GFEVD
    from returns import get_returns_bundle
    from var_model import fit_var
    from config import WINDOW

    bundle = get_returns_bundle(use_cache=True)
    rets = bundle.returns

    w = rets.iloc[:WINDOW]
    var_out = fit_var(w, lag=None)

    sp = compute_gfevd(var_out.model_result, H=10, scale_to_100=True)
    # 1) Total connectedness should NOT be 100%
    print("Total connectedness (%):", sp.total_connectedness)

    # 2) Diagonal of returned W_df should be zero (if zero_diagonal True)
    print("Diagonal (should be 0):", np.diag(sp.W_df.values)[:5])

    # 3) Row sums of W_df should now be <= 100 (since diagonal removed, not renormalized)
    print("Row sums (should be <= 100):", sp.W_df.sum(axis=1).head())

    print("Spillover matrix W shape:", sp.W.shape)
    print("Total connectedness (%):", sp.total_connectedness)
    print("\nTop 3 transmitters (to_others):")
    print(sp.to_others.sort_values(ascending=False).head(3))
    print("\nTop 3 receivers (from_others):")
    print(sp.from_others.sort_values(ascending=False).head(3))
    print("\nW (head):")
    print(sp.W_df.iloc[:5, :5])

    own = 100 - sp.W_df.sum(axis=1)
    print(own.sort_values())

