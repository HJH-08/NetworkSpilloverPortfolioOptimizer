import numpy as np
import pandas as pd

from statsmodels.tsa.api import VAR

from fevd import compute_gfevd


def _simulate_var1(T: int = 300, N: int = 3, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    # Small coefficients for stability
    A = 0.2 * rng.standard_normal((N, N))
    eps = rng.standard_normal((T, N)) * 0.5
    x = np.zeros((T, N))
    for t in range(1, T):
        x[t] = x[t - 1] @ A.T + eps[t]
    idx = pd.date_range("2020-01-01", periods=T, freq="B")
    cols = [f"X{i}" for i in range(N)]
    return pd.DataFrame(x, index=idx, columns=cols)


def test_compute_gfevd_basic_properties():
    df = _simulate_var1()
    res = VAR(df).fit(1)

    sp = compute_gfevd(res, H=5, zero_diagonal=True, scale_to_100=False)

    W = sp.W
    assert W.shape[0] == W.shape[1] == df.shape[1]
    assert np.allclose(np.diag(W), 0.0)

    row_sums = W.sum(axis=1)
    assert np.all(row_sums >= -1e-8)
    assert np.all(row_sums <= 1.0 + 1e-6)

    assert 0.0 <= sp.total_connectedness <= 1.0 + 1e-6


def test_fevd_row_normalization_nonneg_and_diag_handling():
    df = _simulate_var1()
    res = VAR(df).fit(1)

    # W_full with diagonal retained
    sp_full = compute_gfevd(res, H=5, zero_diagonal=False, scale_to_100=False)
    W_full = sp_full.W

    # Row sums ~ 1.0
    row_sums = W_full.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-6)

    # Non-negativity
    assert np.all(W_full >= -1e-12)

    # Diagonal removal should not change off-diagonal values
    sp_zero = compute_gfevd(res, H=5, zero_diagonal=True, scale_to_100=False)
    W_zero = sp_zero.W
    off = ~np.eye(W_full.shape[0], dtype=bool)
    assert np.allclose(W_full[off], W_zero[off], atol=1e-12)


def test_directional_connectedness_matches_manual():
    df = _simulate_var1()
    res = VAR(df).fit(1)
    sp = compute_gfevd(res, H=5, zero_diagonal=False, scale_to_100=False)

    W = sp.W
    diag = np.diag(W)
    manual_from = W.sum(axis=1) - diag
    manual_to = W.sum(axis=0) - diag
    manual_net = manual_to - manual_from

    assert np.allclose(sp.from_others.values, manual_from, atol=1e-8)
    assert np.allclose(sp.to_others.values, manual_to, atol=1e-8)
    assert np.allclose(sp.net.values, manual_net, atol=1e-8)
