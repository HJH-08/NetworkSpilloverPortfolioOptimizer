import numpy as np
import pytest

pytest.importorskip("cvxpy")

from spillover_aware_optimizer import OptConfig, optimize_min_variance, optimize_spillover_aware, optimize_mean_variance


def test_identity_covariance_gives_equal_weights():
    N = 5
    Sigma = np.eye(N)
    cfg = OptConfig(lam=0.0, w_max=1.0, long_only=True, fully_invested=True)
    res = optimize_min_variance(Sigma, cfg=cfg)
    w = res.w
    assert np.allclose(w, np.full(N, 1.0 / N), atol=1e-3)
    assert np.isclose(w.sum(), 1.0, atol=1e-8)


def test_extreme_penalty_reduces_high_risk_asset():
    N = 4
    Sigma = np.eye(N)
    s = np.array([10.0, 1.0, 1.0, 1.0])
    cfg = OptConfig(lam=100.0, w_max=1.0, long_only=True, fully_invested=True)
    res = optimize_spillover_aware(Sigma, s, cfg=cfg, penalty="linear")
    w = res.w
    assert w[0] < w[1]
    assert w[0] < w[2]
    assert w[0] < w[3]


def test_constraints_respected():
    N = 3
    Sigma = np.eye(N)
    s = np.array([1.0, 2.0, 3.0])
    cfg = OptConfig(lam=1.0, w_max=0.6, w_min=0.0, long_only=True, fully_invested=True)
    res = optimize_spillover_aware(Sigma, s, cfg=cfg, penalty="linear")
    w = res.w
    assert np.all(w >= -1e-8)
    assert np.all(w <= 0.6 + 1e-8)
    assert np.isclose(w.sum(), 1.0, atol=1e-8)


def test_mean_variance_tilts_to_high_return_asset():
    N = 3
    Sigma = np.eye(N)
    mu = np.array([0.01, 0.02, 0.05])
    cfg = OptConfig(lam=0.0, w_max=1.0, long_only=True, fully_invested=True)
    res = optimize_mean_variance(Sigma, mu, cfg=cfg, risk_aversion=0.1)
    w = res.w
    assert w[2] == w.max()
