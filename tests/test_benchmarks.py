import numpy as np
import pandas as pd
import pytest

from spillover_aware_optimizer import OptConfig

pytest.importorskip("cvxpy")

from benchmarks import compute_rebalance_dates, equal_weight_over_time, mean_variance_over_time


def test_compute_rebalance_dates_basic():
    idx = pd.date_range("2020-01-01", periods=10, freq="B")
    dates = compute_rebalance_dates(idx, window=3, step=2)
    # positions: 2,4,6,8
    assert list(dates) == list(idx[[2, 4, 6, 8]])


def test_equal_weight_respects_w_max():
    rebal = pd.date_range("2020-01-01", periods=3, freq="B")
    assets = ["A", "B", "C", "D"]
    # 1/4 = 0.25, set w_max below
    with pytest.raises(ValueError):
        equal_weight_over_time(rebal, assets, w_max=0.20)


def test_mean_variance_over_time_runs():
    idx = pd.date_range("2020-01-01", periods=30, freq="B")
    rets = pd.DataFrame(
        {
            "A": 0.001,
            "B": 0.0,
            "C": -0.001,
        },
        index=idx,
    )
    rebal = compute_rebalance_dates(idx, window=10, step=10)
    cfg = OptConfig(lam=0.0, w_max=1.0, long_only=True, fully_invested=True)
    out = mean_variance_over_time(rets, rebal, window=10, opt_cfg=cfg, risk_aversion=1.0)
    assert not out.empty
    assert np.allclose(out.sum(axis=1).values, 1.0, atol=1e-8)
