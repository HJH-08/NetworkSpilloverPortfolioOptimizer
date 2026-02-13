import numpy as np
import pandas as pd

from backtest import run_backtest


def _make_returns(dates, r1, r2):
    return pd.DataFrame({"A": r1, "B": r2}, index=pd.to_datetime(dates))


def test_rebalance_costs_only_on_rebalance_days():
    dates = pd.date_range("2020-01-01", periods=10, freq="B")
    rets = _make_returns(dates, [0.0] * 10, [0.0] * 10)

    # Set targets on day 0 and day 5
    w = pd.DataFrame(
        [[1.0, 0.0], [0.0, 1.0]],
        index=[dates[0], dates[5]],
        columns=["A", "B"],
    )

    res = run_backtest(weights_df=w, returns_df=rets, tcost_bps=10.0)

    # Weights apply one day after targets (shifted)
    expected_rebalance_days = pd.to_datetime([dates[1], dates[6]])
    turnover_days = res.turnover.dropna().index

    assert list(turnover_days) == list(expected_rebalance_days)
    assert list(res.tcosts.dropna().index) == list(expected_rebalance_days)


def test_weight_timing_shift_applies_next_day():
    dates = pd.date_range("2020-01-01", periods=4, freq="B")
    # Only day 1 has return on asset A
    rets = _make_returns(dates, [0.0, 0.10, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0])

    # Target weights on day 0: fully in A
    w = pd.DataFrame([[1.0, 0.0]], index=[dates[0]], columns=["A", "B"])
    res = run_backtest(weights_df=w, returns_df=rets, tcost_bps=0.0)

    # First portfolio return should be on dates[1] because of shift
    assert res.portfolio_returns.index[0] == dates[1]
    assert np.isclose(res.portfolio_returns.iloc[0], 0.10)


def test_single_asset_portfolio_matches_asset_returns_after_shift():
    dates = pd.date_range("2020-01-01", periods=5, freq="B")
    rA = [0.0, 0.01, -0.02, 0.03, 0.0]
    rB = [0.0, 0.0, 0.0, 0.0, 0.0]
    rets = _make_returns(dates, rA, rB)

    w = pd.DataFrame([[1.0, 0.0]], index=[dates[0]], columns=["A", "B"])
    res = run_backtest(weights_df=w, returns_df=rets, tcost_bps=0.0)

    expected = pd.Series(rA[1:], index=dates[1:], name="portfolio_return")
    assert np.allclose(res.portfolio_returns.values, expected.values, atol=1e-12)


def test_no_rebalance_weight_drift():
    dates = pd.date_range("2020-01-01", periods=3, freq="B")
    # Nonzero return on a NON-rebalance day (dates[2])
    rets = _make_returns(dates, [0.0, 0.0, 0.10], [0.0, 0.0, 0.0])

    # Single rebalance at t0: 50/50
    w = pd.DataFrame([[0.5, 0.5]], index=[dates[0]], columns=["A", "B"])
    res = run_backtest(weights_df=w, returns_df=rets, tcost_bps=0.0)

    # After day 2 return (dates[2]), weight in A should increase
    w_day2 = res.weights_drifted.loc[dates[2]].values
    assert w_day2[0] > 0.5
    assert np.isclose(w_day2.sum(), 1.0, atol=1e-12)


def test_transaction_cost_calculation():
    dates = pd.date_range("2020-01-01", periods=6, freq="B")
    rets = _make_returns(dates, [0.0] * 6, [0.0] * 6)

    # Rebalance from 100% A to 100% B
    w = pd.DataFrame(
        [[1.0, 0.0], [0.0, 1.0]],
        index=[dates[0], dates[3]],
        columns=["A", "B"],
    )
    res = run_backtest(weights_df=w, returns_df=rets, tcost_bps=10.0)

    # First applied target (dates[1]) equals initial holdings -> no cost
    assert np.isclose(res.tcosts.loc[dates[1]], 0.0)

    # Rebalance applies next day (dates[4]) for the target at dates[3]
    # Turnover = 0.5 * sum |w_target - w_drift|; full switch => 1.0
    expected_cost = 10.0 / 10000.0
    assert np.isclose(res.tcosts.loc[dates[4]], expected_cost)


def test_rebalance_occurs_even_if_targets_unchanged():
    dates = pd.date_range("2020-01-01", periods=6, freq="B")
    # Asset A outperforms on a non-rebalance day so drift occurs
    rets = _make_returns(dates, [0.0, 0.0, 0.10, 0.0, 0.0, 0.0], [0.0] * 6)

    # Equal-weight targets on two dates (same weights)
    w = pd.DataFrame(
        [[0.5, 0.5], [0.5, 0.5]],
        index=[dates[0], dates[3]],
        columns=["A", "B"],
    )
    res = run_backtest(weights_df=w, returns_df=rets, tcost_bps=0.0)

    # Rebalance should occur on dates[1] and dates[4]
    turnover_days = res.turnover.dropna().index
    assert list(turnover_days) == [dates[1], dates[4]]
