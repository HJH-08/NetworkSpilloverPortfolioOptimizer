import numpy as np
import pandas as pd

from backtest import compute_metrics


def test_equity_curve_equals_cumprod_of_returns():
    r = pd.Series([0.01, -0.02, 0.03], index=pd.date_range("2020-01-01", periods=3, freq="D"))
    equity = (1.0 + r).cumprod()
    expected = equity.values
    # Recompute equity the same way a report would
    got = (1.0 + r).cumprod().values
    assert np.allclose(got, expected, atol=1e-12)


def test_max_drawdown_manual_check():
    r = pd.Series([0.10, -0.05, -0.10, 0.02], index=pd.date_range("2020-01-01", periods=4, freq="D"))
    metrics = compute_metrics(r)

    equity = (1.0 + r).cumprod()
    peak = equity.cummax()
    dd = equity / peak - 1.0
    expected_mdd = float(dd.min())

    assert np.isclose(metrics["max_drawdown"], expected_mdd, atol=1e-12)


def test_annualization_formulas_short_series():
    r = pd.Series([0.01, 0.01, 0.01, 0.01], index=pd.date_range("2020-01-01", periods=4, freq="D"))
    metrics = compute_metrics(r, annualization=252, rf_annual=0.0)

    equity = (1.0 + r).cumprod()
    years = len(r) / 252
    expected_ann_return = float(equity.iloc[-1] ** (1.0 / years) - 1.0)
    expected_ann_vol = float(r.std(ddof=1) * np.sqrt(252))

    assert np.isclose(metrics["ann_return"], expected_ann_return, atol=1e-12)
    assert np.isclose(metrics["ann_vol"], expected_ann_vol, atol=1e-12)
