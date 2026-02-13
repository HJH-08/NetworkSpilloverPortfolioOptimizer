import numpy as np
import pandas as pd
import pytest

from var_model import VarFitError, fit_var, select_var_lag


def test_fit_var_raises_on_nan():
    df = pd.DataFrame(
        {"A": [0.0, 1.0, np.nan, 0.5], "B": [0.1, 0.2, 0.3, 0.4]},
        index=pd.date_range("2020-01-01", periods=4, freq="D"),
    )
    with pytest.raises(VarFitError):
        fit_var(df, lag=1)


def test_fit_var_raises_on_short_window():
    df = pd.DataFrame(
        {"A": [0.0, 0.1, 0.2], "B": [0.0, 0.1, 0.2]},
        index=pd.date_range("2020-01-01", periods=3, freq="D"),
    )
    with pytest.raises(VarFitError):
        fit_var(df, lag=2)


def test_select_var_lag_returns_at_least_one():
    df = pd.DataFrame(
        {"A": np.random.randn(20), "B": np.random.randn(20)},
        index=pd.date_range("2020-01-01", periods=20, freq="D"),
    )
    p = select_var_lag(df, maxlags=3, criterion="bic")
    assert isinstance(p, int)
    assert p >= 1
