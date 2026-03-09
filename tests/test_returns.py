import numpy as np
import pandas as pd

from returns import (
    _winsorize_df,
    _chronological_splits,
    _flatten_columns,
    compute_return_descriptive_stats,
    compute_return_cross_section_summary,
)


def test_winsorize_clips_outliers():
    df = pd.DataFrame({"A": [0.0, 0.0, 100.0, 0.0, 0.0]})
    out = _winsorize_df(df, 0.2, 0.8)
    assert out["A"].max() <= df["A"].quantile(0.8) + 1e-12
    assert out["A"].min() >= df["A"].quantile(0.2) - 1e-12


def test_chronological_splits_non_overlapping():
    idx = pd.date_range("2020-01-01", periods=10, freq="D")
    df = pd.DataFrame({"A": np.arange(10)}, index=idx)
    splits = _chronological_splits(df, "2020-01-03", "2020-01-06", "2020-01-10")

    train = splits["train"]
    valid = splits["valid"]
    test = splits["test"]

    assert train.index.max() <= pd.Timestamp("2020-01-03")
    assert valid.index.min() >= pd.Timestamp("2020-01-04")
    assert valid.index.max() <= pd.Timestamp("2020-01-06")
    assert test.index.min() >= pd.Timestamp("2020-01-07")

    assert train.index.intersection(valid.index).empty
    assert train.index.intersection(test.index).empty
    assert valid.index.intersection(test.index).empty


def test_flatten_columns_handles_multiindex_and_strings():
    cols = pd.MultiIndex.from_product([["TRDPRC_1"], ["XLB", "XLK"]])
    flat = _flatten_columns(cols)
    assert list(flat) == ["XLB", "XLK"]

    cols2 = ["('TRDPRC_1', 'XLE')", "('TRDPRC_1', 'XLF')"]
    flat2 = _flatten_columns(cols2)
    assert list(flat2) == ["XLE", "XLF"]


def test_compute_return_descriptive_stats_columns_and_sizes():
    idx = pd.date_range("2020-01-01", periods=5, freq="D")
    rets = pd.DataFrame(
        {
            "A": [0.01, -0.02, 0.00, 0.03, 0.01],
            "B": [-0.01, 0.01, 0.02, -0.01, 0.00],
        },
        index=idx,
    )
    out = compute_return_descriptive_stats(rets, annualization_factor=252)
    assert list(out.index) == ["A", "B"]
    assert int(out.loc["A", "sample_size"]) == 5
    assert "annualised_volatility" in out.columns
    assert "pct_positive_days" in out.columns


def test_compute_return_cross_section_summary_pairwise_corr():
    idx = pd.date_range("2021-01-01", periods=4, freq="D")
    rets = pd.DataFrame(
        {
            "A": [0.01, 0.02, 0.03, 0.04],
            "B": [0.01, 0.02, 0.03, 0.04],
        },
        index=idx,
    )
    summary = compute_return_cross_section_summary(rets, annualization_factor=252)
    assert int(summary.loc[0, "num_assets"]) == 2
    assert np.isclose(summary.loc[0, "average_pairwise_correlation"], 1.0)
