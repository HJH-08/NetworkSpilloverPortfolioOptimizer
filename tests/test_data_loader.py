import pandas as pd

import data_loader as dl


def test_apply_missing_policy_drop_any(monkeypatch):
    df = pd.DataFrame(
        {
            "A": [1.0, 2.0, None, 4.0],
            "B": [1.0, None, 3.0, 4.0],
        },
        index=pd.date_range("2020-01-01", periods=4, freq="D"),
    )

    monkeypatch.setattr(dl, "MISSING_POLICY", "drop_any")
    out = dl._apply_missing_policy(df)
    # Only rows with no NaNs should remain
    assert out.isna().sum().sum() == 0
    assert len(out) == 2


def test_apply_missing_policy_ffill_then_drop(monkeypatch):
    df = pd.DataFrame(
        {
            "A": [1.0, None, None, 4.0],
            "B": [1.0, 2.0, None, 4.0],
        },
        index=pd.date_range("2020-01-01", periods=4, freq="D"),
    )

    monkeypatch.setattr(dl, "MISSING_POLICY", "ffill_then_drop")
    monkeypatch.setattr(dl, "FFILL_LIMIT", 1)
    out = dl._apply_missing_policy(df)
    assert out.isna().sum().sum() == 0


def test_pivot_long_to_wide_and_datetime_index():
    df = pd.DataFrame(
        {
            "Date": ["2020-01-01", "2020-01-01", "2020-01-02", "2020-01-02"],
            "Instrument": ["A", "B", "A", "B"],
            "TRDPRC_1": [10, 20, 11, 21],
        }
    )
    wide = dl._pivot_long_to_wide(df)
    wide = dl._ensure_datetime_index(wide, date_col="Date")

    assert list(wide.columns) == ["A", "B"]
    assert isinstance(wide.index, pd.DatetimeIndex)
