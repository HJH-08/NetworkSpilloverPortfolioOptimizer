import os
import numpy as np
import pandas as pd
import pytest

from config import CACHE_DIR, REPORTS_DIR, STRESS_PERIODS


def _pick_latest_npz() -> str:
    files = [os.path.join(CACHE_DIR, f) for f in os.listdir(CACHE_DIR) if f.endswith(".npz")]
    if not files:
        raise FileNotFoundError("No spillover .npz files found.")
    files.sort(key=lambda p: os.path.getmtime(p))
    return files[-1]


def _load_weights(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing weights file: {path}")
    return pd.read_csv(path, index_col=0, parse_dates=True).sort_index()


@pytest.mark.parametrize("k", [3])
def test_spillover_reduces_transmitter_exposure_in_stress_windows(k: int):
    """
    Empirical check (option A):
    Spillover-aware portfolios should, on average, reduce allocation
    to top transmitters during stress windows versus min-variance.
    """
    weights_sp_path = os.path.join(REPORTS_DIR, "weights_spillover.csv")
    weights_mv_path = os.path.join(REPORTS_DIR, "weights_minvar.csv")

    try:
        w_sp = _load_weights(weights_sp_path)
        w_mv = _load_weights(weights_mv_path)
    except FileNotFoundError:
        pytest.skip("Missing weights outputs; run run_experiments.py first.")

    try:
        npz_path = _pick_latest_npz()
    except FileNotFoundError:
        pytest.skip("Missing spillover cache; run rolling_spillover.py first.")

    blob = np.load(npz_path, allow_pickle=False)
    dates = pd.to_datetime(blob["dates"].astype("datetime64[ns]"))
    assets = blob["assets"].astype(str).tolist()
    to_others = pd.DataFrame(blob["to_others"], index=dates, columns=assets)

    # Align assets and dates
    common_assets = [a for a in assets if a in w_sp.columns and a in w_mv.columns]
    if len(common_assets) < k:
        pytest.skip("Not enough overlapping assets for transmitter exposure test.")

    w_sp = w_sp[common_assets]
    w_mv = w_mv[common_assets]
    to_others = to_others[common_assets]

    # Check each stress window
    n_checked = 0
    n_better = 0

    for _, start, end in STRESS_PERIODS:
        w_dates = w_sp.index.intersection(w_mv.index)
        s_dates = to_others.index
        dates_win = w_dates.intersection(s_dates)
        dates_win = dates_win[(dates_win >= pd.to_datetime(start)) & (dates_win <= pd.to_datetime(end))]

        if len(dates_win) < 5:
            continue

        for dt in dates_win:
            top_tx = to_others.loc[dt].sort_values(ascending=False).head(k).index
            sp_exposure = float(w_sp.loc[dt, top_tx].sum())
            mv_exposure = float(w_mv.loc[dt, top_tx].sum())
            n_checked += 1
            if sp_exposure <= mv_exposure + 1e-6:
                n_better += 1

    if n_checked < 10:
        pytest.skip("Not enough overlapping stress-window dates for a reliable check.")

    # Require spillover-aware to reduce exposure on a majority of checked dates
    assert (n_better / n_checked) >= 0.6
