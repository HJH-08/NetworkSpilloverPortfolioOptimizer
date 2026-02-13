import numpy as np
import pandas as pd

import rolling_spillover as rs
from returns import ReturnsBundle


def _synthetic_returns(T=120, N=3, seed=0):
    rng = np.random.default_rng(seed)
    rets = rng.normal(0, 0.01, size=(T, N))
    idx = pd.date_range("2020-01-01", periods=T, freq="B")
    cols = [f"A{i}" for i in range(N)]
    df = pd.DataFrame(rets, index=idx, columns=cols)
    return df


def test_rolling_spillovers_cache_reuse(tmp_path, monkeypatch):
    rets = _synthetic_returns()
    bundle = ReturnsBundle(prices=rets.cumsum() + 100.0, returns=rets, splits={"train": rets})

    # Redirect results to temp
    monkeypatch.setattr(rs, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(rs, "CACHE_DIR", tmp_path / "results" / "cache")
    rs.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rs.CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Smaller settings for speed
    monkeypatch.setattr(rs, "WINDOW", 50)
    monkeypatch.setattr(rs, "REBALANCE_EVERY_N_DAYS", 10)
    monkeypatch.setattr(rs, "FEVD_HORIZON", 5)
    monkeypatch.setattr(rs, "RUN_TAG", "test_smoke")

    # Monkeypatch data source (rolling_spillover imports get_returns_bundle directly)
    monkeypatch.setattr(rs, "get_returns_bundle", lambda use_cache=True: bundle)

    out1 = rs.run_rolling_spillovers(use_cache=False, force_recompute=True)
    assert out1.W_stack.ndim == 3
    assert out1.meta["loaded_from_cache"] is False

    out2 = rs.run_rolling_spillovers(use_cache=True, force_recompute=False)
    assert out2.meta["loaded_from_cache"] is True
    assert np.allclose(out1.W_stack, out2.W_stack)
