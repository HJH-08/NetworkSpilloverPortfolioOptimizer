import numpy as np
import pytest

pytest.importorskip("matplotlib")

from network_metrics import compute_metrics_for_W


def test_compute_metrics_for_W_zeroes_diagonal():
    W = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ],
        dtype=float,
    )
    to_others, from_others, net = compute_metrics_for_W(W)

    # Diagonal should be excluded: row sums without diag
    assert np.allclose(from_others, np.array([2.0 + 3.0, 4.0 + 6.0, 7.0 + 8.0]))
    assert np.allclose(to_others, np.array([4.0 + 7.0, 2.0 + 8.0, 3.0 + 6.0]))
    assert np.allclose(net, to_others - from_others)
