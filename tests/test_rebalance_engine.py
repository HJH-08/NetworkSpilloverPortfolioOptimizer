import pytest

from rebalance_engine import compute_weights_over_time


def test_objective_form_validation_rejects_invalid_value():
    with pytest.raises(ValueError, match="objective_form must be 'linear' or 'quadratic'"):
        compute_weights_over_time(
            spillover_npz_path="dummy.npz",
            objective_form="not_valid",
            window=2,
        )
