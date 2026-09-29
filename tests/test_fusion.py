import numpy as np
import pytest

from reefformer import (
    FusionParameters,
    apply_reef_former,
    damped_anomaly_persistence,
    fit_lead_weight,
    lead_weights,
)


def test_weights_are_bounded_and_monotone():
    weight = lead_weights(0.2, 7.0, 30)
    assert np.all((0.0 <= weight) & (weight <= 1.0))
    assert np.all(np.diff(weight) <= 0.0)


def test_damped_persistence_shape_and_values():
    last = np.asarray([[2.0, -1.0]])
    rho = np.asarray([0.5, 1.0])
    forecast = damped_anomaly_persistence(last, rho, 3)
    assert forecast.shape == (1, 3, 2)
    assert np.allclose(forecast[0, :, 0], [1.0, 0.5, 0.25])
    assert np.allclose(forecast[0, :, 1], [-1.0, -1.0, -1.0])


def test_fit_improves_over_both_endpoints():
    rng = np.random.default_rng(9)
    truth = rng.normal(size=(150, 30, 5))
    lead = np.arange(1, 31)[None, :, None]
    left = truth + rng.normal(scale=0.10 + 0.018 * lead, size=truth.shape)
    right = truth + rng.normal(scale=0.47 - 0.010 * lead, size=truth.shape)
    a, tau, weight = fit_lead_weight(left, right, truth)
    fused = weight[None, :, None] * left + (1.0 - weight[None, :, None]) * right
    assert 0.0 <= a <= 1.0
    assert tau > 0.0
    assert np.mean((fused - truth) ** 2) < np.mean((left - truth) ** 2)
    assert np.mean((fused - truth) ** 2) < np.mean((right - truth) ** 2)


def test_apply_rejects_shape_mismatch():
    parameters = FusionParameters(0.2, 4.0, 0.1, 6.0)
    good = np.zeros((2, 30, 3))
    with pytest.raises(ValueError):
        apply_reef_former(good, good[:, :, :2], good, parameters)
