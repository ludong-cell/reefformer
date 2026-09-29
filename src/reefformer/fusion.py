"""Lead-dependent context and persistence fusion.

Arrays use the convention [forecast origin, forecast lead, spatial unit].
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FusionParameters:
    """Four scalar parameters defining the two monotone weight curves."""

    a_context: float
    tau_context_days: float
    a_persistence: float
    tau_persistence_days: float


def _forecast_array(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 3:
        raise ValueError(f"{name} must have shape [origin, lead, spatial_unit]")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def lead_weights(a: float, tau_days: float, horizon: int) -> np.ndarray:
    """Return w(h) = a + (1-a) exp(-h/tau) for h=1,...,horizon."""

    if not 0.0 <= a <= 1.0:
        raise ValueError("a must lie in [0, 1]")
    if not np.isfinite(tau_days) or tau_days <= 0:
        raise ValueError("tau_days must be positive and finite")
    if horizon < 1:
        raise ValueError("horizon must be positive")
    lead = np.arange(1, horizon + 1, dtype=np.float64)
    return a + (1.0 - a) * np.exp(-lead / tau_days)


def fit_lead_weight(
    left: np.ndarray,
    right: np.ndarray,
    truth: np.ndarray,
    tau_min: float = 0.1,
    tau_max: float = 1000.0,
    grid_points: int = 4001,
    refinement_steps: int = 100,
) -> tuple[float, float, np.ndarray]:
    """Fit a monotone convex-combination weight by pooled validation MSE.

    The fitted forecast is `weight * left + (1 - weight) * right`. For a fixed
    time constant, the optimal asymptote has a closed-form least-squares
    solution and is clipped to [0, 1]. The time constant is selected on a
    logarithmic grid and refined by golden-section search.
    """

    left = _forecast_array(left, "left")
    right = _forecast_array(right, "right")
    truth = _forecast_array(truth, "truth")
    if left.shape != right.shape or left.shape != truth.shape:
        raise ValueError("left, right, and truth must have identical shapes")
    if tau_min <= 0 or tau_max <= tau_min:
        raise ValueError("tau bounds must satisfy 0 < tau_min < tau_max")

    delta = left.astype(np.float64) - right.astype(np.float64)
    residual = right.astype(np.float64) - truth.astype(np.float64)
    aa = np.mean(delta * delta, axis=(0, 2))
    bb = np.mean(delta * residual, axis=(0, 2))
    cc = np.mean(residual * residual, axis=(0, 2))
    lead = np.arange(1, left.shape[1] + 1, dtype=np.float64)

    def solve(log_tau: float) -> tuple[float, float, np.ndarray]:
        decay = np.exp(-lead / np.exp(log_tau))
        complement = 1.0 - decay
        denominator = max(float(np.sum(aa * complement * complement)), 1e-15)
        asymptote = float(
            np.clip(
                -np.sum(complement * (aa * decay + bb)) / denominator,
                0.0,
                1.0,
            )
        )
        weight = decay + asymptote * complement
        loss = float(np.mean(aa * weight * weight + 2.0 * bb * weight + cc))
        return loss, asymptote, weight

    grid = np.linspace(np.log(tau_min), np.log(tau_max), grid_points)
    losses = np.asarray([solve(value)[0] for value in grid])
    index = int(np.argmin(losses))
    low = float(grid[max(0, index - 1)])
    high = float(grid[min(len(grid) - 1, index + 1)])
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    x1 = high - (high - low) / phi
    x2 = low + (high - low) / phi
    f1 = solve(x1)[0]
    f2 = solve(x2)[0]
    for _ in range(refinement_steps):
        if f1 <= f2:
            high, x2, f2 = x2, x1, f1
            x1 = high - (high - low) / phi
            f1 = solve(x1)[0]
        else:
            low, x1, f1 = x1, x2, f2
            x2 = low + (high - low) / phi
            f2 = solve(x2)[0]
    log_tau = (low + high) / 2.0
    _, asymptote, weight = solve(log_tau)
    return asymptote, float(np.exp(log_tau)), weight


def damped_anomaly_persistence(
    last_anomaly: np.ndarray,
    rho: np.ndarray,
    horizon: int,
) -> np.ndarray:
    """Create pixel-wise damped anomaly-persistence forecasts."""

    last = np.asarray(last_anomaly, dtype=np.float64)
    coefficient = np.asarray(rho, dtype=np.float64)
    if last.ndim != 2:
        raise ValueError("last_anomaly must have shape [origin, spatial_unit]")
    if coefficient.ndim != 1 or coefficient.shape[0] != last.shape[1]:
        raise ValueError("rho must contain one value per spatial unit")
    if not np.isfinite(last).all() or not np.isfinite(coefficient).all():
        raise ValueError("persistence inputs must be finite")
    if np.any((coefficient < 0.0) | (coefficient > 1.0)):
        raise ValueError("rho must lie in [0, 1]")
    lead = np.arange(1, horizon + 1, dtype=np.float64)
    return last[:, None, :] * coefficient[None, None, :] ** lead[None, :, None]


def fit_reef_former(
    short_validation: np.ndarray,
    long_validation: np.ndarray,
    persistence_validation: np.ndarray,
    truth_validation: np.ndarray,
) -> FusionParameters:
    """Fit LCF followed by LPF on validation forecasts only."""

    a_context, tau_context, context_weight = fit_lead_weight(
        short_validation, long_validation, truth_validation
    )
    context = (
        context_weight[None, :, None] * short_validation
        + (1.0 - context_weight[None, :, None]) * long_validation
    )
    a_persistence, tau_persistence, _ = fit_lead_weight(
        persistence_validation, context, truth_validation
    )
    return FusionParameters(
        a_context=a_context,
        tau_context_days=tau_context,
        a_persistence=a_persistence,
        tau_persistence_days=tau_persistence,
    )


def apply_reef_former(
    short_forecast: np.ndarray,
    long_forecast: np.ndarray,
    persistence_forecast: np.ndarray,
    parameters: FusionParameters,
) -> np.ndarray:
    """Apply frozen ReefFormer parameters to a forecast split."""

    short = _forecast_array(short_forecast, "short_forecast")
    long = _forecast_array(long_forecast, "long_forecast")
    persistence = _forecast_array(persistence_forecast, "persistence_forecast")
    if short.shape != long.shape or short.shape != persistence.shape:
        raise ValueError("all forecasts must have identical shapes")
    q = lead_weights(parameters.a_context, parameters.tau_context_days, short.shape[1])
    context = q[None, :, None] * short + (1.0 - q[None, :, None]) * long
    w = lead_weights(
        parameters.a_persistence,
        parameters.tau_persistence_days,
        short.shape[1],
    )
    return w[None, :, None] * persistence + (1.0 - w[None, :, None]) * context
