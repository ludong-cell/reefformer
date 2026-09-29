"""Public ReefFormer API."""

from .fusion import (
    FusionParameters,
    apply_reef_former,
    damped_anomaly_persistence,
    fit_lead_weight,
    fit_reef_former,
    lead_weights,
)

__all__ = [
    "FusionParameters",
    "apply_reef_former",
    "damped_anomaly_persistence",
    "fit_lead_weight",
    "fit_reef_former",
    "lead_weights",
]

__version__ = "1.0.0"
