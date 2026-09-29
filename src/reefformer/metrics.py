"""Evaluation metrics used in the archived experiments."""

from __future__ import annotations

import numpy as np


def score(prediction: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    """Return pooled RMSE/MAE and the three prespecified lead bands."""

    prediction = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    if prediction.shape != truth.shape or prediction.ndim != 3:
        raise ValueError("prediction and truth must share [origin, lead, spatial] shape")
    error = prediction - truth

    def rmse(values: np.ndarray) -> float:
        return float(np.sqrt(np.mean(values * values)))

    return {
        "RMSE": rmse(error),
        "MAE": float(np.mean(np.abs(error))),
        "D1_7": rmse(error[:, :7]),
        "D8_14": rmse(error[:, 7:14]),
        "D15_30": rmse(error[:, 14:]),
    }
