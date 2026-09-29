"""Fit and apply ReefFormer on a small deterministic synthetic example."""

from __future__ import annotations

import numpy as np

from reefformer import apply_reef_former, fit_reef_former


def main() -> None:
    rng = np.random.default_rng(20260928)
    origins, horizon, pixels = 80, 30, 6
    truth = rng.normal(size=(origins, horizon, pixels))
    lead = np.arange(1, horizon + 1)[None, :, None]
    short = truth + rng.normal(scale=0.15 + 0.015 * lead, size=truth.shape)
    long = truth + rng.normal(scale=0.42 - 0.008 * lead, size=truth.shape)
    persistence = truth + rng.normal(scale=0.10 + 0.020 * lead, size=truth.shape)
    parameters = fit_reef_former(short, long, persistence, truth)
    prediction = apply_reef_former(short, long, persistence, parameters)
    print(parameters)
    print(f"synthetic RMSE: {np.sqrt(np.mean((prediction - truth) ** 2)):.6f}")


if __name__ == "__main__":
    main()
