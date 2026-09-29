# User guide

## Inputs

The fusion API expects forecast arrays shaped
`[forecast_origin, lead, spatial_unit]`.

- `short_forecast`: forecast from the recent-context branch;
- `long_forecast`: forecast from the long-context branch;
- `last_anomaly`: most recent anomaly at each spatial unit;
- `rho`: training-period lag-1 damping coefficient for each spatial unit;
- `truth`: validation targets used only when fitting the fusion curves.

All arrays must use the same anomaly units. Absolute SST can be reconstructed
after fusion by adding the target-date training climatology.

## Fit and apply

```python
from reefformer import fit_reef_former, apply_reef_former

parameters = fit_reef_former(
    short_validation,
    long_validation,
    persistence_validation,
    truth_validation,
)

forecast = apply_reef_former(
    short_test,
    long_test,
    persistence_test,
    parameters,
)
```

`fit_reef_former` returns `a_context`, `tau_context_days`, `a_persistence`, and
`tau_persistence_days`. The same fitted values must be frozen before any test
targets are accessed.

## Expected behavior

Both weights are constrained to `[0, 1]` and decrease monotonically with lead.
The context weight multiplies the short-context forecast. The persistence
weight multiplies the damped anomaly-persistence forecast. Small `tau` values
produce rapid transitions; large values produce gradual transitions.

## Common errors

- Do not fit weights on development or test targets.
- Do not add climatology before fitting if the branches use anomaly units.
- Do not mix forecast arrays with different origin sets or spatial ordering.
- Verify the pixel inventory and dataset hash before loading a checkpoint.
