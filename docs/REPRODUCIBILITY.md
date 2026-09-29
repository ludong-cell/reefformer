# Reproducibility runbook

## Level 1: unit and synthetic checks

```bash
python scripts/run_demo.py
pytest
```

This validates the fusion equations, monotonic weights, parameter recovery,
shape checks, and checkpoint-compatible model loading.

## Level 2: archived checkpoint reproduction

```bash
python scripts/reproduce_reference.py --domain hainan_coraltemp --device auto
python scripts/reproduce_reference.py --domain gbr_coraltemp --device auto
python scripts/reproduce_reference.py --domain hainan_ostia --device auto
```

The workflow performs these steps for seeds 42, 43, and 44:

1. verify dataset and checkpoint paths;
2. infer 90-day and 365-day forecasts;
3. estimate pixel-wise damped anomaly persistence from archived coefficients;
4. fit LCF on Validation only;
5. fit LPF on Validation only;
6. freeze both weight curves;
7. score Validation, Development (2022--2025), and Test2026;
8. compare generated summaries with the archived tables.

## Level 3: model retraining

The manuscript used deterministic CUDA training with Adam, seeds 42--44,
batch size 32, at most 40 epochs, and patience 8. Retrain one component with:

```bash
python scripts/train_backbone.py --domain hainan_coraltemp --lookback 90 --seed 42 --device cuda
```

Repeat for lookbacks 90 and 365, seeds 42--44, and each required domain. The
command reads only `Train` for gradient updates and only `Val` for checkpoint
selection. `Dev` and `Test2026` are not loaded by the trainer. New checkpoints
can differ at the last digits across PyTorch, CUDA, and GPU versions; the
archived checkpoints remain the reference artifacts for exact table
reproduction.

## Expected numerical tolerance

CPU and CUDA inference may differ at the last floating-point digits. The
verification script treats an absolute metric difference below `1e-4 deg C`
as a successful reproduction.
