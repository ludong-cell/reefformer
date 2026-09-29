# ReefFormer

ReefFormer is a low-dimensional forecast-combination method for direct
multi-horizon sea surface temperature (SST) prediction. It combines a
short-context forecast, a long-context forecast, and damped anomaly
persistence using two smooth lead-dependent weights. The fusion stage is
independent of the forecasting backbone and adds four fitted scalars.

This repository is the public reproducibility package for the associated
*Computers & Geosciences* manuscript. It contains:

- the reusable fusion implementation;
- the exact compact iTransformer implementation needed to reload the archived
  checkpoints;
- processed daily anomaly datasets for Hainan CoralTemp, Great Barrier Reef
  CoralTemp, and Hainan OSTIA;
- the 18 checkpoints needed to reproduce the principal fusion results;
- frozen reference tables, protocols, and data inventories;
- automated tests and a synthetic quick-start example.

The repository intentionally excludes multi-gigabyte per-origin prediction
arrays. They are regenerated from the included data and checkpoints.

## Quick start

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
python scripts/run_demo.py
pytest
```

The demo uses synthetic data and finishes in seconds.

## Retrain a backbone

```bash
python scripts/train_backbone.py --domain hainan_coraltemp --lookback 90 --seed 42 --device auto
```

The training command implements the archived Adam, early-stopping, and
validation-only checkpoint-selection protocol. Run it for lookbacks 90 and
365 and seeds 42--44 to rebuild a domain's six component checkpoints. CUDA was
used for the archived runs; CPU training is supported as a slower functional
check.

## Reproduce a paper domain

```bash
python scripts/reproduce_reference.py --domain hainan_coraltemp --device auto
python scripts/reproduce_reference.py --domain gbr_coraltemp --device auto
python scripts/reproduce_reference.py --domain hainan_ostia --device auto
```

Each command reloads the archived 90-day and 365-day checkpoints, fits both
fusion stages on the 2018--2021 validation period only, and evaluates the
frozen forecasts on 2022--2025 and the partial-year 2026 holdout. Generated
files are written under `outputs/` and checked against the archived reference
tables. CPU execution is supported; CUDA is faster.

## Repository map

| Path | Purpose |
|---|---|
| `src/reefformer/` | Reusable fusion, metrics, and checkpoint-compatible model code |
| `scripts/` | Training, synthetic demo, and full reference reproduction |
| `data/processed/` | Compact processed datasets used by the archived runs |
| `data/inventories/` | Frozen pixel inventories and mapping audit |
| `checkpoints/` | Validation-selected neural checkpoints |
| `reference_results/` | Frozen result tables and generalization summaries |
| `docs/` | Installation, data provenance, user guide, and full runbook |
| `tests/` | Unit and smoke tests |
| `third_party/` | Third-party notices and licenses |

## Scope of the archived evidence

The package distinguishes four evaluation settings:

1. future-period evaluation over fixed pixels;
2. buffered leave-one-sector-out evaluation over unseen Hainan pixels;
3. independent-domain replication and frozen-weight transfer between Hainan
   and the Great Barrier Reef;
4. same-domain replication with the OSTIA SST product.

See `reference_results/generalization/GENERALIZATION_EVIDENCE_SUMMARY.md` for
the exact interpretation boundaries.

## Data and software licenses

Original ReefFormer software is released under the MIT License. This license
does not relicense the bundled processed data. The compact
iTransformer reimplementation follows the upstream MIT-licensed architecture;
the upstream license is retained under `third_party/iTransformer/`.

Processed data remain subject to source attribution requirements:

- NOAA Coral Reef Watch content is public domain; credit NOAA Coral Reef Watch.
- AIMS monitoring data are used under CC BY attribution terms.
- OSTIA data require the Copernicus Marine acknowledgement documented in
  `docs/DATA.md`.

See `docs/DATA.md` before redistributing derived data.

## Citation

The final article DOI and permanent repository DOI will be added at release.
Until then, use the metadata in `CITATION.cff`, identify version `1.0.0`, and
link to `https://github.com/ludong-cell/reefformer`.

## Contact and authorship

The manuscript source still contains author placeholders. The approved article
authors, corresponding-author email, article DOI, and archival DOI will be
added when supplied. No personal contact details are invented in this release.
