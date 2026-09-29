# OSTIA second-product replication

Status: **complete**

- Product: Copernicus Marine OSTIA L4 reprocessed SST, 0.05 degree.
- Frozen Hainan locations: 154.
- Seeds: 42, 43, 44.
- Train: 2000-2017; Validation: 2018-2021; Development: 2022-2025.
- Available 2026 origins with complete 30-day targets: 61.
- Dataset SHA256: `1d3fae8ebb966c56580057d9dbd28812b26356ae1e5e1a5ce714cd8876cb08cd`.
- Checkpoint selection and fusion fitting used Validation only.

## Development results

| Model        | Split   |   RMSE_mean |   RMSE_SD |   MAE_mean |   MAE_SD |   D1_7_mean |   D8_14_mean |   D15_30_mean |
|:-------------|:--------|------------:|----------:|-----------:|---------:|------------:|-------------:|--------------:|
| iT90         | Dev     |      0.7697 |    0.0079 |     0.5787 |   0.0060 |      0.5760 |       0.7767 |        0.8377 |
| iT365        | Dev     |      0.7723 |    0.0064 |     0.5918 |   0.0062 |      0.6567 |       0.7770 |        0.8159 |
| DAP          | Dev     |      0.7710 |    0.0000 |     0.5677 |   0.0000 |      0.5472 |       0.7821 |        0.8462 |
| Dual Context | Dev     |      0.7484 |    0.0056 |     0.5651 |   0.0042 |      0.5712 |       0.7594 |        0.8094 |
| ReefFormer   | Dev     |      0.7392 |    0.0029 |     0.5478 |   0.0021 |      0.5416 |       0.7558 |        0.8038 |

## Claim boundary

This is a same-domain, second-SST-product replication. It tests sensitivity to the
input product and preprocessing chain, not geographic transfer. Geographic evidence
is supplied separately by the GBR and spatial leave-one-sector-out experiments.
