# GBR independent-domain generalization result

## Frozen protocol

- Pixels: 292
- Seeds: 42, 43, 44
- Train: 2000-2017; Validation: 2018-2021; Development: 2022-2025
- Fusion fitting: Validation only, independently for each seed
- Development used for fitting or selection: no
- Dataset SHA256: `d67c0ed5776c0c0cdc1a5aff25d538f23e190e896f448da4fb3150fec95e60a8`

## Three-seed development results

| Model        | Split   |   RMSE_mean |   RMSE_std |   MAE_mean |   MAE_std |   D1_7_mean |   D1_7_std |   D8_14_mean |   D8_14_std |   D15_30_mean |   D15_30_std |
|:-------------|:--------|------------:|-----------:|-----------:|----------:|------------:|-----------:|-------------:|------------:|--------------:|-------------:|
| iT90         | Dev     |      0.5648 |     0.0037 |     0.4346 |    0.0029 |      0.3917 |     0.0085 |       0.5420 |      0.0047 |        0.6343 |       0.0022 |
| iT365        | Dev     |      0.5570 |     0.0033 |     0.4340 |    0.0021 |      0.4401 |     0.0041 |       0.5473 |      0.0045 |        0.6050 |       0.0033 |
| DAP          | Dev     |      0.6203 |     0.0000 |     0.4721 |    0.0000 |      0.3557 |     0.0000 |       0.5702 |      0.0000 |        0.7237 |       0.0000 |
| Dual Context | Dev     |      0.5544 |     0.0039 |     0.4276 |    0.0026 |      0.3908 |     0.0079 |       0.5376 |      0.0034 |        0.6190 |       0.0066 |
| ReefFormer   | Dev     |      0.5395 |     0.0037 |     0.4090 |    0.0026 |      0.3478 |     0.0003 |       0.5263 |      0.0006 |        0.6096 |       0.0063 |

ReefFormer - iT365 overall mean ΔRMSE = -0.0175 °C, paired 60-day moving-block 95% CI [-0.0529, 0.0172].

This is an independent-domain replication. It is not a zero-shot transfer of Hainan checkpoints and was not used to redesign the Hainan architecture.
