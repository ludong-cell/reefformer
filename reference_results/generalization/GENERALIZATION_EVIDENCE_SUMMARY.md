# Generalization evidence summary

Status: **complete**

Negative Delta RMSE favors ReefFormer (or the frozen transferred ReefFormer in
the two bidirectional rows). All intervals are paired 60-day moving-block 95%
intervals with 10,000 resamples.

## Overall contrasts

| Setting               | Reference   |   Delta_RMSE |   CI_low |   CI_high | Resolved   | Source                                        |
|:----------------------|:------------|-------------:|---------:|----------:|:-----------|:----------------------------------------------|
| GBR independent fit   | iT365       |      -0.0175 |  -0.0529 |    0.0172 | False      | GBR three-seed independent-domain replication |
| Hainan to GBR frozen  | iT365       |      -0.0245 |  -0.0564 |    0.0023 | False      | Bidirectional frozen-weight transfer          |
| GBR to Hainan frozen  | iT365       |      -0.0474 |  -0.0773 |   -0.0154 | True       | Bidirectional frozen-weight transfer          |
| Hainan unseen sectors | iT365       |      -0.0378 |  -0.0570 |   -0.0208 | True       | Four-fold 25 km buffered leave-one-sector-out |
| Hainan OSTIA product  | iT365       |      -0.0332 |  -0.0532 |   -0.0108 | True       | OSTIA second-product replication              |
| GBR independent fit   | DAP         |      -0.0808 |  -0.1112 |   -0.0514 | True       | GBR three-seed independent-domain replication |
| Hainan to GBR frozen  | DAP         |      -0.0932 |  -0.1356 |   -0.0460 | True       | Bidirectional frozen-weight transfer          |
| GBR to Hainan frozen  | DAP         |      -0.0654 |  -0.1060 |   -0.0302 | True       | Bidirectional frozen-weight transfer          |
| Hainan unseen sectors | DAP         |      -0.0717 |  -0.1219 |   -0.0260 | True       | Four-fold 25 km buffered leave-one-sector-out |
| Hainan OSTIA product  | DAP         |      -0.0319 |  -0.0498 |   -0.0170 | True       | OSTIA second-product replication              |

## Cross-region fusion coefficient transfer

The transferred coefficients are compared directly with independently fitted
target-domain ReefFormer coefficients.

| Direction     |   Delta_RMSE_vs_target_fit |   CI_low |   CI_high |
|:--------------|---------------------------:|---------:|----------:|
| Hainan to GBR |                    -0.0124 |  -0.0271 |    0.0018 |
| GBR to Hainan |                     0.0050 |  -0.0068 |    0.0171 |

## Interpretation boundary

- GBR independent fitting tests geographic replication under the same product.
- Bidirectional frozen-weight transfer tests whether learned shared model weights
  remain useful across domains; target train-only climatology, scale, and DAP
  coefficients are retained as historical preprocessing, so this is not strict
  source-statistics zero-shot transfer.
- Buffered leave-one-sector-out is the direct unseen-pixel spatial test.
- OSTIA is a same-domain second-product test, not a geographic test.
