"""GBR: paired 60-day moving-block bootstrap of ReefFormer against the GBR DLinear and PatchTST baselines.

Same resampling as the paper (paired_bootstrap; seeds 154420 Dev / 154421 Test2026, 10,000 replicates).
Writes runs/gbr_generalization_v1/strong_baselines/REEFFORMER_VS_STRONG_60D_BOOTSTRAP.csv and SUMMARY_MEAN.csv.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from build_final154_complete_results import block_counts, paired_bootstrap

ROOT = Path(__file__).resolve().parent
G = ROOT / "runs/gbr_generalization_v1"
SEEDS, H = (42, 43, 44), 30
with np.load(ROOT / "data/gbr_generalization_v1/dataset_daily.npz") as z:
    ssta = z["ssta_c"].astype("float32"); ids = {s: z[s].astype(int) for s in ("Dev", "Test2026")}
leads = np.arange(1, H + 1)
pred, rows = {}, []
for s in ids:
    pred[s] = {seed: {"ReefFormer": np.load(G / f"results_three_seed/ReefFormer_seed{seed}_{s}.npy"),
                      "DLinear": np.load(G / f"strong_baselines/dlinear/DLinear_seed{seed}_{s}.npy"),
                      "PatchTST": np.load(G / f"strong_baselines/patchtst/PatchTST_seed{seed}_{s}.npy")} for seed in SEEDS}
boot = []
for k, s in enumerate(ids):
    truth = ssta[ids[s][:, None] + leads[None, :]].astype("float64")
    boot += paired_bootstrap(pred, truth, s, block_counts(len(truth), 154420 + k),
                             [("ReefFormer", m, f"ReefFormer - {m}") for m in ("DLinear", "PatchTST")])
    for m in ("DLinear", "PatchTST", "ReefFormer"):
        e = [pred[s][sd][m].astype("float64") - truth for sd in SEEDS]
        r = lambda sl: float(np.mean([np.sqrt(np.mean(x[:, sl] ** 2)) for x in e]))
        rows.append({"split": s, "Model": m, "RMSE": r(slice(0, 30)), "MAE": float(np.mean([np.mean(np.abs(x)) for x in e])),
                     "D1_7": r(slice(0, 7)), "D8_14": r(slice(7, 14)), "D15_30": r(slice(14, 30)),
                     "RMSE_SD": float(np.std([np.sqrt(np.mean(x ** 2)) for x in e], ddof=1))})
b = pd.DataFrame(boot); b.to_csv(G / "strong_baselines/REEFFORMER_VS_STRONG_60D_BOOTSTRAP.csv", index=False)
pd.DataFrame(rows).to_csv(G / "strong_baselines/SUMMARY_MEAN.csv", index=False)
print(pd.DataFrame(rows).round(4).to_string(index=False))
print(b[b.Metric == "RMSE"][["split", "Contrast", "Band", "Delta_A_minus_B", "CI95_low", "CI95_high"]].round(4).to_string(index=False))
