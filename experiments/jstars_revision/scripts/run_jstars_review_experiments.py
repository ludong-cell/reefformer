"""Low-cost review experiments for the JSTARS submission (no retraining).

Uses only archived predictions under runs/final_154_v1 and reuses the paper's own fitting
(analyze_final154_core.fit_smooth) and bootstrap (build_final154_complete_results) code, with the
paper's bootstrap seeds, so every number is directly comparable with the manuscript tables.

Experiments
  E1  X + LPF for X in {PatchTST, DLinear, iT90, iT365}; LPF fitted on 2018-2021 validation only.
  E2  Context-fusion forms + LPF: equal-weight (0.5/0.5), validation-fitted constant, and the
      archived ReefFormer (smooth); paired 60-day block-bootstrap intervals vs ReefFormer.
  E3  Skill scores: MSSS vs climatology, ACC by lead band; mean signed error by target year.
  E4  2026 Diebold-Mariano tests (Newey-West HAC, lag 29, Harvey small-sample correction).
  E5  Warm-season (target month May-Sep) RMSE.
Outputs: runs/final_154_v1/review_experiments/*.csv
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_final154_core import fit_smooth
from build_final154_complete_results import BANDS, block_counts, paired_bootstrap

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data/frozen_154_ssta_v1/dataset_daily.npz"
RUN = ROOT / "runs/final_154_v1"
OUT = RUN / "review_experiments"
OUT.mkdir(exist_ok=True)
SEEDS, H = (42, 43, 44), 30
SPLITS = ("Val", "Dev", "Test2026")
LOOKBACK = {"DLinear": 180, "PatchTST": 365}   # validation-selected, from strong_baselines/*/SELECTION.json

with np.load(DATA) as z:
    dates = pd.DatetimeIndex(z["dates_all"])
    ssta = z["ssta_all_c"].astype("float32")
    rho = z["dap_rho_train"].astype("float32")
    ids = {s: z[s].astype(int) for s in SPLITS}
leads = np.arange(1, H + 1)
truth = {s: ssta[ids[s][:, None] + leads[None, :]] for s in SPLITS}
dap = {s: ssta[ids[s], None, :] * rho[None, None, :] ** leads[None, :, None] for s in SPLITS}
clim = {s: np.zeros_like(truth[s]) for s in SPLITS}
pers = {s: np.repeat(ssta[ids[s], None, :], H, axis=1) for s in SPLITS}
_tidx = {s: ids[s][:, None] + leads[None, :] for s in SPLITS}                  # origins x leads
target_year = {s: dates.year.to_numpy()[_tidx[s]] for s in SPLITS}
target_month = {s: dates.month.to_numpy()[_tidx[s]] for s in SPLITS}


def load(model, seed, split):
    f32 = lambda p: np.load(p).astype("float32")
    if model in ("DLinear", "PatchTST"):
        name = f"{model}_L{LOOKBACK[model]}_seed{seed}_Val.npy" if split == "Val" else f"{model}_seed{seed}_{split}.npy"
        return f32(RUN / "strong_baselines" / model.lower() / name)
    if model in ("iT90", "iT365"):
        return f32(RUN / f"itransformer/FULL/iT_L{model[2:]}_seed{seed}_{split}.npy")
    if model in ("ReefFormer", "CTX"):
        return f32(RUN / f"evidence/{model}_seed{seed}_{split}.npy")
    raise KeyError(model)


def blend(w, a, b):                                  # w: (30,) weight on a
    return w[None, :, None] * a + (1 - w[None, :, None]) * b


def const_weight(left, right, tr):
    d = (left - right).astype("float64"); r = (tr - right).astype("float64")
    return float(np.clip((d * r).sum() / (d * d).sum(), 0, 1))


rmse = lambda p, t: float(np.sqrt(np.mean((p.astype("float64") - t) ** 2)))

# ---------------------------------------------------------------- build predictions
pred = {s: {seed: {} for seed in SEEDS} for s in SPLITS}
fits = []
for seed in SEEDS:
    raw = {m: {s: load(m, seed, s) for s in SPLITS} for m in ("DLinear", "PatchTST", "iT90", "iT365", "ReefFormer")}
    # context-fusion variants (weights from validation only)
    q_fix = const_weight(raw["iT90"]["Val"], raw["iT365"]["Val"], truth["Val"])
    ctx = {"Avg CTX": {s: 0.5 * raw["iT90"][s] + 0.5 * raw["iT365"][s] for s in SPLITS},
           "Fixed CTX": {s: q_fix * raw["iT90"][s] + (1 - q_fix) * raw["iT365"][s] for s in SPLITS}}
    bases = {"DLinear": raw["DLinear"], "PatchTST": raw["PatchTST"], "iT90": raw["iT90"],
             "iT365": raw["iT365"], **ctx}
    for name, series in bases.items():
        a, tau, w = fit_smooth(dap["Val"], series["Val"], truth["Val"])
        fits.append({"seed": seed, "base": name, "a_out": a, "tau_out_d": tau, "w_D1": w[0], "w_D30": w[29],
                     **({"q_fixed": q_fix} if name == "Fixed CTX" else {})})
        for s in SPLITS:
            pred[s][seed][f"{name}+LPF"] = blend(w, dap[s], series[s]).astype("float32")
    for s in SPLITS:
        pred[s][seed].update({"Climatology": clim[s], "Persistence": pers[s], "DAP": dap[s],
                              "DLinear": raw["DLinear"][s], "PatchTST": raw["PatchTST"][s],
                              "iTransformer": raw["iT365"][s], "ReefFormer": raw["ReefFormer"][s]})
pd.DataFrame(fits).to_csv(OUT / "E1E2_LPF_FITS.csv", index=False)

MODELS = ["Climatology", "Persistence", "DAP", "DLinear", "PatchTST", "iTransformer",
          "DLinear+LPF", "PatchTST+LPF", "iT90+LPF", "iT365+LPF", "Avg CTX+LPF", "Fixed CTX+LPF", "ReefFormer"]

# ---------------------------------------------------------------- E1/E2 accuracy table + E3 skill
rows = []
for s in ("Dev", "Test2026"):
    t = truth[s]
    mse_clim = np.mean(t.astype("float64") ** 2)
    for m in MODELS:
        seeds = (42,) if m in ("Climatology", "Persistence", "DAP") else SEEDS
        r = {"split": s, "Model": m}
        vals = {k: [] for k in ("RMSE", "MAE", "MSSS", "ACC", "D1_7", "D8_14", "D15_30", "ACC_D1_7", "ACC_D15_30")}
        for seed in seeds:
            p = pred[s][seed][m].astype("float64"); e = p - t
            vals["RMSE"].append(math.sqrt(np.mean(e ** 2))); vals["MAE"].append(np.mean(np.abs(e)))
            vals["MSSS"].append(1 - np.mean(e ** 2) / mse_clim)
            for b, sl in (("D1_7", slice(0, 7)), ("D8_14", slice(7, 14)), ("D15_30", slice(14, 30))):
                vals[b].append(math.sqrt(np.mean(e[:, sl] ** 2)))
            acc = []
            for h in range(H):   # anomaly correlation per lead, pooled over origins x pixels
                ph, th = p[:, h].ravel(), t[:, h].ravel().astype("float64")
                acc.append(np.nan if ph.std() == 0 else np.corrcoef(ph, th)[0, 1])
            acc = np.array(acc)
            vals["ACC"].append(np.nanmean(acc)); vals["ACC_D1_7"].append(np.nanmean(acc[:7]))
            vals["ACC_D15_30"].append(np.nanmean(acc[14:]))
        r.update({k: float(np.mean(v)) for k, v in vals.items()})
        r["RMSE_SD"] = float(np.std([math.sqrt(np.mean((pred[s][sd][m].astype("float64") - t) ** 2)) for sd in seeds], ddof=1)) if len(seeds) > 1 else 0.0
        rows.append(r)
acc_tab = pd.DataFrame(rows)
acc_tab.to_csv(OUT / "E1E2E3_ACCURACY_SKILL.csv", index=False)

# ---------------------------------------------------------------- E1/E2 paired bootstrap vs ReefFormer
contrasts = [("ReefFormer", m, f"ReefFormer - {m}") for m in
             ("DLinear+LPF", "PatchTST+LPF", "iT90+LPF", "iT365+LPF", "Avg CTX+LPF", "Fixed CTX+LPF",
              "DLinear", "PatchTST", "iTransformer")]
contrasts += [("PatchTST+LPF", "PatchTST", "PatchTST+LPF - PatchTST"), ("DLinear+LPF", "DLinear", "DLinear+LPF - DLinear")]
boot = []
for index, s in enumerate(("Dev", "Test2026")):
    counts = block_counts(len(ids[s]), 154420 + index)       # same resamples as the manuscript tables
    boot += paired_bootstrap(pred, truth[s], s, counts, contrasts)
boot = pd.DataFrame(boot)
boot.to_csv(OUT / "E1E2_PAIRED_60D_BOOTSTRAP.csv", index=False)

# ---------------------------------------------------------------- E3 mean signed error by target year
bias = []
for s in ("Dev", "Test2026"):
    yrs = target_year[s]
    for m in ("Climatology", "DAP", "DLinear", "PatchTST", "iTransformer", "PatchTST+LPF", "ReefFormer"):
        seeds = (42,) if m in ("Climatology", "DAP") else SEEDS
        for y in np.unique(yrs):
            mask = yrs == y
            bias.append({"split": s, "Model": m, "target_year": int(y),
                         "Bias": float(np.mean([np.mean((pred[s][sd][m] - truth[s])[mask]) for sd in seeds]))})
pd.DataFrame(bias).to_csv(OUT / "E3_BIAS_BY_YEAR.csv", index=False)

# ---------------------------------------------------------------- E4 Diebold-Mariano on 2026
def dm_test(a, b, s="Test2026", lag=H - 1):
    d = np.mean([np.mean((pred[s][sd][a].astype("float64") - truth[s]) ** 2 -
                         (pred[s][sd][b].astype("float64") - truth[s]) ** 2, axis=(1, 2)) for sd in SEEDS], axis=0)
    n, dbar = len(d), d.mean(); u = d - dbar
    gamma = [np.dot(u[k:], u[:n - k]) / n for k in range(lag + 1)]
    var = gamma[0] + 2 * sum((1 - k / (lag + 1)) * gamma[k] for k in range(1, lag + 1))
    dm = dbar / math.sqrt(var / n)
    hln = dm * math.sqrt((n + 1 - 2 * (lag + 1) + (lag + 1) * lag / n) / n)   # Harvey-Leybourne-Newbold
    from scipy.stats import t as student_t
    p = 2 * student_t.sf(abs(hln), df=n - 1)
    return {"Model_A": a, "Model_B": b, "n_origins": n, "mean_dMSE": dbar, "DM": dm, "HLN": hln, "p_value": p}


dm = pd.DataFrame([dm_test("ReefFormer", m) for m in
                   ("DLinear", "PatchTST", "iTransformer", "DLinear+LPF", "PatchTST+LPF", "iT365+LPF", "Fixed CTX+LPF")])
dm.to_csv(OUT / "E4_DM_2026.csv", index=False)

# ---------------------------------------------------------------- E5 warm-season RMSE (target month May-Sep)
warm = []
for s in ("Dev", "Test2026"):
    mask = np.isin(target_month[s], [5, 6, 7, 8, 9])
    for m in MODELS:
        seeds = (42,) if m in ("Climatology", "Persistence", "DAP") else SEEDS
        warm.append({"split": s, "Model": m, "warm_RMSE": float(np.mean(
            [rmse(pred[s][sd][m][mask], truth[s][mask]) for sd in seeds])), "share_of_targets": float(mask.mean())})
pd.DataFrame(warm).to_csv(OUT / "E5_WARM_SEASON.csv", index=False)

# ---------------------------------------------------------------- console summary
pd.set_option("display.width", 200)
print(acc_tab[["split", "Model", "RMSE", "RMSE_SD", "MAE", "MSSS", "ACC", "D1_7", "D8_14", "D15_30"]].round(4).to_string(index=False))
b = boot[(boot.Metric == "RMSE") & (boot.Band.isin(["Overall", "D1-7"]))]
print(b[["split", "Contrast", "Band", "Delta_A_minus_B", "CI95_low", "CI95_high"]].round(4).to_string(index=False))
print(dm.round(4).to_string(index=False))
print(pd.DataFrame(warm).pivot(index="Model", columns="split", values="warm_RMSE").round(4).to_string())
