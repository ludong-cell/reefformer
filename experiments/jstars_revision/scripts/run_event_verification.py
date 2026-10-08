"""Event-based (threshold-exceedance) verification of archived forecasts, Hainan 154 and GBR 292.

Definitions fixed before any score was computed (exploratory, added after 2026 access; no refitting):
  E1 HotSpot>=1  : target SST >= MMM_p + 1 degC, with MMM_p the maximum of the 12 monthly means of the
                   training-only (2000-2017) 365-bin climatology C_p. Mirrors the CRW HotSpot>=1 stress level
                   using this study's climatology. Forecast SST = C_p(target day) + predicted SSTA.
  E2 Upper decile: target SSTA >= per-pixel 2000-2017 90th percentile (identical to the High-SSTA subset).
Scores pool all origin x lead x pixel cells of a split: POD = H/(H+M), FAR = F/(H+F),
PSS (Peirce skill score) = POD - F/(F+CN), frequency bias B = (H+F)/(H+M).
Paired 60-day moving-block bootstrap of PSS(ReefFormer) - PSS(model), recomputing the contingency table in each
replicate from per-origin counts, per seed, then averaging the replicate series over seeds (same resampling seeds
as the RMSE intervals: 154420 Dev, 154421 Test2026).
Outputs: runs/event_verification_v1/<domain>/EVENT_SCORES.csv, EVENT_PSS_BOOTSTRAP.csv, EVENT_BASE_RATES.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from build_final154_complete_results import block_counts

ROOT = Path(__file__).resolve().parent
SEEDS, H = (42, 43, 44), 30
SPLITS = ("Dev", "Test2026")
OUT = ROOT / "runs/event_verification_v1"


def hainan():
    run = ROOT / "runs/final_154_v1"
    z = np.load(ROOT / "data/frozen_154_ssta_v1/dataset_daily.npz")
    d = dict(sst=z["sst_all_c"], ssta=z["ssta_all_c"], dates=pd.DatetimeIndex(z["dates_all"]), clim=z["climatology365_c"],
             rho=z["dap_rho_train"], ids={s: z[s].astype(int) for s in SPLITS})
    sb = lambda m: (lambda seed, s: np.load(run / f"strong_baselines/{m.lower()}/{m}_seed{seed}_{s}.npy"))
    learned = {"DLinear": sb("DLinear"), "PatchTST": sb("PatchTST"),
               "iTransformer": lambda seed, s: np.load(run / f"itransformer/FULL/iT_L365_seed{seed}_{s}.npy"),
               "Dual Context": lambda seed, s: np.load(run / f"evidence/CTX_seed{seed}_{s}.npy"),
               "DAP Smooth": lambda seed, s: np.load(run / f"complete_results/DAP_Smooth_seed{seed}_{s}.npy"),
               "ReefFormer": lambda seed, s: np.load(run / f"evidence/ReefFormer_seed{seed}_{s}.npy")}
    return d, learned


def gbr():
    g = ROOT / "runs/gbr_generalization_v1"
    z = np.load(ROOT / "data/gbr_generalization_v1/dataset_daily.npz")
    d = dict(sst=z["sst_c"], ssta=z["ssta_c"], dates=pd.DatetimeIndex(z["dates"]), clim=z["climatology365_c"],
             rho=z["dap_rho_train"], ids={s: z[s].astype(int) for s in SPLITS})
    learned = {}
    for m in ("DLinear", "PatchTST"):
        if (g / f"strong_baselines/{m.lower()}/{m}_seed42_Dev.npy").exists():
            learned[m] = (lambda m: lambda seed, s: np.load(g / f"strong_baselines/{m.lower()}/{m}_seed{seed}_{s}.npy"))(m)
    learned.update({"iT90": lambda seed, s: np.load(g / f"models/iT_L90_seed{seed}_{s}.npy"),
                    "iT365": lambda seed, s: np.load(g / f"models/iT_L365_seed{seed}_{s}.npy"),
                    "Dual Context": lambda seed, s: np.load(g / f"results_three_seed/Dual_Context_seed{seed}_{s}.npy"),
                    "ReefFormer": lambda seed, s: np.load(g / f"results_three_seed/ReefFormer_seed{seed}_{s}.npy")})
    return d, learned


def main(domain):
    d, learned = {"hainan": hainan, "gbr": gbr}[domain]()
    out = OUT / domain; out.mkdir(parents=True, exist_ok=True)
    dates, ssta, sst = d["dates"], d["ssta"].astype("float64"), d["sst"].astype("float64")
    month_of_bin = pd.date_range("2001-01-01", "2001-12-31").month.to_numpy()          # 365 non-leap bins
    clim = d["clim"].astype("float64")
    mmm = np.max([clim[month_of_bin == m].mean(0) for m in range(1, 13)], axis=0)      # (P,)
    q90 = np.quantile(ssta[dates.year <= 2017], 0.9, axis=0)
    leads = np.arange(1, H + 1)
    rows, boots, base = [], [], []
    for k, s in enumerate(SPLITS):
        tidx = d["ids"][s][:, None] + leads[None, :]
        c_target = sst[tidx] - ssta[tidx]                                               # climatology at target
        truth = ssta[tidx]
        obs = {"HotSpot>=1": truth + c_target >= mmm + 1.0, "Upper decile": truth >= q90}
        preds = {"Persistence": {0: np.repeat(ssta[d["ids"][s], None, :], H, axis=1)},
                 "DAP": {0: ssta[d["ids"][s], None, :] * d["rho"][None, None, :] ** leads[None, :, None]}}
        for name, f in learned.items():
            preds[name] = {seed: f(seed, s).astype("float64") for seed in SEEDS}
        counts = block_counts(len(tidx), 154420 + k)
        for ev, o in obs.items():
            base.append({"split": s, "event": ev, "base_rate": float(o.mean()), "n_cells": int(o.size),
                         "origins_with_event": int(o.any(axis=(1, 2)).sum())})
            thr = (mmm + 1.0 - c_target) if ev == "HotSpot>=1" else np.broadcast_to(q90, truth.shape)
            per_origin = {}
            for name, ps in preds.items():
                per_origin[name] = {}
                vals = []
                for seed, p in ps.items():
                    fc = p >= thr
                    tab = [(fc & o), (~fc & o), (fc & ~o), (~fc & ~o)]                 # H, M, F, CN
                    po = np.stack([t.sum(axis=(1, 2)) for t in tab], axis=1).astype("float64")   # (n_origins, 4)
                    per_origin[name][seed] = po
                    Hh, M, F, CN = po.sum(0)
                    vals.append(dict(POD=Hh / (Hh + M), FAR=F / (Hh + F) if Hh + F else np.nan,
                                     PSS=Hh / (Hh + M) - F / (F + CN), Bias=(Hh + F) / (Hh + M)))
                rows.append({"split": s, "event": ev, "Model": name,
                             **{m: float(np.nanmean([v[m] for v in vals])) for m in ("POD", "FAR", "PSS", "Bias")}})

            def pss_samples(po):
                t = counts @ po                                                        # (reps, 4)
                return t[:, 0] / (t[:, 0] + t[:, 1]) - t[:, 2] / (t[:, 2] + t[:, 3])

            ref = per_origin["ReefFormer"]
            for name in preds:
                if name == "ReefFormer":
                    continue
                other = per_origin[name]
                samp = np.mean([pss_samples(ref[seed]) - pss_samples(other.get(seed, other[next(iter(other))]))
                                for seed in SEEDS], axis=0)
                point = np.mean([(lambda a, b: (a[0] / (a[0] + a[1]) - a[2] / (a[2] + a[3])) -
                                  (b[0] / (b[0] + b[1]) - b[2] / (b[2] + b[3])))(ref[seed].sum(0),
                                                                              other.get(seed, other[next(iter(other))]).sum(0))
                                 for seed in SEEDS])
                lo, hi = np.nanquantile(samp, (0.025, 0.975))
                boots.append({"split": s, "event": ev, "Contrast": f"ReefFormer - {name}", "Delta_PSS": float(point),
                              "CI95_low": float(lo), "CI95_high": float(hi)})
    pd.DataFrame(rows).to_csv(out / "EVENT_SCORES.csv", index=False)
    pd.DataFrame(boots).to_csv(out / "EVENT_PSS_BOOTSTRAP.csv", index=False)
    pd.DataFrame(base).to_csv(out / "EVENT_BASE_RATES.csv", index=False)
    pd.set_option("display.width", 200)
    print(pd.DataFrame(base).to_string(index=False))
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    print(pd.DataFrame(boots).round(3).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1])
