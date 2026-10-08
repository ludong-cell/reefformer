"""Weather-residual module on top of the frozen ReefFormer, strict chronological OOF protocol (Hainan 154 / GBR 292).

Port of run_stage5_weather_residual_strict.py (frozen Stage-5 protocol) with two changes only:
  1. base forecast = full ReefFormer (two iTransformer branches L90/L365 -> LCF -> LPF), rebuilt inside every fold;
  2. domain = current 154-pixel Hainan inventory (and the 292-pixel GBR replication).
Protocol (identical to Stage 5): four chronological folds; fold k trains both branches on 2000..end_k, early-stops
on one internal validation year, and produces out-of-fold (OOF) ReefFormer forecasts for the next three years.
Fold preprocessing (climatology, RMS scale, DAP rho) uses fold-training years only. Residual networks are trained
on OOF origins 2006-2014 and early-stopped on OOF 2015-2017. No 2018-2026 observation is used before Dev/2026
evaluation; the archived, frozen ReefFormer Dev/2026 forecasts are the evaluation base (W0).
  W1  state-only residual (7-day mean/std/slope of SSTA + lead)
  W2  state + weather residual (14-day along-shore wind, cross-shore wind, ssrd via causal Conv1D)
Coast geometry: Hainan as in Stage 5 (offshore = radial from 110E, 19.2N); GBR: uniform offshore bearing 60 deg
(mainland coast trending ~150/330 deg), fixed a priori.
"""
from __future__ import annotations

import copy
import json
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from itransformer_feature_pilot import Model, CONFIG as BASE_CONFIG
from train_backbone import calendar_index
from analyze_final154_core import fit_smooth
from build_final154_complete_results import block_counts, paired_bootstrap

ROOT = Path(__file__).resolve().parent
SEEDS, H, BATCH, LR, MAXE, PATIENCE = (42, 43, 44), 30, 32, 1e-3, 40, 8
FOLDS = [("F1", 2004, 2005, 2006, 2008), ("F2", 2007, 2008, 2009, 2011),
         ("F3", 2010, 2011, 2012, 2014), ("F4", 2013, 2014, 2015, 2017)]
DOMAINS = {
    "hainan": dict(data=ROOT / "data/frozen_154_ssta_v1/dataset_daily.npz", sst="sst_all_c", ssta="ssta_all_c",
                   dates="dates_all", weather=ROOT / "data/frozen_154_weather_v1/weather_daily.npz",
                   reef=lambda s, sp: ROOT / f"runs/final_154_v1/evidence/ReefFormer_seed{s}_{sp}.npy",
                   masks=ROOT / "runs/final_154_v1/complete_results/EXTREME_MASKS_{}.npz",
                   geometry="radial", out=ROOT / "runs/weather_residual_154_v1"),
    "gbr": dict(data=ROOT / "data/gbr_generalization_v1/dataset_daily.npz", sst="sst_c", ssta="ssta_c", dates="dates",
                weather=ROOT / "data/gbr_weather_v1/weather_daily.npz",
                reef=lambda s, sp: ROOT / f"runs/gbr_generalization_v1/results_three_seed/ReefFormer_seed{s}_{sp}.npy",
                masks=None, geometry="bearing60", out=ROOT / "runs/weather_residual_gbr_v1"),
}


def make_ssta(sst, dates, end):
    mask = (dates >= pd.Timestamp("2000-01-01")) & (dates <= pd.Timestamp(f"{end}-12-31")); doy = calendar_index(dates)
    clim = np.stack([sst[mask & (doy == i)].mean(0) for i in range(365)])
    clim = np.mean([np.roll(clim, k, 0) for k in range(-15, 16)], 0)
    x = sst - clim[doy]; scale = float(np.sqrt(np.mean(x[mask] ** 2)))
    past, fut = x[mask][:-1], x[mask][1:]
    rho = np.clip(np.sum(past * fut, 0) / np.maximum(np.sum(past * past, 0), 1e-12), 0, 1)
    return x.astype("float32"), scale, rho.astype("float64")


def origins(dates, a, b, L=365):
    return np.array([i for i in range(L - 1, len(dates) - H)
                     if dates[i + 1] >= pd.Timestamp(f"{a}-01-01") and dates[i + H] <= pd.Timestamp(f"{b}-12-31")], dtype=int)


class ResidualNet(nn.Module):                      # unchanged from Stage 5
    def __init__(self, weather):
        super().__init__(); self.weather = weather
        self.state = nn.Sequential(nn.Linear(4, 32), nn.GELU())
        if weather:
            self.wc1 = nn.Conv1d(3, 16, 3); self.wc2 = nn.Conv1d(16, 32, 3)
        self.head = nn.Sequential(nn.Linear(32, 32), nn.GELU(), nn.Linear(32, 1))
        nn.init.zeros_(self.head[-1].weight); nn.init.zeros_(self.head[-1].bias)

    def forward(self, state, weather):
        b, p, _ = state.shape
        h = torch.arange(1, H + 1, device=state.device).float() / H
        ss = state[:, None].expand(-1, H, -1, -1); hh = h[None, :, None, None].expand(b, -1, p, 1)
        z = self.state(torch.cat([ss, hh], -1))
        if self.weather:
            w = weather.reshape(b * p, 14, 3).transpose(1, 2)
            w = F.gelu(self.wc1(F.pad(w, (2, 0)))); w = F.gelu(self.wc2(F.pad(w, (2, 0))))[:, :, -1].reshape(b, p, 32)
            z = z + w[:, None]
        return self.head(z).squeeze(-1)


def train_branch(x, scale, tr, va, L, seed, cfg_base, device):
    cfg = dict(cfg_base); cfg.update(pred_len=H, seq_len=L)
    series = torch.tensor(x / scale, device=device); hist = torch.arange(-L + 1, 1, device=device)
    lead = torch.arange(1, H + 1, device=device)
    truthv = np.stack([x[i + 1:i + H + 1] for i in va])
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); gen = torch.Generator().manual_seed(seed)
    model = Model(SimpleNamespace(**cfg)).to(device); opt = torch.optim.Adam(model.parameters(), lr=LR)
    batch = lambda ids: (series[torch.as_tensor(ids, device=device)[:, None] + hist],
                         series[torch.as_tensor(ids, device=device)[:, None] + lead])

    def infer(ids):
        model.eval(); q = []
        with torch.no_grad():
            for st in range(0, len(ids), BATCH):
                q.append(model(batch(ids[st:st + BATCH])[0], None, None, None).cpu().numpy())
        return np.concatenate(q) * scale

    best = float(np.mean((infer(va) - truthv) ** 2)); state = copy.deepcopy(model.state_dict()); bad = 0
    for ep in range(1, MAXE + 1):
        model.train(); perm = torch.randperm(len(tr), generator=gen).numpy()
        for st in range(0, len(perm), BATCH):
            bx, by = batch(tr[perm[st:st + BATCH]]); opt.zero_grad(set_to_none=True)
            F.mse_loss(model(bx, None, None, None), by).backward(); opt.step()
        mse = float(np.mean((infer(va) - truthv) ** 2))
        if mse < best:
            best, state, bad = mse, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
        if bad >= PATIENCE:
            break
    model.load_state_dict(state)
    return infer


def main(name):
    cfg = DOMAINS[name]; out = cfg["out"]; out.mkdir(parents=True, exist_ok=True)
    with np.load(cfg["data"]) as z:
        dates = pd.DatetimeIndex(z[cfg["dates"]]); sst = z[cfg["sst"]].astype("float64")
        xf = z[cfg["ssta"]].astype("float64"); rhof = z["dap_rho_train"].astype("float64")
        lat, lon = z["pixel_latitude"].astype(float), z["pixel_longitude"].astype(float)
        ids_eval = {s: z[s].astype(int) for s in ("Dev", "Test2026")}
    cfg_base = {k: BASE_CONFIG[k] for k in ["output_attention", "use_norm", "d_model", "embed", "freq", "dropout",
                                            "class_strategy", "factor", "n_heads", "d_ff", "e_layers", "activation"]}
    device = "cuda"; torch.use_deterministic_algorithms(True); torch.backends.cudnn.benchmark = False

    # ---------------- OOF ReefFormer forecasts (fold-internal two branches + LCF + LPF)
    oof = {s: [] for s in SEEDS}; meta = []
    cache = out / "oof_cache.npz"
    if cache.exists():
        c = np.load(cache, allow_pickle=True); oof = c["oof"].item(); print("OOF cache loaded", flush=True)
    else:
        L_ = np.arange(1, H + 1)
        for fname, end, ival, oa, ob in FOLDS:
            x, scale, rho = make_ssta(sst, dates, end)
            tr = origins(dates, 2000, end); va = origins(dates, ival, ival); oo = origins(dates, oa, ob)
            truthv = np.stack([x[i + 1:i + H + 1] for i in va]); trutho = np.stack([x[i + 1:i + H + 1] for i in oo])
            dv = x[va, None, :] * rho[None, None, :] ** L_[None, :, None]; do = x[oo, None, :] * rho[None, None, :] ** L_[None, :, None]
            for seed in SEEDS:
                inf90 = train_branch(x, scale, tr, va, 90, seed, cfg_base, device); p90v, p90o = inf90(va), inf90(oo)
                inf365 = train_branch(x, scale, tr, va, 365, seed, cfg_base, device); p365v, p365o = inf365(va), inf365(oo)
                _, _, q = fit_smooth(p90v, p365v, truthv)                       # LCF on the fold's internal val year
                ctxv = q[None, :, None] * p90v + (1 - q[None, :, None]) * p365v
                ctxo = q[None, :, None] * p90o + (1 - q[None, :, None]) * p365o
                a, tau, w = fit_smooth(dv, ctxv, truthv)                        # LPF
                base = w[None, :, None] * do + (1 - w[None, :, None]) * ctxo
                oof[seed].append({"ids": oo, "ssta": x, "base": base.astype("float32"), "truth": trutho.astype("float32")})
                meta.append({"fold": fname, "seed": seed, "train": f"2000-{end}", "internal_val": ival, "OOF": f"{oa}-{ob}",
                             "q_D1": q[0], "q_D30": q[-1], "w_D1": w[0], "w_D30": w[-1],
                             "OOF_RMSE": float(np.sqrt(np.mean((base - trutho) ** 2)))})
                torch.cuda.empty_cache(); print(name, fname, "seed", seed, "OOF RMSE", round(meta[-1]["OOF_RMSE"], 4), flush=True)
        pd.DataFrame(meta).to_csv(out / "OOF_FOLD_METADATA.csv", index=False)
        np.savez(cache, oof=np.array(oof, dtype=object))

    # ---------------- weather features (the OOF stage above does not need them)
    if not Path(cfg["weather"]).exists():
        print("OOF stage complete; weather file not built yet -> stop here and re-run later", flush=True)
        return
    with np.load(cfg["weather"]) as wz:
        assert np.array_equal(pd.DatetimeIndex(wz["dates"]).normalize(), dates.normalize())
        weather = wz["weather_daily"].astype("float32")
    if cfg["geometry"] == "radial":
        dx = (lon - 110) * np.cos(np.deg2rad(19.2)); dy = lat - 19.2; nrm = np.sqrt(dx * dx + dy * dy); nx, ny = dx / nrm, dy / nrm
    else:
        nx = np.full_like(lon, math.sin(math.radians(60))); ny = np.full_like(lat, math.cos(math.radians(60)))
    along = weather[:, :, 0] * (-ny[None, :]) + weather[:, :, 1] * nx[None, :]
    off = weather[:, :, 0] * nx[None, :] + weather[:, :, 1] * ny[None, :]

    def feats(ids, x):
        d7 = np.stack([x[ids - k] for k in range(6, -1, -1)], 1); t = np.arange(7) - 3
        state = np.stack([d7.mean(1), d7.std(1), np.sum(d7 * t[None, :, None], 1) / 28], -1)
        ix = ids[:, None] + np.arange(-13, 1)
        ww = np.stack([along[ix], off[ix], weather[ix, :, 2]], -1).transpose(0, 2, 1, 3)
        assert np.isfinite(ww).all()
        return state.astype("float32"), ww.astype("float32")

    truth_eval = {s: xf[ids_eval[s][:, None] + np.arange(1, H + 1)[None, :]] for s in ids_eval}
    preds = {s: {seed: {} for seed in SEEDS} for s in ids_eval}
    for seed in SEEDS:
        ids = np.concatenate([q["ids"] for q in oof[seed]]); base = np.concatenate([q["base"] for q in oof[seed]])
        truth = np.concatenate([q["truth"] for q in oof[seed]])
        st_, ww_ = zip(*[feats(q["ids"], q["ssta"]) for q in oof[seed]]); state = np.concatenate(st_); ww = np.concatenate(ww_)
        years = dates[ids + 1].year; trm = years <= 2014; vam = years >= 2015
        sm, ss = state[trm].mean((0, 1)), state[trm].std((0, 1)) + 1e-6
        wm, ws = ww[trm].mean((0, 1, 2)), ww[trm].std((0, 1, 2)) + 1e-6
        state = (state - sm) / ss; ww = (ww - wm) / ws
        nets = {}
        for kind, usew in (("W1", False), ("W2", True)):
            torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); net = ResidualNet(usew).to(device)
            opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4); gen = torch.Generator().manual_seed(seed)
            tri, vai = np.flatnonzero(trm), np.flatnonzero(vam); best = np.inf; bst = copy.deepcopy(net.state_dict()); bad = 0

            def pred_res(sel):
                net.eval(); parts = []
                with torch.no_grad():
                    for k in range(0, len(sel), 8):
                        q = sel[k:k + 8]
                        parts.append(net(torch.tensor(state[q], device=device), torch.tensor(ww[q], device=device)).cpu().numpy())
                return np.concatenate(parts)
            for ep in range(1, 51):
                net.train(); perm = tri[torch.randperm(len(tri), generator=gen).numpy()]
                for k in range(0, len(perm), 8):
                    q = perm[k:k + 8]; opt.zero_grad(set_to_none=True)
                    loss = F.mse_loss(net(torch.tensor(state[q], device=device), torch.tensor(ww[q], device=device)),
                                      torch.tensor(truth[q] - base[q], device=device)); loss.backward(); opt.step()
                mse = float(np.mean((base[vai] + pred_res(vai) - truth[vai]) ** 2))
                if mse < best:
                    best, bst, bad = mse, copy.deepcopy(net.state_dict()), 0
                else:
                    bad += 1
                if bad >= 8:
                    break
            net.load_state_dict(bst); nets[kind] = net
            torch.save({"state_dict": bst, "seed": seed, "weather": usew}, out / f"{kind}_seed{seed}.pt")
        (out / f"PRE_EVAL_FREEZE_seed{seed}.json").write_text(json.dumps(
            {"seed": seed, "residual_train": "OOF 2006-2014", "residual_val": "OOF 2015-2017", "Dev_or_2026_used": False}))
        for s in ids_eval:
            reef = np.load(cfg["reef"](seed, s)).astype("float64")
            ds, dw = feats(ids_eval[s], xf); ds = (ds - sm) / ss; dw = (dw - wm) / ws
            preds[s][seed]["W0 ReefFormer"] = reef.astype("float32")
            for kind, net in nets.items():
                net.eval(); parts = []
                with torch.no_grad():
                    for k in range(0, len(ds), 8):
                        parts.append(net(torch.tensor(ds[k:k + 8], device=device), torch.tensor(dw[k:k + 8], device=device)).cpu().numpy())
                p = reef + np.concatenate(parts)
                preds[s][seed][{"W1": "W1 State", "W2": "W2 State+Weather"}[kind]] = p.astype("float32")
                np.save(out / f"{kind}_seed{seed}_{s}.npy", p.astype("float32"))

    rows = []
    for s in ids_eval:
        for m in ("W0 ReefFormer", "W1 State", "W2 State+Weather"):
            e = [preds[s][sd][m].astype("float64") - truth_eval[s] for sd in SEEDS]
            r = {"domain": name, "split": s, "Model": m, "RMSE": float(np.mean([math.sqrt(np.mean(v ** 2)) for v in e])),
                 "RMSE_SD": float(np.std([math.sqrt(np.mean(v ** 2)) for v in e], ddof=1)),
                 "MAE": float(np.mean([np.mean(np.abs(v)) for v in e])), "Bias": float(np.mean([v.mean() for v in e]))}
            for b, sl in (("D1_7", slice(0, 7)), ("D8_14", slice(7, 14)), ("D15_30", slice(14, 30))):
                r[b] = float(np.mean([math.sqrt(np.mean(v[:, sl] ** 2)) for v in e]))
            if cfg["masks"] is not None:
                with np.load(str(cfg["masks"]).format(s)) as mk:
                    for key, lab in (("high_ssta_mask", "High"), ("mhw_like_mask", "Spell")):
                        r[f"{lab}_RMSE"] = float(np.mean([math.sqrt(np.mean(v[mk[key]] ** 2)) for v in e]))
            rows.append(r)
    tab = pd.DataFrame(rows); tab.to_csv(out / "MAIN_TABLE.csv", index=False)
    contrasts = [("W2 State+Weather", "W0 ReefFormer", "W2 - W0"), ("W1 State", "W0 ReefFormer", "W1 - W0"),
                 ("W2 State+Weather", "W1 State", "W2 - W1")]
    boot = []
    for index, s in enumerate(("Dev", "Test2026")):
        boot += paired_bootstrap(preds, truth_eval[s], s, block_counts(len(ids_eval[s]), 154420 + index), contrasts)
    boot = pd.DataFrame(boot); boot.to_csv(out / "PAIRED_60D_BOOTSTRAP.csv", index=False)
    pd.set_option("display.width", 250)
    print(tab.round(4).to_string(index=False))
    b = boot[boot.Metric == "RMSE"]
    print(b[["split", "Contrast", "Band", "Delta_A_minus_B", "CI95_low", "CI95_high"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1])
