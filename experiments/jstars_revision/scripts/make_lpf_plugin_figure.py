"""LPF as a plug-in: RMSE of each backbone before and after Lead-dependent Persistence Fusion.

Hainan: DLinear, PatchTST, iT90, iT365 (E1 of run_jstars_review_experiments.py) and Dual Context
(archived CTX -> ReefFormer). GBR: iT90, iT365 and Dual Context -> ReefFormer (archived three-seed run).
LPF is fitted per seed on the 2018-2021 validation period only (paper's fit_smooth). Intervals are the
paper's paired 60-day moving-block bootstrap (seeds 154420 Dev / 154421 Test2026, 10,000 replicates).
Writes LPF_PLUGIN.csv next to the figure and fig_lpf_plugin.pdf/.png in the manuscript folder.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE.parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from analyze_final154_core import fit_smooth  # noqa: E402
from build_final154_complete_results import block_counts, paired_bootstrap  # noqa: E402
from make_results_figures_jstars import STYLE, WH_BLUE, WH_PINK, panel  # noqa: E402

plt.rcParams.update(STYLE)
SEEDS, H, SPLITS = (42, 43, 44), 30, ("Val", "Dev", "Test2026")
leads = np.arange(1, H + 1)
CSV = HERE / "LPF_PLUGIN.csv"


def domain_inputs(path, dates_key, ssta_key):
    with np.load(path) as z:
        ssta = z[ssta_key].astype("float32"); rho = z["dap_rho_train"].astype("float32")
        ids = {s: z[s].astype(int) for s in SPLITS}
    truth = {s: ssta[ids[s][:, None] + leads[None, :]] for s in SPLITS}
    dap = {s: ssta[ids[s], None, :] * rho[None, None, :] ** leads[None, :, None] for s in SPLITS}
    return truth, dap


def blend(w, a, b):
    return w[None, :, None] * a + (1 - w[None, :, None]) * b


def evaluate(domain, truth, dap, loaders, archived_pairs):
    """loaders: name -> f(seed, split) raw backbone; archived_pairs: name -> (f_before, f_after)."""
    pred = {s: {seed: {} for seed in SEEDS} for s in SPLITS}
    for seed in SEEDS:
        for name, f in loaders.items():
            raw = {s: f(seed, s).astype("float32") for s in SPLITS}
            _, _, w = fit_smooth(dap["Val"], raw["Val"], truth["Val"])
            for s in SPLITS:
                pred[s][seed][name] = raw[s]; pred[s][seed][name + "+LPF"] = blend(w, dap[s], raw[s]).astype("float32")
        for name, (fb, fa) in archived_pairs.items():
            for s in SPLITS:
                pred[s][seed][name] = fb(seed, s).astype("float32"); pred[s][seed][name + "+LPF"] = fa(seed, s).astype("float32")
    names = list(loaders) + list(archived_pairs)
    rows = []
    for k, s in enumerate(("Dev", "Test2026")):
        t = truth[s].astype("float64")
        boot = paired_bootstrap(pred, t, s, block_counts(len(t), 154420 + k), [(n + "+LPF", n, n) for n in names])
        boot = {b["Contrast"]: b for b in boot if b["Metric"] == "RMSE" and b["Band"] == "Overall"}
        for n in names:
            r = lambda m: float(np.mean([np.sqrt(np.mean((pred[s][sd][m].astype("float64") - t) ** 2)) for sd in SEEDS]))
            b = boot[n]
            rows.append({"domain": domain, "split": s, "backbone": n, "RMSE_before": r(n), "RMSE_after": r(n + "+LPF"),
                         "delta": b["Delta_A_minus_B"], "lo": b["CI95_low"], "hi": b["CI95_high"]})
    return rows


def compute():
    rows = []
    run = ROOT / "runs/final_154_v1"
    truth, dap = domain_inputs(ROOT / "data/frozen_154_ssta_v1/dataset_daily.npz", "dates_all", "ssta_all_c")
    look = {"DLinear": 180, "PatchTST": 365}

    def sb(model):
        def f(seed, s):
            name = f"{model}_L{look[model]}_seed{seed}_Val.npy" if s == "Val" else f"{model}_seed{seed}_{s}.npy"
            return np.load(run / "strong_baselines" / model.lower() / name)
        return f
    it = lambda L: (lambda seed, s: np.load(run / f"itransformer/FULL/iT_L{L}_seed{seed}_{s}.npy"))
    ev = lambda m: (lambda seed, s: np.load(run / f"evidence/{m}_seed{seed}_{s}.npy"))
    rows += evaluate("Hainan", truth, dap, {"DLinear": sb("DLinear"), "PatchTST": sb("PatchTST"),
                                           "iT90": it(90), "iT365": it(365)},
                     {"Dual Context": (ev("CTX"), ev("ReefFormer"))})
    g = ROOT / "runs/gbr_generalization_v1"
    truth, dap = domain_inputs(ROOT / "data/gbr_generalization_v1/dataset_daily.npz", "dates", "ssta_c")
    git = lambda L: (lambda seed, s: np.load(g / f"models/iT_L{L}_seed{seed}_{s}.npy"))
    gr = lambda m: (lambda seed, s: np.load(g / f"results_three_seed/{m}_seed{seed}_{s}.npy"))

    def gsb(model):                       # GBR strong baselines: both select L=90 on validation (SELECTION.json)
        def f(seed, s):
            name = f"{model}_L90_seed{seed}_Val.npy" if s == "Val" else f"{model}_seed{seed}_{s}.npy"
            return np.load(g / "strong_baselines" / model.lower() / name)
        return f
    rows += evaluate("GBR", truth, dap, {"DLinear": gsb("DLinear"), "PatchTST": gsb("PatchTST"),
                                        "iT90": git(90), "iT365": git(365)},
                     {"Dual Context": (gr("Dual_Context"), gr("ReefFormer"))})
    df = pd.DataFrame(rows); df.to_csv(CSV, index=False)
    return df


def draw(df):
    order = {"Hainan": ["DLinear", "PatchTST", "iT90", "iT365", "Dual Context"],
             "GBR": ["DLinear", "PatchTST", "iT90", "iT365", "Dual Context"]}
    label = {}
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.6), gridspec_kw={"height_ratios": [5, 5], "hspace": 0.55,
                                                                     "wspace": 0.30})
    titles = {("Hainan", "Dev"): "Hainan, 2022–2025", ("Hainan", "Test2026"): "Hainan, 2026 holdout",
              ("GBR", "Dev"): "GBR, 2022–2025", ("GBR", "Test2026"): "GBR, 2026 holdout"}
    letters = iter("abcd")
    for i, dom in enumerate(("Hainan", "GBR")):
        for j, s in enumerate(("Dev", "Test2026")):
            ax = axes[i, j]; d = df[(df.domain == dom) & (df.split == s)].set_index("backbone").loc[order[dom]]
            y = np.arange(len(d))[::-1]
            col = WH_BLUE if dom == "Hainan" else WH_PINK
            for yy, (_, r) in zip(y, d.iterrows()):
                ax.annotate("", xy=(r.RMSE_after, yy), xytext=(r.RMSE_before, yy),
                            arrowprops=dict(arrowstyle="-", color="#9A9A9A", lw=1.6, shrinkA=3.5, shrinkB=3.5))
                ax.plot(r.RMSE_before, yy, "o", ms=5.5, mfc="white", mec="#555555", mew=1.0, zorder=3)
                ax.plot(r.RMSE_after, yy, "o", ms=5.5, color=col, zorder=4)
                sig = "*" if (r.hi < 0 or r.lo > 0) else ""
                ax.text(1.0, yy, f"{r.delta:+.3f}{sig}".replace("-", "−"), transform=ax.get_yaxis_transform(), ha="left",
                        va="center", fontsize=7.8)
            ax.set_yticks(y); ax.set_yticklabels([label.get(n, n) for n in d.index] if j == 0 else [])
            lo = min(d.RMSE_before.min(), d.RMSE_after.min()); hi = max(d.RMSE_before.max(), d.RMSE_after.max())
            pad = 0.08 * (hi - lo); ax.set_xlim(lo - pad, hi + pad); ax.set_ylim(-0.6, len(d) - 0.4)
            ax.set_xlabel("RMSE (°C)"); panel(ax, next(letters), titles[(dom, s)])
    h = [plt.Line2D([], [], ls="", marker="o", ms=5.5, mfc="white", mec="#555555", label="Backbone"),
         plt.Line2D([], [], ls="", marker="o", ms=5.5, color=WH_BLUE, label="+ LPF (Hainan)"),
         plt.Line2D([], [], ls="", marker="o", ms=5.5, color=WH_PINK, label="+ LPF (GBR)")]
    fig.legend(handles=h, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.045))
    fig.savefig(OUT / "fig_lpf_plugin.pdf", bbox_inches="tight", pad_inches=0.03)
    fig.savefig(OUT / "fig_lpf_plugin.png", dpi=300, bbox_inches="tight", pad_inches=0.03)


if __name__ == "__main__":
    df = pd.read_csv(CSV) if CSV.exists() and "--recompute" not in sys.argv else compute()
    print(df.round(4).to_string(index=False))
    draw(df)
