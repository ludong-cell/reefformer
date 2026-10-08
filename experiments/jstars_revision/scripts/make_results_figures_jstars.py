"""Redraw the Hainan result figures (Figs. 3-6) in the unified JSTARS style.

Style: Times New Roman, black text, no grid, panel labels "(a) Title" in bold at the
top-left of each panel, aligned with the axes' left edge. Every plotted number is read
from the same archived CSVs used by the earlier build scripts; nothing is recomputed
beyond seed means.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
OUT = HERE.parent                                   # JSTARS_submission_v1
ROOT = HERE.parents[2]                              # hainan_sst_forecast
RUN = ROOT / "runs/final_154_v1"
EV, CR = RUN / "evidence", RUN / "complete_results"

STYLE = {
    "font.family": "Times New Roman", "mathtext.fontset": "stix", "font.size": 8.5,
    "axes.labelsize": 8.5, "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "text.color": "black", "axes.labelcolor": "black", "xtick.color": "black", "ytick.color": "black",
    "axes.edgecolor": "black", "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.direction": "out", "ytick.direction": "out",
    "legend.frameon": False, "pdf.fonttype": 42, "svg.fonttype": "none", "savefig.facecolor": "white",
}
plt.rcParams.update(STYLE)

BLACK = "#111111"
BLUE, GREEN = "#2F6DB5", "#4E9A3F"                  # context / predictor, as in Figs. 2 and 7
# Fig. 3 palette (WenHai-style): steel blue, coral pink, grey dashed for persistence.
WH_BLUE, WH_PINK, WH_GREY = "#3D6DB5", "#EF6F7B", "#8C8C8C"
C = {"DAP": "#8c6b3f", "iTransformer": "#3167a3", "iT90": "#7FB2DE", "iT180": "#8b8799",
     "iT365": "#1F4E8C", "Dual Context": "#2b9a9a", "DAP Smooth": "#9060a6", "ReefFormer": "#c65a2e",
     "DLinear": "#626b72", "PatchTST": "#769c3e"}


def panel(ax, letter, title):
    """Unified panel label: bold '(a) Title' at the top-left of the axes."""
    ax.set_title(f"({letter}) {title}", loc="left", fontsize=9.5, fontweight="bold", pad=7)


def better_arrow(ax, x, y0, y1, label, rotate=True, side="right"):
    ax.annotate("", xy=(x, y1), xytext=(x, y0),
                arrowprops=dict(arrowstyle="-|>", color=BLACK, lw=1.1, mutation_scale=9))
    if rotate:
        ax.text(x + 0.8, (y0 + y1) / 2, label, rotation=90, ha="left", va="center", fontsize=8.5)
    else:
        ax.text(x - 0.8, (y0 + y1) / 2, label, ha="right", va="center", fontsize=8.5)


def contrast_panel(ax, lead, val, lo, hi, color, ylim, up, down, up_rotate=True):
    x = lead.to_numpy(float)
    ax.fill_between(x, lo, hi, color=color, alpha=0.18, lw=0)
    ax.plot(x, val, color=color, lw=1.8)
    ax.axhline(0, color=BLACK, lw=1.2)
    v = np.asarray(val)
    k = next(i for i in range(1, len(v)) if v[i - 1] > 0 >= v[i])          # first sign change
    cross = x[k - 1] + 0.5
    ax.axvline(cross, color=BLACK, lw=0.8, ls=(0, (3, 2)))
    ax.text(cross + 0.4, ylim[1], f"day {int(x[k - 1])}/{int(x[k])}", ha="left", va="top", fontsize=8.5)
    ax.set(xlim=(1, 30), ylim=ylim, xlabel="Lead time (days)")
    ax.set_xticks([1, 7, 14, 21, 30])
    better_arrow(ax, up[0], ylim[1] * 0.12, ylim[1] * 0.85, up[1], rotate=up_rotate)
    better_arrow(ax, down[0], ylim[0] * 0.12, ylim[0] * 0.85, down[1])


def weight_panel(ax, curves, color, ylabel):
    h = np.arange(1, 31)
    for c in curves:
        ax.plot(h, c, color=color, lw=0.9, alpha=0.55)
    ax.plot(h, np.mean(curves, axis=0), color=color, lw=2.0)
    ax.set(xlim=(1, 30), ylim=(0, 1), xlabel="Lead time (days)", ylabel=ylabel)
    ax.set_xticks([1, 7, 14, 21, 30]); ax.set_yticks([0, 0.5, 1])
    ax.legend([Line2D([], [], color=color, lw=0.9, alpha=0.55), Line2D([], [], color=color, lw=2.0)],
              ["Seeds 42–44", "Seed mean"], loc="upper right")


def save(fig, stem):
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.03)
    fig.savefig(OUT / f"{stem}.png", dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def fig3_lead_evidence():
    core = pd.read_csv(EV / "CORE_METRICS_SUMMARY.csv")
    comp = pd.read_csv(EV / "LEAD_COMPLEMENTARITY_2018_2025.csv")
    w = pd.read_csv(EV / "FROZEN_WEIGHTS.csv")
    lead = pd.read_csv(CR / "LEADWISE.csv")
    h = np.arange(1, 31)

    fig, axs = plt.subplots(2, 3, figsize=(7.16, 4.7))
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.09, top=0.93, wspace=0.42, hspace=0.62)

    ax = axs[0, 0]
    for m, lab, col, ls in (("iT90", "90 d", WH_PINK, "-"), ("iT180", "180 d", WH_GREY, "--"),
                            ("iT365", "365 d", WH_BLUE, "-")):
        r = core[(core.split == "Val") & (core.Model == m)].iloc[0]
        ax.plot([1, 2, 3], [r.D1_7, r.D8_14, r.D15_30], marker="o", ms=4, lw=1.8, ls=ls, color=col, label=lab)
    ax.set_xticks([1, 2, 3], ["D1–7", "D8–14", "D15–30"]); ax.set_xlim(0.7, 3.3)
    ax.set_ylabel(r"Validation RMSE ($^\circ$C)"); ax.legend(loc="upper left")
    panel(ax, "a", "Context-length screening")

    ax = axs[0, 1]
    contrast_panel(ax, comp.lead, comp.Delta_ctx_MSE, comp.Delta_ctx_CI_low, comp.Delta_ctx_CI_high,
                   WH_BLUE, (-0.17, 0.2), (26.5, "iT90 better"), (2.5, "iT365 better"))
    ax.set_ylabel(r"$\Delta$MSE ($^\circ$C$^2$)")
    panel(ax, "b", "Context: iT365 vs. iT90")

    ax = axs[0, 2]
    weight_panel(ax, [r.a_ctx + (1 - r.a_ctx) * np.exp(-h / r.tau_ctx_d) for r in w.itertuples()],
                 WH_BLUE, "Short-context weight $q(h)$")
    panel(ax, "c", "Fitted LCF weight")

    ax = axs[1, 0]
    dev = lead[lead.split == "Dev"]
    for m, col, ls in (("Dual Context", WH_BLUE, "-"), ("DAP", WH_GREY, "--")):
        g = dev[dev.Model == m].groupby("Lead").RMSE.mean()
        ax.plot(g.index, g.values, color=col, lw=1.8, ls=ls, label=m)
    ax.set(xlim=(1, 30), xlabel="Lead time (days)", ylabel=r"Development RMSE ($^\circ$C)")
    ax.set_xticks([1, 7, 14, 21, 30]); ax.legend(loc="lower right")
    panel(ax, "d", "Predictor skill by lead")

    ax = axs[1, 1]
    contrast_panel(ax, comp.lead, comp.Delta_pred_MSE, comp.Delta_pred_CI_low, comp.Delta_pred_CI_high,
                   WH_PINK, (-0.42, 0.14), (28.5, "DAP better"), (2.5, "Dual Context better"), up_rotate=False)
    ax.set_ylabel(r"$\Delta$MSE ($^\circ$C$^2$)")
    panel(ax, "e", "Predictor: Dual Context vs. DAP")

    ax = axs[1, 2]
    weight_panel(ax, [r.a_out + (1 - r.a_out) * np.exp(-h / r.tau_out_d) for r in w.itertuples()],
                 WH_PINK, "DAP weight $w(h)$")
    panel(ax, "f", "Fitted LPF weight")
    save(fig, "fig3_lead_evidence")


def fig5_heterogeneity():
    fig, axs = plt.subplots(2, 1, figsize=(3.5, 4.6))          # single IEEE column
    fig.subplots_adjust(left=0.17, right=0.97, bottom=0.07, top=0.86, hspace=0.42)
    for ax, key, file, order, letter, title in (
            (axs[0], "Year", "YEARWISE.csv", [2022, 2023, 2024, 2025], "a", "Year-wise RMSE"),
            (axs[1], "Region", "REGIONWISE.csv", ["West", "North", "East", "South"], "b", "Coastal-sector RMSE")):
        d = pd.read_csv(CR / file); d = d[d.split == "Dev"]
        for m in ("DAP", "DLinear", "PatchTST", "iTransformer", "ReefFormer"):
            g = d[d.Model == m].groupby(key).RMSE.mean().reindex(order)
            ax.plot(range(4), g.values, marker="o", ms=3.5, lw=1.5 if m != "ReefFormer" else 2.0,
                    color=C[m], label=m)
        ax.set_xticks(range(4), [str(o) for o in order]); ax.set_ylabel(r"Development RMSE ($^\circ$C)")
        panel(ax, letter, title)
    hd, lb = axs[0].get_legend_handles_labels()
    fig.legend(hd, lb, loc="upper center", bbox_to_anchor=(0.55, 1.0), ncol=3, handlelength=2.0,
               columnspacing=1.2)
    save(fig, "fig5_heterogeneity")


def fig6_limit_holdout():
    ext = pd.read_csv(CR / "EXTREME_SUMMARY.csv"); ext = ext[ext.split == "Dev"]
    sb = pd.read_csv(CR / "STRONG_BASELINE_PAIRED_60D_BOOTSTRAP.csv")
    sb = sb[(sb.split == "Test2026") & (sb.Band == "Overall") & (sb.Metric == "RMSE")]
    fig, axs = plt.subplots(2, 1, figsize=(3.5, 4.9))          # single IEEE column
    fig.subplots_adjust(left=0.25, right=0.97, bottom=0.08, top=0.84, hspace=0.5)

    ax = axs[0]
    models_a = ("PatchTST", "iTransformer", "Dual Context", "DAP Smooth", "ReefFormer")
    for i, subset in enumerate(("High-SSTA", "MHW-like")):
        for j, m in enumerate(models_a):
            r = ext[(ext.subset == subset) & (ext.Model == m)].iloc[0]
            ax.errorbar(i + (j - 2) * 0.09, r.RMSE_mean, yerr=r.RMSE_SD, fmt="o", ms=4.5, capsize=2.5,
                        color=C[m], elinewidth=1.2)
    ax.set_xticks([0, 1], ["High anomaly", "Warm spell"]); ax.set_xlim(-0.5, 1.5)
    ax.set_ylabel(r"Conditional RMSE ($^\circ$C)")
    panel(ax, "a", "Development warm subsets")

    ax = axs[1]
    order_b = ("iTransformer", "PatchTST", "DAP Smooth", "DLinear", "Dual Context")
    for i, m in enumerate(order_b):
        r = sb[sb.Model_B == m].iloc[0]; v = r.Delta_A_minus_B
        ax.errorbar(v, 4 - i, xerr=[[v - r.CI95_low], [r.CI95_high - v]], fmt="o", ms=4.5, capsize=2.5,
                    color=C[m], elinewidth=1.4)
    ax.set_yticks(range(5), list(order_b)[::-1]); ax.tick_params(axis="y", length=0)
    ax.axvline(0, color=BLACK, ls=(0, (3, 2)), lw=0.9)
    ax.set_xlabel(r"$\Delta$RMSE, ReefFormer $-$ baseline ($^\circ$C)")
    panel(ax, "b", "Frozen 2026 holdout")

    names = ("PatchTST", "iTransformer", "DLinear", "Dual Context", "DAP Smooth", "ReefFormer")
    hd = [Line2D([], [], color=C[m], marker="o", ms=4.5, lw=1.2) for m in names]
    fig.legend(hd, names, loc="upper center", bbox_to_anchor=(0.55, 1.0), ncol=3, handlelength=1.6,
               columnspacing=1.0)
    save(fig, "fig6_limit_holdout")


def fig4_spatial_map():
    """Reuse the archived map routine, switching only fonts/colour of text and the output path."""
    sys.path.insert(0, str(ROOT / "04_manuscript/EJRS_submission"))
    import make_spatial_delta_map as smap
    plt.rcParams.update(STYLE)
    smap.OUTPUT_STEM = OUT / "fig4_spatial_delta"
    smap.draw_map()


if __name__ == "__main__":
    fig3_lead_evidence()
    fig5_heterogeneity()
    fig6_limit_holdout()
    fig4_spatial_map()
    print("ok")
