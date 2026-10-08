"""Supplementary Tables S13-S17 for the JSTARS revision, generated directly from archived result CSVs.

S13 OSTIA second-product replication      runs/ostia_hainan_v1
S14 buffered leave-one-sector-out          runs/hainan_loso_v1
S15 context-fusion forms after LPF         runs/final_154_v1/review_experiments
S16 LPF attached to learned backbones      figure_scripts/LPF_PLUGIN.csv (make_lpf_plugin_figure.py)
S17 ERA5 weather-residual extension        runs/weather_residual_{154,gbr}_v1
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "supp_tables"
RUNS = HERE.parents[2] / "runs"
f4 = lambda x: f"{x:.4f}".replace("-", "$-$")


def ci(lo, hi, bold=False):
    s = f"$[{lo:+.4f},{hi:+.4f}]$"
    return f"\\textbf{{{s}}}" if bold else s


def sig(lo, hi):
    return lo > 0 or hi < 0


def table(label, caption, colspec, header, rows, note):
    body = "\n".join(rows)
    return (f"\\begin{{table}}[H]\\centering\n\\caption{{{caption}}}\\label{{{label}}}\n"
            f"\\scriptsize\\setlength{{\\tabcolsep}}{{3pt}}\n"
            f"\\begin{{tabular*}}{{\\textwidth}}{{@{{\\extracolsep{{\\fill}}}}{colspec}@{{}}}}\\toprule\n{header}\\\\\\midrule\n"
            f"{body}\n\\bottomrule\n\\end{{tabular*}}\n\\par\\smallskip{{\\scriptsize {note}}}\n\\end{{table}}\n")


MODELS = ["iT90", "iT365", "DAP", "Dual Context", "ReefFormer"]


def s13():
    d = RUNS / "ostia_hainan_v1"; m = pd.read_csv(d / "SUMMARY_MEAN_SD.csv"); b = pd.read_csv(d / "PAIRED_60D_BOOTSTRAP_DEV.csv")
    b = b[b.Band == "Overall"].set_index("Contrast")
    rows = []
    for mod in MODELS:
        dv = m[(m.Split == "Dev") & (m.Model == mod)].iloc[0]; ts = m[(m.Split == "Test2026") & (m.Model == mod)].iloc[0]
        if mod == "ReefFormer":
            c = "--"
        else:
            r = b.loc[f"ReefFormer - {mod}"]; c = f"{f4(r.Delta_RMSE)} {ci(r.CI_low, r.CI_high, sig(r.CI_low, r.CI_high))}"
        rows.append(f"{mod} & {f4(dv.RMSE_mean)} & {f4(dv.MAE_mean)} & {f4(dv.D1_7_mean)} & {f4(dv.D8_14_mean)} & "
                    f"{f4(dv.D15_30_mean)} & {c} & {f4(ts.RMSE_mean)} \\\\")
    fo = pd.read_csv(d / "FUSION_PARAMETERS_BY_SEED.csv"); fc = pd.read_csv(RUNS / "final_154_v1/evidence/FROZEN_WEIGHTS.csv")
    lpf = (f"Seed-mean LPF coefficients: $a_{{\\rm out}}={fo.a_out.mean():.2f}$ and $\\tau_{{\\rm out}}={fo.tau_out_days.mean():.1f}$~d on OSTIA, "
           f"against $a_{{\\rm out}}={fc.a_out.mean():.2f}$ and $\\tau_{{\\rm out}}={fc.tau_out_d.mean():.1f}$~d on CoralTemp. ")
    return table("tab:ostia", "Second-product replication on the 0.05$^{\\circ}$ reprocessed OSTIA analysis at the 154 Hainan locations (errors in $^{\\circ}$C).",
                 "lccccccc", "& \\multicolumn{6}{c}{Development 2022--2025} & 2026 \\\\\\cmidrule(lr){2-7}\\cmidrule(l){8-8}\n"
                 "Model & RMSE & MAE & D1--7 & D8--14 & D15--30 & ReefFormer $-$ model & RMSE", rows,
                 "Same splits, seeds, architecture, and validation-only fitting as the CoralTemp experiment; every fitted quantity is recomputed from OSTIA. "
                 "Learned models report means over seeds 42--44. Intervals are paired 60-day moving-block 95\\% intervals of $\\Delta$RMSE; bold intervals exclude zero. "
                 + lpf + "Only 61 complete 2026 forecast windows were available for OSTIA at the time of analysis, so the 2026 column is exploratory; on these windows DAP has the lowest RMSE.")


def s14():
    d = RUNS / "hainan_loso_v1"; sec = pd.read_csv(d / "SUMMARY_BY_SECTOR.csv"); mac = pd.read_csv(d / "SUMMARY_MACRO_FOLD_SEED.csv")
    b = pd.read_csv(d / "PAIRED_60D_BOOTSTRAP_DEV.csv"); b = b[b.Band == "Overall"].set_index("Contrast")
    rows = []
    for mod in MODELS:
        vals = [f4(sec[(sec.target_sector == s) & (sec.Split == "Dev") & (sec.Model == mod)].RMSE_mean.iloc[0])
                for s in ("West", "North", "East", "South")]
        mdev = mac[(mac.Split == "Dev") & (mac.Model == mod)].RMSE_mean.iloc[0]
        m26 = mac[(mac.Split == "Test2026") & (mac.Model == mod)].RMSE_mean.iloc[0]
        if mod == "ReefFormer":
            c = "--"
        else:
            r = b.loc[f"ReefFormer - {mod}"]; c = f"{f4(r.Delta_RMSE)} {ci(r.CI_low, r.CI_high, sig(r.CI_low, r.CI_high))}"
        rows.append(f"{mod} & {' & '.join(vals)} & {f4(mdev)} & {c} & {f4(m26)} \\\\")
    return table("tab:loso", "Buffered leave-one-sector-out evaluation around Hainan: RMSE ($^{\\circ}$C) on each withheld sector.",
                 "lccccccc", "& \\multicolumn{6}{c}{Development 2022--2025} & 2026 \\\\\\cmidrule(lr){2-7}\\cmidrule(l){8-8}\n"
                 "Model & West & North & East & South & Macro & ReefFormer $-$ model & Macro", rows,
                 "For each target sector (West 36, North 35, East 45, South 38 pixels), the sector and every source pixel within 25~km of it are excluded from neural training, "
                 "checkpoint selection, and fusion fitting; the shared iTransformer weights are applied to the unseen target-sector tokens. Target-pixel training-period climatology, "
                 "anomaly scaling, and DAP coefficients are retained as historical preprocessing. Macro values weight the four folds equally. Intervals are paired 60-day "
                 "moving-block 95\\% intervals with equal sector weight; bold intervals exclude zero. Seed means over 42--44.")


def s15():
    d = RUNS / "final_154_v1/review_experiments"; acc = pd.read_csv(d / "E1E2E3_ACCURACY_SKILL.csv")
    b = pd.read_csv(d / "E1E2_PAIRED_60D_BOOTSTRAP.csv"); b = b[(b.Band == "Overall") & (b.Metric == "RMSE")]
    rows = []
    for mod, lab in (("Avg CTX+LPF", "Equal-weight average + LPF"), ("Fixed CTX+LPF", "Validation-fitted constant + LPF"),
                     ("ReefFormer", "Lead-dependent (LCF) + LPF (ReefFormer)")):
        cells = []
        for s in ("Dev", "Test2026"):
            cells.append(f4(acc[(acc.split == s) & (acc.Model == mod)].RMSE.iloc[0]))
            if mod == "ReefFormer":
                cells.append("--")
            else:
                r = b[(b.split == s) & (b.Contrast == f"ReefFormer - {mod}")].iloc[0]
                cells.append(f"{f4(r.Delta_A_minus_B)} {ci(r.CI95_low, r.CI95_high, sig(r.CI95_low, r.CI95_high))}")
        rows.append(f"{lab} & {' & '.join(cells)} \\\\")
    return table("tab:ctxlpf", "Context-fusion forms after LPF around Hainan (RMSE in $^{\\circ}$C).",
                 "lcccc", "& \\multicolumn{2}{c}{Development 2022--2025} & \\multicolumn{2}{c}{2026 holdout} \\\\\\cmidrule(lr){2-3}\\cmidrule(l){4-5}\n"
                 "Context combination & RMSE & ReefFormer $-$ form & RMSE & ReefFormer $-$ form", rows,
                 "All forms combine the same iT90 and iT365 forecasts; each context weight and each LPF layer is fitted per seed on 2018--2021 validation forecasts only. "
                 "Seed means over 42--44. Intervals are paired 60-day moving-block 95\\% intervals; none excludes zero. Without LPF, the lead-dependent weight is more accurate "
                 "than the constant weight (Table~\\ref{tab:ctxcontrols}).")


def s16():
    p = pd.read_csv(HERE / "LPF_PLUGIN.csv")
    order = {"Hainan": ["DLinear", "PatchTST", "iT90", "iT365", "Dual Context"],
             "GBR": ["DLinear", "PatchTST", "iT90", "iT365", "Dual Context"]}
    rows = []
    for dom in ("Hainan", "GBR"):
        rows.append(f"\\multicolumn{{7}}{{@{{}}l}}{{\\textit{{{dom if dom == 'Hainan' else 'Great Barrier Reef'}}}}}\\\\")
        for bb in order[dom]:
            cells = []
            for s in ("Dev", "Test2026"):
                r = p[(p.domain == dom) & (p.split == s) & (p.backbone == bb)].iloc[0]
                cells += [f4(r.RMSE_before), f4(r.RMSE_after), f"{f4(r.delta)} {ci(r.lo, r.hi, sig(r.lo, r.hi))}"]
            rows.append(f"{bb} & {' & '.join(cells)} \\\\")
    return table("tab:plugin", "LPF attached to learned backbones: RMSE ($^{\\circ}$C) before and after LPF.",
                 "lcccccc", "& \\multicolumn{3}{c}{Development 2022--2025} & \\multicolumn{3}{c}{2026 holdout} \\\\\\cmidrule(lr){2-4}\\cmidrule(l){5-7}\n"
                 "Backbone & Backbone & +LPF & $\\Delta$RMSE [95\\% interval] & Backbone & +LPF & $\\Delta$RMSE [95\\% interval]", rows,
                 "LPF is refitted per backbone and per seed on 2018--2021 validation forecasts only; $\\Delta$RMSE is backbone+LPF minus backbone. Dual Context+LPF is ReefFormer. "
                 "Seed means over 42--44; intervals are paired 60-day moving-block 95\\% intervals (seeds 154420 for development and 154421 for 2026); bold intervals exclude zero. "
                 "The Hainan Dual Context development interval differs in the fourth decimal from $M_{11}-M_{10}$ in Table~II of the main text, which uses the factorial resample set.")


def s17():
    rows = []
    for dom, run in (("Hainan", "weather_residual_154_v1"), ("Great Barrier Reef", "weather_residual_gbr_v1")):
        m = pd.read_csv(RUNS / run / "MAIN_TABLE.csv"); b = pd.read_csv(RUNS / run / "PAIRED_60D_BOOTSTRAP.csv")
        b = b[(b.Band == "Overall") & (b.Metric == "RMSE")]
        rows.append(f"\\multicolumn{{5}}{{@{{}}l}}{{\\textit{{{dom}}}}}\\\\")
        for key, lab in (("W0", "W0: ReefFormer"), ("W1", "W1: + state residual"), ("W2", "W2: + state and ERA5 forcing residual")):
            cells = []
            for s in ("Dev", "Test2026"):
                cells.append(f4(m[(m.split == s) & (m.Model.str.startswith(key))].RMSE.iloc[0]))
                if key == "W0":
                    cells.append("--")
                else:
                    r = b[(b.split == s) & (b.Contrast == f"{key} - W0")].iloc[0]
                    cells.append(f"{f4(r.Delta_A_minus_B)} {ci(r.CI95_low, r.CI95_high, sig(r.CI95_low, r.CI95_high))}")
            rows.append(f"{lab} & {' & '.join(cells)} \\\\")
    return table("tab:weather", "Exploratory ERA5 forcing extension: residual correction of ReefFormer (RMSE in $^{\\circ}$C).",
                 "lcccc", "& \\multicolumn{2}{c}{Development 2022--2025} & \\multicolumn{2}{c}{2026 holdout} \\\\\\cmidrule(lr){2-3}\\cmidrule(l){4-5}\n"
                 "Configuration & RMSE & $-$ W0 & RMSE & $-$ W0", rows,
                 "W1 adds a residual network on the 7-day mean, standard deviation, and slope of the SST anomaly and the lead; W2 adds 14-day daily ERA5 along-shore wind, "
                 "cross-shore wind, and downward shortwave radiation at the nearest 0.25$^{\\circ}$ cell through a causal one-dimensional convolution. Base forecasts for training "
                 "are out-of-fold ReefFormer forecasts from four chronological folds within 2000--2017; the residual networks are trained on 2006--2014 and early-stopped on "
                 "2015--2017, so no 2018--2026 observation is used. Coast geometry was fixed in advance: offshore direction radial from 110$^{\\circ}$E, 19.2$^{\\circ}$N for "
                 "Hainan and a uniform 60$^{\\circ}$ bearing for the GBR. The protocol was specified before evaluation, but the 2026 holdout had already been used for "
                 "ReefFormer. Seed means over 42--44; paired 60-day moving-block 95\\% intervals; bold intervals exclude zero.")


EVENTS = RUNS / "event_verification_v1"
EV_ORDER = {"hainan": ["Persistence", "DAP", "DLinear", "PatchTST", "iTransformer", "Dual Context", "DAP Smooth", "ReefFormer"],
            "gbr": ["Persistence", "DAP", "DLinear", "PatchTST", "iT365", "iT90", "Dual Context", "ReefFormer"]}
EV_LABEL = {"iT365": "iTransformer (iT365)"}


def s18():
    rows = []
    for dom, lab in (("hainan", "Hainan"), ("gbr", "Great Barrier Reef")):
        sc = pd.read_csv(EVENTS / dom / "EVENT_SCORES.csv"); br = pd.read_csv(EVENTS / dom / "EVENT_BASE_RATES.csv")
        rates = " & ".join(f"\\multicolumn{{2}}{{c}}{{{100 * br[(br.split == s) & (br.event == e)].base_rate.iloc[0]:.1f}\\%}}"
                           for s in ("Dev", "Test2026") for e in ("HotSpot>=1", "Upper decile"))
        rows.append(f"\\multicolumn{{9}}{{@{{}}l}}{{\\textit{{{lab}}}}}\\\\")
        rows.append(f"Base rate & {rates} \\\\")
        for m in EV_ORDER[dom]:
            cells = []
            for s in ("Dev", "Test2026"):
                for e in ("HotSpot>=1", "Upper decile"):
                    r = sc[(sc.split == s) & (sc.event == e) & (sc.Model == m)].iloc[0]
                    cells += [f"{r.PSS:.3f}", f"{r.Bias:.2f}"]
            rows.append(f"{EV_LABEL.get(m, m)} & {' & '.join(cells)} \\\\")
        rows.append("\\addlinespace")
    rows.pop()
    return table("tab:events", "Threshold-exceedance verification: Peirce skill score (PSS) and frequency bias (B).",
                 "lcccccccc", "& \\multicolumn{4}{c}{Development 2022--2025} & \\multicolumn{4}{c}{2026 holdout} \\\\"
                 "\\cmidrule(lr){2-5}\\cmidrule(l){6-9}\n& \\multicolumn{2}{c}{HotSpot} & \\multicolumn{2}{c}{Upper decile} & "
                 "\\multicolumn{2}{c}{HotSpot} & \\multicolumn{2}{c}{Upper decile} \\\\\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}"
                 "\\cmidrule(lr){6-7}\\cmidrule(l){8-9}\nModel & PSS & B & PSS & B & PSS & B & PSS & B", rows,
                 "HotSpot: SST $\\geq$ maximum monthly mean of the 2000--2017 climatology $+1^{\\circ}$C; upper decile: SSTA $\\geq$ per-pixel "
                 "2000--2017 90th percentile. Scores pool all origin--lead--pixel cells; base rate is the observed event frequency. "
                 "PSS $=$ POD $-$ probability of false detection; B $=$ forecast events / observed events. Seed means over 42--44 for learned models. "
                 "The 2026 HotSpot events fall within one warm season (106 forecast origins around Hainan, 63 on the GBR). "
                 "POD and FAR for every entry are archived with the code.")


def s19():
    rows = []
    for dom, lab, models in (("hainan", "Hainan", ["Persistence", "DAP", "DLinear", "PatchTST", "iTransformer", "Dual Context"]),
                             ("gbr", "Great Barrier Reef", ["Persistence", "DAP", "DLinear", "PatchTST", "iT365", "Dual Context"])):
        b = pd.read_csv(EVENTS / dom / "EVENT_PSS_BOOTSTRAP.csv")
        rows.append(f"\\multicolumn{{4}}{{@{{}}l}}{{\\textit{{{lab}}}}}\\\\")
        for m in models:
            cells = []
            for s in ("Dev", "Test2026"):
                r = b[(b.split == s) & (b.event == "Upper decile") & (b.Contrast == f"ReefFormer - {m}")].iloc[0]
                cells.append(f"{f4(r.Delta_PSS)} {ci(r.CI95_low, r.CI95_high, sig(r.CI95_low, r.CI95_high))}")
                if s == "Dev":
                    h = b[(b.split == s) & (b.event == "HotSpot>=1") & (b.Contrast == f"ReefFormer - {m}")].iloc[0]
                    cells.insert(0, f"{f4(h.Delta_PSS)} {ci(h.CI95_low, h.CI95_high, sig(h.CI95_low, h.CI95_high))}")
            rows.append(f"ReefFormer $-$ {EV_LABEL.get(m, m)} & {' & '.join(cells)} \\\\")
    return table("tab:eventboot", "Paired differences in Peirce skill score, ReefFormer minus each model.",
                 "lccc", "& \\multicolumn{2}{c}{Development 2022--2025} & 2026 holdout \\\\\\cmidrule(lr){2-3}\\cmidrule(l){4-4}\n"
                 "Contrast & HotSpot & Upper decile & Upper decile", rows,
                 "Positive values favor ReefFormer. Paired 60-day moving-block 95\\% intervals, recomputing the contingency table in each "
                 "replicate (seeds 154420 and 154421, 10,000 replicates); bold intervals exclude zero. 2026 HotSpot contrasts are omitted "
                 "because the events fall within one warm season, which leaves many replicates without events.")


if __name__ == "__main__":
    for name, fn in (("supp_ostia", s13), ("supp_loso", s14), ("supp_ctx_after_lpf", s15), ("supp_lpf_plugin", s16), ("supp_weather", s17),
                     ("supp_events", s18), ("supp_event_boot", s19)):
        (OUT / f"{name}.tex").write_text(fn(), encoding="utf-8"); print("wrote", name)
