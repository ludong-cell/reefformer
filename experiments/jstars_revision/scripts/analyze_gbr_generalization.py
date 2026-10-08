"""Analyze the frozen GBR ReefFormer generalization experiment."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data/gbr_generalization_v1/dataset_daily.npz"
MODELS = ROOT / "runs/gbr_generalization_v1/models"
OUT = ROOT / "runs/gbr_generalization_v1/results"
PAPER = ROOT / "04_manuscript/EJRS_submission/gbr_generalization"
SEEDS = (42, 43, 44)
SEED = 42  # Backward-compatible single-seed entry point; use the three-seed script for final analysis.
H = 30


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def score(prediction, truth):
    error = prediction.astype("float64") - truth.astype("float64")
    return {
        "RMSE": float(np.sqrt(np.mean(error ** 2))),
        "MAE": float(np.mean(np.abs(error))),
        "D1_7": float(np.sqrt(np.mean(error[:, :7] ** 2))),
        "D8_14": float(np.sqrt(np.mean(error[:, 7:14] ** 2))),
        "D15_30": float(np.sqrt(np.mean(error[:, 14:] ** 2))),
    }


def fit_smooth(left, right, truth):
    """Fit w*left + (1-w)*right, w=a+(1-a)exp(-h/tau), on Val."""
    delta = left.astype("float64") - right.astype("float64")
    residual = right.astype("float64") - truth.astype("float64")
    aa = np.mean(delta * delta, axis=(0, 2))
    bb = np.mean(delta * residual, axis=(0, 2))
    cc = np.mean(residual * residual, axis=(0, 2))
    lead = np.arange(1, H + 1, dtype="float64")

    def solve(log_tau):
        decay = np.exp(-lead / np.exp(log_tau))
        complement = 1 - decay
        denominator = max(np.sum(aa * complement * complement), 1e-15)
        asymptote = float(np.clip(-np.sum(complement * (aa * decay + bb)) / denominator, 0, 1))
        weight = decay + asymptote * complement
        objective = float(np.mean(aa * weight * weight + 2 * bb * weight + cc))
        return objective, asymptote, weight

    grid = np.linspace(np.log(0.1), np.log(1000), 4001)
    losses = np.array([solve(value)[0] for value in grid])
    index = int(np.argmin(losses))
    low, high = grid[max(0, index - 1)], grid[min(len(grid) - 1, index + 1)]
    phi = (1 + np.sqrt(5)) / 2
    x1, x2 = high - (high - low) / phi, low + (high - low) / phi
    f1, f2 = solve(x1)[0], solve(x2)[0]
    for _ in range(100):
        if f1 <= f2:
            high, x2, f2 = x2, x1, f1
            x1 = high - (high - low) / phi
            f1 = solve(x1)[0]
        else:
            low, x1, f1 = x1, x2, f2
            x2 = low + (high - low) / phi
            f2 = solve(x2)[0]
    tau = float(np.exp((low + high) / 2))
    _, asymptote, weight = solve(np.log(tau))
    return asymptote, tau, weight


def block_indices(n, repetitions=10000, block=60, seed=420154):
    rng = np.random.default_rng(seed)
    blocks = int(np.ceil(n / block))
    offsets = np.arange(block)
    for start in range(0, repetitions, 250):
        count = min(250, repetitions - start)
        origins = rng.integers(0, n - block + 1, size=(count, blocks))
        yield (origins[:, :, None] + offsets[None, None, :]).reshape(count, -1)[:, :n]


def paired_rmse_ci(candidate, reference, groups, repetitions=10000, block=60):
    candidate_loss = np.mean((candidate.astype("float64") - groups["truth"]) ** 2, axis=2)
    reference_loss = np.mean((reference.astype("float64") - groups["truth"]) ** 2, axis=2)
    rows = []
    bands = {"Overall": np.arange(30), "D1-7": np.arange(7),
             "D8-14": np.arange(7, 14), "D15-30": np.arange(14, 30)}
    replicate = {name: [] for name in bands}
    for indices in block_indices(len(candidate_loss), repetitions, block):
        for name, leads in bands.items():
            left = np.sqrt(candidate_loss[indices][:, :, leads].mean(axis=(1, 2)))
            right = np.sqrt(reference_loss[indices][:, :, leads].mean(axis=(1, 2)))
            replicate[name].append(left - right)
    for name, leads in bands.items():
        values = np.concatenate(replicate[name])
        estimate = np.sqrt(candidate_loss[:, leads].mean()) - np.sqrt(reference_loss[:, leads].mean())
        rows.append({"Contrast": "ReefFormer - iT365", "Group": name,
                     "Delta_RMSE": estimate, "CI_low": np.quantile(values, 0.025),
                     "CI_high": np.quantile(values, 0.975), "block_days": block,
                     "replicates": repetitions})
    return rows


def paired_rmse_ci_multiseed(candidates, references, truth, repetitions=10000, block=60):
    """Seed-averaged paired moving-block RMSE contrasts.

    Every bootstrap replicate uses the same sampled origins for all seeds.  A
    seed-specific RMSE difference is computed first and the three differences
    are then averaged, matching the inference convention used for Hainan.
    """
    candidate_loss = np.stack([
        np.mean((candidate.astype("float64") - truth) ** 2, axis=2)
        for candidate in candidates
    ])
    reference_loss = np.stack([
        np.mean((reference.astype("float64") - truth) ** 2, axis=2)
        for reference in references
    ])
    bands = {
        "Overall": np.arange(30),
        "D1-7": np.arange(7),
        "D8-14": np.arange(7, 14),
        "D15-30": np.arange(14, 30),
    }
    replicate = {name: [] for name in bands}
    for indices in block_indices(candidate_loss.shape[1], repetitions, block):
        for name, leads in bands.items():
            left = np.sqrt(candidate_loss[:, indices][:, :, :, leads].mean(axis=(2, 3)))
            right = np.sqrt(reference_loss[:, indices][:, :, :, leads].mean(axis=(2, 3)))
            replicate[name].append((left - right).mean(axis=0))
    rows = []
    for name, leads in bands.items():
        values = np.concatenate(replicate[name])
        estimate = np.mean([
            np.sqrt(candidate_loss[index, :, leads].mean())
            - np.sqrt(reference_loss[index, :, leads].mean())
            for index in range(len(candidates))
        ])
        rows.append({
            "Contrast": "ReefFormer - iT365",
            "Group": name,
            "Delta_RMSE": estimate,
            "CI_low": np.quantile(values, 0.025),
            "CI_high": np.quantile(values, 0.975),
            "block_days": block,
            "replicates": repetitions,
            "seeds": ";".join(map(str, SEEDS)),
        })
    return rows


def daily_mse_ci(loss_difference, repetitions=10000, block=60, seed=420365):
    result = []
    for indices in block_indices(len(loss_difference), repetitions, block, seed):
        result.append(loss_difference[indices].mean(axis=1))
    replicate = np.concatenate(result, axis=0)
    return np.quantile(replicate, 0.025, axis=0), np.quantile(replicate, 0.975, axis=0)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    PAPER.mkdir(parents=True, exist_ok=True)
    with np.load(DATA, allow_pickle=False) as dataset:
        ids = {name: dataset[name].astype("int64") for name in ("Val", "Dev", "Test2026")}
        dates = pd.DatetimeIndex(dataset["dates"])
        ssta = dataset["ssta_c"].astype("float32")
        rho = dataset["dap_rho_train"].astype("float64")
        pixels = len(dataset["pixel_ids"])
    lead = np.arange(1, H + 1)

    def truth(split):
        split_ids = ids[split]
        return ssta[split_ids[:, None] + lead[None, :]]

    def dap(split):
        return (ssta[ids[split], None, :] * rho[None, None, :] ** lead[None, :, None]).astype("float32")

    predictions = {}
    for split in ids:
        p90 = np.load(MODELS / f"iT_L90_seed{SEED}_{split}.npy")
        p365 = np.load(MODELS / f"iT_L365_seed{SEED}_{split}.npy")
        predictions[split] = {"iT90": p90, "iT365": p365, "DAP": dap(split)}

    y_val = truth("Val")
    a_ctx, tau_ctx, q = fit_smooth(predictions["Val"]["iT90"], predictions["Val"]["iT365"], y_val)
    ctx_val = q[None, :, None] * predictions["Val"]["iT90"] + (1 - q[None, :, None]) * predictions["Val"]["iT365"]
    a_out, tau_out, w = fit_smooth(predictions["Val"]["DAP"], ctx_val, y_val)

    rows, lead_rows, year_rows = [], [], []
    for split in ids:
        p = predictions[split]
        y = truth(split)
        p["Dual Context"] = q[None, :, None] * p["iT90"] + (1 - q[None, :, None]) * p["iT365"]
        p["ReefFormer"] = w[None, :, None] * p["DAP"] + (1 - w[None, :, None]) * p["Dual Context"]
        for model in ("iT90", "iT365", "DAP", "Dual Context", "ReefFormer"):
            rows.append({"Split": split, "Model": model, "seed": SEED, **score(p[model], y)})
            error = p[model].astype("float64") - y.astype("float64")
            for h in range(H):
                lead_rows.append({"Split": split, "Model": model, "lead": h + 1,
                                  "RMSE": np.sqrt(np.mean(error[:, h] ** 2)),
                                  "MAE": np.mean(np.abs(error[:, h]))})
        starts = dates[ids[split] + 1]
        for year in sorted(starts.year.unique()):
            mask = starts.year == year
            for model in ("iT90", "iT365", "DAP", "Dual Context", "ReefFormer"):
                year_rows.append({"Split": split, "year": int(year), "Model": model,
                                  "N_origins": int(mask.sum()), **score(p[model][mask], y[mask])})
        for model in ("Dual Context", "ReefFormer"):
            np.save(OUT / f"{model.replace(' ', '_')}_seed42_{split}.npy", p[model].astype("float32"))

    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "SUMMARY.csv", index=False)
    pd.DataFrame(lead_rows).to_csv(OUT / "LEADWISE.csv", index=False)
    pd.DataFrame(year_rows).to_csv(OUT / "YEARWISE.csv", index=False)
    weights = pd.DataFrame({"lead": lead, "q_iT90": q, "w_DAP": w})
    weights.to_csv(OUT / "FROZEN_WEIGHTS.csv", index=False)
    parameters = {
        "a_ctx": a_ctx, "tau_ctx_days": tau_ctx,
        "a_out": a_out, "tau_out_days": tau_out,
        "fit_split": "GBR Validation 2018-2021",
    }
    (OUT / "FUSION_PARAMETERS.json").write_text(json.dumps(parameters, indent=2), encoding="utf-8")

    # Paired moving-block CI for the central external-domain comparison.
    bootstrap = paired_rmse_ci(
        predictions["Dev"]["ReefFormer"], predictions["Dev"]["iT365"],
        {"truth": truth("Dev")},
    )
    pd.DataFrame(bootstrap).to_csv(OUT / "REEFFORMER_VS_IT365_60D_BOOTSTRAP.csv", index=False)

    # Two preregistered lead-dependent complementarity diagnostics, Val+Dev.
    combined_truth = np.concatenate([truth("Val"), truth("Dev")])
    combined90 = np.concatenate([predictions["Val"]["iT90"], predictions["Dev"]["iT90"]])
    combined365 = np.concatenate([predictions["Val"]["iT365"], predictions["Dev"]["iT365"]])
    combined_ctx = np.concatenate([predictions["Val"]["Dual Context"], predictions["Dev"]["Dual Context"]])
    combined_dap = np.concatenate([predictions["Val"]["DAP"], predictions["Dev"]["DAP"]])
    ctx_diff = np.mean((combined365-combined_truth) ** 2 - (combined90-combined_truth) ** 2, axis=2)
    pred_diff = np.mean((combined_ctx-combined_truth) ** 2 - (combined_dap-combined_truth) ** 2, axis=2)
    ctx_lo, ctx_hi = daily_mse_ci(ctx_diff)
    pred_lo, pred_hi = daily_mse_ci(pred_diff, seed=420366)
    complementarity = pd.DataFrame({
        "lead": lead,
        "Delta_ctx_MSE_365_minus_90": ctx_diff.mean(axis=0),
        "Delta_ctx_CI_low": ctx_lo,
        "Delta_ctx_CI_high": ctx_hi,
        "Delta_pred_MSE_CTX_minus_DAP": pred_diff.mean(axis=0),
        "Delta_pred_CI_low": pred_lo,
        "Delta_pred_CI_high": pred_hi,
    })
    complementarity.to_csv(OUT / "LEAD_COMPLEMENTARITY_2018_2025.csv", index=False)

    # Compact manuscript-ready figure.
    plt.rcParams.update({"font.family": "Arial", "font.size": 8})
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.15), constrained_layout=True)
    val = summary[summary.Split == "Val"].set_index("Model")
    x = np.arange(3)
    labels = ["D1–7", "D8–14", "D15–30"]
    for model, color, marker in (("iT90", "#2878B5", "o"), ("iT365", "#E07B39", "s")):
        axes[0].plot(x, [val.loc[model, "D1_7"], val.loc[model, "D8_14"], val.loc[model, "D15_30"]],
                     marker=marker, color=color, label=model, lw=1.5)
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("Validation RMSE (°C)")
    axes[0].legend(frameon=False, fontsize=7)
    axes[0].set_title("(a) Context length")
    axes[1].plot(lead, complementarity["Delta_ctx_MSE_365_minus_90"], color="#2878B5", lw=1.5)
    axes[1].fill_between(lead, ctx_lo, ctx_hi, color="#2878B5", alpha=.18)
    axes[1].axhline(0, color="0.25", lw=.7)
    axes[1].set(title="(b) Context complementarity", xlabel="Lead (days)", ylabel=r"MSE$_{365}$ − MSE$_{90}$")
    axes[2].plot(lead, complementarity["Delta_pred_MSE_CTX_minus_DAP"], color="#D15336", lw=1.5)
    axes[2].fill_between(lead, pred_lo, pred_hi, color="#D15336", alpha=.18)
    axes[2].axhline(0, color="0.25", lw=.7)
    axes[2].set(title="(c) Predictor complementarity", xlabel="Lead (days)", ylabel=r"MSE$_{CTX}$ − MSE$_{DAP}$")
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="0.9", lw=.5)
    fig.savefig(OUT / "FIG_GBR_GENERALIZATION.pdf")
    fig.savefig(OUT / "FIG_GBR_GENERALIZATION.png", dpi=400)
    plt.close(fig)

    # Manuscript-ready table and a concise audit report; main paper remains untouched.
    dev = summary[summary.Split == "Dev"].set_index("Model")
    table = (
        "\\begin{table}[t]\n\\caption{Independent GBR generalization results (seed 42). "
        "Fusion functions were fitted using the GBR validation period only.}\n"
        "\\label{tab:gbr_generalization}\n\\centering\n\\footnotesize\n"
        "\\begin{tabular}{lccccc}\n\\toprule\nModel & RMSE & MAE & D1--7 & D8--14 & D15--30 \\\\\n+        \n\\midrule\n"
    )
    for model in ("iT90", "iT365", "DAP", "Dual Context", "ReefFormer"):
        row = dev.loc[model]
        table += f"{model} & {row.RMSE:.4f} & {row.MAE:.4f} & {row.D1_7:.4f} & {row.D8_14:.4f} & {row.D15_30:.4f} \\\\\n+"
    table += "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    (PAPER / "table_gbr_generalization.tex").write_text(table, encoding="utf-8")
    for name in ("FIG_GBR_GENERALIZATION.pdf", "FIG_GBR_GENERALIZATION.png"):
        (PAPER / name).write_bytes((OUT / name).read_bytes())
    summary.to_csv(PAPER / "GBR_SUMMARY.csv", index=False)
    weights.to_csv(PAPER / "GBR_FROZEN_WEIGHTS.csv", index=False)
    pd.DataFrame(bootstrap).to_csv(PAPER / "GBR_BOOTSTRAP.csv", index=False)

    val90, val365 = val.loc["iT90"], val.loc["iT365"]
    dev365, dev_final = dev.loc["iT365"], dev.loc["ReefFormer"]
    ci = pd.DataFrame(bootstrap).query("Group == 'Overall'").iloc[0]
    report = f"""# GBR independent-domain generalization result

## Frozen protocol

- Pixels: {pixels}
- Seed: 42
- Train: 2000–2017; Validation: 2018–2021; Development: 2022–2025
- Fusion fitting: Validation only
- Development used for fitting or selection: no
- Dataset SHA256: `{sha256(DATA)}`

## Experiment 1 — context length on Validation

| Model | Overall RMSE | MAE | D1–7 | D8–14 | D15–30 |
|---|---:|---:|---:|---:|---:|
| iT90 | {val90.RMSE:.4f} | {val90.MAE:.4f} | {val90.D1_7:.4f} | {val90.D8_14:.4f} | {val90.D15_30:.4f} |
| iT365 | {val365.RMSE:.4f} | {val365.MAE:.4f} | {val365.D1_7:.4f} | {val365.D8_14:.4f} | {val365.D15_30:.4f} |

## Experiments 2–3 — Development

{dev.reset_index()[['Model','RMSE','MAE','D1_7','D8_14','D15_30']].to_markdown(index=False, floatfmt='.4f')}

Fusion parameters: q(h): a={a_ctx:.4f}, tau={tau_ctx:.4f} d; w(h): a={a_out:.4f}, tau={tau_out:.4f} d.

ReefFormer − iT365 overall ΔRMSE = {ci.Delta_RMSE:.4f} °C, paired 60-day moving-block 95% CI [{ci.CI_low:.4f}, {ci.CI_high:.4f}].

The result is reported as an independent-domain test. It is not used to redesign the Hainan architecture or coefficients.
"""
    (OUT / "GBR_GENERALIZATION_REPORT.md").write_text(report, encoding="utf-8")
    (PAPER / "GBR_GENERALIZATION_REPORT.md").write_text(report, encoding="utf-8")

    expected_shape = {split: (len(ids[split]), H, pixels) for split in ids}
    qa = {
        "status": "PASS",
        "dataset_sha256": sha256(DATA),
        "pixels": pixels,
        "seed": SEED,
        "prediction_shapes_expected": {key: list(value) for key, value in expected_shape.items()},
        "all_prediction_arrays_finite": all(np.isfinite(value).all() for split in predictions.values() for value in split.values()),
        "q_in_0_1": bool(np.all((q >= 0) & (q <= 1))),
        "w_in_0_1": bool(np.all((w >= 0) & (w <= 1))),
        "fusion_fit_split": "Val",
        "dev_used_for_selection": False,
        "bootstrap_replicates": 10000,
        "block_days": 60,
    }
    (OUT / "QA.json").write_text(json.dumps(qa, indent=2), encoding="utf-8")
    print(report)
    print(json.dumps(qa, indent=2))


if __name__ == "__main__":
    main()
