"""Analyze the five frozen 154-pixel priority experiments without manuscript edits."""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data/frozen_154_ssta_v1/dataset_daily.npz"
PRED = ROOT / "runs/final_154_v1/itransformer"
OUT = ROOT / "runs/final_154_v1/evidence"
SEEDS = (42, 43, 44)
H = 30


def score(pred, truth):
    error = pred.astype("float64") - truth.astype("float64")
    return {
        "RMSE": float(np.sqrt(np.mean(error ** 2))),
        "MAE": float(np.mean(np.abs(error))),
        "D1_7": float(np.sqrt(np.mean(error[:, :7] ** 2))),
        "D8_14": float(np.sqrt(np.mean(error[:, 7:14] ** 2))),
        "D15_30": float(np.sqrt(np.mean(error[:, 14:] ** 2))),
    }


def fit_smooth(left, right, truth):
    """Fit left*w + right*(1-w) by pooled validation MSE."""
    delta = left.astype("float64") - right.astype("float64")
    residual = right.astype("float64") - truth.astype("float64")
    aa = np.mean(delta * delta, axis=(0, 2))
    bb = np.mean(delta * residual, axis=(0, 2))
    cc = np.mean(residual * residual, axis=(0, 2))
    lead = np.arange(1, H + 1, dtype=float)

    def solve(log_tau):
        decay = np.exp(-lead / np.exp(log_tau))
        complement = 1 - decay
        denominator = max(np.sum(aa * complement * complement), 1e-15)
        asymptote = float(np.clip(-np.sum(complement * (aa * decay + bb)) / denominator, 0, 1))
        weight = decay + asymptote * complement
        loss = float(np.mean(aa * weight * weight + 2 * bb * weight + cc))
        return loss, asymptote, weight

    grid = np.linspace(np.log(0.1), np.log(1000), 4001)
    losses = np.array([solve(value)[0] for value in grid])
    index = int(np.argmin(losses))
    low = grid[max(0, index - 1)]
    high = grid[min(len(grid) - 1, index + 1)]
    phi = (1 + np.sqrt(5)) / 2
    x1 = high - (high - low) / phi
    x2 = low + (high - low) / phi
    f1 = solve(x1)[0]
    f2 = solve(x2)[0]
    for _ in range(100):
        if f1 <= f2:
            high, x2, f2 = x2, x1, f1
            x1 = high - (high - low) / phi
            f1 = solve(x1)[0]
        else:
            low, x1, f1 = x1, x2, f2
            x2 = low + (high - low) / phi
            f2 = solve(x2)[0]
    log_tau = (low + high) / 2
    _, asymptote, weight = solve(log_tau)
    return asymptote, float(np.exp(log_tau)), weight


def moving_block_ci(values, block=60, repetitions=10000, seed=1542026):
    """CI for the time mean of an [origin, lead] paired-loss array."""
    values = np.asarray(values, dtype="float64")
    n = len(values)
    blocks = int(np.ceil(n / block))
    rng = np.random.default_rng(seed)
    replicates = np.empty((repetitions, values.shape[1]), dtype="float64")
    offsets = np.arange(block)
    for start in range(0, repetitions, 500):
        count = min(500, repetitions - start)
        starts = rng.integers(0, n - block + 1, size=(count, blocks))
        indices = (starts[:, :, None] + offsets[None, None, :]).reshape(count, -1)[:, :n]
        replicates[start:start + count] = values[indices].mean(axis=1)
    return np.quantile(replicates, [0.025, 0.975], axis=0)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with np.load(DATA, allow_pickle=False) as z:
        dates = pd.DatetimeIndex(z["dates_all"])
        ssta = z["ssta_all_c"].astype("float32")
        rho = z["dap_rho_train"].astype("float64")
        ids = {name: z[name].astype(int) for name in ("Val", "Dev", "Test2026")}
    offsets = np.arange(1, H + 1)

    def truth(indices):
        return ssta[indices[:, None] + offsets[None, :]]

    def dap(indices):
        return (ssta[indices, None, :] * rho[None, None, :] ** offsets[None, :, None]).astype("float32")

    # Main full-Validation fusion and forward evaluations.
    rows = []
    weights = []
    predictions = {split: {} for split in ids}
    val_truth = truth(ids["Val"])
    for seed in SEEDS:
        full = PRED / "FULL"
        val90 = np.load(full / f"iT_L90_seed{seed}_Val.npy")
        val365 = np.load(full / f"iT_L365_seed{seed}_Val.npy")
        a_ctx, tau_ctx, q = fit_smooth(val90, val365, val_truth)
        ctx_val = q[None, :, None] * val90 + (1 - q[None, :, None]) * val365
        dap_val = dap(ids["Val"])
        a_out, tau_out, w = fit_smooth(dap_val, ctx_val, val_truth)
        weights.append({
            "seed": seed, "a_ctx": a_ctx, "tau_ctx_d": tau_ctx,
            "q_D1": q[0], "q_D7": q[6], "q_D14": q[13], "q_D21": q[20], "q_D30": q[29],
            "a_out": a_out, "tau_out_d": tau_out,
            "w_D1": w[0], "w_D7": w[6], "w_D14": w[13], "w_D21": w[20], "w_D30": w[29],
        })
        for split, split_ids in ids.items():
            target = truth(split_ids)
            label = split
            p90 = np.load(full / f"iT_L90_seed{seed}_{label}.npy")
            p180 = np.load(full / f"iT_L180_seed{seed}_{label}.npy")
            p365 = np.load(full / f"iT_L365_seed{seed}_{label}.npy")
            p_dap = dap(split_ids)
            p_ctx = q[None, :, None] * p90 + (1 - q[None, :, None]) * p365
            p_final = w[None, :, None] * p_dap + (1 - w[None, :, None]) * p_ctx
            model_predictions = {
                "iT90": p90, "iT180": p180, "iT365": p365,
                "DAP": p_dap, "Dual Context": p_ctx, "ReefFormer": p_final,
            }
            predictions[split][seed] = model_predictions
            for model, prediction in model_predictions.items():
                rows.append({"split": split, "seed": seed, "Model": model, **score(prediction, target)})
            np.save(OUT / f"CTX_seed{seed}_{split}.npy", p_ctx.astype("float32"))
            np.save(OUT / f"ReefFormer_seed{seed}_{split}.npy", p_final.astype("float32"))
    pd.DataFrame(rows).to_csv(OUT / "CORE_METRICS_ALL_SEEDS.csv", index=False)
    pd.DataFrame(weights).to_csv(OUT / "FROZEN_WEIGHTS.csv", index=False)
    summary = pd.DataFrame(rows).groupby(["split", "Model"], sort=False).agg(
        RMSE_mean=("RMSE", "mean"), RMSE_SD=("RMSE", "std"),
        MAE_mean=("MAE", "mean"), MAE_SD=("MAE", "std"),
        D1_7=("D1_7", "mean"), D8_14=("D8_14", "mean"), D15_30=("D15_30", "mean"),
    ).reset_index()
    summary.to_csv(OUT / "CORE_METRICS_SUMMARY.csv", index=False)

    # Lead-dependent complementarity from 2018 through 2025.
    combined_ids = np.concatenate([ids["Val"], ids["Dev"]])
    combined_truth = truth(combined_ids)
    combined_dates = dates[combined_ids + 1]
    context_loss = []
    predictor_loss = []
    for seed in SEEDS:
        p90 = np.concatenate([predictions["Val"][seed]["iT90"], predictions["Dev"][seed]["iT90"]])
        p365 = np.concatenate([predictions["Val"][seed]["iT365"], predictions["Dev"][seed]["iT365"]])
        pctx = np.concatenate([predictions["Val"][seed]["Dual Context"], predictions["Dev"][seed]["Dual Context"]])
        pdap = np.concatenate([predictions["Val"][seed]["DAP"], predictions["Dev"][seed]["DAP"]])
        context_loss.append(np.mean((p365 - combined_truth) ** 2 - (p90 - combined_truth) ** 2, axis=2))
        predictor_loss.append(np.mean((pctx - combined_truth) ** 2 - (pdap - combined_truth) ** 2, axis=2))
    context_loss = np.mean(context_loss, axis=0)
    predictor_loss = np.mean(predictor_loss, axis=0)
    context_ci = moving_block_ci(context_loss)
    predictor_ci = moving_block_ci(predictor_loss, seed=1542027)
    lead_rows = []
    for lead_index in range(H):
        lead_rows.append({
            "lead": lead_index + 1,
            "Delta_ctx_MSE": context_loss[:, lead_index].mean(),
            "Delta_ctx_CI_low": context_ci[0, lead_index],
            "Delta_ctx_CI_high": context_ci[1, lead_index],
            "Delta_pred_MSE": predictor_loss[:, lead_index].mean(),
            "Delta_pred_CI_low": predictor_ci[0, lead_index],
            "Delta_pred_CI_high": predictor_ci[1, lead_index],
        })
    pd.DataFrame(lead_rows).to_csv(OUT / "LEAD_COMPLEMENTARITY_2018_2025.csv", index=False)
    year_rows = []
    for year in range(2018, 2026):
        mask = combined_dates.year == year
        for lead_index in range(H):
            year_rows.append({
                "year": year, "lead": lead_index + 1,
                "Delta_ctx_MSE": context_loss[mask, lead_index].mean(),
                "Delta_pred_MSE": predictor_loss[mask, lead_index].mean(),
            })
    pd.DataFrame(year_rows).to_csv(OUT / "YEAR_LEAD_COMPLEMENTARITY.csv", index=False)

    # Strong cross-period validation-reuse analysis with disjoint checkpoint selection.
    cross_rows = []
    cross_weights = []
    selection = {item["scheme"]: item["selected_lookback"] for item in json.loads((PRED / "SELECTIONS.json").read_text())}
    val_year = dates[ids["Val"] + 1].year
    scheme_masks = {
        "FIT_2018_2019": {"fit": val_year <= 2019, "hold": val_year >= 2020},
        "FIT_2020_2021": {"fit": val_year >= 2020, "hold": val_year <= 2019},
    }
    for scheme, masks in scheme_masks.items():
        fit_ids = ids["Val"][masks["fit"]]
        hold_ids = ids["Val"][masks["hold"]]
        fit_truth = truth(fit_ids)
        hold_truth = truth(hold_ids)
        for seed in SEEDS:
            folder = PRED / scheme
            fit90 = np.load(folder / f"iT_L90_seed{seed}_Fit.npy")
            fit365 = np.load(folder / f"iT_L365_seed{seed}_Fit.npy")
            hold90 = np.load(folder / f"iT_L90_seed{seed}_Holdout.npy")
            hold365 = np.load(folder / f"iT_L365_seed{seed}_Holdout.npy")
            a_ctx, tau_ctx, q = fit_smooth(fit90, fit365, fit_truth)
            fit_ctx = q[None, :, None] * fit90 + (1 - q[None, :, None]) * fit365
            hold_ctx = q[None, :, None] * hold90 + (1 - q[None, :, None]) * hold365
            fit_dap = dap(fit_ids)
            hold_dap = dap(hold_ids)
            a_out, tau_out, w = fit_smooth(fit_dap, fit_ctx, fit_truth)
            hold_final = w[None, :, None] * hold_dap + (1 - w[None, :, None]) * hold_ctx
            selected_l = int(selection[scheme])
            hold_selected = np.load(folder / f"iT_L{selected_l}_seed{seed}_Holdout.npy")
            cross_weights.append({
                "scheme": scheme, "seed": seed, "selected_L": selected_l,
                "a_ctx": a_ctx, "tau_ctx_d": tau_ctx, "q_D1": q[0], "q_D14": q[13], "q_D30": q[29],
                "a_out": a_out, "tau_out_d": tau_out, "w_D1": w[0], "w_D14": w[13], "w_D30": w[29],
            })
            for model, prediction in {
                f"Selected iT L{selected_l}": hold_selected,
                "iT90": hold90, "iT365": hold365, "DAP": hold_dap,
                "Dual Context": hold_ctx, "ReefFormer": hold_final,
            }.items():
                cross_rows.append({"scheme": scheme, "seed": seed, "Model": model, **score(prediction, hold_truth)})
    pd.DataFrame(cross_weights).to_csv(OUT / "CROSS_VALIDATION_WEIGHTS.csv", index=False)
    pd.DataFrame(cross_rows).to_csv(OUT / "CROSS_VALIDATION_ALL_SEEDS.csv", index=False)
    pd.DataFrame(cross_rows).groupby(["scheme", "Model"], sort=False).agg(
        RMSE_mean=("RMSE", "mean"), RMSE_SD=("RMSE", "std"), MAE_mean=("MAE", "mean"),
        D1_7=("D1_7", "mean"), D8_14=("D8_14", "mean"), D15_30=("D15_30", "mean"),
    ).reset_index().to_csv(OUT / "CROSS_VALIDATION_SUMMARY.csv", index=False)

    # Calendar-matched Dev years and frozen 2026.
    matched_rows = []
    cutoff = pd.Timestamp("2024-08-09").dayofyear
    for split in ("Dev", "Test2026"):
        split_ids = ids[split]
        starts = dates[split_ids + 1]
        target = truth(split_ids)
        for year in sorted(np.unique(starts.year)):
            mask = (starts.year == year) & (starts.dayofyear <= cutoff)
            for seed in SEEDS:
                for model in ("iT365", "DAP", "Dual Context", "ReefFormer"):
                    matched_rows.append({
                        "year": int(year), "seed": seed, "Model": model,
                        "N_origins": int(mask.sum()), **score(predictions[split][seed][model][mask], target[mask]),
                    })
    matched = pd.DataFrame(matched_rows)
    matched.to_csv(OUT / "MATCHED_SEASON_ALL_SEEDS.csv", index=False)
    matched.groupby(["year", "Model"], sort=False).agg(
        N_origins=("N_origins", "first"), RMSE_mean=("RMSE", "mean"), RMSE_SD=("RMSE", "std"),
        MAE_mean=("MAE", "mean"),
    ).reset_index().to_csv(OUT / "MATCHED_SEASON_SUMMARY.csv", index=False)

    # Domain-transition audit: retrained 154 models versus old 163 models filtered to 154.
    old_path = ROOT / "runs/reviewer_revision_v1/GEOGRAPHIC_SENSITIVITY_154.csv"
    if old_path.exists():
        old = pd.read_csv(old_path)
        old = old[old.Model.isin(["iTransformer", "Dual Context", "ReefFormer"])]
        old = old.rename(columns={"RMSE_154": "Old163_model_filtered154_RMSE", "MAE_154": "Old163_model_filtered154_MAE"})
        new = pd.DataFrame(rows)
        new = new[(new.split == "Dev") & new.Model.isin(["iT365", "Dual Context", "ReefFormer"])].copy()
        new["Model"] = new.Model.replace({"iT365": "iTransformer"})
        audit = old.merge(new[["seed", "Model", "RMSE", "MAE"]], on=["seed", "Model"], how="inner")
        audit = audit.rename(columns={"RMSE": "Retrained154_RMSE", "MAE": "Retrained154_MAE"})
        audit["Retraining_RMSE_change"] = audit.Retrained154_RMSE - audit.Old163_model_filtered154_RMSE
        audit.to_csv(OUT / "DOMAIN_TRANSITION_AUDIT.csv", index=False)

    # Review figures stay outside the manuscript until user approval.
    lead_table = pd.DataFrame(lead_rows)
    year_table = pd.DataFrame(year_rows)
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.8), constrained_layout=True)
    h = lead_table.lead
    axes[0, 0].plot(h, lead_table.Delta_ctx_MSE, color="#2474b5")
    axes[0, 0].fill_between(h, lead_table.Delta_ctx_CI_low, lead_table.Delta_ctx_CI_high, color="#2474b5", alpha=0.18)
    axes[0, 0].axhline(0, color="black", lw=0.8)
    axes[0, 0].set(title="Context complementarity", ylabel=r"$MSE_{365}-MSE_{90}$", xlabel="Lead (days)")
    axes[0, 1].plot(h, lead_table.Delta_pred_MSE, color="#cc5a2e")
    axes[0, 1].fill_between(h, lead_table.Delta_pred_CI_low, lead_table.Delta_pred_CI_high, color="#cc5a2e", alpha=0.18)
    axes[0, 1].axhline(0, color="black", lw=0.8)
    axes[0, 1].set(title="Predictor complementarity", ylabel=r"$MSE_{CTX}-MSE_{DAP}$", xlabel="Lead (days)")
    for ax, value, title in zip(axes[1], ("Delta_ctx_MSE", "Delta_pred_MSE"), ("Context: year × lead", "Predictor: year × lead")):
        matrix = year_table.pivot(index="year", columns="lead", values=value).sort_index()
        limit = np.quantile(np.abs(matrix.to_numpy()), 0.98)
        image = ax.imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=-limit, vmax=limit, origin="lower")
        ax.set(title=title, xlabel="Lead (days)", ylabel="Target year")
        ax.set_xticks([0, 6, 13, 20, 29], [1, 7, 14, 21, 30])
        ax.set_yticks(range(len(matrix.index)), matrix.index)
        fig.colorbar(image, ax=ax, shrink=0.8, label="Paired MSE difference")
    fig.savefig(OUT / "FIG_COMPLEMENTARITY.png", dpi=240)
    fig.savefig(OUT / "FIG_COMPLEMENTARITY.pdf")
    plt.close(fig)

    audit = {
        "pixels": 154,
        "five_priority_experiments_completed": True,
        "neural_runs": 27,
        "main_context_runs": 9,
        "disjoint_validation_runs": 18,
        "bootstrap": "10,000 paired 60-day moving-block replicates",
        "manuscript_modified": False,
        "development_used_for_selection": False,
        "outputs": sorted(path.name for path in OUT.iterdir()),
    }
    (OUT / "AUDIT.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print("CORE SUMMARY\n", summary.to_string(index=False))
    print("\nCROSS VALIDATION\n", pd.read_csv(OUT / "CROSS_VALIDATION_SUMMARY.csv").to_string(index=False))
    print("\nMATCHED SEASON\n", pd.read_csv(OUT / "MATCHED_SEASON_SUMMARY.csv").to_string(index=False))


if __name__ == "__main__":
    main()
