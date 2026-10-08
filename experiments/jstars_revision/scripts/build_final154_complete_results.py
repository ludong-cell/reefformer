"""Build all frozen 154-pixel paper evidence without editing the manuscript."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_final154_core import fit_smooth
from train_backbone import calendar_index


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "frozen_154_ssta_v1" / "dataset_daily.npz"
CORE = ROOT / "runs" / "final_154_v1" / "evidence"
BASE = ROOT / "runs" / "final_154_v1" / "strong_baselines"
OUT = ROOT / "runs" / "final_154_v1" / "complete_results"
SEEDS = (42, 43, 44)
H = 30
BLOCK = 60
REPS = 10000
BANDS = {"Overall": slice(0, 30), "D1-7": slice(0, 7),
         "D8-14": slice(7, 14), "D15-30": slice(14, 30)}
ORDER = ("Climatology", "Persistence", "DAP", "DLinear", "PatchTST",
         "iTransformer", "Dual Context", "DAP Smooth", "ReefFormer")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def score(prediction, truth, mask=None):
    error = prediction.astype("float64") - truth.astype("float64")
    values = error.ravel() if mask is None else error[mask]
    return {"RMSE": float(np.sqrt(np.mean(values ** 2))),
            "MAE": float(np.mean(np.abs(values))), "Bias": float(np.mean(values)),
            "N_elements": int(values.size)}


def run_mask(values, min_length=5):
    output = np.zeros_like(values, dtype=bool)
    for pixel in range(values.shape[1]):
        series = values[:, pixel]
        edges = np.diff(np.r_[False, series, False].astype("int8"))
        starts = np.flatnonzero(edges == 1)
        ends = np.flatnonzero(edges == -1)
        for start, end in zip(starts, ends):
            if end - start >= min_length:
                output[start:end, pixel] = True
    return output


def block_counts(n, seed):
    rng = np.random.default_rng(seed)
    counts = np.empty((REPS, n), dtype="float32")
    n_blocks = int(np.ceil(n / BLOCK))
    offsets = np.arange(BLOCK)
    for start in range(0, REPS, 500):
        count = min(500, REPS - start)
        origins = rng.integers(0, n - BLOCK + 1, size=(count, n_blocks))
        indices = (origins[:, :, None] + offsets).reshape(count, -1)[:, :n]
        for row, sampled in enumerate(indices):
            counts[start + row] = np.bincount(sampled, minlength=n)
    return counts


def paired_bootstrap(predictions, truth, split, counts, comparisons):
    rows = []
    for model_a, model_b, label in comparisons:
        for band, band_slice in BANDS.items():
            rmse_samples = []
            mae_samples = []
            point_rmse = []
            point_mae = []
            for seed in SEEDS:
                error_a = predictions[split][seed][model_a][:, band_slice].astype("float64") - truth[:, band_slice]
                error_b = predictions[split][seed][model_b][:, band_slice].astype("float64") - truth[:, band_slice]
                mse_a = np.mean(error_a ** 2, axis=(1, 2))
                mse_b = np.mean(error_b ** 2, axis=(1, 2))
                mae_a = np.mean(np.abs(error_a), axis=(1, 2))
                mae_b = np.mean(np.abs(error_b), axis=(1, 2))
                denominator = counts.sum(axis=1)
                rmse_samples.append(np.sqrt(counts @ mse_a / denominator) -
                                    np.sqrt(counts @ mse_b / denominator))
                mae_samples.append(counts @ (mae_a - mae_b) / denominator)
                point_rmse.append(np.sqrt(mse_a.mean()) - np.sqrt(mse_b.mean()))
                point_mae.append(mae_a.mean() - mae_b.mean())
            for metric, samples, point in (
                ("RMSE", np.mean(rmse_samples, axis=0), float(np.mean(point_rmse))),
                ("MAE", np.mean(mae_samples, axis=0), float(np.mean(point_mae))),
            ):
                low, high = np.quantile(samples, (0.025, 0.975))
                rows.append({"split": split, "Contrast": label, "Model_A": model_a,
                             "Model_B": model_b, "Band": band, "Metric": metric,
                             "Delta_A_minus_B": point, "CI95_low": low, "CI95_high": high,
                             "CI_excludes_zero": bool(low > 0 or high < 0),
                             "Favours_A": bool(high < 0), "Block_days": BLOCK,
                             "Replicates": REPS})
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with np.load(DATA, allow_pickle=False) as z:
        dates = pd.DatetimeIndex(z["dates_all"])
        sst = z["sst_all_c"].astype("float64")
        ssta = z["ssta_all_c"].astype("float64")
        climatology = z["climatology365_c"].astype("float64")
        rho = z["dap_rho_train"].astype("float64")
        sectors = z["sectors"].astype(str)
        ids_by_split = {name: z[name].astype(int) for name in ("Val", "Dev", "Test2026")}
    leads = np.arange(1, H + 1)
    selected_lookback = {
        "DLinear": int(json.loads((BASE / "dlinear" / "SELECTION.json").read_text())["best_lookback"]),
        "PatchTST": int(json.loads((BASE / "patchtst" / "SELECTION.json").read_text())["best_lookback"]),
    }

    def truth(indices):
        return ssta[indices[:, None] + leads[None, :]]

    def persistence(indices):
        return np.repeat(ssta[indices, None, :], H, axis=1)

    def dap(indices):
        return ssta[indices, None, :] * rho[None, None, :] ** leads[None, :, None]

    truth_by_split = {split: truth(indices) for split, indices in ids_by_split.items()}
    prediction = {split: {seed: {} for seed in SEEDS} for split in ids_by_split}
    fit_rows = []
    for seed in SEEDS:
        val_truth = truth_by_split["Val"]
        val_dap = dap(ids_by_split["Val"])
        val_it365 = np.load(ROOT / "runs" / "final_154_v1" / "itransformer" / "FULL" /
                            f"iT_L365_seed{seed}_Val.npy")
        a_old, tau_old, w_old = fit_smooth(val_dap, val_it365, val_truth)
        fit_rows.append({"seed": seed, "a_DAP_Smooth": a_old, "tau_DAP_Smooth_d": tau_old,
                         "w_D1": w_old[0], "w_D7": w_old[6], "w_D14": w_old[13],
                         "w_D21": w_old[20], "w_D30": w_old[29]})
        for split, indices in ids_by_split.items():
            stat = {"Climatology": np.zeros_like(truth_by_split[split]),
                    "Persistence": persistence(indices), "DAP": dap(indices)}
            if split == "Val":
                dlinear = np.load(BASE / "dlinear" /
                                  f"DLinear_L{selected_lookback['DLinear']}_seed{seed}_Val.npy")
                patch = np.load(BASE / "patchtst" /
                                f"PatchTST_L{selected_lookback['PatchTST']}_seed{seed}_Val.npy")
            else:
                dlinear = np.load(BASE / "dlinear" / f"DLinear_seed{seed}_{split}.npy")
                patch = np.load(BASE / "patchtst" / f"PatchTST_seed{seed}_{split}.npy")
            it365 = np.load(ROOT / "runs" / "final_154_v1" / "itransformer" / "FULL" /
                            f"iT_L365_seed{seed}_{split}.npy")
            ctx = np.load(CORE / f"CTX_seed{seed}_{split}.npy")
            reef = np.load(CORE / f"ReefFormer_seed{seed}_{split}.npy")
            old_smooth = w_old[None, :, None] * stat["DAP"] + (1 - w_old[None, :, None]) * it365
            prediction[split][seed] = {**stat, "DLinear": dlinear, "PatchTST": patch,
                                       "iTransformer": it365, "Dual Context": ctx,
                                       "DAP Smooth": old_smooth, "ReefFormer": reef}
            np.save(OUT / f"DAP_Smooth_seed{seed}_{split}.npy", old_smooth.astype("float32"))
    pd.DataFrame(fit_rows).to_csv(OUT / "DAP_SMOOTH_FROZEN_WEIGHTS.csv", index=False)

    parameters = {
        "Climatology": ("--", 0, 0), "Persistence": ("--", 0, 0),
        "DAP": ("--", 0, 154), "DLinear": ("180", 10860, 0),
        "PatchTST": ("365", 62612, 0), "iTransformer": ("365", 29854, 0),
        "Dual Context": ("90+365", 50910, 0), "DAP Smooth": ("365", 29856, 154),
        "ReefFormer": ("90+365", 50912, 154),
    }
    all_rows, lead_rows, year_rows, region_rows = [], [], [], []
    for split, indices in ids_by_split.items():
        target = truth_by_split[split]
        target_year = dates[indices + 1].year
        for model in ORDER:
            model_seeds = (42,) if model in ("Climatology", "Persistence", "DAP") else SEEDS
            for seed in model_seeds:
                pred = prediction[split][seed][model]
                assert pred.shape == target.shape and np.isfinite(pred).all()
                error = pred.astype("float64") - target
                entry = {"split": split, "seed": seed, "Model": model,
                         "Best_L": parameters[model][0], "Neural_and_blend_params": parameters[model][1],
                         "Statistical_coefficients": parameters[model][2], **score(pred, target)}
                entry.update({"D1_7": float(np.sqrt(np.mean(error[:, :7] ** 2))),
                              "D8_14": float(np.sqrt(np.mean(error[:, 7:14] ** 2))),
                              "D15_30": float(np.sqrt(np.mean(error[:, 14:] ** 2)))})
                all_rows.append(entry)
                for h in range(H):
                    lead_rows.append({"split": split, "seed": seed, "Model": model, "Lead": h + 1,
                                      "RMSE": float(np.sqrt(np.mean(error[:, h] ** 2))),
                                      "MAE": float(np.mean(np.abs(error[:, h]))),
                                      "Bias": float(np.mean(error[:, h]))})
                for year in np.unique(target_year):
                    mask = target_year == year
                    year_rows.append({"split": split, "seed": seed, "Model": model, "Year": int(year),
                                      **score(pred[mask], target[mask])})
                for region in ("West", "North", "East", "South"):
                    mask = sectors == region
                    region_rows.append({"split": split, "seed": seed, "Model": model, "Region": region,
                                        **score(pred[:, :, mask], target[:, :, mask])})
    all_frame = pd.DataFrame(all_rows)
    all_frame.to_csv(OUT / "ALL_SEED_SPLIT_METRICS.csv", index=False)
    pd.DataFrame(lead_rows).to_csv(OUT / "LEADWISE.csv", index=False)
    pd.DataFrame(year_rows).to_csv(OUT / "YEARWISE.csv", index=False)
    pd.DataFrame(region_rows).to_csv(OUT / "REGIONWISE.csv", index=False)
    summary = all_frame.groupby(["split", "Model"], sort=False).agg(
        Best_L=("Best_L", "first"), Neural_and_blend_params=("Neural_and_blend_params", "first"),
        Statistical_coefficients=("Statistical_coefficients", "first"),
        RMSE_mean=("RMSE", "mean"), RMSE_SD=("RMSE", "std"),
        MAE_mean=("MAE", "mean"), MAE_SD=("MAE", "std"),
        D1_7=("D1_7", "mean"), D8_14=("D8_14", "mean"), D15_30=("D15_30", "mean")
    ).reset_index().fillna(0)
    summary["ReefFormer_improvement_pct"] = np.nan
    for split in ids_by_split:
        final = float(summary[(summary.split == split) & (summary.Model == "ReefFormer")].RMSE_mean.iloc[0])
        mask = summary.split == split
        summary.loc[mask, "ReefFormer_improvement_pct"] = 100 * (summary.loc[mask, "RMSE_mean"] - final) / summary.loc[mask, "RMSE_mean"]
    summary.to_csv(OUT / "MAIN_TABLE_ALL_SPLITS.csv", index=False)
    summary[summary.split == "Dev"].to_csv(OUT / "TABLE_MAIN_DEV_2022_2025.csv", index=False)
    summary[summary.split == "Test2026"].to_csv(OUT / "TABLE_FROZEN_2026.csv", index=False)

    comparisons = [("ReefFormer", model, f"ReefFormer - {model}") for model in ORDER[:-1]]
    bootstrap_rows = []
    for index, split in enumerate(("Dev", "Test2026")):
        counts = block_counts(len(ids_by_split[split]), 154420 + index)
        bootstrap_rows.extend(paired_bootstrap(prediction, truth_by_split[split], split, counts, comparisons))
    pd.DataFrame(bootstrap_rows).to_csv(OUT / "STRONG_BASELINE_PAIRED_60D_BOOTSTRAP.csv", index=False)

    # Factorial module ablation on Dev.
    factorial_models = {"M00": "iTransformer", "M10": "Dual Context",
                        "M01": "DAP Smooth", "M11": "ReefFormer"}
    factorial_comparisons = [
        ("Dual Context", "iTransformer", "M10-M00"),
        ("DAP Smooth", "iTransformer", "M01-M00"),
        ("ReefFormer", "Dual Context", "M11-M10"),
        ("ReefFormer", "DAP Smooth", "M11-M01"),
    ]
    counts_dev = block_counts(len(ids_by_split["Dev"]), 154600)
    factorial_rows = paired_bootstrap(prediction, truth_by_split["Dev"], "Dev", counts_dev,
                                      factorial_comparisons)
    fact = pd.DataFrame(factorial_rows)
    # RMSE-scale diminishing-return interaction with the same paired blocks.
    for band in BANDS:
        band_slice = BANDS[band]
        per_seed_samples = []
        per_seed_points = []
        denominator = counts_dev.sum(axis=1)
        for seed in SEEDS:
            origin_mse = {}
            for code, model in factorial_models.items():
                error = (prediction["Dev"][seed][model][:, band_slice].astype("float64") -
                         truth_by_split["Dev"][:, band_slice])
                origin_mse[code] = np.mean(error ** 2, axis=(1, 2))
            samples = ((np.sqrt(counts_dev @ origin_mse["M11"] / denominator) -
                        np.sqrt(counts_dev @ origin_mse["M10"] / denominator)) -
                       (np.sqrt(counts_dev @ origin_mse["M01"] / denominator) -
                        np.sqrt(counts_dev @ origin_mse["M00"] / denominator)))
            per_seed_samples.append(samples)
            per_seed_points.append((np.sqrt(origin_mse["M11"].mean()) - np.sqrt(origin_mse["M10"].mean())) -
                                   (np.sqrt(origin_mse["M01"].mean()) - np.sqrt(origin_mse["M00"].mean())))
        interaction_samples = np.mean(per_seed_samples, axis=0)
        low, high = np.quantile(interaction_samples, (0.025, 0.975))
        fact.loc[len(fact)] = {"split": "Dev", "Contrast": "Interaction", "Model_A": "(M11-M10)",
                              "Model_B": "(M01-M00)", "Band": band, "Metric": "RMSE",
                              "Delta_A_minus_B": float(np.mean(per_seed_points)),
                              "CI95_low": low, "CI95_high": high,
                              "CI_excludes_zero": bool(low > 0 or high < 0),
                              "Favours_A": np.nan, "Block_days": BLOCK, "Replicates": REPS}
    fact.to_csv(OUT / "FACTORIAL_PAIRED_60D_BOOTSTRAP.csv", index=False)
    summary[(summary.split == "Dev") & summary.Model.isin(factorial_models.values())].to_csv(
        OUT / "TABLE_FACTORIAL_ABLATION.csv", index=False)

    # Train-defined extreme-warm subsets.
    day_of_year = calendar_index(dates)
    train_days = dates.year <= 2017
    high_threshold = np.quantile(ssta[train_days], 0.9, axis=0)
    warm_threshold = np.empty((365, sst.shape[1]), dtype="float64")
    for day in range(365):
        distance = np.minimum((day_of_year - day) % 365, (day - day_of_year) % 365)
        warm_threshold[day] = np.quantile(sst[train_days & (distance <= 15)], 0.9, axis=0)
    observed_exceedance = sst > warm_threshold[day_of_year]
    warm_daily = run_mask(observed_exceedance, 5)
    extreme_rows = []
    diagnostic_rows = []
    extreme_bootstrap = []
    for split in ("Dev", "Test2026"):
        indices = ids_by_split[split]
        target_indices = indices[:, None] + leads[None, :]
        target = truth_by_split[split]
        masks = {"High-SSTA": target >= high_threshold[None, None, :],
                 "MHW-like": warm_daily[target_indices]}
        counts = block_counts(len(indices), 155000 + (split == "Test2026"))
        for subset, mask in masks.items():
            for model in ORDER:
                model_seeds = (42,) if model in ("Climatology", "Persistence", "DAP") else SEEDS
                for seed in model_seeds:
                    pred = prediction[split][seed][model]
                    values = score(pred, target, mask)
                    true_values = target[mask]
                    pred_values = pred[mask]
                    slope, intercept = np.polyfit(true_values, pred_values, 1)
                    extreme_rows.append({"split": split, "subset": subset, "seed": seed,
                                         "Model": model, **values})
                    diagnostic_rows.append({"split": split, "subset": subset, "seed": seed,
                                            "Model": model, "Regression_slope": slope,
                                            "Regression_intercept": intercept,
                                            "Mean_true_SSTA": float(true_values.mean()),
                                            "Mean_predicted_SSTA": float(pred_values.mean()),
                                            "Amplitude_shrinkage_1_minus_slope": 1 - slope})
            # Conditional paired bootstrap versus ReefFormer for learnable strong baselines.
            for baseline in ("DLinear", "PatchTST", "iTransformer", "Dual Context", "DAP Smooth"):
                samples_by_seed, points = [], []
                mask_count = mask.sum(axis=(1, 2)).astype("float64")
                denominator = counts @ mask_count
                valid = denominator > 0
                for seed in SEEDS:
                    errors = {}
                    for model in ("ReefFormer", baseline):
                        err = prediction[split][seed][model].astype("float64") - target
                        errors[model] = np.sum(np.where(mask, err ** 2, 0), axis=(1, 2))
                    a = np.sqrt((counts[valid] @ errors["ReefFormer"]) / denominator[valid])
                    b = np.sqrt((counts[valid] @ errors[baseline]) / denominator[valid])
                    samples_by_seed.append(a - b)
                    points.append(np.sqrt(errors["ReefFormer"].sum() / mask_count.sum()) -
                                  np.sqrt(errors[baseline].sum() / mask_count.sum()))
                samples = np.mean(samples_by_seed, axis=0)
                low, high = np.quantile(samples, (0.025, 0.975))
                extreme_bootstrap.append({"split": split, "subset": subset,
                                          "Contrast": f"ReefFormer - {baseline}",
                                          "Delta_RMSE": float(np.mean(points)), "CI95_low": low,
                                          "CI95_high": high, "CI_excludes_zero": bool(low > 0 or high < 0),
                                          "Block_days": BLOCK, "Replicates": REPS})
        np.savez_compressed(OUT / f"EXTREME_MASKS_{split}.npz", target_indices=target_indices,
                            high_ssta_mask=masks["High-SSTA"], mhw_like_mask=masks["MHW-like"])
    extreme = pd.DataFrame(extreme_rows)
    extreme.to_csv(OUT / "EXTREME_ALL_SEEDS.csv", index=False)
    extreme.groupby(["split", "subset", "Model"], sort=False).agg(
        RMSE_mean=("RMSE", "mean"), RMSE_SD=("RMSE", "std"),
        MAE_mean=("MAE", "mean"), Bias_mean=("Bias", "mean"),
        N_elements=("N_elements", "first")
    ).reset_index().fillna(0).to_csv(OUT / "EXTREME_SUMMARY.csv", index=False)
    pd.DataFrame(diagnostic_rows).to_csv(OUT / "EXTREME_WARM_SHRINKAGE_DIAGNOSTIC.csv", index=False)
    pd.DataFrame(extreme_bootstrap).to_csv(OUT / "EXTREME_PAIRED_60D_BOOTSTRAP.csv", index=False)

    # Same Jan 1--Aug 9 forecast-origin season for every year.
    matched_rows = []
    cutoff = pd.Timestamp("2024-08-09").dayofyear
    for split in ("Dev", "Test2026"):
        indices = ids_by_split[split]
        starts = dates[indices + 1]
        target = truth_by_split[split]
        for year in np.unique(starts.year):
            mask = (starts.year == year) & (starts.dayofyear <= cutoff)
            for model in ORDER:
                model_seeds = (42,) if model in ("Climatology", "Persistence", "DAP") else SEEDS
                for seed in model_seeds:
                    matched_rows.append({"year": int(year), "seed": seed, "Model": model,
                                         "N_origins": int(mask.sum()),
                                         **score(prediction[split][seed][model][mask], target[mask])})
    matched = pd.DataFrame(matched_rows)
    matched.to_csv(OUT / "MATCHED_SEASON_ALL_SEEDS.csv", index=False)
    matched.groupby(["year", "Model"], sort=False).agg(
        N_origins=("N_origins", "first"), RMSE_mean=("RMSE", "mean"),
        RMSE_SD=("RMSE", "std"), MAE_mean=("MAE", "mean")
    ).reset_index().fillna(0).to_csv(OUT / "MATCHED_SEASON_SUMMARY.csv", index=False)

    manifest_files = [DATA]
    for model in ("dlinear", "patchtst"):
        manifest_files.append(BASE / model / "PRE_DEV_FREEZE_MANIFEST.json")
    manifest_files.extend([
        ROOT / "runs" / "final_154_v1" / "itransformer" / "PROTOCOL.json",
        CORE / "FROZEN_WEIGHTS.csv",
    ])
    audit = {
        "pixels": 154, "seeds": list(SEEDS), "forecast_horizon_days": H,
        "dataset_sha256": sha256(DATA), "all_prediction_shapes_verified": True,
        "all_predictions_finite": True, "baseline_selection_split": "2018-2021 Validation only",
        "Dev_used_for_model_or_lookback_selection": False,
        "bootstrap": {"block_days": BLOCK, "replicates": REPS, "paired": True},
        "high_ssta_definition": "target SSTA >= per-pixel Train 2000-2017 90th percentile",
        "mhw_like_definition": "target SST above per-pixel calendar-day Train 90th percentile (+/-15-day window) for >=5 consecutive observed days",
        "mhw_label_limit": "MHW-like because the training baseline is 18 years",
        "source_hashes": {str(path.relative_to(ROOT)): sha256(path) for path in manifest_files},
        "manuscript_modified": False,
    }
    (OUT / "PROTOCOL_AND_AUDIT.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print("DEV MAIN\n", summary[summary.split == "Dev"].to_string(index=False), flush=True)
    print("\n2026 MAIN\n", summary[summary.split == "Test2026"].to_string(index=False), flush=True)
    print("\nFACTORIAL OVERALL RMSE\n", fact[(fact.Band == "Overall") & (fact.Metric == "RMSE")].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
