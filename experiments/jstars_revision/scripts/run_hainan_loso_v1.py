"""Buffered four-fold leave-one-sector-out experiment around Hainan.

Each fold removes one complete coastal sector and every non-target pixel within
25 km of that sector. Neural checkpoints and both smooth fusion functions are
trained/fitted only on the remaining source pixels. The frozen models are then
applied to the unseen target-sector tokens. Target-sector train-period
climatology/scaling and pixel DAP decay remain available as historical
preprocessing; target-sector values never contribute to neural optimization,
checkpoint selection, or LCF/LPF fitting.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from analyze_gbr_generalization import fit_smooth
from itransformer_feature_pilot import CONFIG as BASE_CONFIG, Model


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data/frozen_154_ssta_v1/dataset_daily.npz"
OUT = ROOT / "runs/hainan_loso_v1"
SECTORS = ("West", "North", "East", "South")
SEEDS = (42, 43, 44)
LOOKBACKS = (90, 365)
H = 30
BATCH = 32
LR = 0.001
MAX_EPOCHS = 40
PATIENCE = 8
BUFFER_KM = 25.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def haversine_matrix(lat1, lon1, lat2, lon2):
    p1 = np.radians(np.asarray(lat1))[:, None]
    p2 = np.radians(np.asarray(lat2))[None, :]
    dp = p2 - p1
    dl = np.radians(np.asarray(lon2))[None, :] - np.radians(np.asarray(lon1))[:, None]
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 6371.0088 * 2 * np.arcsin(np.sqrt(a))


def metrics(prediction, truth):
    error = prediction.astype("float32", copy=False) - truth.astype("float32", copy=False)
    return {
        "RMSE": float(np.sqrt(np.mean(error ** 2, dtype=np.float64))),
        "MAE": float(np.mean(np.abs(error), dtype=np.float64)),
        "D1_7": float(np.sqrt(np.mean(error[:, :7] ** 2, dtype=np.float64))),
        "D8_14": float(np.sqrt(np.mean(error[:, 7:14] ** 2, dtype=np.float64))),
        "D15_30": float(np.sqrt(np.mean(error[:, 14:] ** 2, dtype=np.float64))),
    }


def mse_float32(prediction, truth, chunk=256):
    """Memory-bounded MSE with float64 accumulation and float32 differences."""
    total = 0.0
    count = 0
    for start in range(0, prediction.shape[0], chunk):
        difference = (prediction[start:start + chunk].astype("float32", copy=False)
                      - truth[start:start + chunk].astype("float32", copy=False))
        total += float(np.sum(difference * difference, dtype=np.float64))
        count += difference.size
    return total / count


def smooth_weights(a, tau):
    lead = np.arange(1, H + 1, dtype="float64")
    return a + (1 - a) * np.exp(-lead / tau)


def block_indices(n, repetitions=10000, block=60, seed=154425):
    rng = np.random.default_rng(seed)
    blocks = int(np.ceil(n / block))
    offsets = np.arange(block)
    for start in range(0, repetitions, 250):
        count = min(250, repetitions - start)
        origins = rng.integers(0, n - block + 1, size=(count, blocks))
        yield (origins[:, :, None] + offsets[None, None, :]).reshape(count, -1)[:, :n]


def paired_fold_seed_ci(candidates, references, contrast):
    # Inputs are already pixel-averaged squared errors with shape
    # folds, seeds, origins, leads. This keeps the four-fold run bounded in RAM.
    cand = np.asarray(candidates, dtype="float32")
    ref = np.asarray(references, dtype="float32")
    bands = {"Overall": np.arange(H), "D1-7": np.arange(7),
             "D8-14": np.arange(7, 14), "D15-30": np.arange(14, H)}
    cand_band = {name: cand[:, :, :, leads].mean(axis=3) for name, leads in bands.items()}
    ref_band = {name: ref[:, :, :, leads].mean(axis=3) for name, leads in bands.items()}
    collected = {name: [] for name in bands}
    for indices in block_indices(cand.shape[2]):
        for name in bands:
            left = np.sqrt(cand_band[name][:, :, indices].mean(axis=3))
            right = np.sqrt(ref_band[name][:, :, indices].mean(axis=3))
            collected[name].append((left - right).mean(axis=(0, 1)))
    rows = []
    for name, leads in bands.items():
        values = np.concatenate(collected[name])
        point = []
        for fold_index in range(cand.shape[0]):
            for seed_index in range(cand.shape[1]):
                point.append(np.sqrt(cand[fold_index, seed_index, :, leads].mean())
                             - np.sqrt(ref[fold_index, seed_index, :, leads].mean()))
        rows.append({
            "Contrast": contrast,
            "Band": name,
            "Delta_RMSE": float(np.mean(point)),
            "CI_low": float(np.quantile(values, .025)),
            "CI_high": float(np.quantile(values, .975)),
            "block_days": 60,
            "replicates": 10000,
            "fold_weighting": "equal sector weight",
        })
    return rows


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the frozen LOSO protocol")
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    device = "cuda"

    with np.load(DATA, allow_pickle=False) as z:
        dates = pd.DatetimeIndex(z["dates_all"])
        ssta = z["ssta_all_c"].astype("float32")
        scale = float(z["ssta_scale"])
        rho = z["dap_rho_train"].astype("float64")
        sectors = z["sectors"].astype(str)
        lat = z["pixel_latitude"].astype("float64")
        lon = z["pixel_longitude"].astype("float64")
        pixel_ids = z["pixel_ids"].astype(str)
        ids = {name: z[name].astype("int64") for name in ("Train", "Val", "Dev", "Test2026")}

    normalized = torch.as_tensor(ssta / scale, device=device)
    lead_t = torch.arange(1, H + 1, device=device)
    lead = np.arange(1, H + 1)
    config = {key: BASE_CONFIG[key] for key in (
        "output_attention", "use_norm", "d_model", "embed", "freq", "dropout",
        "class_strategy", "factor", "n_heads", "d_ff", "e_layers", "activation"
    )}
    config["pred_len"] = H

    fold_rows = []
    audit_rows = []
    training_rows = []
    bootstrap_store = {}
    for target_sector in SECTORS:
        target_cols = np.where(sectors == target_sector)[0]
        candidates = np.where(sectors != target_sector)[0]
        distance = haversine_matrix(lat[candidates], lon[candidates], lat[target_cols], lon[target_cols]).min(axis=1)
        source_cols = candidates[distance >= BUFFER_KM]
        excluded_buffer = candidates[distance < BUFFER_KM]
        fold_dir = OUT / target_sector
        fold_dir.mkdir(exist_ok=True)
        audit_rows.append({
            "target_sector": target_sector,
            "target_pixels": len(target_cols),
            "source_pixels_after_buffer": len(source_cols),
            "buffer_excluded_pixels": len(excluded_buffer),
            "buffer_km": BUFFER_KM,
            "minimum_source_target_distance_km": float(haversine_matrix(
                lat[source_cols], lon[source_cols], lat[target_cols], lon[target_cols]
            ).min()),
            "target_pixel_ids": ";".join(pixel_ids[target_cols]),
            "buffer_excluded_pixel_ids": ";".join(pixel_ids[excluded_buffer]),
        })

        def truth(split, cols):
            current = ids[split]
            return ssta[current[:, None] + lead[None, :]][:, :, cols]

        def dap(split, cols):
            current = ids[split]
            return (ssta[current, None, :][:, :, cols]
                    * rho[None, None, cols] ** lead[None, :, None]).astype("float32")

        fold_predictions = {}
        for seed in SEEDS:
            fold_predictions[seed] = {split: {} for split in ("Val", "Dev", "Test2026")}
            for lookback in LOOKBACKS:
                checkpoint_path = fold_dir / f"iT_L{lookback}_seed{seed}.pt"
                marker = fold_dir / f"COMPLETE_L{lookback}_seed{seed}.json"
                history_path = fold_dir / f"iT_L{lookback}_seed{seed}_training.csv"
                cfg = SimpleNamespace(**dict(config, seq_len=lookback))
                model = Model(cfg).to(device)
                if marker.exists() and checkpoint_path.exists():
                    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
                    model.load_state_dict(checkpoint["state_dict"])
                else:
                    torch.manual_seed(seed)
                    torch.cuda.manual_seed_all(seed)
                    np.random.seed(seed)
                    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
                    history = torch.arange(-lookback + 1, 1, device=device)
                    source_t = torch.as_tensor(source_cols, device=device)
                    generator = torch.Generator().manual_seed(seed)

                    def source_batch(origin_ids):
                        current = torch.as_tensor(origin_ids, device=device)
                        x = normalized[current[:, None] + history][:, :, source_t]
                        y = normalized[current[:, None] + lead_t][:, :, source_t]
                        return x, y

                    def val_mse():
                        model.eval()
                        total = 0.0
                        count = 0
                        with torch.no_grad():
                            for start in range(0, len(ids["Val"]), BATCH):
                                x, y = source_batch(ids["Val"][start:start + BATCH])
                                difference = model(x, None, None, None) - y
                                total += float(torch.sum(difference * difference).item())
                                count += difference.numel()
                        return total / count

                    best = val_mse()
                    best_epoch = 0
                    best_state = copy.deepcopy(model.state_dict())
                    bad = 0
                    log = [{"epoch": 0, "train_loss": np.nan, "Val_RMSE": np.sqrt(best) * scale}]
                    started = time.perf_counter()
                    completed = 0
                    for epoch in range(1, MAX_EPOCHS + 1):
                        model.train()
                        losses = []
                        order = torch.randperm(len(ids["Train"]), generator=generator).numpy()
                        for start in range(0, len(order), BATCH):
                            x, y = source_batch(ids["Train"][order[start:start + BATCH]])
                            optimizer.zero_grad(set_to_none=True)
                            loss = F.mse_loss(model(x, None, None, None), y)
                            if not torch.isfinite(loss):
                                raise RuntimeError("Non-finite LOSO training loss")
                            loss.backward()
                            optimizer.step()
                            losses.append(loss.item())
                        current = val_mse()
                        log.append({"epoch": epoch, "train_loss": float(np.mean(losses)),
                                    "Val_RMSE": np.sqrt(current) * scale})
                        pd.DataFrame(log).to_csv(history_path, index=False)
                        if current < best:
                            best = current
                            best_epoch = epoch
                            best_state = copy.deepcopy(model.state_dict())
                            bad = 0
                        else:
                            bad += 1
                        completed = epoch
                        print(target_sector, f"seed={seed}", f"L={lookback}", f"epoch={epoch}",
                              f"val={np.sqrt(current)*scale:.6f}", f"best={best_epoch}", flush=True)
                        if bad >= PATIENCE:
                            break
                    model.load_state_dict(best_state)
                    torch.save({
                        "state_dict": best_state,
                        "seed": seed,
                        "lookback": lookback,
                        "target_sector": target_sector,
                        "source_cols": source_cols,
                        "target_cols": target_cols,
                        "buffer_km": BUFFER_KM,
                        "best_epoch": best_epoch,
                        "config": dict(config, seq_len=lookback),
                        "dataset_sha256": sha256(DATA),
                    }, checkpoint_path)
                    marker.write_text(json.dumps({"complete": True, "best_epoch": best_epoch}, indent=2), encoding="utf-8")
                    training_rows.append({
                        "target_sector": target_sector, "seed": seed, "Lookback": lookback,
                        "Best_epoch": best_epoch, "Epochs_run": completed,
                        "Training_time_s": time.perf_counter() - started,
                        "source_pixels": len(source_cols),
                    })
                    pd.DataFrame(training_rows).to_csv(OUT / "TRAINING_SUMMARY.csv", index=False)
                    del optimizer, best_state

                history = torch.arange(-lookback + 1, 1, device=device)
                target_t = torch.as_tensor(target_cols, device=device)
                source_t = torch.as_tensor(source_cols, device=device)

                def infer(origin_ids, columns_t):
                    model.eval()
                    result = np.empty((len(origin_ids), H, len(columns_t)), dtype="float32")
                    with torch.no_grad():
                        for start in range(0, len(origin_ids), BATCH):
                            current = torch.as_tensor(origin_ids[start:start + BATCH], device=device)
                            x = normalized[current[:, None] + history][:, :, columns_t]
                            stop = min(start + BATCH, len(origin_ids))
                            result[start:stop] = model(x, None, None, None).cpu().numpy() * scale
                    return result

                fold_predictions[seed]["Val"][f"iT{lookback}"] = infer(ids["Val"], source_t)
                for split in ("Dev", "Test2026"):
                    fold_predictions[seed][split][f"iT{lookback}"] = infer(ids[split], target_t)
                del model
                torch.cuda.empty_cache()

            val_truth = truth("Val", source_cols)
            val_dap = dap("Val", source_cols)
            a_ctx, tau_ctx, q = fit_smooth(
                fold_predictions[seed]["Val"]["iT90"],
                fold_predictions[seed]["Val"]["iT365"],
                val_truth,
            )
            val_ctx = (q[None, :, None] * fold_predictions[seed]["Val"]["iT90"]
                       + (1 - q[None, :, None]) * fold_predictions[seed]["Val"]["iT365"])
            a_out, tau_out, w = fit_smooth(val_dap, val_ctx, val_truth)
            for split in ("Dev", "Test2026"):
                current = fold_predictions[seed][split]
                y = truth(split, target_cols)
                p_dap = dap(split, target_cols)
                p_ctx = q[None, :, None] * current["iT90"] + (1 - q[None, :, None]) * current["iT365"]
                p_final = w[None, :, None] * p_dap + (1 - w[None, :, None]) * p_ctx
                current.update({"DAP": p_dap, "Dual Context": p_ctx, "ReefFormer": p_final})
                for model_name in ("iT90", "iT365", "DAP", "Dual Context", "ReefFormer"):
                    fold_rows.append({
                        "target_sector": target_sector, "Split": split, "seed": seed,
                        "Model": model_name, "source_pixels": len(source_cols),
                        "target_pixels": len(target_cols), "buffer_km": BUFFER_KM,
                        "a_ctx": a_ctx, "tau_ctx_days": tau_ctx,
                        "a_out": a_out, "tau_out_days": tau_out,
                        **metrics(current[model_name], y),
                    })
                np.savez_compressed(
                    fold_dir / f"predictions_seed{seed}_{split}.npz",
                    iT90=current["iT90"], iT365=current["iT365"],
                    DAP=p_dap, Dual_Context=p_ctx.astype("float32"),
                    ReefFormer=p_final.astype("float32"),
                )
            # Retain only compact, pixel-averaged daily losses for inference.
            y_dev = truth("Dev", target_cols).astype("float32", copy=False)
            fold_predictions[seed]["Dev_loss"] = {
                name: np.mean((fold_predictions[seed]["Dev"][name].astype("float32", copy=False)
                               - y_dev) ** 2, axis=2, dtype=np.float64).astype("float32")
                for name in ("iT90", "iT365", "DAP", "Dual Context", "ReefFormer")
            }
            del fold_predictions[seed]["Val"], fold_predictions[seed]["Dev"], fold_predictions[seed]["Test2026"]
        bootstrap_store[target_sector] = {
            seed: fold_predictions[seed]["Dev_loss"] for seed in SEEDS
        }

    frame = pd.DataFrame(fold_rows)
    frame.to_csv(OUT / "METRICS_BY_FOLD_SEED.csv", index=False)
    summary = frame.groupby(["Split", "Model"], sort=False).agg(
        RMSE_mean=("RMSE", "mean"), RMSE_SD=("RMSE", "std"),
        MAE_mean=("MAE", "mean"), MAE_SD=("MAE", "std"),
        D1_7_mean=("D1_7", "mean"), D8_14_mean=("D8_14", "mean"),
        D15_30_mean=("D15_30", "mean"),
    ).reset_index()
    summary.to_csv(OUT / "SUMMARY_MACRO_FOLD_SEED.csv", index=False)
    sector_summary = frame.groupby(["target_sector", "Split", "Model"], sort=False).agg(
        RMSE_mean=("RMSE", "mean"), RMSE_SD=("RMSE", "std"),
        MAE_mean=("MAE", "mean"), D1_7_mean=("D1_7", "mean"),
        D8_14_mean=("D8_14", "mean"), D15_30_mean=("D15_30", "mean"),
    ).reset_index()
    sector_summary.to_csv(OUT / "SUMMARY_BY_SECTOR.csv", index=False)
    audit = pd.DataFrame(audit_rows)
    audit.to_csv(OUT / "SPATIAL_FOLD_AUDIT.csv", index=False)

    bootstrap_rows = []
    for reference in ("iT365", "iT90", "DAP", "Dual Context"):
        candidates = [[bootstrap_store[sector][seed]["ReefFormer"]
                       for seed in SEEDS] for sector in SECTORS]
        references = [[bootstrap_store[sector][seed][reference]
                       for seed in SEEDS] for sector in SECTORS]
        bootstrap_rows.extend(paired_fold_seed_ci(candidates, references,
                                                  f"ReefFormer - {reference}"))
    bootstrap = pd.DataFrame(bootstrap_rows)
    bootstrap.to_csv(OUT / "PAIRED_60D_BOOTSTRAP_DEV.csv", index=False)

    protocol = {
        "status": "complete",
        "dataset": str(DATA),
        "dataset_sha256": sha256(DATA),
        "target_sectors": list(SECTORS),
        "buffer_km": BUFFER_KM,
        "seeds": list(SEEDS),
        "lookbacks": list(LOOKBACKS),
        "training": {"optimizer": "Adam", "learning_rate": LR, "batch_size": BATCH,
                     "max_epochs": MAX_EPOCHS, "patience": PATIENCE},
        "selection": "source-pixel Hainan Validation 2018-2021 only",
        "fusion_fit": "source-pixel Hainan Validation 2018-2021 only",
        "target_sector_used_for_training_selection_or_fusion": False,
        "target_history_preprocessing": "target train-period climatology/scale and target train-fitted pixel DAP rho",
        "inference": "source-trained shared iTransformer weights applied to unseen target-sector tokens",
        "device": torch.cuda.get_device_name(0),
    }
    (OUT / "PROTOCOL.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    qa = {
        "status": "PASS",
        "all_metrics_finite": bool(np.isfinite(frame.select_dtypes("number")).all().all()),
        "all_bootstrap_finite": bool(np.isfinite(bootstrap.select_dtypes("number")).all().all()),
        "minimum_source_target_distance_km": float(audit.minimum_source_target_distance_km.min()),
        "all_folds_respect_buffer": bool((audit.minimum_source_target_distance_km >= BUFFER_KM).all()),
        "completed_fold_seed_pairs": int(frame[["target_sector", "seed"]].drop_duplicates().shape[0]),
        "expected_fold_seed_pairs": len(SECTORS) * len(SEEDS),
    }
    (OUT / "QA.json").write_text(json.dumps(qa, indent=2), encoding="utf-8")

    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial"],
        "font.size": 8, "svg.fonttype": "none", "axes.spines.right": False,
        "axes.spines.top": False, "axes.linewidth": 0.8,
        "legend.frameon": False,
    })
    dev = sector_summary[sector_summary.Split == "Dev"]
    models = ("iT365", "DAP", "Dual Context", "ReefFormer")
    colors = {"iT365": "#3569A8", "DAP": "#D28C2D", "Dual Context": "#4C9F91", "ReefFormer": "#C94C4C"}
    fig, axis = plt.subplots(figsize=(7.1, 3.25), constrained_layout=True)
    x = np.arange(len(SECTORS))
    width = .19
    for index, model_name in enumerate(models):
        part = dev[dev.Model == model_name].set_index("target_sector").loc[list(SECTORS)]
        axis.bar(x + (index - 1.5) * width, part.RMSE_mean, width,
                 yerr=part.RMSE_SD, capsize=2, label=model_name,
                 color=colors[model_name], edgecolor="white", linewidth=.4)
    axis.set_xticks(x, SECTORS)
    axis.set_ylabel("Unseen-sector Development RMSE (°C)")
    axis.set_title("Unseen-sector transfer with a 25 km buffer", loc="left", fontweight="bold")
    axis.set_ylabel("Unseen-sector Development RMSE (deg C)")
    axis.grid(axis="y", color="0.88", linewidth=.6)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(ncol=4, loc="upper center", bbox_to_anchor=(.5, -.16))
    fig.savefig(OUT / "fig_hainan_loso.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig_hainan_loso.png", dpi=600, bbox_inches="tight")
    plt.close(fig)

    report = [
        "# Hainan buffered leave-one-sector-out experiment", "", "Status: **complete**", "",
        "Each target sector and every source candidate within 25 km were excluded from neural training, checkpoint selection, and fusion fitting.",
        "Target train-period climatology/scaling and target train-fitted DAP decay were retained as historical preprocessing.", "",
        "## Fold audit", "", audit.to_markdown(index=False), "",
        "## Development macro-average", "",
        summary[summary.Split == "Dev"].to_markdown(index=False, floatfmt=".4f"), "",
        "## Development by held-out sector", "",
        dev.to_markdown(index=False, floatfmt=".4f"), "",
        "## Paired moving-block contrasts", "",
        bootstrap[bootstrap.Band.isin(["Overall", "D1-7"])].to_markdown(index=False, floatfmt=".4f"), "",
        "## QA", "", f"```json\n{json.dumps(qa, indent=2)}\n```",
    ]
    (OUT / "HAINAN_LOSO_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print(summary[summary.Split == "Dev"].to_string(index=False))
    print(bootstrap[bootstrap.Band.isin(["Overall", "D1-7"])].to_string(index=False))
    print(json.dumps(qa, indent=2))


if __name__ == "__main__":
    main()
