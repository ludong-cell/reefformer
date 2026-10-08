"""Train and evaluate the three-seed Hainan OSTIA product replication."""
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
from itransformer_feature_pilot import Model, CONFIG as BASE_CONFIG


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data/ostia_hainan_v1/dataset_daily.npz"
OUT = ROOT / "runs/ostia_hainan_v1"
MODELS = OUT / "models"
LOOKBACKS = (90, 365)
SEEDS = (42, 43, 44)
H = 30
BATCH = 32
LR = 0.001
MAX_EPOCHS = 40
PATIENCE = 8
MODEL_NAMES = ("iT90", "iT365", "DAP", "Dual Context", "ReefFormer")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def score(prediction, truth):
    error = prediction.astype("float32", copy=False) - truth.astype("float32", copy=False)
    return {
        "RMSE": float(np.sqrt(np.mean(error * error, dtype=np.float64))),
        "MAE": float(np.mean(np.abs(error), dtype=np.float64)),
        "D1_7": float(np.sqrt(np.mean(error[:, :7] ** 2, dtype=np.float64))),
        "D8_14": float(np.sqrt(np.mean(error[:, 7:14] ** 2, dtype=np.float64))),
        "D15_30": float(np.sqrt(np.mean(error[:, 14:] ** 2, dtype=np.float64))),
    }


def block_indices(n, repetitions=10000, block=60, seed=420925):
    rng = np.random.default_rng(seed)
    blocks = int(np.ceil(n / block))
    offsets = np.arange(block)
    for start in range(0, repetitions, 250):
        count = min(250, repetitions - start)
        origins = rng.integers(0, n - block + 1, size=(count, blocks))
        yield (origins[:, :, None] + offsets).reshape(count, -1)[:, :n]


def paired_ci(candidate_loss, reference_loss, contrast):
    bands = {"Overall": np.arange(H), "D1-7": np.arange(7),
             "D8-14": np.arange(7, 14), "D15-30": np.arange(14, H)}
    rows = []
    for name, leads in bands.items():
        cand = candidate_loss[:, :, leads].mean(axis=2)
        ref = reference_loss[:, :, leads].mean(axis=2)
        values = []
        for indices in block_indices(cand.shape[1]):
            left = np.sqrt(cand[:, indices].mean(axis=2))
            right = np.sqrt(ref[:, indices].mean(axis=2))
            values.append((left - right).mean(axis=0))
        values = np.concatenate(values)
        point = np.sqrt(cand.mean(axis=1)) - np.sqrt(ref.mean(axis=1))
        rows.append({
            "Contrast": contrast, "Band": name, "Delta_RMSE": float(point.mean()),
            "CI_low": float(np.quantile(values, .025)),
            "CI_high": float(np.quantile(values, .975)),
            "block_days": 60, "replicates": 10000,
        })
    return rows


def main():
    if not DATA.exists():
        raise FileNotFoundError("Run build_ostia_hainan_v1.py first")
    OUT.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(exist_ok=True)
    with np.load(DATA, allow_pickle=False) as dataset:
        ids = {name: dataset[name].astype("int64") for name in ("Train", "Val", "Dev", "Test2026")}
        dates = pd.DatetimeIndex(dataset["dates_all"])
        ssta = dataset["ssta_all_c"].astype("float32")
        scale = float(dataset["ssta_scale"])
        rho = dataset["dap_rho_train"].astype("float32")
        pixels = len(dataset["pixel_ids"])

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = "cuda"
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    normalized = torch.as_tensor(ssta / scale, device=device)
    lead_t = torch.arange(1, H + 1, device=device)
    lead = np.arange(1, H + 1)
    config = {key: BASE_CONFIG[key] for key in (
        "output_attention", "use_norm", "d_model", "embed", "freq", "dropout",
        "class_strategy", "factor", "n_heads", "d_ff", "e_layers", "activation")}
    config["pred_len"] = H

    history_rows = []
    for seed in SEEDS:
        for lookback in LOOKBACKS:
            marker = MODELS / f"COMPLETE_L{lookback}_seed{seed}.json"
            checkpoint_path = MODELS / f"iT_L{lookback}_seed{seed}.pt"
            if marker.exists() and checkpoint_path.exists():
                print(f"SKIP OSTIA L={lookback} seed={seed}", flush=True)
                continue
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)
            cfg = SimpleNamespace(**dict(config, seq_len=lookback))
            model = Model(cfg).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=LR)
            history = torch.arange(-lookback + 1, 1, device=device)
            generator = torch.Generator().manual_seed(seed)

            def batch(origin_ids):
                current = torch.as_tensor(origin_ids, device=device)
                return (normalized[current[:, None] + history],
                        normalized[current[:, None] + lead_t])

            def val_mse():
                model.eval()
                total, count = 0.0, 0
                with torch.no_grad():
                    for start in range(0, len(ids["Val"]), BATCH):
                        x, y = batch(ids["Val"][start:start + BATCH])
                        difference = model(x, None, None, None) - y
                        total += float(torch.sum(difference * difference).item())
                        count += difference.numel()
                return total / count

            best = val_mse()
            best_epoch, bad = 0, 0
            best_state = copy.deepcopy(model.state_dict())
            log = [{"epoch": 0, "train_loss": np.nan, "Val_RMSE": np.sqrt(best) * scale}]
            started = time.perf_counter()
            completed = 0
            for epoch in range(1, MAX_EPOCHS + 1):
                model.train()
                losses = []
                order = torch.randperm(len(ids["Train"]), generator=generator).numpy()
                for start in range(0, len(order), BATCH):
                    x, y = batch(ids["Train"][order[start:start + BATCH]])
                    optimizer.zero_grad(set_to_none=True)
                    loss = F.mse_loss(model(x, None, None, None), y)
                    loss.backward()
                    optimizer.step()
                    losses.append(loss.item())
                current = val_mse()
                log.append({"epoch": epoch, "train_loss": float(np.mean(losses)),
                            "Val_RMSE": np.sqrt(current) * scale})
                pd.DataFrame(log).to_csv(MODELS / f"training_L{lookback}_seed{seed}.csv", index=False)
                if current < best:
                    best, best_epoch = current, epoch
                    best_state = copy.deepcopy(model.state_dict())
                    bad = 0
                else:
                    bad += 1
                completed = epoch
                print(f"OSTIA seed={seed} L={lookback} epoch={epoch} val={np.sqrt(current)*scale:.6f} best={best_epoch}", flush=True)
                if bad >= PATIENCE:
                    break
            model.load_state_dict(best_state)

            def infer(origin_ids):
                result = np.empty((len(origin_ids), H, pixels), dtype="float32")
                model.eval()
                with torch.no_grad():
                    for start in range(0, len(origin_ids), BATCH):
                        x, _ = batch(origin_ids[start:start + BATCH])
                        stop = min(start + BATCH, len(origin_ids))
                        result[start:stop] = model(x, None, None, None).cpu().numpy() * scale
                return result

            for split in ("Val", "Dev", "Test2026"):
                np.save(MODELS / f"iT_L{lookback}_seed{seed}_{split}.npy", infer(ids[split]))
            torch.save({"state_dict": best_state, "seed": seed, "lookback": lookback,
                        "best_epoch": best_epoch, "config": dict(config, seq_len=lookback),
                        "dataset_sha256": sha256(DATA)}, checkpoint_path)
            marker.write_text(json.dumps({"complete": True, "best_epoch": best_epoch}, indent=2), encoding="utf-8")
            history_rows.append({"seed": seed, "Lookback": lookback, "Best_epoch": best_epoch,
                                 "Epochs_run": completed, "Training_time_s": time.perf_counter() - started})
            pd.DataFrame(history_rows).to_csv(MODELS / "RUN_SUMMARY.csv", index=False)
            del model, optimizer, best_state
            torch.cuda.empty_cache()

    def truth(split):
        return ssta[ids[split][:, None] + lead[None, :]]

    def dap(split):
        return (ssta[ids[split], None, :] * rho[None, None, :] ** lead[None, :, None]).astype("float32")

    rows, parameter_rows = [], []
    loss_store = {model: [] for model in MODEL_NAMES}
    for seed in SEEDS:
        predictions = {split: {
            "iT90": np.load(MODELS / f"iT_L90_seed{seed}_{split}.npy"),
            "iT365": np.load(MODELS / f"iT_L365_seed{seed}_{split}.npy"),
            "DAP": dap(split),
        } for split in ("Val", "Dev", "Test2026")}
        a_ctx, tau_ctx, q = fit_smooth(predictions["Val"]["iT90"], predictions["Val"]["iT365"], truth("Val"))
        val_ctx = q[None, :, None] * predictions["Val"]["iT90"] + (1 - q[None, :, None]) * predictions["Val"]["iT365"]
        a_out, tau_out, w = fit_smooth(predictions["Val"]["DAP"], val_ctx, truth("Val"))
        parameter_rows.append({"seed": seed, "a_ctx": a_ctx, "tau_ctx_days": tau_ctx,
                               "a_out": a_out, "tau_out_days": tau_out,
                               "fit_split": "OSTIA Validation 2018-2021"})
        for split in predictions:
            current = predictions[split]
            current["Dual Context"] = q[None, :, None] * current["iT90"] + (1 - q[None, :, None]) * current["iT365"]
            current["ReefFormer"] = w[None, :, None] * current["DAP"] + (1 - w[None, :, None]) * current["Dual Context"]
            y = truth(split)
            for name in MODEL_NAMES:
                rows.append({"Split": split, "Model": name, "seed": seed, **score(current[name], y)})
                if split == "Dev":
                    loss_store[name].append(np.mean((current[name] - y) ** 2, axis=2, dtype=np.float64).astype("float32"))
            np.save(MODELS / f"ReefFormer_seed{seed}_{split}.npy", current["ReefFormer"].astype("float32"))
        del predictions

    metrics = pd.DataFrame(rows)
    summary = metrics.groupby(["Split", "Model"], sort=False).agg(
        RMSE_mean=("RMSE", "mean"), RMSE_SD=("RMSE", "std"),
        MAE_mean=("MAE", "mean"), MAE_SD=("MAE", "std"),
        D1_7_mean=("D1_7", "mean"), D8_14_mean=("D8_14", "mean"),
        D15_30_mean=("D15_30", "mean"),
    ).reset_index()
    metrics.to_csv(OUT / "METRICS_BY_SEED.csv", index=False)
    summary.to_csv(OUT / "SUMMARY_MEAN_SD.csv", index=False)
    pd.DataFrame(parameter_rows).to_csv(OUT / "FUSION_PARAMETERS_BY_SEED.csv", index=False)
    bootstrap_rows = []
    candidate = np.stack(loss_store["ReefFormer"])
    for reference in ("iT365", "iT90", "DAP", "Dual Context"):
        bootstrap_rows.extend(paired_ci(candidate, np.stack(loss_store[reference]),
                                        f"ReefFormer - {reference}"))
    bootstrap = pd.DataFrame(bootstrap_rows)
    bootstrap.to_csv(OUT / "PAIRED_60D_BOOTSTRAP_DEV.csv", index=False)

    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial"], "font.size": 8,
        "svg.fonttype": "none", "axes.spines.right": False, "axes.spines.top": False,
        "axes.linewidth": 0.8, "legend.frameon": False,
    })
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.7), constrained_layout=True)
    dev = summary[summary.Split == "Dev"].set_index("Model")
    shown = ("iT365", "DAP", "Dual Context", "ReefFormer")
    colors = ("#7297B3", "#D6A35C", "#75A99C", "#C75B5B")
    axes[0].bar(np.arange(len(shown)), [dev.loc[name, "RMSE_mean"] for name in shown],
                yerr=[dev.loc[name, "RMSE_SD"] for name in shown], color=colors,
                edgecolor="white", linewidth=.5, capsize=2)
    axes[0].set_xticks(np.arange(len(shown)), ("iT365", "DAP", "Dual\nContext", "ReefFormer"))
    axes[0].set_ylabel("Development RMSE (deg C)")
    axes[0].set_title("a  OSTIA product replication", loc="left", fontweight="bold")
    axes[0].grid(axis="y", color="0.9", linewidth=.5)
    forest = bootstrap[(bootstrap.Band == "Overall")].iloc[::-1]
    y = np.arange(len(forest))
    axes[1].errorbar(forest.Delta_RMSE, y,
                     xerr=np.vstack([forest.Delta_RMSE - forest.CI_low,
                                     forest.CI_high - forest.Delta_RMSE]),
                     fmt="o", color="#C75B5B", ecolor="#555555", capsize=2, markersize=4)
    axes[1].axvline(0, color="0.25", linewidth=.8)
    axes[1].set_yticks(y, [value.replace("ReefFormer - ", "vs ") for value in forest.Contrast])
    axes[1].set_xlabel("Delta RMSE (deg C; negative favors ReefFormer)")
    axes[1].set_title("b  Paired 60-day block intervals", loc="left", fontweight="bold")
    axes[1].grid(axis="x", color="0.9", linewidth=.5)
    fig.savefig(OUT / "FIG_OSTIA_PRODUCT_REPLICATION.pdf", bbox_inches="tight")
    fig.savefig(OUT / "FIG_OSTIA_PRODUCT_REPLICATION.png", dpi=600, bbox_inches="tight")
    plt.close(fig)

    dev_table = dev.reset_index().to_markdown(index=False, floatfmt=".4f")
    report = f"""# OSTIA second-product replication

Status: **complete**

- Product: Copernicus Marine OSTIA L4 reprocessed SST, 0.05 degree.
- Frozen Hainan locations: {pixels}.
- Seeds: 42, 43, 44.
- Train: 2000-2017; Validation: 2018-2021; Development: 2022-2025.
- Available 2026 origins with complete 30-day targets: {len(ids['Test2026'])}.
- Dataset SHA256: `{sha256(DATA)}`.
- Checkpoint selection and fusion fitting used Validation only.

## Development results

{dev_table}

## Claim boundary

This is a same-domain, second-SST-product replication. It tests sensitivity to the
input product and preprocessing chain, not geographic transfer. Geographic evidence
is supplied separately by the GBR and spatial leave-one-sector-out experiments.
"""
    (OUT / "OSTIA_REPLICATION_REPORT.md").write_text(report, encoding="utf-8")
    qa = {
        "status": "PASS", "dataset_sha256": sha256(DATA), "pixels": pixels,
        "seeds": list(SEEDS), "completed_neural_runs": len(SEEDS) * len(LOOKBACKS),
        "all_metrics_finite": bool(np.isfinite(metrics.select_dtypes("number")).all().all()),
        "all_bootstrap_finite": bool(np.isfinite(bootstrap.select_dtypes("number")).all().all()),
        "dev_used_for_training_selection_or_fusion": False,
        "test2026_origins": len(ids["Test2026"]),
    }
    (OUT / "QA.json").write_text(json.dumps(qa, indent=2), encoding="utf-8")
    print(report)
    print(json.dumps(qa, indent=2))


if __name__ == "__main__":
    main()
