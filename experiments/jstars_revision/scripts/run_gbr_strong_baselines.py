"""GBR port of run_final154_strong_baselines.py: DLinear and PatchTST on the frozen 292-pixel GBR domain.

Identical protocol, hyperparameters, seeds and selection rule; only the dataset, array keys, output folder
and the (absent) sector breakdown differ.

For each model, input context is selected from {90, 180, 365} days by the
mean 2018--2021 Validation RMSE over seeds 42--44.  Dev and Test2026 are
opened only after the selected lookback and checkpoint hashes are frozen.
The script is resumable at completed run boundaries.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "gbr_generalization_v1" / "dataset_daily.npz"
BASE_OUT = ROOT / "runs" / "gbr_generalization_v1" / "strong_baselines"
SEEDS = (42, 43, 44)
LOOKBACKS = (90, 180, 365)
H = 30
BATCH = 32
LR = 1e-3
MAX_EPOCHS = 40
PATIENCE = 8


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def metrics(prediction: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    error = prediction.astype("float64") - truth.astype("float64")
    return {
        "RMSE": float(np.sqrt(np.mean(error ** 2))),
        "MAE": float(np.mean(np.abs(error))),
        "D1_7": float(np.sqrt(np.mean(error[:, :7] ** 2))),
        "D8_14": float(np.sqrt(np.mean(error[:, 7:14] ** 2))),
        "D15_30": float(np.sqrt(np.mean(error[:, 14:] ** 2))),
    }


def load_model(model_name: str, lookback: int, pixels: int):
    if model_name == "LSTM":                       # Zhang et al. (2017)-style SST LSTM, see baselines_lstm.py
        from baselines_lstm import Model as LSTMModel
        config = {"seq_len": lookback, "pred_len": H, "enc_in": pixels, "d_model": 64, "e_layers": 2, "dropout": .1}
        return LSTMModel(SimpleNamespace(**config)), config, "in-house (baselines_lstm.py)"
    if model_name == "DLinear":
        model_file = ROOT / "external" / "DLinear_official" / "models" / "DLinear.py"
        spec = importlib.util.spec_from_file_location("official_dlinear_154", model_file)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        config = {"seq_len": lookback, "pred_len": H, "individual": False, "enc_in": pixels}
        return module.Model(SimpleNamespace(**config)), config, "0c113668a3b88c4c4ee586b8c5ec3e539c4de5a6"

    official = ROOT / "external" / "PatchTST_official" / "PatchTST_supervised"
    if str(official) not in sys.path:
        sys.path.insert(0, str(official))
    from models.PatchTST import Model
    config = {
        "enc_in": pixels, "seq_len": lookback, "pred_len": H,
        "e_layers": 2, "n_heads": 4, "d_model": 32, "d_ff": 64,
        "dropout": .1, "fc_dropout": .1, "head_dropout": 0.,
        "individual": False, "patch_len": 16, "stride": 8,
        "padding_patch": "end", "revin": True, "affine": True,
        "subtract_last": False, "decomposition": False, "kernel_size": 25,
    }
    return Model(SimpleNamespace(**config)), config, "204c21efe0b39603ad6e2ca640ef5896646ab1a9"


def main(model_name: str):
    output = BASE_OUT / model_name.lower()
    output.mkdir(parents=True, exist_ok=True)
    with np.load(DATA, allow_pickle=False) as z:
        train_ids = z["Train"].astype("int64")
        val_ids = z["Val"].astype("int64")
        dev_ids = z["Dev"].astype("int64")
        test_ids = z["Test2026"].astype("int64")
        ssta = z["ssta_c"].astype("float32")
        scale = float(z["ssta_scale"])
        dates = pd.DatetimeIndex(z["dates"])
    pixels = ssta.shape[1]
    assert pixels == 292
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("CUDA is required by the frozen training protocol")
    torch.set_num_threads(4)
    if model_name == "DLinear":
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
    else:
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.benchmark = True
    series = torch.tensor(ssta / scale, device=device)
    lead = torch.arange(1, H + 1, device=device)

    runs_file = output / "LOOKBACK_VALIDATION.csv"
    rows = pd.read_csv(runs_file).to_dict("records") if runs_file.exists() else []
    completed_keys = {(int(row["seed"]), int(row["Lookback"])) for row in rows}
    official_commit = None
    for seed in SEEDS:
        for lookback in LOOKBACKS:
            if (seed, lookback) in completed_keys:
                print(f"SKIP completed {model_name} seed={seed} L={lookback}", flush=True)
                continue
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)
            generator = torch.Generator().manual_seed(seed)
            model, config, official_commit = load_model(model_name, lookback, pixels)
            model = model.to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=LR)
            amp = model_name == "PatchTST"
            scaler = torch.amp.GradScaler("cuda", enabled=amp)
            history = torch.arange(-lookback + 1, 1, device=device)

            def batch(ids):
                ids_t = torch.as_tensor(ids, device=device)
                return series[ids_t[:, None] + history], series[ids_t[:, None] + lead]

            def infer(ids):
                model.eval()
                pieces = []
                with torch.no_grad():
                    for start in range(0, len(ids), BATCH):
                        features, _ = batch(ids[start:start + BATCH])
                        with torch.amp.autocast("cuda", enabled=amp):
                            prediction = model(features)
                        pieces.append(prediction.float().cpu().numpy())
                return np.concatenate(pieces, axis=0)

            truth_val = ssta[val_ids[:, None] + np.arange(1, H + 1)[None, :]]
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            started = time.perf_counter()
            initial = infer(val_ids)
            best_mse = float(np.mean((initial.astype("float64") - truth_val / scale) ** 2))
            best_epoch = 0
            best_state = copy.deepcopy(model.state_dict())
            bad = 0
            completed_epoch = 0
            log = [{"epoch": 0, "train_loss": None, "Val_RMSE": np.sqrt(best_mse) * scale}]
            for epoch in range(1, MAX_EPOCHS + 1):
                model.train()
                losses = []
                order = torch.randperm(len(train_ids), generator=generator).numpy()
                for start in range(0, len(order), BATCH):
                    features, target = batch(train_ids[order[start:start + BATCH]])
                    optimizer.zero_grad(set_to_none=True)
                    with torch.amp.autocast("cuda", enabled=amp):
                        loss = F.mse_loss(model(features), target)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    losses.append(loss.item())
                val_prediction = infer(val_ids)
                mse = float(np.mean((val_prediction.astype("float64") - truth_val / scale) ** 2))
                completed_epoch = epoch
                log.append({"epoch": epoch, "train_loss": float(np.mean(losses)),
                            "Val_RMSE": np.sqrt(mse) * scale})
                if mse < best_mse:
                    best_mse = mse
                    best_epoch = epoch
                    best_state = copy.deepcopy(model.state_dict())
                    bad = 0
                else:
                    bad += 1
                pd.DataFrame(log).to_csv(
                    output / f"{model_name}_L{lookback}_seed{seed}_training.csv", index=False
                )
                print(model_name, f"seed={seed}", f"L={lookback}", f"epoch={epoch}",
                      f"val={np.sqrt(mse) * scale:.6f}", f"best={best_epoch}", flush=True)
                if bad >= PATIENCE:
                    break
            model.load_state_dict(best_state)
            val_prediction = (infer(val_ids) * scale).astype("float32")
            checkpoint = {
                "state_dict": best_state, "model": model_name, "seed": seed,
                "lookback": lookback, "best_epoch": best_epoch, "config": config,
                "dataset_sha256": sha256(DATA), "pixels": pixels,
            }
            torch.save(checkpoint, output / f"{model_name}_L{lookback}_seed{seed}.pt")
            np.save(output / f"{model_name}_L{lookback}_seed{seed}_Val.npy", val_prediction)
            rows.append({
                "seed": seed, "Lookback": lookback, **metrics(val_prediction, truth_val),
                "Best_epoch": best_epoch, "Epochs_run": completed_epoch,
                "Training_time_s": time.perf_counter() - started,
                "Peak_GPU_MiB": torch.cuda.max_memory_allocated() / 1024 ** 2,
                "Params": sum(parameter.numel() for parameter in model.parameters()),
            })
            pd.DataFrame(rows).sort_values(["seed", "Lookback"]).to_csv(runs_file, index=False)
            del model, optimizer, best_state
            torch.cuda.empty_cache()

    frame = pd.DataFrame(rows)
    summary = frame.groupby("Lookback").agg(
        Val_RMSE_mean=("RMSE", "mean"), Val_RMSE_SD=("RMSE", "std"),
        Val_MAE_mean=("MAE", "mean"), Params=("Params", "first"),
        Training_time_mean_s=("Training_time_s", "mean"),
        Peak_GPU_MiB=("Peak_GPU_MiB", "max"),
    ).reset_index()
    minimum = summary.Val_RMSE_mean.min()
    selected = int(summary[summary.Val_RMSE_mean <= minimum * 1.005].Lookback.min())
    selection = {
        "rule": "lowest mean 2018-2021 Validation RMSE; choose shorter context when within 0.5%",
        "best_lookback": selected, "minimum_val_rmse": float(minimum),
        "within_0.5pct_threshold": float(minimum * 1.005), "dev_used": False,
        "mean_val_rmse_by_lookback": {
            str(int(row.Lookback)): float(row.Val_RMSE_mean) for row in summary.itertuples()
        },
    }
    summary.to_csv(output / "LOOKBACK_SUMMARY.csv", index=False)
    (output / "SELECTION.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
    if official_commit is None:
        _, _, official_commit = load_model(model_name, selected, pixels)
    freeze = {
        "model": model_name, "official_commit": official_commit,
        "dataset_sha256": sha256(DATA), "pixels": pixels,
        "selected_lookback": selected, "selection_split": "2018-2021 Validation only",
        "checkpoints": {
            str(seed): sha256(output / f"{model_name}_L{selected}_seed{seed}.pt") for seed in SEEDS
        },
    }
    (output / "PRE_DEV_FREEZE_MANIFEST.json").write_text(
        json.dumps(freeze, indent=2), encoding="utf-8"
    )
    print("FROZEN BEFORE DEV/2026", json.dumps(selection), flush=True)

    split_rows = []
    day_rows = []
    year_rows = []
    region_rows = []
    for seed in SEEDS:
        checkpoint = torch.load(
            output / f"{model_name}_L{selected}_seed{seed}.pt",
            weights_only=False, map_location="cpu"
        )
        model, _, _ = load_model(model_name, selected, pixels)
        model = model.to(device)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        amp = model_name == "PatchTST"
        history = torch.arange(-selected + 1, 1, device=device)
        for split, ids in (("Dev", dev_ids), ("Test2026", test_ids)):
            pieces = []
            with torch.no_grad():
                for start in range(0, len(ids), BATCH):
                    ids_t = torch.as_tensor(ids[start:start + BATCH], device=device)
                    with torch.amp.autocast("cuda", enabled=amp):
                        prediction = model(series[ids_t[:, None] + history])
                    pieces.append(prediction.float().cpu().numpy())
            prediction = (np.concatenate(pieces) * scale).astype("float32")
            truth = ssta[ids[:, None] + np.arange(1, H + 1)[None, :]]
            np.save(output / f"{model_name}_seed{seed}_{split}.npy", prediction)
            split_rows.append({"Model": model_name, "split": split, "seed": seed,
                               "Best_L": selected, "Params": sum(p.numel() for p in model.parameters()),
                               **metrics(prediction, truth)})
            error = prediction.astype("float64") - truth.astype("float64")
            for h in range(H):
                day_rows.append({"Model": model_name, "split": split, "seed": seed, "lead": h + 1,
                                 "RMSE": float(np.sqrt(np.mean(error[:, h] ** 2))),
                                 "MAE": float(np.mean(np.abs(error[:, h])))})
            target_years = dates[ids + 1].year
            for year in np.unique(target_years):
                subset = error[target_years == year]
                year_rows.append({"Model": model_name, "split": split, "seed": seed, "Year": int(year),
                                  "RMSE": float(np.sqrt(np.mean(subset ** 2))),
                                  "MAE": float(np.mean(np.abs(subset)))})
            for region in ():
                subset = error[:, :, sectors == region]
                region_rows.append({"Model": model_name, "split": split, "seed": seed, "Region": region,
                                    "RMSE": float(np.sqrt(np.mean(subset ** 2))),
                                    "MAE": float(np.mean(np.abs(subset)))})
        del model
        torch.cuda.empty_cache()
    pd.DataFrame(split_rows).to_csv(output / "FROZEN_SPLIT_METRICS.csv", index=False)
    pd.DataFrame(day_rows).to_csv(output / "FROZEN_DAY1_30.csv", index=False)
    pd.DataFrame(year_rows).to_csv(output / "FROZEN_YEARWISE.csv", index=False)
    pd.DataFrame(region_rows).to_csv(output / "FROZEN_REGIONWISE.csv", index=False)
    protocol = {
        **freeze, "seeds": list(SEEDS), "lookbacks": list(LOOKBACKS),
        "optimizer": "Adam", "learning_rate": LR, "batch_size": BATCH,
        "max_epochs": MAX_EPOCHS, "patience": PATIENCE,
        "PatchTST_AMP": model_name == "PatchTST",
        "Dev_and_2026": "forward only after Validation selection and manifest freeze",
    }
    (output / "PROTOCOL.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    print(summary.to_string(index=False), flush=True)
    print(pd.DataFrame(split_rows).to_string(index=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=("DLinear", "PatchTST", "LSTM"))
    args = parser.parse_args()
    main(args.model)
