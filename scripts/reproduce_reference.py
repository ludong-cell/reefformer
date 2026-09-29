"""Reproduce the principal archived ReefFormer metrics from compact artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from reefformer import (
    apply_reef_former,
    damped_anomaly_persistence,
    fit_reef_former,
    lead_weights,
)
from reefformer.itransformer import ITransformer
from reefformer.metrics import score


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (42, 43, 44)
LOOKBACKS = (90, 365)
HORIZON = 30

DOMAINS = {
    "hainan_coraltemp": {
        "data": ROOT / "data/processed/hainan_coraltemp_daily.npz",
        "checkpoints": ROOT / "checkpoints/hainan_coraltemp",
        "ssta_key": "ssta_all_c",
        "reference": ROOT / "reference_results/hainan_coraltemp/CORE_METRICS_SUMMARY.csv",
    },
    "gbr_coraltemp": {
        "data": ROOT / "data/processed/gbr_coraltemp_daily.npz",
        "checkpoints": ROOT / "checkpoints/gbr_coraltemp",
        "ssta_key": "ssta_c",
        "reference": ROOT / "reference_results/gbr_coraltemp/SUMMARY_MEAN_SD.csv",
    },
    "hainan_ostia": {
        "data": ROOT / "data/processed/hainan_ostia_daily.npz",
        "checkpoints": ROOT / "checkpoints/hainan_ostia",
        "ssta_key": "ssta_all_c",
        "reference": ROOT / "reference_results/hainan_ostia/SUMMARY_MEAN_SD.csv",
    },
}


def choose_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return requested


def load_data(config: dict) -> dict:
    with np.load(config["data"], allow_pickle=False) as archive:
        return {
            "ssta": archive[config["ssta_key"]].astype(np.float32),
            "scale": float(archive["ssta_scale"]),
            "rho": archive["dap_rho_train"].astype(np.float64),
            "ids": {name: archive[name].astype(np.int64) for name in ("Val", "Dev", "Test2026")},
        }


def truth(data: dict, split: str) -> np.ndarray:
    leads = np.arange(1, HORIZON + 1)
    ids = data["ids"][split]
    return data["ssta"][ids[:, None] + leads[None, :]]


def persistence(data: dict, split: str) -> np.ndarray:
    ids = data["ids"][split]
    return damped_anomaly_persistence(data["ssta"][ids], data["rho"], HORIZON).astype(np.float32)


def infer(
    data: dict,
    checkpoint_path: Path,
    split: str,
    device: str,
    batch_size: int,
) -> np.ndarray:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = ITransformer(checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    lookback = int(checkpoint["lookback"])
    history = torch.arange(-lookback + 1, 1, device=device)
    normalized = torch.as_tensor(data["ssta"] / data["scale"], device=device)
    ids = data["ids"][split]
    pieces = []
    with torch.no_grad():
        for start in range(0, len(ids), batch_size):
            current = torch.as_tensor(ids[start : start + batch_size], device=device)
            features = normalized[current[:, None] + history]
            pieces.append(model(features).cpu().numpy())
    prediction = np.concatenate(pieces, axis=0) * data["scale"]
    del model, checkpoint, normalized
    if device == "cuda":
        torch.cuda.empty_cache()
    return prediction.astype(np.float32)


def normalize_reference(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path).rename(
        columns={
            "Split": "split",
            "RMSE_std": "RMSE_SD",
            "MAE_std": "MAE_SD",
            "D1_7_mean": "D1_7",
            "D8_14_mean": "D8_14",
            "D15_30_mean": "D15_30",
        }
    )
    return frame


def verify(summary: pd.DataFrame, reference_path: Path, tolerance: float = 1e-4) -> dict:
    reference = normalize_reference(reference_path)
    keys = ["split", "Model"]
    metrics = ["RMSE_mean", "MAE_mean", "D1_7", "D8_14", "D15_30"]
    available = reference[reference["Model"].isin(summary["Model"])][keys + metrics]
    merged = summary.merge(available, on=keys, suffixes=("_new", "_reference"))
    differences = {}
    for metric in metrics:
        differences[metric] = float(
            np.max(np.abs(merged[f"{metric}_new"] - merged[f"{metric}_reference"]))
        )
    return {
        "rows_compared": int(len(merged)),
        "tolerance_c": tolerance,
        "maximum_absolute_differences_c": differences,
        "status": "PASS" if all(value <= tolerance for value in differences.values()) else "FAIL",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", choices=sorted(DOMAINS), required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    config = DOMAINS[args.domain]
    device = choose_device(args.device)
    data = load_data(config)
    output = ROOT / "outputs" / args.domain
    output.mkdir(parents=True, exist_ok=True)

    predictions = {split: {} for split in data["ids"]}
    rows = []
    parameter_rows = []
    for seed in SEEDS:
        for lookback in LOOKBACKS:
            for split in data["ids"]:
                path = config["checkpoints"] / f"iT_L{lookback}_seed{seed}.pt"
                predictions[split][(seed, lookback)] = infer(
                    data, path, split, device, args.batch_size
                )

        val_truth = truth(data, "Val")
        val_dap = persistence(data, "Val")
        parameters = fit_reef_former(
            predictions["Val"][(seed, 90)],
            predictions["Val"][(seed, 365)],
            val_dap,
            val_truth,
        )
        q = lead_weights(parameters.a_context, parameters.tau_context_days, HORIZON)
        w = lead_weights(parameters.a_persistence, parameters.tau_persistence_days, HORIZON)
        parameter_rows.append(
            {
                "seed": seed,
                **parameters.__dict__,
                "q_day1": q[0],
                "q_day30": q[-1],
                "w_day1": w[0],
                "w_day30": w[-1],
            }
        )

        for split in data["ids"]:
            target = truth(data, split)
            short = predictions[split][(seed, 90)]
            long = predictions[split][(seed, 365)]
            dap = persistence(data, split)
            context = q[None, :, None] * short + (1.0 - q[None, :, None]) * long
            reef = apply_reef_former(short, long, dap, parameters)
            for model_name, forecast in {
                "iT90": short,
                "iT365": long,
                "DAP": dap,
                "Dual Context": context,
                "ReefFormer": reef,
            }.items():
                rows.append({"split": split, "seed": seed, "Model": model_name, **score(forecast, target)})

    per_seed = pd.DataFrame(rows)
    summary = (
        per_seed.groupby(["split", "Model"], sort=False)
        .agg(
            RMSE_mean=("RMSE", "mean"),
            RMSE_SD=("RMSE", "std"),
            MAE_mean=("MAE", "mean"),
            MAE_SD=("MAE", "std"),
            D1_7=("D1_7", "mean"),
            D8_14=("D8_14", "mean"),
            D15_30=("D15_30", "mean"),
        )
        .reset_index()
    )
    audit = verify(summary, config["reference"])
    per_seed.to_csv(output / "metrics_by_seed.csv", index=False)
    summary.to_csv(output / "metrics_summary.csv", index=False)
    pd.DataFrame(parameter_rows).to_csv(output / "fusion_parameters.csv", index=False)
    (output / "verification.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    print(json.dumps(audit, indent=2))
    if audit["status"] != "PASS":
        raise SystemExit("Reference verification failed")


if __name__ == "__main__":
    main()
