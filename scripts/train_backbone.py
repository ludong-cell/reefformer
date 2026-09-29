"""Train one archived-protocol iTransformer backbone from processed data."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from reefformer.itransformer import ITransformer


os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path(__file__).resolve().parents[1]
HORIZON = 30
DOMAINS = {
    "hainan_coraltemp": ("hainan_coraltemp_daily.npz", "ssta_all_c"),
    "gbr_coraltemp": ("gbr_coraltemp_daily.npz", "ssta_c"),
    "hainan_ostia": ("hainan_ostia_daily.npz", "ssta_all_c"),
}
BASE_CONFIG = {
    "output_attention": False,
    "use_norm": True,
    "d_model": 32,
    "embed": "timeF",
    "freq": "d",
    "dropout": 0.1,
    "class_strategy": "projection",
    "factor": 1,
    "n_heads": 4,
    "d_ff": 64,
    "e_layers": 2,
    "activation": "gelu",
    "pred_len": HORIZON,
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def choose_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return requested


def limit(ids: np.ndarray, count: int | None) -> np.ndarray:
    return ids if count is None else ids[:count]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", choices=sorted(DOMAINS), required=True)
    parser.add_argument("--lookback", choices=(90, 365), type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--max-epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--limit-train-origins", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--limit-val-origins", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()

    device = choose_device(args.device)
    data_name, anomaly_key = DOMAINS[args.domain]
    data_path = ROOT / "data" / "processed" / data_name
    output = args.output or (
        ROOT / "outputs" / "training" / args.domain / f"iT_L{args.lookback}_seed{args.seed}"
    )
    output.mkdir(parents=True, exist_ok=True)

    with np.load(data_path, allow_pickle=False) as archive:
        anomalies = archive[anomaly_key].astype(np.float32)
        scale = float(archive["ssta_scale"])
        train_ids = limit(archive["Train"].astype(np.int64), args.limit_train_origins)
        val_ids = limit(archive["Val"].astype(np.int64), args.limit_val_origins)
    if not len(train_ids) or not len(val_ids):
        raise ValueError("Training and validation origin sets must be non-empty")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(4)

    config = dict(BASE_CONFIG, seq_len=args.lookback)
    model = ITransformer(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    normalized = torch.as_tensor(anomalies / scale, device=device)
    history = torch.arange(-args.lookback + 1, 1, device=device)
    leads = torch.arange(1, HORIZON + 1, device=device)
    shuffle_generator = torch.Generator().manual_seed(args.seed)

    def batch(ids: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        origins = torch.as_tensor(ids, device=device)
        return (
            normalized[origins[:, None] + history],
            normalized[origins[:, None] + leads],
        )

    def validation_mse() -> float:
        model.eval()
        total = 0.0
        count = 0
        with torch.no_grad():
            for start in range(0, len(val_ids), args.batch_size):
                features, target = batch(val_ids[start : start + args.batch_size])
                squared_error = F.mse_loss(model(features), target, reduction="sum")
                total += float(squared_error)
                count += target.numel()
        return total / count

    started = time.perf_counter()
    best_loss = validation_mse()
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    bad_epochs = 0
    log = [{"epoch": 0, "train_mse": None, "validation_rmse_c": np.sqrt(best_loss) * scale}]
    for epoch in range(1, args.max_epochs + 1):
        model.train()
        losses = []
        order = torch.randperm(len(train_ids), generator=shuffle_generator).numpy()
        for start in range(0, len(order), args.batch_size):
            features, target = batch(train_ids[order[start : start + args.batch_size]])
            optimizer.zero_grad(set_to_none=True)
            loss = F.mse_loss(model(features), target)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        current_loss = validation_mse()
        log.append(
            {
                "epoch": epoch,
                "train_mse": float(np.mean(losses)),
                "validation_rmse_c": float(np.sqrt(current_loss) * scale),
            }
        )
        pd.DataFrame(log).to_csv(output / "training_log.csv", index=False)
        print(json.dumps(log[-1]), flush=True)
        if current_loss < best_loss:
            best_loss = current_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
        if bad_epochs >= args.patience:
            break

    checkpoint = {
        "state_dict": best_state,
        "seed": args.seed,
        "lookback": args.lookback,
        "best_epoch": best_epoch,
        "config": config,
        "dataset_sha256": file_sha256(data_path),
    }
    torch.save(checkpoint, output / "checkpoint.pt")
    run = {
        "domain": args.domain,
        "dataset": data_path.relative_to(ROOT).as_posix(),
        "dataset_sha256": checkpoint["dataset_sha256"],
        "device": device,
        "seed": args.seed,
        "lookback": args.lookback,
        "horizon": HORIZON,
        "optimizer": "Adam",
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "best_epoch": best_epoch,
        "best_validation_rmse_c": float(np.sqrt(best_loss) * scale),
        "train_origins": len(train_ids),
        "validation_origins": len(val_ids),
        "elapsed_seconds": time.perf_counter() - started,
        "torch_version": torch.__version__,
        "limited_smoke_run": bool(args.limit_train_origins or args.limit_val_origins),
    }
    (output / "run.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
    print(json.dumps(run, indent=2), flush=True)


if __name__ == "__main__":
    main()
