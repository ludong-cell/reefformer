"""Build the frozen Hainan OSTIA second-product replication dataset.

The 154 locations are inherited unchanged from the frozen CoralTemp domain.
Official Copernicus Marine OSTIA L4 reprocessed SST is downloaded in annual,
restartable blocks. No spatial or temporal interpolation is permitted.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import concurrent.futures
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

import download_full_region as hainan_download
from train_backbone import calendar_index


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "data/frozen_154_ssta_v1/dataset_daily.npz"
OUT = ROOT / "data/ostia_hainan_v1"
RAW = OUT / "raw"
DATASET = OUT / "dataset_daily.npz"
DATASET_ID = "METOFFICE-GLO-SST-L4-REP-OBS-SST"
DATASET_VERSION = "202003"
START = pd.Timestamp("2000-01-01")
END = pd.Timestamp("2026-03-31")
TRAIN_END = pd.Timestamp("2017-12-31")
H = 30
LMAX = 365
SPLITS = {
    "Train": ("2000-01-01", "2017-12-31"),
    "Val": ("2018-01-01", "2021-12-31"),
    "Dev": ("2022-01-01", "2025-12-31"),
}
BOUNDS = (108.30, 111.30, 17.80, 20.30)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def open_nc(path: Path):
    return hainan_download.reader.open_nc(str(path))


def copernicus_executable() -> str:
    executable = shutil.which("copernicusmarine")
    if executable:
        return executable
    candidate = Path(r"D:\ancomda\Scripts\copernicusmarine.exe")
    if candidate.exists():
        return str(candidate)
    raise RuntimeError("copernicusmarine CLI is not installed")


def download_year(year: int) -> dict:
    first = max(START, pd.Timestamp(year, 1, 1))
    last = min(END, pd.Timestamp(year, 12, 31))
    target = RAW / f"ostia_hainan_{year}.nc"
    manifest = target.with_suffix(".json")
    if target.exists() and manifest.exists():
        record = json.loads(manifest.read_text(encoding="utf-8"))
        if record.get("sha256") == sha256(target):
            print(f"SKIP OSTIA {year}", flush=True)
            return record
    command = [
        copernicus_executable(), "subset", "-i", DATASET_ID,
        "--dataset-version", DATASET_VERSION, "-v", "analysed_sst",
        "-t", str(first.date()), "-T", str(last.date()),
        "-x", str(BOUNDS[0]), "-X", str(BOUNDS[1]),
        "-y", str(BOUNDS[2]), "-Y", str(BOUNDS[3]),
        "-s", "geoseries", "-o", str(RAW), "-f", target.name,
        "--overwrite", "--disable-progress-bar", "--log-level", "ERROR",
        "--netcdf-compression-level", "4",
    ]
    completed = subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
    dataset = open_nc(target)
    try:
        dates = pd.DatetimeIndex(dataset.time.values).normalize()
        expected = pd.date_range(first, last, freq="D")
        if not dates.equals(expected):
            raise RuntimeError(f"OSTIA {year} time axis is not complete and consecutive")
        if dataset.analysed_sst.attrs.get("units", "").lower() != "kelvin":
            raise RuntimeError("OSTIA analysed_sst unit is not kelvin")
    finally:
        dataset.close()
    record = {
        "year": year, "start": str(first.date()), "end": str(last.date()),
        "dataset_id": DATASET_ID, "dataset_version": DATASET_VERSION,
        "service": "arco-geo-series", "sha256": sha256(target),
        "bytes": target.stat().st_size, "cli_response": completed.stdout.strip(),
    }
    manifest.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"OSTIA {year}: {target.stat().st_size / 1024**2:.2f} MiB", flush=True)
    return record


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.deg2rad, (lat1, lon1, lat2, lon2))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    value = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 6371.0088 * 2 * np.arctan2(np.sqrt(value), np.sqrt(1 - value))


def build_dataset(raw_records: list[dict]) -> dict:
    with np.load(SOURCE, allow_pickle=False) as source:
        lat = source["pixel_latitude"].astype("float32")
        lon = source["pixel_longitude"].astype("float32")
        pixel_ids = source["pixel_ids"].astype(str)
        sectors = source["sectors"].astype(str)

    dates_parts, sst_parts = [], []
    mapped_lat = mapped_lon = None
    for year in range(START.year, END.year + 1):
        dataset = open_nc(RAW / f"ostia_hainan_{year}.nc")
        try:
            query = dataset.analysed_sst.sel(
                latitude=xr.DataArray(lat, dims="pixel"),
                longitude=xr.DataArray(lon, dims="pixel"), method="nearest",
            )
            if mapped_lat is None:
                mapped_lat = query.latitude.values.astype("float64")
                mapped_lon = query.longitude.values.astype("float64")
            dates_parts.append(pd.DatetimeIndex(dataset.time.values).normalize())
            sst_parts.append((query.transpose("time", "pixel").values - 273.15).astype("float32"))
        finally:
            dataset.close()
    dates = dates_parts[0].append(dates_parts[1:])
    sst = np.concatenate(sst_parts)
    expected = pd.date_range(START, END, freq="D")
    if not dates.equals(expected) or dates.duplicated().any():
        raise RuntimeError("Final OSTIA time axis is not unique and consecutive")
    if not np.isfinite(sst).all():
        locations = np.argwhere(~np.isfinite(sst))
        raise RuntimeError(f"OSTIA has {len(locations)} non-finite selected values; no imputation allowed")

    distance = haversine_km(lat, lon, mapped_lat, mapped_lon)
    mapping = pd.DataFrame({
        "pixel_id": pixel_ids, "coraltemp_lat": lat, "coraltemp_lon": lon,
        "ostia_lat": mapped_lat, "ostia_lon": mapped_lon,
        "mapping_distance_km": distance, "sector": sectors,
    })
    mapping.to_csv(OUT / "PIXEL_MAPPING_AUDIT.csv", index=False)
    if mapping[["ostia_lat", "ostia_lon"]].duplicated().any():
        raise RuntimeError("Multiple CoralTemp locations mapped to the same OSTIA cell")

    train_days = dates <= TRAIN_END
    day_index = calendar_index(dates)
    climatology365 = np.stack([
        sst[train_days & (day_index == day)].mean(axis=0) for day in range(365)
    ])
    climatology365 = np.mean(
        [np.roll(climatology365, shift, axis=0) for shift in range(-15, 16)], axis=0
    ).astype("float32")
    climatology = climatology365[day_index]
    ssta = (sst - climatology).astype("float32")
    scale = np.float32(np.sqrt(np.mean(ssta[train_days].astype("float64") ** 2)))
    train_ssta = ssta[train_days]
    rho = np.clip(
        np.sum(train_ssta[:-1].astype("float64") * train_ssta[1:].astype("float64"), axis=0)
        / np.sum(train_ssta[:-1].astype("float64") ** 2, axis=0), 0, 1,
    ).astype("float32")

    origins = {}
    for split, (first, last) in SPLITS.items():
        origins[split] = np.array([
            index for index in range(LMAX - 1, len(dates) - H)
            if dates[index + 1] >= pd.Timestamp(first) and dates[index + H] <= pd.Timestamp(last)
        ], dtype="int32")
    test = np.array([
        index for index in range(LMAX - 1, len(dates) - H)
        if dates[index + 1] >= pd.Timestamp("2026-01-01")
    ], dtype="int32")
    np.savez_compressed(
        DATASET, dates=dates[dates <= pd.Timestamp("2025-12-31")].values,
        dates_all=dates.values, sst_all_c=sst, ssta_all_c=ssta,
        climatology365_c=climatology365, ssta_scale=scale,
        dap_rho_train=rho, pixel_ids=pixel_ids.astype("U16"),
        pixel_latitude=mapped_lat.astype("float32"),
        pixel_longitude=mapped_lon.astype("float32"), sectors=sectors.astype("U16"),
        history_offsets_L90=np.arange(-89, 1, dtype="int16"),
        history_offsets_L365=np.arange(-364, 1, dtype="int16"),
        target_offsets_H30=np.arange(1, 31, dtype="int16"),
        Test2026=test, **origins,
    )
    protocol = {
        "status": "complete", "dataset_id": DATASET_ID,
        "dataset_version": DATASET_VERSION,
        "source": "Copernicus Marine OSTIA global L4 reprocessed foundation SST, 0.05 degree",
        "coverage": [str(START.date()), str(END.date())],
        "pixels": len(pixel_ids), "location_rule": "nearest OSTIA cell to each frozen Hainan cell",
        "maximum_mapping_distance_km": float(distance.max()),
        "duplicate_mapped_cells": 0, "imputation_used": False,
        "origin_counts": {**{key: len(value) for key, value in origins.items()}, "Test2026": len(test)},
        "climatology": "per-pixel Train-only 365-bin mean with circular 31-day smoothing",
        "scaling": "single Train-only domain RMS scalar",
        "dap": "per-pixel Train-only lag-1 coefficient clipped to [0,1]",
        "dataset_sha256": sha256(DATASET), "source_dataset_sha256": sha256(SOURCE),
        "raw_blocks": raw_records,
    }
    (OUT / "PROTOCOL.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    return protocol


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    RAW.mkdir(exist_ok=True)
    years = list(range(START.year, END.year + 1))
    # Three bounded workers substantially reduce wall time while keeping the
    # low-memory workstation within its available RAM.
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        records = list(pool.map(download_year, years))
    protocol = build_dataset(records)
    print(json.dumps({key: protocol[key] for key in (
        "status", "pixels", "coverage", "origin_counts", "maximum_mapping_distance_km",
        "dataset_sha256")}, indent=2))


if __name__ == "__main__":
    main()
