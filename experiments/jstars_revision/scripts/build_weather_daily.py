"""Daily ERA5 weather on the fixed SST pixels of a domain, aligned to that domain's dataset dates.

Processing follows freeze_l365_weather14.py exactly:
  * u10, v10, t2m: mean of the 24 hourly values of each UTC day (t2m in degC);
  * ssrd: each hourly accumulation is assigned to date(valid_time - 1 h), summed over 24 intervals and divided by
    86400 (W m-2); a day with fewer than 24 intervals is stored as NaN and must lie outside every input window;
  * each SST pixel takes the nearest 0.25-degree ERA5 grid cell (shared cells are intentional).
Output: data/<domain weather dir>/weather_daily.npz with weather_daily [n_days, n_pixels, 4]
(u10, v10, ssrd, t2m) on exactly the dates of the domain dataset, plus the ERA5 cell mapping and file SHA256s.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parent
from download_full_region import reader as raw_reader  # noqa: E402
DOMAINS = {
    "hainan": dict(data=ROOT / "data/frozen_154_ssta_v1/dataset_daily.npz", dates="dates_all",
                   files=sorted((ROOT / "data/full_region_v2/era5").glob("era5_*.nc")) +
                   sorted((ROOT / "data/era5_cds_regions/hainan_2026").glob("era5_*.nc")),
                   out=ROOT / "data/frozen_154_weather_v1"),
    "gbr": dict(data=ROOT / "data/gbr_generalization_v1/dataset_daily.npz", dates="dates",
                files=sorted((ROOT / "data/era5_arco_regions/gbr_2000_2026").glob("block_*.nc")),
                out=ROOT / "data/gbr_weather_v1"),
}


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main(name):
    cfg = DOMAINS[name]
    with np.load(cfg["data"]) as z:
        dates = pd.DatetimeIndex(z[cfg["dates"]]).normalize()
        plat, plon = z["pixel_latitude"].astype(float), z["pixel_longitude"].astype(float)
    n, p = len(dates), len(plat)
    plat_da, plon_da = xr.DataArray(plat, dims="pixel"), xr.DataArray(plon % 360, dims="pixel")
    sums = np.zeros((n, p, 3)); counts = np.zeros((n, 3), int)          # u10, v10, t2m by UTC day
    ssrd_sum = np.zeros((n, p)); ssrd_cnt = np.zeros(n, int)
    seen_hours = set(); era_lat = era_lon = None; inventory = []
    for f in cfg["files"]:
        # Project reader (used for the frozen 163-pixel dataset): reads bytes, so non-ASCII paths are safe, and
        # handles NetCDF3, NetCDF4 and the zip-of-stepTypes that CDS returns for mixed instant/accumulated requests.
        ds = raw_reader.open_nc(str(f))
        if "valid_time" in ds.dims:
            ds = ds.rename({"valid_time": "time"})
        q = ds[["u10", "v10", "ssrd", "t2m"]].sel(latitude=plat_da, longitude=plon_da, method="nearest").load()
        lat_now, lon_now = q.latitude.values.astype("float32"), q.longitude.values.astype("float32")
        if era_lat is None:
            era_lat, era_lon = lat_now, lon_now
        assert np.array_equal(era_lat, lat_now) and np.array_equal(era_lon, lon_now)
        t = pd.DatetimeIndex(q.time.values)
        keep = np.array([ts not in seen_hours for ts in t])               # overlapping files: count each hour once
        seen_hours.update(t[keep]); t = t[keep]
        inventory.append({"file": str(f.relative_to(ROOT)), "sha256": sha(f), "hours_used": int(keep.sum())})
        if not keep.any():
            ds.close(); continue
        day = ((t.normalize() - dates[0]) / pd.Timedelta(days=1)).to_numpy().astype(int)
        sday = (((t - pd.Timedelta(hours=1)).normalize() - dates[0]) / pd.Timedelta(days=1)).to_numpy().astype(int)
        for c, v in enumerate(("u10", "v10", "t2m")):
            vals = q[v].transpose("time", "pixel").values[keep].astype("float64")
            assert np.isfinite(vals).all()
            ok = (day >= 0) & (day < n)
            np.add.at(sums[:, :, c], day[ok], vals[ok]); np.add.at(counts[:, c], day[ok], 1)
        vals = q["ssrd"].transpose("time", "pixel").values[keep].astype("float64")
        ok = (sday >= 0) & (sday < n)
        np.add.at(ssrd_sum, sday[ok], vals[ok]); np.add.at(ssrd_cnt, sday[ok], 1)
        ds.close()
        print("READ", f.name, flush=True)
    weather = np.full((n, p, 4), np.nan, dtype="float32")
    for c, ch in ((0, 0), (1, 1), (2, 3)):
        full = counts[:, c] == 24
        weather[full, :, ch] = (sums[full, :, c] / 24).astype("float32")
    weather[:, :, 3] -= 273.15
    full = ssrd_cnt == 24
    weather[full, :, 2] = (ssrd_sum[full] / 86400).astype("float32")
    missing = np.where(~np.isfinite(weather).all(axis=(1, 2)))[0]
    print("days with any missing channel:", len(missing), [str(dates[i].date()) for i in missing[:5]])
    cfg["out"].mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cfg["out"] / "weather_daily.npz", weather_daily=weather, dates=dates.values.astype("datetime64[D]"),
                        weather_names=np.array(["u10_mean", "v10_mean", "ssrd_mean_flux", "t2m_mean_c"]),
                        era5_latitude=era_lat, era5_longitude=era_lon)
    (cfg["out"] / "manifest.json").write_text(json.dumps({
        "domain_dataset": str(cfg["data"].relative_to(ROOT)), "n_days": n, "n_pixels": p,
        "date_range": [str(dates[0].date()), str(dates[-1].date())], "missing_days": [str(dates[i].date()) for i in missing],
        "unique_era5_cells": int(len(set(zip(era_lat.tolist(), era_lon.tolist())))), "inputs": inventory,
        "processing": "see module docstring (identical to freeze_l365_weather14.py)"}, indent=1), encoding="utf-8")
    print("SAVED", cfg["out"] / "weather_daily.npz", weather.shape)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("domain", choices=DOMAINS); main(ap.parse_args().domain)
