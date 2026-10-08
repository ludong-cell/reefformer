"""Generic ERA5 ARCO downloader (hourly u10, v10, ssrd, t2m) for a lat/lon box and time range.

Same access pattern as download_era5_arco.py (ARCO geoChunked Zarr, bearer token from ~/.cdsapirc, token never
logged). Saves one validated NetCDF per source time chunk with a JSON manifest (hours, variables, SHA256);
existing valid blocks are skipped, so the script can be re-run to resume.

Usage:
  python download_era5_arco_region.py --probe
  python download_era5_arco_region.py --name hainan_2026 --north 20.5 --west 108.25 --south 17.75 --east 111.5 \
      --start 2025-12-31T00 --end 2026-09-07T23
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cdsapi
import numpy as np
import pandas as pd
import xarray as xr

URL = ("https://arco.datastores.ecmwf.int/cadl-arco-geo-002/arco/"
       "reanalysis_era5_single_levels/sfc/geoChunked.zarr")
VARS = ["u10", "v10", "ssrd", "t2m"]
ROOT = Path(__file__).resolve().parent / "data/era5_arco_regions"


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def open_remote():
    c = cdsapi.Client(quiet=True, debug=False)
    token = c.key[-1] if isinstance(c.key, (tuple, list)) else c.key
    return xr.open_zarr(URL, consolidated=True, chunks={},
                        storage_options={"headers": {"Authorization": f"Bearer {token}"}})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--name"); ap.add_argument("--north", type=float); ap.add_argument("--west", type=float)
    ap.add_argument("--south", type=float); ap.add_argument("--east", type=float)
    ap.add_argument("--start"); ap.add_argument("--end")
    a = ap.parse_args()
    remote = open_remote()
    try:
        t = pd.DatetimeIndex(remote.time.values)
        if a.probe:
            print("ARCO time range:", t[0], "->", t[-1], "| hours:", len(t), "| vars ok:", set(VARS) <= set(remote.data_vars))
            return
        out = ROOT / a.name; out.mkdir(parents=True, exist_ok=True)
        start, end = pd.Timestamp(a.start), pd.Timestamp(a.end)
        assert end <= t[-1], f"ARCO ends at {t[-1]}, requested {end}"
        lat = remote.latitude.values; lon = remote.longitude.values
        lat = lat[(lat >= a.south) & (lat <= a.north)]
        lon = lon[(lon >= (a.west % 360)) & (lon <= (a.east % 360))]
        chunks = remote["u10"].chunks[0]
        starts = np.cumsum((0,) + chunks[:-1]); ends = np.cumsum(chunks) - 1
        for i, (l, r) in enumerate(zip(starts, ends), 1):
            lo, hi = max(t[l], start), min(t[r], end)
            if lo > hi:
                continue
            path = out / f"block_{i:02d}_{lo:%Y%m%d%H}_{hi:%Y%m%d%H}.nc"
            man = path.with_suffix(".json")
            if path.exists() and man.exists() and json.loads(man.read_text())["sha256"] == sha(path):
                print("SKIP valid", path.name, flush=True); continue
            sub = remote[VARS].sel(time=slice(lo, hi), latitude=lat, longitude=lon).load()
            assert pd.DatetimeIndex(sub.time.values).equals(pd.date_range(lo, hi, freq="h"))
            assert all(np.isfinite(sub[v].values).all() for v in VARS), "non-finite values"
            part = path.with_suffix(".part"); sub.to_netcdf(part, engine="scipy")
            rec = {"source": "ECMWF ERA5 ARCO geoChunked Zarr", "url": URL, "start": str(lo), "end": str(hi),
                   "hours": int(sub.sizes["time"]), "variables": VARS, "lat": [float(lat.min()), float(lat.max()), len(lat)],
                   "lon": [float(lon.min()), float(lon.max()), len(lon)], "bytes": part.stat().st_size, "sha256": sha(part)}
            part.replace(path); man.write_text(json.dumps(rec, indent=1))
            print("BLOCK VALIDATED", path.name, rec["hours"], "h", flush=True)
        print("COMPLETE", a.name, flush=True)
    finally:
        remote.close()


if __name__ == "__main__":
    main()
