"""ERA5 hourly u10, v10, ssrd, t2m from the CDS API (reanalysis-era5-single-levels), quarterly requests.

Fallback for the ARCO mirror (whose time axis was found corrupted on 2026-10-01). Request format is the one
validated in download_full_region.py. Each quarter is checked for an exact hourly time axis and finite values
before it is promoted from .part to .nc with a JSON manifest (SHA256); valid quarters are skipped on re-run.

  python download_era5_cds_region.py --name hainan_2026 --area 20.5 108.25 17.75 111.5 --start 2026-01-01 --end 2026-09-07
  python download_era5_cds_region.py --name gbr_2000_2026 --area -14.25 145.25 -19.75 148.25 --start 2000-01-01 --end 2026-09-07
"""
from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cdsapi
import numpy as np
import pandas as pd
import xarray as xr

VARS = ["10m_u_component_of_wind", "10m_v_component_of_wind", "surface_solar_radiation_downwards", "2m_temperature"]
ROOT = Path(__file__).resolve().parent / "data/era5_cds_regions"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def open_any(path):
    """CDS may return a zip with separate instant/accumulated files; merge them."""
    if zipfile.is_zipfile(path):
        d = path.with_suffix(".unz"); d.mkdir(exist_ok=True)
        with zipfile.ZipFile(path) as z:
            z.extractall(d)
        # h5netcdf (pure Python) because the netCDF4 C library cannot open non-ASCII Windows paths
        return xr.merge([xr.open_dataset(f, engine="h5netcdf").load() for f in sorted(d.glob("*.nc"))], compat="override")
    return xr.open_dataset(path, engine="h5netcdf").load()


def fetch(client, area, lo, hi, out):
    target = out / f"era5_{lo:%Y%m%d}_{hi:%Y%m%d}.nc"; man = target.with_suffix(".json")
    if target.exists() and man.exists() and json.loads(man.read_text())["sha256"] == sha(target):
        return f"SKIP {target.name}"
    days = pd.date_range(lo, hi, freq="D")
    req = {"product_type": ["reanalysis"], "variable": VARS, "year": sorted({f"{d.year}" for d in days}),
           "month": sorted({f"{d.month:02d}" for d in days}), "day": sorted({f"{d.day:02d}" for d in days}),
           "time": [f"{h:02d}:00" for h in range(24)], "area": area,
           "data_format": "netcdf", "download_format": "unarchived"}
    part = target.with_suffix(".part")
    for attempt in range(4):
        try:
            client.retrieve("reanalysis-era5-single-levels", req, str(part))
            ds = open_any(part)
            tname = "valid_time" if "valid_time" in ds.dims else "time"
            ds = ds.sel({tname: slice(lo, hi + pd.Timedelta(hours=23))})
            assert pd.DatetimeIndex(ds[tname].values).equals(pd.date_range(lo, hi + pd.Timedelta(hours=23), freq="h"))
            for v in ("u10", "v10", "ssrd", "t2m"):
                assert np.isfinite(ds[v].values).all(), v
            ds = ds.rename({tname: "valid_time"})
            for name in ds.variables:
                ds[name].encoding = {}
            ds = ds[["u10", "v10", "ssrd", "t2m"]].drop_vars([c for c in ("number", "expver") if c in ds.coords])
            ds.to_netcdf(target, engine="scipy")                 # NetCDF3 via scipy: safe with non-ASCII paths
            man.write_text(json.dumps({"source": "CDS reanalysis-era5-single-levels", "request": req, "start": str(lo.date()),
                                       "end": str(hi.date()), "sha256": sha(target)}, indent=1))
            part.unlink(missing_ok=True)
            return f"OK {target.name}"
        except Exception as e:                                   # noqa: BLE001
            err = repr(e)[:200]
    return f"FAILED {target.name}: {err}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True); ap.add_argument("--area", nargs=4, type=float, required=True)
    ap.add_argument("--start", required=True); ap.add_argument("--end", required=True)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args(); out = ROOT / a.name; out.mkdir(parents=True, exist_ok=True)
    start, end = pd.Timestamp(a.start), pd.Timestamp(a.end)
    q = pd.date_range(start.to_period("Q").start_time, end, freq="QS")
    jobs = [(max(s, start), min(s + pd.offsets.QuarterEnd(0), end)) for s in q]
    client = cdsapi.Client(quiet=True, debug=False)
    with ThreadPoolExecutor(a.workers) as ex:
        for f in as_completed([ex.submit(fetch, client, a.area, lo, hi, out) for lo, hi in jobs]):
            print(f.result(), flush=True)
    print("DONE", a.name, flush=True)


if __name__ == "__main__":
    main()
