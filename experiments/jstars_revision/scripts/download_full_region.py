"""Resumable raw NOAA/ERA5 acquisition for the frozen 163-pixel domain.

Files are promoted from .part only after coordinate/time/variable checks.
No training, climatology fitting, imputation or test-set selection occurs here.
"""
from pathlib import Path
import argparse
import concurrent.futures
import hashlib
import importlib.util
import json
import logging
import time
import calendar
import os
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'data/full_region_v2'
spec = importlib.util.spec_from_file_location('era_reader', ROOT.parent/'hainan_upwelling/02_code/era5_download.py')
reader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reader)
NOAA = 'https://coastwatch.noaa.gov/erddap/griddap/noaacrwsstDaily.nc'
OCEANWATCH = 'https://oceanwatch.pifsc.noaa.gov/erddap/griddap/CRW_sst_v3_1.nc'
NOAA_DELAY = 60
VARS = ['10m_u_component_of_wind', '10m_v_component_of_wind',
        'surface_solar_radiation_downwards', '2m_temperature']

def atomic_json(path, obj):
    obj = dict(obj, updated_utc=pd.Timestamp.now(tz='UTC').isoformat())
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(path)

def inspect(path, source, start, end):
    ds = reader.open_nc(str(path))
    try:
        t = pd.DatetimeIndex(ds['time' if source == 'noaa' else 'valid_time'].values)
        expected = pd.date_range(start, end, freq='D') + pd.Timedelta(hours=12) if source == 'noaa' else pd.date_range(start, end+pd.Timedelta(hours=23), freq='h')
        assert t.equals(expected), 'Time coverage mismatch'
        variables = ['analysed_sst'] if source == 'noaa' else ['u10', 'v10', 'ssrd', 't2m']
        assert set(variables).issubset(ds.data_vars), 'Variables missing'
        p = pd.read_csv(ROOT/'data/reef_study_region_v2/FINAL_OCEAN_PIXELS.csv')
        assert float(ds.latitude.min()) <= p.lat.min() and float(ds.latitude.max()) >= p.lat.max()
        assert float(ds.longitude.min()) <= p.lon.min() and float(ds.longitude.max()) >= p.lon.max()
        # Force actual variable decoding to detect truncated downloads.
        for v in variables:
            ds[v].load()
        return {'times': len(t), 'variables': {v: ds[v].attrs.get('units') for v in variables},
                'latitude_range': [float(ds.latitude.min()), float(ds.latitude.max())],
                'longitude_range': [float(ds.longitude.min()), float(ds.longitude.max())]}
    finally:
        ds.close()

def worker(source, first, last):
    folder = OUT/source
    folder.mkdir(parents=True, exist_ok=True)
    lock = OUT/f'{source}.lock'
    # A stale lock is retained after a crash and must be checked before removal.
    with lock.open('x', encoding='utf-8') as f:
        f.write(str(os.getpid()))
    client = None
    if source == 'era5':
        import cdsapi
        client = cdsapi.Client(quiet=True, debug=False)
    failures = []
    # Chronological chunks are bounded to limit memory and request cost.
    for year in range(first, last+1):
        for month in (1, 4, 7, 10):
            start = pd.Timestamp(year, month, 1)
            end = pd.Timestamp(year, month+2, calendar.monthrange(year, month+2)[1])
            tag = f'{year}_{month:02d}-{month+2:02d}'
            target = folder/f'{source}_{tag}.nc'
            manifest = target.with_suffix('.json')
            if target.exists() and manifest.exists():
                record = json.loads(manifest.read_text(encoding='utf-8'))
                if hashlib.sha256(target.read_bytes()).hexdigest() == record['sha256']:
                    continue
            part = target.with_suffix('.part')
            if source == 'noaa':
                # Includes a real 3x3 SST neighborhood around every selected pixel.
                request = NOAA+f'?analysed_sst[({start.date()}T12:00:00Z):1:({end.date()}T12:00:00Z)][(18.075):1:(20.125)][(108.525):1:(111.125)]'
            else:
                request = {'product_type': ['reanalysis'], 'variable': VARS,
                    'year': [str(year)], 'month': [f'{m:02d}' for m in range(month, month+3)],
                    'day': [f'{d:02d}' for d in range(1,32)],
                    'time': [f'{h:02d}:00' for h in range(24)],
                    'area': [20.5,108.25,17.75,111.5],
                    'data_format': 'netcdf', 'download_format': 'unarchived'}
            for attempt in range(3):
                atomic_json(OUT/f'{source}_status.json', {'state':'downloading','chunk':tag,'attempt':attempt+1,'failed_chunks':failures})
                print(source, tag, 'request', attempt+1, flush=True)
                try:
                    if source == 'noaa':
                        with requests.get(request, stream=True, timeout=(30,180), headers={'User-Agent':'HainanSSTResearch/1.0'}) as response:
                            response.raise_for_status()
                            with part.open('wb') as f:
                                for block in response.iter_content(1024*1024):
                                    f.write(block)
                    else:
                        client.retrieve('reanalysis-era5-single-levels', request, str(part))
                    # netCDF C library is not thread-safe; file validation is serialized.
                    with NC_LOCK:
                        repair_record = None
                        if source == 'noaa':
                            from repair_noaa_dates import repair
                            repair_record = repair(part, start, end, reader)
                        record = inspect(part, source, start, end)
                    if repair_record is not None:
                        record['repair'] = repair_record
                    record.update(source=source, request=request, start=str(start.date()), end=str(end.date()),
                        sha256=hashlib.sha256(part.read_bytes()).hexdigest(), bytes=part.stat().st_size)
                    part.replace(target)
                    atomic_json(manifest, record)
                    print(source, tag, 'validated', record['bytes'], flush=True)
                    if source == 'noaa':
                        atomic_json(OUT/f'{source}_status.json', {'state':'throttling', 'last_completed_chunk':tag,
                            'validated_chunks':len(list(folder.glob('*.json'))),'expected_chunks':(last-first+1)*4,
                            'delay_seconds':NOAA_DELAY, 'provider':NOAA})
                        time.sleep(NOAA_DELAY)
                    break
                except Exception as exc:
                    # Do not log credentials, server response bodies or signed URLs.
                    status = exc.response.status_code if isinstance(exc, requests.HTTPError) and exc.response is not None else None
                    print(source, tag, 'failed', type(exc).__name__, 'HTTP',status, flush=True)
                    if status in (401,403):
                        atomic_json(OUT/f'{source}_status.json', {'state':'access_denied','chunk':tag,
                            'http_status':status,'provider':NOAA,'validated_chunks':len(list(folder.glob('*.json'))),
                            'expected_chunks':(last-first+1)*4})
                        lock.unlink()
                        return source, [tag]
                    if attempt < 2:
                        time.sleep(10*(attempt+1))
            else:
                failures.append(tag)
                # Stop after three failed chunks rather than flood a unavailable server.
                if len(failures) >= 3:
                    break
        if len(failures) >= 3:
            break
    atomic_json(OUT/f'{source}_status.json', {'state':'incomplete' if failures else 'complete',
        'failed_chunks':failures,'validated_chunks':len(list(folder.glob('*.json'))),
        'expected_chunks':(last-first+1)*4})
    lock.unlink()
    return source, failures

if __name__ == '__main__':
    import threading
    NC_LOCK = threading.Lock()
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=int, default=2000)
    parser.add_argument('--end', type=int, default=2025)
    parser.add_argument('--source', choices=['both','noaa','era5'], default='both')
    parser.add_argument('--noaa-provider', choices=['coastwatch','oceanwatch'], default='coastwatch')
    parser.add_argument('--noaa-delay', type=int, default=60)
    args = parser.parse_args()
    NOAA = OCEANWATCH if args.noaa_provider == 'oceanwatch' else NOAA
    NOAA_DELAY = max(30,args.noaa_delay)
    OUT.mkdir(parents=True, exist_ok=True)
    logging.getLogger('ecmwf').setLevel(logging.ERROR)
    atomic_json(OUT/('download_plan.json' if args.source=='both' else f'{args.source}_resume_plan.json'), {'start_year':args.start,'end_year':args.end,
        'selected_pixels':163,'noaa_grid_deg':.05,'sst_context':'3x3 neighborhoods included; land/missing cells retained as missing',
        'era5_sampling':'all 24 hours UTC; raw accumulations preserved',
        'sources':[NOAA,'https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels'],
        'daily_processing':'SST at nominal 12 UTC. Future processing must align ssrd hourly interval-end timestamps before daily aggregation.',
        'climatology':'not downloaded or fitted; fit from training years only after split audit',
        'existing_cache_audit':'2000 NOAA cache contains 88/163 targets; new rectangle additionally supplies spatial neighbors',
        'final_test':'not yet assigned; audit historical experimental usage before freezing years'})
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        jobs = [pool.submit(worker, s, args.start, args.end) for s in (['noaa','era5'] if args.source=='both' else [args.source])]
        for job in concurrent.futures.as_completed(jobs):
            try:
                print('FINISHED', job.result(), flush=True)
            except Exception as exc:
                print('WORKER FAILED', type(exc).__name__, flush=True)
