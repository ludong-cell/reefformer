"""Small, reproducible SST-only experiment. No proposed modules are enabled."""
import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent


def calendar_index(dates):
    d = pd.DatetimeIndex(dates)
    return np.asarray(d.dayofyear - (d.is_leap_year & (d.dayofyear >= 60)), int) - 1


class ResidualBlock(nn.Module):
    def __init__(self, cin, hidden, dilation):
        super().__init__()
        self.pad = 2 * dilation
        self.c1 = nn.Conv1d(cin, hidden, 3, dilation=dilation)
        self.c2 = nn.Conv1d(hidden, hidden, 3, dilation=dilation)
        self.skip = nn.Conv1d(cin, hidden, 1) if cin != hidden else nn.Identity()
        self.drop = nn.Dropout(0.1)

    def forward(self, x):
        z = self.drop(torch.relu(self.c1(nn.functional.pad(x, (self.pad, 0)))))
        z = self.drop(torch.relu(self.c2(nn.functional.pad(z, (self.pad, 0)))))
        return torch.relu(z + self.skip(x))


class SharedTCN(nn.Module):
    def __init__(self, horizon=28, hidden=32):
        super().__init__()
        self.encoder = nn.Sequential(ResidualBlock(3, hidden, 1),
                                     ResidualBlock(hidden, hidden, 2),
                                     ResidualBlock(hidden, hidden, 4))
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, horizon))

    def encode(self, x):
        # Full-history mean pooling matches the architecture drawing.
        return self.encoder(x).mean(dim=-1)

    def forward(self, x):
        return self.head(self.encode(x))


def prepare(source, out, lookback, horizon, train_start='2010-01-01'):
    raw = pd.read_parquet(source)
    assert not raw.duplicated(['pixel_id', 'date']).any()
    assert np.isfinite(raw.sst_c).all()
    # Limit processing to the development experiment. Old 2022+ test is not used.
    raw = raw[raw.date.between(train_start, '2018-12-31')].copy()
    splits = {'train': (train_start, '2016-12-31'),
              'val': ('2017-01-01', '2017-12-31'),
              'dev_eval': ('2018-01-01', '2018-12-31')}
    buffers = {s: {k: [] for k in ['x', 'y', 'clim', 'last', 'last_anom', 'pixel', 'origin']} for s in splits}
    clim_rows, train_anoms = [], []
    series = []
    for pixel, group in raw.groupby('pixel_id', sort=True):
        group = group.sort_values('date')
        dates = pd.DatetimeIndex(group.date)
        assert np.all(np.diff(dates.values).astype('timedelta64[D]').astype(int) == 1)
        values = group.sst_c.to_numpy(np.float32)
        doy = calendar_index(dates)
        train = dates <= pd.Timestamp('2016-12-31')
        c = pd.Series(values[train]).groupby(doy[train]).mean().reindex(range(365)).to_numpy()
        assert np.isfinite(c).all()
        c = np.mean([np.roll(c, shift) for shift in range(-15, 16)], axis=0).astype(np.float32)
        anomaly = values - c[doy]
        train_anoms.append(anomaly[train])
        clim_rows.extend({'pixel_id': pixel, 'calendar_day': k+1, 'climatology_c': float(v)} for k, v in enumerate(c))
        series.append((pixel, dates, values, doy, c, anomaly))
    scale = float(np.std(np.concatenate(train_anoms)))
    assert scale > 0
    for pixel, dates, values, doy, c, anomaly in series:
        history = np.stack([anomaly / scale, np.sin(2*np.pi*doy/365), np.cos(2*np.pi*doy/365)], axis=0).astype(np.float32)
        for origin in range(lookback-1, len(dates)-horizon):
            start, end = dates[origin+1], dates[origin+horizon]
            for split, (lo, hi) in splits.items():
                if start >= pd.Timestamp(lo) and end <= pd.Timestamp(hi):
                    b = buffers[split]
                    b['x'].append(history[:, origin-lookback+1:origin+1])
                    b['y'].append(values[origin+1:origin+horizon+1])
                    b['clim'].append(c[doy[origin+1:origin+horizon+1]])
                    b['last'].append(values[origin]); b['last_anom'].append(anomaly[origin])
                    b['pixel'].append(pixel); b['origin'].append(dates[origin].to_datetime64())
                    break
    data = {s: {k: np.asarray(v) for k, v in b.items()} for s, b in buffers.items()}
    for s, b in data.items():
        assert len(b['x']) > 0 and b['x'].shape[1:] == (3, lookback)
        assert np.isfinite(b['x']).all() and np.isfinite(b['y']).all()
        assert (b['origin'] + np.timedelta64(1, 'D')).min() >= np.datetime64(splits[s][0])
        assert (b['origin'] + np.timedelta64(horizon, 'D')).max() <= np.datetime64(splits[s][1])
    pd.DataFrame(clim_rows).to_csv(out/'climatology_train.csv', index=False)
    raw.to_parquet(out/'development_daily.parquet', index=False)
    return data, scale, splits


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--patience', type=int, default=6)
    p.add_argument('--batch-size', type=int, default=512)
    p.add_argument('--horizon', type=int, default=28)
    p.add_argument('--lookback', type=int, default=60)
    p.add_argument('--run-name', default='backbone_small_seed42')
    args = p.parse_args()
    out = ROOT/'runs'/args.run_name
    out.mkdir(parents=True, exist_ok=False)
    t0 = time.perf_counter()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    source = ROOT.parent/'HAINAN_NOAA_SST_FORECAST_STAGE1/outputs/hainan_noaa_sst_daily.parquet'
    data, scale, splits = prepare(source, out, args.lookback, args.horizon)
    model = SharedTCN(args.horizon).to(device)
    config = {**vars(args), 'source': str(source), 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'splits': splits, 'samples': {s: len(d['x']) for s, d in data.items()},
              'pixels': sorted(set(data['train']['pixel'])), 'scale_c': scale,
              'device': str(device), 'gpu': torch.cuda.get_device_name() if device.type == 'cuda' else None,
              'parameters': sum(p.numel() for p in model.parameters()),
              'torch': str(torch.__version__), 'learning_rate': 0.001,
              'modules_enabled': [], 'climatology': '2010-2016 per pixel; circular 31-day smoothing; Feb29 folded to Feb28',
              'inputs': 'anomaly/scale, sin(calendar_day), cos(calendar_day)',
              'output': 'direct multi-horizon anomaly/scale, plus training climatology',
              'evaluation': 'development-only; one seed; equal-pixel macro MAE/RMSE'}
    (out/'config.json').write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({k: config[k] for k in ['samples','parameters','device','gpu'] }), flush=True)
    tensors = {s: (torch.tensor(d['x'], device=device),
                    torch.tensor((d['y']-d['clim'])/scale, device=device)) for s,d in data.items()}
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    lossfn = nn.MSELoss()
    best, bad, log = float('inf'), 0, []
    for epoch in range(1, args.epochs+1):
        te = time.perf_counter(); model.train(); total = 0
        x, y = tensors['train']; order = torch.randperm(len(x), device=device)
        for idx in order.split(args.batch_size):
            opt.zero_grad(set_to_none=True)
            loss = lossfn(model(x[idx]), y[idx])
            assert torch.isfinite(loss), 'Nonfinite loss'
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            total += loss.item()*len(idx)
        model.eval()
        with torch.no_grad():
            vx, vy = tensors['val']
            pred = torch.cat([model(b) for b in vx.split(args.batch_size)])
            val = lossfn(pred, vy).item()
        row = {'epoch': epoch, 'train_mse_c2': total/len(x)*scale**2,
               'val_rmse_c': float(np.sqrt(val)*scale), 'seconds': time.perf_counter()-te}
        log.append(row); print(json.dumps(row), flush=True)
        if val < best:
            best, bad = val, 0
            torch.save({'state_dict': model.state_dict(), 'config': config, 'epoch': epoch}, out/'best_model.pt')
        else: bad += 1
        pd.DataFrame(log).to_csv(out/'training_log.csv', index=False)
        if bad >= args.patience: break
    checkpoint = torch.load(out/'best_model.pt', map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['state_dict']); model.eval()
    clone = SharedTCN(args.horizon).to(device)
    clone.load_state_dict(checkpoint['state_dict']); clone.eval()
    with torch.no_grad():
        assert torch.equal(model(tensors['val'][0][:4]), clone(tensors['val'][0][:4]))
    all_metrics, predictions = [], []
    for split in ['val','dev_eval']:
        d = data[split]
        with torch.no_grad():
            pred = torch.cat([model(b) for b in tensors[split][0].split(args.batch_size)]).cpu().numpy()*scale+d['clim']
        assert pred.shape == d['y'].shape and np.isfinite(pred).all()
        forecasts = {'persistence': np.broadcast_to(d['last'][:,None], pred.shape),
                     'climatology': d['clim'], 'seasonal_anomaly_persistence': d['clim']+d['last_anom'][:,None], 'tcn': pred}
        for name, f in forecasts.items():
            for pixel in sorted(set(d['pixel'])):
                err = f[d['pixel']==pixel]-d['y'][d['pixel']==pixel]
                for h in range(args.horizon):
                    all_metrics.append({'split':split,'model':name,'pixel_id':pixel,'horizon':h+1,
                                        'mae_c':float(np.abs(err[:,h]).mean()),'rmse_c':float(np.sqrt((err[:,h]**2).mean()))})
        for h in range(args.horizon):
            predictions.append(pd.DataFrame({'split':split,'pixel_id':d['pixel'],'origin':d['origin'],
                'target_date':d['origin']+np.timedelta64(h+1,'D'),'horizon':h+1,'observed_c':d['y'][:,h],
                **{name:f[:,h] for name,f in forecasts.items()}}))
    per_pixel = pd.DataFrame(all_metrics)
    per_pixel.to_csv(out/'metrics_by_pixel.csv', index=False)
    macro = per_pixel.groupby(['split','model','horizon'],as_index=False)[['mae_c','rmse_c']].mean()
    macro.to_csv(out/'metrics_macro.csv', index=False)
    pd.concat(predictions,ignore_index=True).to_parquet(out/'predictions.parquet',index=False)
    summary = {'best_epoch':checkpoint['epoch'],'epochs_run':len(log),'elapsed_seconds':time.perf_counter()-t0,
               'checks':{'daily_continuity':True,'no_duplicate_pixel_date':True,'finite_inputs_outputs':True,
                         'labels_within_split':True,'checkpoint_reload_exact':True},
               'verdict':'BACKBONE_PIPELINE_COMPLETE; MODULE_EFFECT_NOT_TESTED'}
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    table = macro[(macro.split=='dev_eval') & macro.horizon.isin([7,14,28])]
    report = ['# 共享 TCN 小数据基线实验','',
        '状态：主干训练、验证选择、开发评估与权重重载检查已完成。两个候选模块均未启用。','',
        '数据：现有 12 个独立 NOAA SST 像元。训练 2010—2016，验证 2017，开发评估 2018；未使用旧项目 2022 年以后的测试期。',
        f'输入 {args.lookback} 天，直接输出未来 {args.horizon} 天。共 {config["parameters"]:,} 个参数；最佳 epoch {checkpoint["epoch"]}。',
        '训练期逐像元气候态、31 日循环平滑、训练期距平尺度；所有预测标签完整落在其所属时间段内。验证和评估可读取预测起点之前的历史。','',
        '## 2018 开发评估：逐像元指标的等权平均','',
        '| 模型 | 提前期 | MAE (°C) | RMSE (°C) |','|---|---:|---:|---:|']
    for r in table.itertuples(): report.append(f'| {r.model} | {r.horizon} | {r.mae_c:.4f} | {r.rmse_c:.4f} |')
    report += ['', '## 使用边界','',
        '- 这是单随机种子、小区域开发基线，不是论文最终测试，也不证明模块有效。',
        '- 2018 结果可以用于诊断；若据此修改模型，必须保留另一个最终确认数据集。',
        '- 12 个稀疏像元不构成规则 3×3 邻域。模块一需要之后读取真实邻域数据，不能把任意像元当作相邻格点。',
        '- 模块二可直接沿用本实验数据、划分、气候态与尺度，在相同预算下与该主干比较。',
        '- 后续比较必须重复相同种子集合；本轮未运行候选模块、显著性检验或参数匹配实验。',
        '- SVG 中主干及平均池化已对应本代码；本轮尚未启用图中的模块与融合分支。','',
        f'总耗时：{summary["elapsed_seconds"]:.1f} 秒。详细配置、逐格点指标和所有提前期预测均已保存。']
    (out/'REPORT.md').write_text('\n'.join(report),encoding='utf-8')
    print(table.to_string(index=False),flush=True)
    print(json.dumps(summary),flush=True)


if __name__ == '__main__': main()
