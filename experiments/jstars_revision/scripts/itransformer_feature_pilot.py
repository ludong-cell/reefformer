"""Official iTransformer, fixed common samples, one feature group at a time."""
import hashlib
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch
from torch import nn

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'vendor/python_deps'))
sys.path.insert(0,str(ROOT/'vendor/iTransformer'))
from model.iTransformer import Model

DATA=ROOT/'data/feature_pilot_v1'
OUT=ROOT/'runs/itransformer_features_v1'
FEATURES=['ssta','climatology','doy_sin','doy_cos']+[f'spatial_{i}' for i in range(9)]+['grad_x','grad_y','u10','v10','ssrd','air_sea_delta']
GROUPS=dict(ssta=[],climatology=['climatology'],doy=['doy_sin','doy_cos'],
    spatial=[f'spatial_{i}' for i in range(9)],grad_x=['grad_x'],grad_y=['grad_y'],
    u10=['u10'],v10=['v10'],ssrd=['ssrd'],air_sea_delta=['air_sea_delta'],all_features=FEATURES[1:])
CONFIG=dict(seq_len=60,pred_len=28,output_attention=False,use_norm=True,d_model=32,
            embed='timeF',freq='d',dropout=.1,class_strategy='projection',factor=1,
            n_heads=4,d_ff=64,e_layers=2,activation='gelu')

def prepare():
    daily=pd.read_parquet(DATA/'daily_features.parquet')
    train=daily[daily.date<='1990-12-31']
    mean=train[FEATURES].mean().values.astype('float32')
    std=train[FEATURES].std(ddof=0).values.astype('float32')
    assert (std>0).all()
    splits=dict(train=('1985-01-01','1990-12-31'),val=('1991-01-01','1991-12-31'),dev_eval=('1992-01-01','1992-12-31'))
    buf={s:{k:[] for k in ['x','y','clim','pixel','origin','last_ssta']} for s in splits}
    for pixel,d in daily.groupby('pixel_id',sort=True):
        d=d.sort_values('date');dates=d.date.values
        assert (np.diff(dates).astype('timedelta64[D]').astype(int)==1).all()
        x=((d[FEATURES].values-mean)/std).astype('float32')
        y=d.ssta.values.astype('float32');clim=d.climatology.values.astype('float32')
        for i in range(59,len(d)-28):
            for s,(lo,hi) in splits.items():
                if dates[i+1]>=np.datetime64(lo) and dates[i+28]<=np.datetime64(hi):
                    if s=='train' and (i-59)%3:continue
                    b=buf[s];b['x'].append(x[i-59:i+1]);b['y'].append((y[i+1:i+29]-mean[0])/std[0]);b['clim'].append(clim[i+1:i+29])
                    b['pixel'].append(pixel);b['origin'].append(dates[i]);b['last_ssta'].append(y[i])
    data={s:{k:np.asarray(v) for k,v in b.items()} for s,b in buf.items()}
    (OUT/'normalization.json').write_text(json.dumps(dict(columns=FEATURES,mean=mean.tolist(),std=std.tolist(),splits=splits),indent=2),encoding='utf-8')
    return data,mean[0],std[0]

def main(model_factory=None):
    OUT.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    device='cuda' if torch.cuda.is_available() else 'cpu'
    data,offset,scale=prepare()
    config=dict(model=CONFIG,feature_groups=GROUPS,seeds=[42,43,44],batch_size=128,epochs=40,patience=8,lr=.001,
        samples={s:len(d['x']) for s,d in data.items()},device=device,torch=str(torch.__version__),
        official_commit='c2426e68ca13f74aaec08045c5c724d8ad328124',
        dataset_sha256=hashlib.sha256((DATA/'daily_features.parquet').read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        selection='1991 validation normalized SSTA MSE; 1992 development diagnostics only',
        adaptation='Official model predicts all input variates; loss and reporting use only first (SSTA) token. All other variables are historical inputs.')
    if model_factory is not None:
        import inspect
        config['factory']=model_factory.__name__
        config['factory_source_sha256']=hashlib.sha256(Path(inspect.getfile(model_factory)).read_bytes()).hexdigest()
        config['adaptation']='Factory model; first output token is SSTA; see factory source for architecture.'
    (OUT/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    print(json.dumps(config['samples']),flush=True)
    tensors={s:(torch.tensor(d['x'],device=device),torch.tensor(d['y'],device=device)) for s,d in data.items()}
    metrics=[];runs=[];logs=[]
    for seed in config['seeds']:
        for name,extra in GROUPS.items():
            t0=time.perf_counter();torch.manual_seed(seed);np.random.seed(seed)
            ids=[FEATURES.index(c) for c in ['ssta']+extra]
            make=lambda: Model(SimpleNamespace(**CONFIG)) if model_factory is None else model_factory(name,SimpleNamespace(**CONFIG))
            m=make().to(device)
            opt=torch.optim.Adam(m.parameters(),lr=.001)
            gen=torch.Generator(device=device).manual_seed(seed+1000)
            torch.manual_seed(seed+2000)
            folder=OUT/f'{name}_seed{seed}';folder.mkdir()
            best=float('inf');bad=0
            for epoch in range(1,41):
                m.train();tx,ty=tensors['train']
                for idx in torch.randperm(len(tx),generator=gen,device=device).split(128):
                    opt.zero_grad(set_to_none=True)
                    pred=m(tx[idx][:,:,ids],None,None,None)[:,:,0]
                    loss=nn.functional.mse_loss(pred,ty[idx]);assert torch.isfinite(loss)
                    loss.backward();nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step()
                m.eval()
                with torch.no_grad():
                    vx,vy=tensors['val'];p=torch.cat([m(b[:,:,ids],None,None,None)[:,:,0] for b in vx.split(256)])
                    val=nn.functional.mse_loss(p,vy).item()
                logs.append(dict(model=name,seed=seed,epoch=epoch,val_mse=val))
                if val<best:
                    best=val;bad=0;best_epoch=epoch
                    torch.save(dict(state_dict=m.state_dict(),features=['ssta']+extra,config=CONFIG,seed=seed,epoch=epoch),folder/'best_model.pt')
                else:bad+=1
                if bad>=8:break
            checkpoint=torch.load(folder/'best_model.pt',weights_only=False,map_location=device)
            m.load_state_dict(checkpoint['state_dict']);m.eval()
            clone=make().to(device);clone.load_state_dict(checkpoint['state_dict']);clone.eval()
            with torch.no_grad():assert torch.equal(m(vx[:2,:,ids],None,None,None),clone(vx[:2,:,ids],None,None,None))
            rows=[]
            for s in ['val','dev_eval']:
                d=data[s]
                with torch.no_grad():pred=torch.cat([m(b[:,:,ids],None,None,None)[:,:,0] for b in tensors[s][0].split(256)]).cpu().numpy()*scale+offset+d['clim']
                observed=d['y']*scale+offset+d['clim']
                forecasts={name:pred}
                if name=='ssta':forecasts.update(climatology_only=d['clim'],anomaly_persistence=d['last_ssta'][:,None]+d['clim'])
                for label,f in forecasts.items():
                    for pixel in np.unique(d['pixel']):
                        e=f[d['pixel']==pixel]-observed[d['pixel']==pixel]
                        for h in range(28):metrics.append(dict(model=label,seed=seed,split=s,pixel_id=pixel,horizon=h+1,rmse=float(np.sqrt(np.mean(e[:,h]**2))),mae=float(np.abs(e[:,h]).mean())))
                for h in range(28):rows.append(pd.DataFrame(dict(split=s,pixel_id=d['pixel'],origin=d['origin'],horizon=h+1,observed=observed[:,h],prediction=pred[:,h])))
            pd.concat(rows,ignore_index=True).to_parquet(folder/'predictions.parquet',index=False)
            r=dict(model=name,seed=seed,features=len(ids),parameters=sum(p.numel() for p in m.parameters()),best_epoch=best_epoch,epochs=epoch,seconds=time.perf_counter()-t0)
            runs.append(r);print(json.dumps(r),flush=True)
    pd.DataFrame(metrics).to_csv(OUT/'metrics.csv',index=False)
    pd.DataFrame(runs).to_csv(OUT/'run_summary.csv',index=False)
    pd.DataFrame(logs).to_csv(OUT/'training_log.csv',index=False)
    (OUT/'COMPLETE.json').write_text(json.dumps(dict(runs=len(runs),checkpoint_reload_checks=True)),encoding='utf-8')

if __name__=='__main__':main()
