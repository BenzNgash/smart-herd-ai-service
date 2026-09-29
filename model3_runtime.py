import os, json, math
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from supabase import create_client

ROOT = Path(__file__).resolve().parent
SUPABASE_URL = os.getenv('SUPABASE_URL','').strip()
SUPABASE_SECRET_KEY = os.getenv('SUPABASE_SECRET_KEY','').strip()
FARM_TIMEZONE = os.getenv('FARM_TIMEZONE','Africa/Nairobi').strip()
MODEL_VERSION = os.getenv('MODEL3_VERSION','model3-decision-v3-shadow').strip()
HISTORY_DAYS = int(os.getenv('MODEL3_HISTORY_DAYS','21'))
MIN_HOURLY_COVERAGE = float(os.getenv('MODEL3_MIN_HOURLY_COVERAGE','0.50'))

BASELINE_DAYS = 14
BASELINE_GAP_HOURS = 48
MIN_BASELINE_OBS = 7
WINDOWS = [6,12,24,48]
MIN_BASELINE_QUALITY = 0.50
MIN_COVERAGE_6H = 0.67
MIN_COVERAGE_24H = 0.70
MIN_COVERAGE_48H = 0.65
ROBUST_Z_CLIP = 12.0
SUP_WATCH = 97.0
SUP_CHECK = 99.5
NOVELTY_MODE = 'persistent_check'
NOVELTY_THRESHOLD = 99.8
NOVELTY_Z_SUPPORT = 3.5

SIGNALS = ['walking_prop','resting_prop','feeding_prop','other_prop','activity_level']
Z_COLS = [f'{s}_z' for s in SIGNALS]
DIRECTIONAL = {'walking_prop':'low','resting_prop':'high','feeding_prop':'low','activity_level':'low'}
SCALE_FLOORS = {'walking_prop':0.01,'resting_prop':0.01,'feeding_prop':0.01,'other_prop':0.01,'activity_level':10.0}

BUNDLE = joblib.load(ROOT/'smart_herd_model3_decision_v3.joblib')
FEATURES = list(BUNDLE['features'])
PREPROCESSOR = BUNDLE['preprocessor']
SUPERVISED_MODEL = BUNDLE['supervised_model']
SUPERVISED_REF = np.asarray(BUNDLE['supervised_score_reference'],dtype=np.float64)
IFOREST = BUNDLE['iforest']
IFOREST_REF = np.asarray(BUNDLE['iforest_score_reference'],dtype=np.float64)
SELFTEST = json.loads((ROOT/'model3_v3_selftest.json').read_text())

ADAPTER_WARNING = (
    'SHADOW_RESEARCH ONLY: Model 3 v3 was benchmarked on public hourly '
    'walking/resting/feeding/activity behavior. Live Smart Herd grazing is used '
    'as a temporary feeding-behavior proxy and activity_intensity is mapped to '
    'the public activity signal. This flags health-related behavioral anomalies '
    'for observation; it does not diagnose disease. DeKUT field calibration is required.'
)

class InsufficientHistoryError(RuntimeError):
    pass

@lru_cache(maxsize=1)
def get_supabase():
    if not SUPABASE_URL or not SUPABASE_SECRET_KEY:
        raise RuntimeError('Supabase is not configured for Model 3.')
    return create_client(SUPABASE_URL,SUPABASE_SECRET_KEY)

def empirical_percentile(scores, ref):
    s=np.asarray(scores); r=np.asarray(ref)
    return np.clip(100.0*np.searchsorted(r,s,side='right')/max(1,len(r)),0,100)

def score_features(frame):
    X=PREPROCESSOR.transform(frame[FEATURES])
    X=np.asarray(X,dtype=np.float32)
    p=SUPERVISED_MODEL.predict_proba(X)[:,1]
    sup=empirical_percentile(p,SUPERVISED_REF)
    raw=-IFOREST.score_samples(X)
    nov=empirical_percentile(raw,IFOREST_REF)
    return p.astype(float),sup.astype(float),nov.astype(float)

def self_test_result():
    frame=pd.DataFrame([SELFTEST['features']],columns=FEATURES)
    p,s,n=score_features(frame)
    actual={'supervised_probability':float(p[0]),'supervised_score':float(s[0]),'novelty_score':float(n[0])}
    expected={'supervised_probability':float(SELFTEST['expected_supervised_probability']),'supervised_score':float(SELFTEST['expected_supervised_score']),'novelty_score':float(SELFTEST['expected_novelty_score'])}
    tol=float(SELFTEST['tolerance'])
    diffs={k:abs(actual[k]-expected[k]) for k in actual}
    return {'passed':all(v<=tol for v in diffs.values()),'feature_count':len(FEATURES),'actual':actual,'expected':expected,'absolute_differences':diffs,'tolerance':tol}

def fetch_rows(cow_id):
    start=(datetime.now(timezone.utc)-timedelta(days=HISTORY_DAYS)).isoformat()
    cols=('farm_id,cow_id,ts,walking_prop,grazing_prop,resting_prop,other_prop,'
          'activity_intensity,behavior_transition_rate,mean_model1_confidence,'
          'collar_temperature,data_quality')
    rows=[]; offset=0; page=1000
    while True:
        q=(get_supabase().table('behavior_15min').select(cols).eq('cow_id',cow_id)
           .gte('ts',start).order('ts').range(offset,offset+page-1).execute())
        batch=q.data or []; rows.extend(batch)
        if len(batch)<page: break
        offset += page
        if offset>=20000: break
    return rows

def prepare_hourly(rows,cow_id):
    if not rows: raise InsufficientHistoryError('No behavior_15min rows found for this cow.')
    df=pd.DataFrame(rows).copy()
    required=['ts','walking_prop','grazing_prop','resting_prop','activity_intensity','data_quality']
    missing=[c for c in required if c not in df.columns]
    if missing: raise RuntimeError(f'behavior_15min rows are missing columns: {missing}')
    df['ts']=pd.to_datetime(df['ts'],utc=True,errors='coerce')
    for c in ['walking_prop','grazing_prop','resting_prop','other_prop','activity_intensity','data_quality']:
        if c not in df.columns: df[c]=np.nan
        df[c]=pd.to_numeric(df[c],errors='coerce')
    derived=(1-df.walking_prop-df.grazing_prop-df.resting_prop).clip(0,1)
    df['other_prop']=df.other_prop.fillna(derived)
    df=df.dropna(subset=['ts','walking_prop','grazing_prop','resting_prop','activity_intensity'])
    if df.empty: raise InsufficientHistoryError('No valid behavior_15min rows remain after cleaning.')
    df['local_ts']=df.ts.dt.tz_convert(FARM_TIMEZONE); df['hour_start']=df.local_ts.dt.floor('h')
    g=(df.groupby('hour_start',as_index=False).agg(
        walking_prop=('walking_prop','mean'),resting_prop=('resting_prop','mean'),
        feeding_prop=('grazing_prop','mean'),other_prop=('other_prop','mean'),
        activity_level=('activity_intensity','mean'),row_count=('ts','size'),
        mean_15min_quality=('data_quality','mean')))
    g['hourly_coverage']=(g.row_count.clip(upper=4)/4.0)*g.mean_15min_quality.fillna(0)
    g['observed']=(g.hourly_coverage>=MIN_HOURLY_COVERAGE).astype('int8')
    g.loc[g.observed==0,SIGNALS]=np.nan
    idx=pd.date_range(g.hour_start.min().floor('h'),g.hour_start.max().floor('h'),freq='1h',tz=g.hour_start.iloc[0].tz)
    out=g.set_index('hour_start').reindex(idx); out.index.name='hour_start'
    out['ts']=out.index; out['cow']=cow_id; out['hour_of_day']=out.index.hour.astype('int8')
    out['observed']=out.observed.fillna(0).astype('int8'); out['hourly_coverage']=out.hourly_coverage.fillna(0).astype('float32')
    return out.reset_index(drop=True)

def add_baselines(frame):
    out=frame.copy()
    for s in SIGNALS:
        for suffix in ['baseline_median','baseline_scale','z','baseline_count']: out[f'{s}_{suffix}']=np.nan
    gap=pd.Timedelta(hours=BASELINE_GAP_HOURS); window=f'{BASELINE_DAYS}D'
    for _,idx in out.groupby(['cow','hour_of_day'],sort=False,observed=True).groups.items():
        sub=out.loc[idx,['ts',*SIGNALS]].sort_values('ts'); target=pd.DatetimeIndex(sub.ts)
        src=sub[SIGNALS].copy(); src.index=target+gap
        work=src.reindex(target.union(src.index).sort_values())
        roll=work.rolling(window=window,min_periods=MIN_BASELINE_OBS,closed='both')
        med=roll.median().reindex(target); q25=roll.quantile(.25).reindex(target); q75=roll.quantile(.75).reindex(target); cnt=roll.count().reindex(target)
        for s in SIGNALS:
            scale=((q75[s]-q25[s])/1.349).clip(lower=SCALE_FLOORS[s])
            z=np.clip((sub[s].to_numpy(dtype='float64')-med[s].to_numpy())/scale.to_numpy(),-ROBUST_Z_CLIP,ROBUST_Z_CLIP)
            out.loc[sub.index,f'{s}_baseline_median']=med[s].to_numpy(); out.loc[sub.index,f'{s}_baseline_scale']=scale.to_numpy()
            out.loc[sub.index,f'{s}_z']=z; out.loc[sub.index,f'{s}_baseline_count']=cnt[s].to_numpy()
    counts=[f'{s}_baseline_count' for s in SIGNALS]
    out['baseline_days_available']=out[counts].min(axis=1)
    out['baseline_quality']=(out.baseline_days_available/BASELINE_DAYS).clip(0,1)
    return out

def add_recent(frame):
    out=frame.copy(); active=out.walking_prop+out.feeding_prop
    probs=out[['walking_prop','resting_prop','feeding_prop','other_prop']].clip(1e-8,1)
    out['active_prop']=active.astype('float32'); out['rest_to_active_ratio']=(out.resting_prop/(active+.02)).clip(0,25).astype('float32')
    out['behavior_entropy']=(-(probs*np.log(probs)).sum(axis=1,min_count=1)).astype('float32')
    out['hour_sin']=np.sin(2*np.pi*out.hour_of_day/24).astype('float32'); out['hour_cos']=np.cos(2*np.pi*out.hour_of_day/24).astype('float32')
    for w in WINDOWS:
        out[f'coverage_{w}h']=np.nan
        for zc in Z_COLS:
            base=zc[:-2]
            for stat in ['mean','std','absmax','vsmean']: out[f'{base}_z_{stat}_{w}h']=np.nan
        for base,direction in DIRECTIONAL.items():
            out[f'{base}_{direction}_frac_{w}h']=np.nan; out[f'{base}_z_range_{w}h']=np.nan
    for _,idx in out.groupby('cow',sort=False,observed=True).groups.items():
        sub=out.loc[idx].sort_values('ts'); ti=pd.DatetimeIndex(sub.ts)
        zdata=sub[Z_COLS].copy(); zdata.index=ti
        obs=pd.Series(sub.observed.astype(float).to_numpy(),index=ti)
        for w in WINDOWS:
            win=f'{w}h'; minp=max(2,int(math.ceil(w*.5)))
            out.loc[sub.index,f'coverage_{w}h']=(obs.rolling(win,min_periods=1,closed='both').sum()/float(w)).clip(0,1).to_numpy()
            roll=zdata.rolling(win,min_periods=minp,closed='both'); mean=roll.mean(); std=roll.std(ddof=0); rmax=roll.max(); rmin=roll.min()
            for zc in Z_COLS:
                base=zc[:-2]; absmax=zdata[zc].abs().rolling(win,min_periods=minp,closed='both').max()
                out.loc[sub.index,f'{base}_z_mean_{w}h']=mean[zc].to_numpy(); out.loc[sub.index,f'{base}_z_std_{w}h']=std[zc].to_numpy()
                out.loc[sub.index,f'{base}_z_absmax_{w}h']=absmax.to_numpy(); out.loc[sub.index,f'{base}_z_vsmean_{w}h']=zdata[zc].to_numpy()-mean[zc].to_numpy()
            for base,direction in DIRECTIONAL.items():
                zc=f'{base}_z'; valid=zdata[zc].notna(); ind=((zdata[zc]<=-2) if direction=='low' else (zdata[zc]>=2)).astype(float).where(valid,np.nan)
                out.loc[sub.index,f'{base}_{direction}_frac_{w}h']=ind.rolling(win,min_periods=minp,closed='both').mean().to_numpy()
                out.loc[sub.index,f'{base}_z_range_{w}h']=(rmax[zc]-rmin[zc]).to_numpy()
    out['model3_data_ok']=((out.baseline_quality>=MIN_BASELINE_QUALITY)&(out.coverage_6h>=MIN_COVERAGE_6H)&(out.coverage_24h>=MIN_COVERAGE_24H)&(out.coverage_48h>=MIN_COVERAGE_48H)).astype('int8')
    return out

def apply_decision(scored):
    out=scored.sort_values(['cow','ts']).copy()
    out['max_abs_personal_z']=out[['walking_prop_z','resting_prop_z','feeding_prop_z','activity_level_z']].abs().max(axis=1)
    out['directional_persistence_6h']=out[['walking_prop_low_frac_6h','resting_prop_high_frac_6h','feeding_prop_low_frac_6h','activity_level_low_frac_6h']].max(axis=1)
    out['sup_watch_seed']=out.supervised_score>=SUP_WATCH; out['sup_check_seed']=out.supervised_score>=SUP_CHECK
    support=(out.max_abs_personal_z>=NOVELTY_Z_SUPPORT)|(out.directional_persistence_6h>=.5)
    out['novelty_seed']=(out.novelty_score>=NOVELTY_THRESHOLD)&support
    out['urgent_seed']=(out.activity_level_z<=-3)&(out.resting_prop_z>=2.5)&(out.walking_prop_z<=-2)&(out.baseline_quality>=MIN_BASELINE_QUALITY)&(out.coverage_6h>=MIN_COVERAGE_6H)
    for c in ['sup_persistent_check','sup_persistent_watch','novelty_persistent','urgent_check']: out[c]=False
    for _,idx in out.groupby('cow',sort=False,observed=True).groups.items():
        sub=out.loc[idx].sort_values('ts'); ti=pd.DatetimeIndex(sub.ts)
        def persistent(col,win,n): return pd.Series(sub[col].astype(int).to_numpy(),index=ti).rolling(win,min_periods=1,closed='both').sum()>=n
        out.loc[sub.index,'sup_persistent_check']=persistent('sup_check_seed','3h',2).to_numpy()
        out.loc[sub.index,'sup_persistent_watch']=persistent('sup_watch_seed','4h',3).to_numpy()
        out.loc[sub.index,'novelty_persistent']=persistent('novelty_seed','3h',2).to_numpy()
        out.loc[sub.index,'urgent_check']=persistent('urgent_seed','3h',2).to_numpy()
    out['primary_actionable']=out.sup_persistent_check|out.sup_persistent_watch
    novelty_actionable=out.novelty_persistent if NOVELTY_MODE=='persistent_check' else pd.Series(False,index=out.index)
    out['actionable']=out.primary_actionable|novelty_actionable|out.urgent_check
    watch=out.sup_watch_seed | (out.novelty_seed if NOVELTY_MODE in ['watch_only','persistent_check'] else False)
    status=np.full(len(out),'NORMAL',dtype=object); status[watch.to_numpy()]='WATCH'; status[out.actionable.to_numpy()]='CHECK'; status[out.urgent_check.to_numpy()]='URGENT_CHECK'; out['status']=status
    path=np.full(len(out),'NONE',dtype=object); path[out.sup_watch_seed.to_numpy()]='SUPERVISED_WATCH'; path[out.novelty_seed.to_numpy()]='NOVELTY_WATCH'; path[out.primary_actionable.to_numpy()]='SUPERVISED_PERSISTENT'; path[out.novelty_persistent.to_numpy()]='NOVELTY_PERSISTENT'; path[(out.primary_actionable&out.novelty_persistent).to_numpy()]='SUPERVISED_PLUS_NOVELTY'; path[out.urgent_check.to_numpy()]='URGENT_INACTIVITY'; out['decision_path']=path
    out['priority_score']=np.maximum(out.supervised_score,out.novelty_score*.95).clip(0,100)
    return out

def reason_pair(row):
    vals=[]
    for col,label in [('walking_prop_z','Walking'),('resting_prop_z','Resting'),('feeding_prop_z','Grazing/feeding proxy'),('activity_level_z','Activity')]:
        v=row.get(col,np.nan)
        if pd.notna(v): vals.append((abs(float(v)),float(v),label))
    vals.sort(reverse=True)
    def phrase(x): return f"{x[2]} {'above' if x[1]>0 else 'below'} personal baseline ({x[1]:+.1f} SD)"
    primary=phrase(vals[0]) if vals else 'Persistent multivariate behavior change'; secondary=phrase(vals[1]) if len(vals)>1 else None
    if row.get('decision_path')=='URGENT_INACTIVITY': primary='Severe persistent inactivity pattern'
    elif row.get('decision_path') in ['NOVELTY_PERSISTENT','SUPERVISED_PLUS_NOVELTY'] and secondary is None: secondary='Novel pattern outside the learned normal range'
    return primary,secondary

def persistence_hours(decisions):
    if decisions.empty: return 0
    d=decisions.sort_values('ts'); latest=d.iloc[-1]
    if latest.status=='NORMAL': return 0
    count=1; prev=pd.Timestamp(latest.ts)
    for i in range(len(d)-2,-1,-1):
        row=d.iloc[i]; ts=pd.Timestamp(row.ts)
        if row.status=='NORMAL' or (prev-ts)>pd.Timedelta(hours=1.5): break
        count+=1; prev=ts
    return count

def build_latest(rows,cow_id):
    e=add_recent(add_baselines(prepare_hourly(rows,cow_id)))
    complete=e[FEATURES].notna().all(axis=1); eligible=(e.observed==1)&(e.model3_data_ok==1)&complete
    er=e.loc[eligible].copy()
    if er.empty:
        maxdays=float(e.baseline_days_available.dropna().max()) if e.baseline_days_available.notna().any() else 0
        raise InsufficientHistoryError(f'No complete Model 3 feature row is available yet. Maximum baseline days currently available: {maxdays:.0f}.')
    latest_ts=er.ts.max(); recent=er[er.ts>=latest_ts-pd.Timedelta(hours=8)].copy()
    p,s,n=score_features(recent); recent['supervised_probability']=p; recent['supervised_score']=s; recent['novelty_score']=n
    dec=apply_decision(recent); latest=dec.sort_values('ts').iloc[-1]
    primary,secondary=reason_pair(latest); persist=persistence_hours(dec)
    dq=float(np.nanmin([latest.get('hourly_coverage',np.nan),latest.get('coverage_6h',np.nan),latest.get('coverage_24h',np.nan),latest.get('coverage_48h',np.nan)]))
    farm_id=next((r.get('farm_id') for r in reversed(rows) if r.get('farm_id')),None)
    return {'farm_id':farm_id,'cow_id':cow_id,'timestamp_local':pd.Timestamp(latest.ts).isoformat(),'priority_score':float(latest.priority_score),'status':str(latest.status),'actionable':bool(latest.actionable),'decision_path':str(latest.decision_path),'supervised_probability':float(latest.supervised_probability),'supervised_score':float(latest.supervised_score),'novelty_score':float(latest.novelty_score),'baseline_days_available':float(latest.baseline_days_available),'baseline_quality':float(latest.baseline_quality),'data_quality':dq,'walking_z':float(latest.walking_prop_z),'resting_z':float(latest.resting_prop_z),'feeding_proxy_z':float(latest.feeding_prop_z),'activity_z':float(latest.activity_level_z),'persistence_hours':int(persist),'primary_reason':primary,'secondary_reason':secondary}

def persist(result):
    payload={**result,'ts':result['timestamp_local'],'inference_run_at':datetime.now(timezone.utc).isoformat(),'model_version':MODEL_VERSION,'adapter_warning':ADAPTER_WARNING}
    payload.pop('timestamp_local',None)
    return get_supabase().table('health_anomaly_predictions').upsert(payload,on_conflict='cow_id,ts,model_version').execute().data

def shadow(cow_id,persist_result=False):
    rows=fetch_rows(cow_id); r=build_latest(rows,cow_id)
    out={'status':'shadow_prediction','cow_id':cow_id,'timestamp':r['timestamp_local'],'research_status':r['status'],'priority_score':r['priority_score'],'actionable_research_flag':r['actionable'],'decision_path':r['decision_path'],'supervised_probability':r['supervised_probability'],'supervised_score':r['supervised_score'],'novelty_score':r['novelty_score'],'model_version':MODEL_VERSION,'feature_count':len(FEATURES),'diagnostics':{k:r[k] for k in ['baseline_days_available','baseline_quality','data_quality','walking_z','resting_z','feeding_proxy_z','activity_z','persistence_hours','primary_reason','secondary_reason']},'adapter_warning':ADAPTER_WARNING,'disease_diagnosis':None,'persisted':False}
    if persist_result: persist(r); out['persisted']=True
    return out
