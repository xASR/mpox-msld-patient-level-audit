"""Frozen-role metadata classification and calibration-only threshold selection."""
from pathlib import Path
import argparse, json, os, time
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits
from stats_ext import CLASSES,METRICS,estimates,weighted,wilson
from features import META

SEEDS=(42,17,73,101,137,211)
TIERS=('1A','1B')
MODELS=('logistic','forest')
PREDICTORS=('ensemble','convnext_tiny','convnext_tiny_scaled','swin_small','swin_small_scaled')
POINTS=('youden','sens85','sens90','secondary_sens95')

def threshold(y,p,rule):
    positive=y==2
    if not positive.any() or positive.all():raise ValueError('Calibration class missing')
    candidates=np.r_[np.nextafter(p.max(),np.inf),np.unique(p)[::-1]]
    call=p[:,None]>=candidates
    sens=call[positive].mean(0)
    spec=(~call[~positive]).mean(0)
    if rule=='youden':
        # Integer numerator of J removes numerical ambiguity at ties.
        objective=call[positive].sum(0)*(~positive).sum()-call[~positive].sum(0)*positive.sum()
        use=np.flatnonzero(objective==objective.max())
        return candidates[use[np.argmax(spec[use])]]
    target={'sens85':.85,'sens90':.90,'secondary_sens95':.95}[rule]
    return candidates[np.flatnonzero(sens>=target)[0]]

def operating(package,out):
    d=pd.read_csv(package/'operating.csv')
    predictions=[]; cuts=[]
    for (seed,fold),rows in d.groupby(['seed','fold']):
        cal=rows[rows.role=='calibration']; test=rows[rows.role=='test']
        assert len(cal) and not set(cal.group)&set(test.group)
        for predictor in PREDICTORS:
            for rule in POINTS:
                t=threshold(cal.label.to_numpy(),cal[predictor].to_numpy(),rule)
                cut=dict(seed=seed,fold=fold,predictor=predictor,rule=rule,threshold=t,
                         calibration_n=len(cal),calibration_positive=sum(cal.label==2),
                         calibration_sensitivity=(cal.loc[cal.label==2,predictor]>=t).mean())
                cuts.append(cut)
                predictions.append(test[['seed','fold','group','label']].assign(predictor=predictor,rule=rule,
                    score=test[predictor],positive=test[predictor]>=t))
    scored=pd.concat(predictions); summary=[]
    for (seed,predictor,rule),rows in scored.groupby(['seed','predictor','rule']):
        assert len(rows)==rows.group.nunique()==520
        y=rows.label==2; call=rows.positive
        tp,fp,fn,tn=[int(mask.sum()) for mask in (y&call,~y&call,y&~call,~y&~call)]
        rec=dict(seed=seed,predictor=predictor,rule=rule,tp=tp,fp=fp,tn=tn,fn=fn)
        for name,x,n in [('sensitivity',tp,tp+fn),('specificity',tn,tn+fp),('ppv',tp,tp+fp),('npv',tn,tn+fn)]:
            rec[name],rec[name+'_lo'],rec[name+'_hi']=wilson(x,n)
        for label,c in enumerate(CLASSES):
            if label!=2:rec['FP_'+c]=int(sum(call&(rows.label==label)))
        summary.append(rec)
    frame=pd.DataFrame(summary)
    fields=['sensitivity','specificity','ppv','npv','fp']+['FP_'+c for c in CLASSES if c!='Mpox']
    stability=frame.groupby(['predictor','rule'])[fields].agg(['mean','std','count'])
    stability.columns=['_'.join(c) for c in stability.columns]
    stability.to_csv(out/'E3_six_seed_summary.csv')
    frame.to_csv(out/'E3_each_seed.csv',index=False)
    frame[frame.seed==42].to_csv(out/'E3_primary.csv',index=False)
    pd.DataFrame(cuts).to_csv(out/'E3_fold_thresholds.csv',index=False)
    scored.to_csv(out/'E3_heldout_decisions.csv',index=False)

def model_for(name):
    if name=='logistic':return make_pipeline(StandardScaler(),LogisticRegression(C=1.,max_iter=2000))
    return RandomForestClassifier(n_estimators=32,min_samples_leaf=2,max_features='sqrt',random_state=42,n_jobs=1)

def predict(x,y,roles,name):
    p=np.full((len(y),6),np.nan)
    for train,test in roles:
        model=model_for(name)
        model.fit(x[train],y[train])
        assert np.array_equal(model.classes_,np.arange(6))
        p[test]=model.predict_proba(x[test])
    assert np.isfinite(p).all()
    return p

def per_group(p,inv,ng):
    g=np.zeros((ng,6))
    np.add.at(g,inv,p)
    return g/np.bincount(inv)[:,None]

def permuted(k,matrices,roles,inv,gy,blocks,first):
    rng=np.random.default_rng(470000+k)
    labels=gy.copy()
    for b in blocks: labels[b]=rng.permutation(gy[b])
    y=labels[inv]; rows=[]
    for tier,matrix in matrices.items():
        for name in MODELS:
            p=predict(matrix,y,roles,name)
            for unit,truth,prob in [('image',y,p),('group',labels,per_group(p,inv,len(gy)))]:
                scores=weighted(truth,prob,np.ones((1,len(truth))))[0]
                rows.append(dict(permutation=k,tier=tier,model=name,unit=unit,**dict(zip(METRICS[:3],scores[:3]))))
    return rows

def metadata(package,out,deadline=float('inf')):
    feat=pd.read_csv(package/'features.csv').set_index('filename').sort_index()
    folds=pd.read_csv(package/'a_roles.csv')
    matrices={'1A':feat[META].to_numpy(),'1B':feat.to_numpy()}
    summaries=[]; prediction_rows=[]
    null_args=None
    for seed in SEEDS:
        roles=[];foldid=np.zeros(len(feat),int)
        for f in range(1,6):
            a=folds[(folds.seed==seed)&(folds.fold==f)].set_index('filename').loc[feat.index]
            tr=a.role.eq('train').to_numpy(); te=a.role.eq('test').to_numpy()
            assert not set(a.patient_cluster[tr])&set(a.patient_cluster[te])
            assert not foldid[te].any()
            foldid[te]=f;roles.append((tr,te))
        assert foldid.min()==1
        y=a.label.to_numpy()
        groups,first,inv=np.unique(a.patient_cluster.to_numpy(),return_index=True,return_inverse=True)
        gy=y[first]
        assert np.array_equal(gy[inv],y)
        for tier,x in matrices.items():
            for name in MODELS:
                p=predict(x,y,roles,name)
                prediction_rows.append(pd.DataFrame(p,columns=CLASSES).assign(filename=feat.index,group=groups[inv],
                    label=y,seed=seed,fold=foldid,tier=tier,model=name))
                for unit,labels,prob,grp in [('image',y,p,groups[inv]),('group',gy,per_group(p,inv,len(gy)),groups)]:
                    summaries.append(dict(seed=seed,tier=tier,model=name,unit=unit,n=len(labels),**estimates(labels,prob,grp)))
        if seed==42:
            blocks=[np.where(foldid[first]==f)[0] for f in range(1,6)]
            null_args=(matrices,roles,inv,gy,blocks,first)
            for unit,labels,grp in [('image',y,groups[inv]),('group',gy,groups)]:
                prior=np.bincount(labels,minlength=6)/len(labels)
                summaries.append(dict(seed=seed,tier='reference',model='fixed_majority',unit=unit,n=len(labels),
                    **estimates(labels,np.tile(prior,(len(labels),1)),grp)))
    obs=pd.DataFrame(summaries);obs.to_csv(out/'E1_metrics.csv',index=False)
    pd.concat(prediction_rows).to_csv(out/'E1_oof.csv',index=False)
    # Immutable blocks allow interruption and continuation from the full output.
    cache=out/'permutations';cache.mkdir(exist_ok=True)
    for start in range(0,999,12):
        dest=cache/f'block_{start:03d}.csv'
        if dest.exists():continue
        if time.time()+max(180,2*locals().get('block_seconds',0))>=deadline:
            return False
        block_start=time.time()
        with Parallel(n_jobs=min(4,os.cpu_count() or 1),backend='loky',inner_max_num_threads=1) as pool:
            result=pool(delayed(permuted)(k,*null_args) for k in range(start,min(start+12,999)))
        tmp=dest.with_suffix('.tmp')
        pd.DataFrame([row for block in result for row in block]).to_csv(tmp,index=False)
        tmp.replace(dest);print('Permutations completed:',min(start+12,999),flush=True)
        block_seconds=time.time()-block_start
    null=pd.concat(pd.read_csv(p) for p in sorted(cache.glob('*.csv')))
    result=[]
    for (tier,model,unit),block in null.groupby(['tier','model','unit']):
        assert sorted(block.permutation)==list(range(999))
        anchor=obs[(obs.seed==42)&(obs.tier==tier)&(obs.model==model)&(obs.unit==unit)].iloc[0]
        for m in METRICS[:3]:
            result.append(dict(seed=42,tier=tier,model=model,unit=unit,metric=m,observed=anchor[m],
                permutations=999,p=(1+(block[m]>=anchor[m]-1e-14).sum())/1000))
    pd.DataFrame(result).to_csv(out/'E1_primary_permutation.csv',index=False)
    return True

if __name__=='__main__':
    import sys
    from package_io import finish, identity
    package,out=map(Path,sys.argv[1:3])
    out.mkdir(parents=True,exist_ok=True)
    identity(package,out)
    start=time.time()
    with threadpool_limits(limits=1):
        operating(package,out)
        complete=metadata(package,out,float(sys.argv[3]) if len(sys.argv)>3 else float('inf'))
    finish(out,'CPU',time.time()-start,complete)

