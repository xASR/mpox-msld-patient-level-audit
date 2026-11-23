"""Consolidate only complete, checksum-verified experiment outputs. See --help."""
import argparse,io,json,os,shutil,tempfile,zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import binomtest
from sklearn.metrics import f1_score,log_loss
from stats_ext import METRICS,estimates,mcnemar,scale,weighted
from package_io import get,put,sha

def holm(values):
    p=np.asarray(values);order=np.argsort(p);a=np.empty(len(p))
    a[order]=np.minimum(1,np.maximum.accumulate(p[order]*(len(p)-np.arange(len(p)))))
    return a

def open_result(path,owner,temp,package):
    dest=temp/owner;dest.mkdir()
    with zipfile.ZipFile(path) as archive:
        members=archive.namelist()
        assert len(set(members))==len(members)
        index=json.loads(archive.read('result_index.json'))
        if index['worker']!=owner or index['complete'] is not True:raise ValueError(f'Incomplete or wrong result: {owner}')
        for name,h in index['files'].items():
            target=(dest/name).resolve()
            assert target.is_relative_to(dest.resolve())
            target.parent.mkdir(parents=True,exist_ok=True)
            with archive.open(name) as src,target.open('wb') as dst:shutil.copyfileobj(src,dst)
            if sha(target)!=h:raise ValueError('Result checksum differs: '+name)
    assert get(dest/'input_identity.json')['input_manifest_sha256']==sha(package/'extension_manifest.json')
    return dest

def table(out,name,frame):
    frame.to_csv(out/(name+'.csv'),index=False)
    view=frame.copy()
    for c in view.select_dtypes('float'):
        view[c]=view[c].map(lambda v:'—' if pd.isna(v) else f'{v:.4f}')
    def line(values):
        return '| '+' | '.join(str(v).replace('|','\\|') for v in values)+' |'
    lines=[line(view.columns),line(['---']*len(view.columns))]
    lines.extend(line(values) for values in view.itertuples(index=False,name=None))
    (out/(name+'.md')).write_text('\n'.join(lines)+'\n',encoding='utf8')

def pooled(files,reference):
    merged=[]
    for side,target in [('val','calibration'),('test','test')]:
        parts=[]
        for f in range(1,6):
            z=np.load(files(f,side))
            df=pd.DataFrame(z['probs'],columns=list('abcdef'))
            parts.append(df.assign(group=z['pids'].astype(str),label=z['labels']))
        d=pd.concat(parts)
        assert d.groupby('group').label.nunique().max()==1
        d=d.groupby('group').mean().sort_index()
        if side=='val':d=d[~d.index.isin(reference['test_groups'])]
        assert np.array_equal(d.index,reference[target+'_groups']) and np.array_equal(d.label,reference[target+'_labels'])
        p=d[list('abcdef')].to_numpy()
        merged.append(p/p.sum(1,keepdims=True))
    return merged

def efficient(pkg,out,workers,jobs):
    z=np.load(pkg/'b_reference.npz');e=np.load(pkg/'e9_reference.npz');y=z['test_labels'];groups=z['test_groups']
    lookup={j['fold']:workers[j['owner']]/j['id'] for j in jobs if j.get('arm')=='2a'}
    dirs={'2a':lookup,'2b':{j['fold']:workers[j['owner']]/j['id'] for j in jobs if j.get('arm')=='2b'}}
    configs={};status=[]
    for origin in ('A','B'):
        native=pooled(lambda f,s:pkg/f'ref_{origin}_{f}_{s}.npz',z)
        configs[origin+'_native']=native
        other='B' if origin=='A' else 'A'
        cross={f:workers['Me']/f'cross_{origin}_{f}' for f in range(1,6)}
        good=[f for f,v in cross.items() if (v/'complete.json').is_file()]
        status.append(dict(origin=origin,verified_folds=','.join(map(str,good)),pooled_available=len(good)==5))
        if len(good)==5:
            configs[origin+'_cross']=pooled(lambda f,s:cross[f]/f'{other}_{s}.npz',z)
    table(out,'E2_crossover_coverage',pd.DataFrame(status))
    for arm,byfold in dirs.items():
        for recipe in ('A','B'):
            configs[arm+'_'+recipe]=pooled(lambda f,s:byfold[f]/f'{recipe}_{s}.npz',z)
    outcomes=[];predictions={};paired=[]
    for name,(cal,test) in configs.items():
        cy=z['calibration_labels']
        fit=minimize_scalar(lambda t:log_loss(cy,scale(cal,t),labels=np.arange(6)),bounds=(.1,5.),method='bounded')
        assert fit.success
        processed=scale(test,fit.x)
        if name in ('A_native','B_native'):
            baseline=e['new_raw'] if name=='A_native' else z['efficientnetv2_s_raw']
            assert np.max(abs(baseline-test))<1e-6
            processed=e['new_scaled'] if name=='A_native' else z['efficientnetv2_s_scaled']
        for form,p in [('raw',test),('scaled',processed)]:
            predictions[name,form]=p
            outcomes.append(dict(configuration=name,form=form,n=len(y),correct=sum(p.argmax(1)==y),
                calibration_refit_T=fit.x,**estimates(y,p,groups)))
    for new,ref in [('2a_A','A_native'),('2b_A','A_native'),('A_cross','A_native'),('B_cross','B_native'),
                    ('2a_B','2a_A'),('2b_B','2b_A'),('A_native','B_native')]:
        if new not in configs:continue
        for form in ('raw','scaled'):
            a=predictions[new,form];b=predictions[ref,form]
            paired.append(dict(new=new,reference=ref,form=form,**mcnemar(y,a,b),**estimates(y,a,groups,b)))
    paired=pd.DataFrame(paired)
    paired['p_holm']=paired.groupby('form').p_exact.transform(holm)
    table(out,'E2_performance',pd.DataFrame(outcomes));table(out,'E2_paired_differences',paired)
    np.savez_compressed(out/'E2_pooled_probabilities.npz',groups=groups,labels=y,**{'_'.join(k):v for k,v in predictions.items()})

def dose(pkg,out,workers,jobs):
    rows=[];long=[]
    for s in range(5):
        anchor=np.load(pkg/f'dose/{s}/anchor.npz')
        test=pd.read_csv(pkg/f'dose/{s}/test.csv')
        for n in (0,10,26,52,104):
            if n:
                j=next(j for j in jobs if j['family']=='dose' and j['seed']==s and j['n']==n)
                path=workers[j['owner']]/j['id'];z=np.load(path/'dose.npz')
                mask=get(pkg/f'dose/{s}/mask{n}.json')
                exposed=np.isin(z['groups'],mask['groups'])
                assert np.array_equal(exposed,z['exposed']) and len(np.unique(z['groups'][exposed]))==n
            else:z=anchor;exposed=np.zeros(len(test),bool)
            for key in ['filenames','labels','groups']:assert np.array_equal(z[key],anchor[key])
            assert np.array_equal(z['filenames'],test.filename)
            y=z['labels'];g=z['groups'];p=z['probs'];a=anchor['probs']
            long.append(pd.DataFrame(dict(seed=s,n=n,group=g,label=y,filename=z['filenames'],exposed=exposed,
                pred=p.argmax(1),clean_pred=a.argmax(1))))
            for unit in ('image','group'):
                if unit=='group':
                    gu,first,ix=np.unique(g,return_index=True,return_inverse=True)
                    def average(mat):
                        arr=np.zeros((len(gu),6));np.add.at(arr,ix,mat);return arr/np.bincount(ix)[:,None]
                    yp,score,clean,seen=y[first],average(p),average(a),exposed[first]
                else:yp,score,clean,seen=y,p,a,exposed
                for subset,take in [('overall',np.ones(len(yp),bool)),('exposed',seen),('unexposed',~seen)]:
                    met=dict(accuracy=np.nan,macro_f1=np.nan,clean_accuracy=np.nan,clean_macro_f1=np.nan)
                    if take.any():
                        for key,mat in [('',score),('clean_',clean)]:
                            met[key+'accuracy']=np.mean(mat[take].argmax(1)==yp[take])
                            met[key+'macro_f1']=f1_score(yp[take],mat[take].argmax(1),labels=range(6),average='macro',zero_division=0)
                    rows.append(dict(seed=s,exposed_groups=n,exposure_fraction=n/104,unit=unit,subset=subset,
                                     denominator=take.sum(),**met))
    frame=pd.DataFrame(rows)
    base=frame.query("exposed_groups==0 and unit=='image' and subset=='overall'").accuracy.mean()
    assert round(100*base,2)==83.03
    for m in ('accuracy','macro_f1'):frame[m+'_change']=frame[m]-frame['clean_'+m]
    table(out,'E4_each_seed_and_subset',frame)
    summary=frame.groupby(['exposed_groups','exposure_fraction','unit','subset'])[['accuracy','macro_f1','accuracy_change','macro_f1_change']].agg(['mean','std','count'])
    summary.columns=['_'.join(k) for k in summary.columns]
    table(out,'E4_curve',summary.reset_index())
    raw=pd.concat(long);raw.to_csv(out/'E4_all_image_decisions.csv',index=False)
    contrasts=[]
    # Shared group multipliers across all five seeds preserve recurring test groups.
    gids=np.unique(raw.group);rng=np.random.default_rng(615)
    draws=rng.multinomial(len(gids),np.full(len(gids),1/len(gids)),size=2000)
    for n in (10,26,52,104):
        for subset in ('overall','exposed','unexposed'):
            sample=raw[raw.n==n]
            if subset!='overall':sample=sample[sample.exposed==(subset=='exposed')]
            if sample.empty:continue
            delta=[];observed=[]
            for s,d in sample.groupby('seed'):
                w=draws[:,np.searchsorted(gids,d.group)]
                D=(d.pred==d.label).to_numpy(int)-(d.clean_pred==d.label).to_numpy(int)
                nz=w.sum(1);delta.append(np.divide(w@D,nz,where=nz>0,out=np.full(len(nz),np.nan)))
                observed.append(D.mean())
            vals=np.mean(delta,axis=0);lo,hi=np.nanquantile(vals,[.025,.975])
            pos=sum(x>0 for x in observed);neg=sum(x<0 for x in observed)
            contrasts.append(dict(exposed_groups=n,subset=subset,paired_accuracy_change=np.mean(observed),
                cluster_boot_lo=lo,cluster_boot_hi=hi,valid_boot=np.isfinite(vals).sum(),
                positive_seeds=pos,negative_seeds=neg,seed_sign_p=binomtest(pos,pos+neg).pvalue if pos+neg else 1.))
    table(out,'E4_paired_image_accuracy',pd.DataFrame(contrasts))

def precision(out,workers,jobs):
    rows=[];epochs=[];initials={};aligned={};preds={}
    failures=[]
    for j in jobs:
        if j['family']!='beit':continue
        path=workers[j['owner']]/j['id'];s,reg=j['seed'],j['regime']
        sig=get(path/'init.json')['initial'];assert initials.setdefault(s,sig)==sig
        terminal=get(path/'complete.json') if (path/'complete.json').exists() else {}
        if terminal.get('outcome')=='diverged':
            failures.append(dict(seed=s,regime=reg,completed_epochs=terminal['epochs'],reason=terminal['reason']))
            continue
        p=np.load(path/'A_test.npz');y=p['labels'];prob=p['probs'];ids=p['pids'];preds[s,reg]=prob
        assert len(y)==105 and aligned.setdefault('groups',list(ids))==list(ids)
        assert aligned.setdefault('labels',list(y))==list(y)
        history=get(path/'history.json');assert len(history)==25
        for row in history:
            steps=pd.read_csv(path/f"steps_{row['epoch']:02d}.csv")
            assert len(steps)==row['scheduled'] and steps.skipped.sum()==row['skipped']
            epochs.append(dict(seed=s,regime=reg,**row,gradient_l2_max=steps.gradient_l2.max(),
                gradient_component_max=steps.gradient_max_abs.max(),nan_steps=steps.gradient_nan.sum(),
                inf_steps=steps.gradient_inf.sum(),scale_start=steps.scale_before.iloc[0],scale_end=steps.scale_after.iloc[-1]))
        rows.append(dict(seed=s,regime=reg,n=105,correct=sum(prob.argmax(1)==y),skips=sum(h['skipped'] for h in history),
            best_selection_logloss=min(h['selection_logloss'] for h in history),**estimates(y,prob,ids)))
    table(out,'E5_numerical_failures',pd.DataFrame(failures,columns=['seed','regime','completed_epochs','reason']))
    df=pd.DataFrame(rows);table(out,'E5_fixed_fold_seed_results',df)
    if not rows:return
    table(out,'E5_epoch_diagnostics',pd.DataFrame(epochs))
    summary=df.groupby('regime')[['accuracy','macro_f1','macro_auc','ece','skips']].agg(['mean','std'])
    summary.columns=['_'.join(k) for k in summary.columns]
    table(out,'E5_regime_summary',summary.reset_index())
    comparisons=[]
    for reg in ('clip','fp32'):
        differences=[]
        for s in sorted(initials):
            if (s,reg) not in preds or (s,'amp') not in preds:continue
            dif=(preds[s,reg].argmax(1)==y).astype(int)-(preds[s,'amp'].argmax(1)==y).astype(int)
            differences.append(dif)
        if not differences:continue
        d=np.asarray(differences);avg=d.mean(0)
        rng=np.random.default_rng(615);boot=avg[rng.integers(0,len(y),(2000,len(y)))].mean(1)
        pos=sum(d.mean(1)>0);neg=sum(d.mean(1)<0)
        comparisons.append(dict(regime=reg,reference='amp',paired_completed_seeds=len(d),accuracy_change=d.mean(),
            group_boot_lo=np.quantile(boot,.025),group_boot_hi=np.quantile(boot,.975),
            positive_seeds=pos,negative_seeds=neg,seed_sign_p=binomtest(pos,pos+neg).pvalue if pos+neg else 1.))
    table(out,'E5_paired_regimes',pd.DataFrame(comparisons))

def main(pkg,archives,out):
    for rel,h in get(pkg/'extension_manifest.json').items():
        file=(pkg/rel).resolve()
        if not file.is_relative_to(pkg.resolve()) or sha(file)!=h:raise ValueError('Input checksum mismatch: '+rel)
    out.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory() as temp:
        workers={s:open_result(archives[s],s,Path(temp),pkg) for s in ('CPU','Me','Friend1','Friend2')}
        jobs=get(pkg/'schedule.json');assert len(jobs)==45 and len({j['id'] for j in jobs})==45
        for j in jobs:
            record=get(workers[j['owner']]/j['id']/'complete.json')
            assert record['job']==j and record['manifest']==sha(pkg/'extension_manifest.json')
            assert record['epochs']==25 or (j['family']=='beit' and record.get('outcome')=='diverged')
        for source in workers['CPU'].glob('*.csv'):
            d=pd.read_csv(source)
            if source.stem=='E1_primary_permutation':d['p_holm_all_tests']=holm(d.p)
            if source.stem in ('E1_oof','E3_heldout_decisions','E3_fold_thresholds'):shutil.copyfile(source,out/source.name)
            else:table(out,source.stem,d)
        efficient(pkg,out,workers,jobs);dose(pkg,out,workers,jobs);precision(out,workers,jobs)
        environments=[get(workers[j['owner']]/j['id']/'complete.json')['environment'] for j in jobs]
        unique={json.dumps(e,sort_keys=True):e for e in environments}
        put(out/'scientific_provenance.json',dict(environments=list(unique.values()),models=['EfficientNetV2-S','ConvNeXt-Tiny','BEiT-Base'],
            protocol_A_split_seeds=[42,17,73,101,137,211],EfficientNet_seeds=list(range(43,48)),
            fraction_holdout_seeds=list(range(5)),BEiT_fold2_training_seeds=list(range(344,349)),
            bootstrap_draws=2000,primary_group_permutations=999))
    publication_tables(out)
    put(out/'publication_manifest.json',dict(input_sha256=sha(pkg/'extension_manifest.json'),
        result_archive_hashes=sorted(sha(p) for p in archives.values()),files={p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file() and p.name!='publication_manifest.json'}))

def publication_tables(out):
    def ci(row,key,factor=1,places=4):
        vals=[row[key+s]*factor for s in ('','_lo','_hi')]
        if pd.isna(vals[0]):return '—'
        return f'{vals[0]:.{places}f} ({vals[1]:.{places}f}–{vals[2]:.{places}f})'
    def score_table(source,labels,filter_primary=False):
        d=pd.read_csv(out/source)
        if filter_primary:d=d[d.seed==42]
        records=[]
        for _,r in d.iterrows():
            rec={name:r[col] for name,col in labels.items()}
            rec['Accuracy, % (95% CI)']=ci(r,'accuracy',100,2)
            rec['Macro F1 (95% CI)']=ci(r,'macro_f1');rec['Macro AUC (95% CI)']=ci(r,'macro_auc')
            records.append(rec)
        return pd.DataFrame(records)
    table(out,'Table_1_Metadata_shortcuts',score_table('E1_metrics.csv',
        {'Tier':'tier','Classifier':'model','Unit':'unit','N':'n'},True))
    table(out,'Table_2_EfficientNet_configurations',score_table('E2_performance.csv',
        {'Configuration':'configuration','Output':'form','Correct':'correct','N':'n'}))
    for rule in ('primary','secondary'):
        d=pd.read_csv(out/'E3_primary.csv');selected=d.rule.str.startswith('secondary')
        d=d[selected if rule=='secondary' else ~selected]
        rows=[]
        for _,r in d.iterrows():
            rows.append({'Score':r.predictor,'Rule':r.rule,**{k.upper()+', % (Wilson 95% CI)':ci(r,k,100,2)
                for k in ('sensitivity','specificity','ppv','npv')},'False positives':r.fp})
        table(out,'Table_3_Operating_points_'+rule,pd.DataFrame(rows))
    d=pd.read_csv(out/'E4_curve.csv');d=d[d.unit=='image']
    rows=[]
    for _,r in d.iterrows():
        def avg(key,factor,places):
            m=r[key+'_mean'];s=r[key+'_std']
            return '—' if pd.isna(m) else f'{m*factor:.{places}f} ± {s*factor:.{places}f}'
        rows.append({'Exposed groups / 104':int(r.exposed_groups),'Subset':r.subset,
            'Accuracy, % (mean ± SD)':avg('accuracy',100,2),'Macro F1 (mean ± SD)':avg('macro_f1',1,4)})
    table(out,'Table_4_Exposure_fraction',pd.DataFrame(rows))
    if (out/'E5_regime_summary.csv').exists():
        table(out,'Table_5_BEiT_precision',pd.read_csv(out/'E5_regime_summary.csv'))

if __name__=='__main__':
    import sys
    if len(sys.argv)!=3:raise SystemExit('Usage: consolidate.py EXTRACTED_EXTENSION_INPUT_FOLDER RESULTS_ZIP_FOLDER')
    pkg,result=map(Path,sys.argv[1:]);paths={k:result/f'EXT_{k}_RESULTS_ONLY.zip' for k in ('CPU','Me','Friend1','Friend2')}
    main(pkg,paths,result/'EXT_Publication')

