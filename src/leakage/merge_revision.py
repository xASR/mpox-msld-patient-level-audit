"""Merge completed account outputs on CPU. Test outcomes never select jobs, groups or publication endpoints."""
import argparse,itertools,json,shutil,stat,tempfile
from pathlib import Path,PurePosixPath
from zipfile import ZipFile,ZIP_DEFLATED
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import binomtest
from sklearn.metrics import log_loss
from revision_io import read,put,sha,check_done,verify_package
from revision_stats import normalized,point,group_intervals

def unpack(zfile,dest):
    with ZipFile(zfile) as z:
        if sum(q.file_size for q in z.infolist())>500_000_000:raise ValueError('Expected only small result ZIPs.')
        for q in z.infolist():
            rel=PurePosixPath(q.filename.replace('\\','/'))
            if rel.is_absolute() or '..' in rel.parts or any(':' in x for x in rel.parts) or stat.S_ISLNK(q.external_attr>>16):raise ValueError('Invalid archive path')
            if not q.is_dir():
                target=dest.joinpath(*rel.parts);target.parent.mkdir(parents=True,exist_ok=True)
                with z.open(q) as a,target.open('wb') as b:shutil.copyfileobj(a,b)
def gather(pkg,search,scratch,manifest):
    pr=read(pkg/'plan.json');out={};source=[]
    for owner in ('alve','friend1','friend2'):
        target='REV_'+owner+'_RESULTS_ONLY.zip';roots=list(search.rglob('account_progress.json'))
        for arc in search.rglob(target):
            dest=scratch/sha(arc);unpack(arc,dest);roots.extend(dest.rglob('account_progress.json'))
        jobs=[j for j in pr['jobs'] if j['owner']==owner];ids={j['id'] for j in jobs}
        matching=[p for p in roots if read(p).get('owner')==owner and read(p).get('manifest')==manifest and read(p).get('complete')]
        if not matching:raise ValueError(f'Missing a complete {owner} output with {len(jobs)} jobs.')
        signatures=set()
        for p in matching:
            d=read(p)
            if set(d['completed'])!=ids or d['failed']:raise ValueError('Complete marker disagrees with the queue')
            arch=[]
            for j in jobs:
                dest=p.parent/j['id'];check_done(dest,j,manifest);arch.append(sha(dest/'complete.json'))
                out[j['id']]=dest
            signatures.add(tuple(arch))
        if len(signatures)!=1:raise ValueError('Conflicting completed runs for '+owner)
        source.append(dict(owner=owner,jobs=len(jobs),job_record_hashes=list(signatures.pop())))
    return pr['jobs'],out,source
def table(out,name,rows):
    df=pd.DataFrame(rows);df.to_csv(out/name,index=False);return df
def data(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def paired(y,p0,p1,groups):
    hit0=p0.argmax(1)==y;hit1=p1.argmax(1)==y;a=int((hit0&~hit1).sum());b=int((hit1&~hit0).sum())
    keys,inv=np.unique(groups,return_inverse=True);delta=hit1.astype(float)-hit0
    num=np.bincount(inv,weights=delta);den=np.bincount(inv)
    rng=np.random.default_rng(20261118);w=rng.multinomial(len(keys),np.full(len(keys),1/len(keys)),size=2000)
    boots=(w@num)/(w@den)
    return dict(n=len(y),groups=len(keys),accuracy_change=float(delta.mean()),accuracy_change_ci_lo=float(np.quantile(boots,.025)),accuracy_change_ci_hi=float(np.quantile(boots,.975)),
      macro_f1_change=point(y,p1)['macro_f1']-point(y,p0)['macro_f1'],first_only_correct=a,second_only_correct=b,
      exact_mcnemar_descriptive=float(binomtest(min(a,b),a+b,.5).pvalue) if a+b else 1.)
def validate_arrays(p,y,df):
    p=normalized(p)
    if len(df)!=len(y) or not np.array_equal(df.label,y):raise ValueError('Mismatch to locked test cohort')
    return p
def group_file(folder,model,fold,split):
    pre=f'{model}_fold{fold}_{split}_'
    ids=np.load(folder/(pre+'pids.npy')).astype(str);y=np.load(folder/(pre+'labels.npy'));p=normalized(np.load(folder/(pre+'probs.npy')))
    if len(set(ids))!=len(ids):raise ValueError('Repeated fold group')
    return pd.DataFrame(dict(group=ids,label=y,**{str(k):p[:,k] for k in range(6)}))
def assemble(pkg,search,out,manifest):
    verify_package(pkg,manifest);out.mkdir(exist_ok=False,parents=True)
    with tempfile.TemporaryDirectory() as tmp:
        jobs,folders,inputs=gather(pkg,search,Path(tmp),manifest)
        shutil.copytree(pkg/'saved_cpu',out/'saved_cpu')
        pool=pd.read_csv(pkg/'design/pool.csv')
        release=[];pairedrows=[];matched=[];diagnostic=[];received=[]
        for j in jobs:
            folder=folders[j['id']];rec=read(folder/'complete.json');received.append(dict(job=j['id'],owner=j['owner'],epochs=rec['epochs'],planned_batches=rec['total_batches'],optimizer_steps_applied=rec['optimizer_steps_applied'],initial_sha256=rec['initial']['tensor_sha256']))
        for family,model,seed in sorted({(j['family'],j['model'],j['index']) for j in jobs if j['family'] in ('release','matched')}):
            arms=['group','random'] if family=='release' else ['clean','contaminated'];arrs=[];inits=[];steps=[]
            for arm in arms:
                folder=folders[f'{family}_{model}_{seed}_{arm}'];rec=read(folder/'complete.json');d=data(folder/'test.npz')
                inits.append(rec['initial']['tensor_sha256']);steps.append(rec['total_batches']);arrs.append(d)
                if family=='release':
                    idx=pd.read_csv(pkg/f'design/release_s{seed}.csv')[arm+'_test'].eq(1);test=pool[idx]
                    p=validate_arrays(d['probs'],d['labels'],test)
                    if not np.array_equal(d['keys'],test.item.to_numpy(str)) or not np.array_equal(d['is_original'],test.is_original) or not np.array_equal(d['groups'],test.group):raise ValueError('Release test rows differ')
                    seen=np.load(folder/'training_items_seen.npy');train=pool[~idx]
                    if not set(seen)<=set(train.item) or len(set(seen))!=len(seen):raise ValueError('Observed fit exposure includes invalid items')
                    for flag,field in [('source','source'),('group','group'),('pixel','rgb')]:
                        for prefix,tr in [('seen',pool[pool.item.isin(seen)]),('eligible',train)]:
                            if not np.array_equal(d[prefix+'_'+flag],test[field].isin(tr[field])):raise ValueError('Exposure disagrees with saved sampling')
                    for scope,mask in [('all_test',np.ones(len(test),bool)),('clinical_originals',d['is_original'])]:
                        row=dict(model=model,seed=seed,arm=arm,scope=scope,eligible_train_items=len(train),actual_fit_items=len(seen),
                            **group_intervals(d['labels'][mask],p[mask],d['groups'][mask]))
                        for field in ('seen_source','seen_group','seen_pixel','eligible_source','eligible_group','eligible_pixel'):
                            row[field+'_n']=int(d[field][mask].sum());row[field+'_fraction']=float(d[field][mask].mean())
                        release.append(row)
                else:
                    for role in ('test','diagnostic'):
                        q=d if role=='test' else data(folder/'diagnostic.npz');df=pd.read_csv(pkg/f'design/matched_s{seed}_{role}.csv')
                        p=validate_arrays(q['probs'],q['labels'],df)
                        if not np.array_equal(q['source'],df.source) or not np.array_equal(q['groups'],df.group):raise ValueError('Matched cohort differs')
                        row=dict(model=model,seed=seed,arm=arm,**group_intervals(q['labels'],p,q['groups']))
                        (matched if role=='test' else diagnostic).append(row)
            if len(set(inits))!=1 or len(set(steps))!=1:raise ValueError('Paired jobs differ in initialization or update budget')
            first,second=arrs
            if family=='release':
                inter=np.intersect1d(first['keys'][first['is_original']],second['keys'][second['is_original']]);scope='same held-out clinical originals'
            else:
                if not np.array_equal(first['keys'],second['keys']):raise ValueError('Matched original test cohort differs')
                inter=first['keys'];scope='all shared clinical originals'
            if not len(inter):
                pairedrows.append(dict(family=family,model=model,seed=seed,scope=scope,n=0));continue
            ia=pd.Index(first['keys']).get_indexer(inter);ib=pd.Index(second['keys']).get_indexer(inter)
            if not np.array_equal(first['labels'][ia],second['labels'][ib]) or not np.array_equal(first['groups'][ia],second['groups'][ib]):raise ValueError('Shared labels differ')
            pairedrows.append(dict(family=family,model=model,seed=seed,scope=scope,first=arms[0],second=arms[1],**paired(first['labels'][ia],first['probs'][ia],second['probs'][ib],first['groups'][ia])))
        rd=table(out,'released_pool_by_seed.csv',release);table(out,'released_pool_mean_over_seeds.csv',rd.groupby(['model','arm','scope'])[['accuracy','macro_f1','seen_source_fraction','seen_group_fraction']].agg(['mean','std']).reset_index().pipe(lambda z:z.set_axis(['_'.join(k).strip('_') for k in z.columns],axis=1)))
        table(out,'shared_originals_paired.csv',pairedrows);table(out,'matched_reserved_diagnostics.csv',diagnostic);table(out,'matched_extension_originals.csv',matched)
        old=pd.read_csv(pkg/'baseline/vit_s0_4.csv');historical=[]
        for _,r in old.iterrows():
            for arm in ['clean','contaminated']:historical.append(dict(model='vit',seed=int(r.seed),arm=arm,n=int(r.n_test_photos),accuracy=r[arm+'_accuracy_percent']/100,macro_f1=r[arm+'_macro_f1']))
        other=pd.read_csv(pkg/'baseline/conv_s0_4.csv')
        for r in other[other.unit.eq('image')].itertuples():historical.append(dict(model='conv',seed=r.seed,arm=r.condition,n=r.n,accuracy=r.accuracy,macro_f1=r.macro_f1))
        ten=pd.concat([pd.DataFrame(historical),pd.DataFrame(matched)[['model','seed','arm','n','accuracy','macro_f1']]],ignore_index=True)
        assert len(ten)==40 and not ten.duplicated(['model','seed','arm']).any()
        table(out,'matched_ten_seeds_image_metrics.csv',ten);signs=[]
        for (m,metric) in itertools.product(['vit','conv'],['accuracy','macro_f1']):
            pv=ten[ten.model.eq(m)].pivot(index='seed',columns='arm',values=metric);d=pv.contaminated-pv.clean
            win=int((d>0).sum());loss=int((d<0).sum())
            signs.append(dict(model=m,metric=metric,seeds=len(d),mean_difference=d.mean(),sd_difference=d.std(),positive=win,negative=loss,tied=len(d)-win-loss,
                      two_sided_sign_p=float(binomtest(win,win+loss,.5).pvalue) if win+loss else 1.))
        table(out,'matched_ten_seed_sign_tests.csv',signs)
        bz=data(pkg/'baseline/b/analysis_probabilities.npz');model='efficientnetv2_s';g={}
        for split in ['val','test']:
            parts=[group_file(folders[f'alignment_{model}_{f}_a_recipe'],model,f,split) for f in range(1,6)]
            frame=pd.concat(parts);assert frame.groupby('group').label.nunique().max()==1
            avg=frame.groupby('group').mean().sort_index()
            if split=='val':avg=avg[~avg.index.isin(bz['test_groups'])]
            target='calibration' if split=='val' else 'test'
            assert np.array_equal(avg.index,bz[target+'_groups']) and np.array_equal(avg.label,bz[target+'_labels'])
            g[split]=avg
        def prob(frame):return normalized(frame[[str(c) for c in range(6)]].to_numpy())
        def scale(p,t):
            z=np.log(np.clip(p,1e-12,1))/t;z=np.exp(z-z.max(1,keepdims=True));return z/z.sum(1,keepdims=True)
        fit=minimize_scalar(lambda t:log_loss(g['val'].label,scale(prob(g['val']),t),labels=np.arange(6)),bounds=(.1,5),method='bounded')
        assert fit.success and len(g['val'])==138 and len(g['test'])==215
        y=bz['test_labels'];raw=prob(g['test']);align=[];pr=read(pkg/'baseline/b/report.json')
        for method,p,t in [('B_original_recipe_raw',bz[model+'_raw'],1.),('B_A_recipe_raw',raw,1.),
                        ('B_original_recipe_scaled',bz[model+'_scaled'],pr['temperatures'][model]),('B_A_recipe_scaled',scale(raw,fit.x),fit.x)]:
            p=normalized(p);align.append(dict(method=method,temperature=float(t),log_loss=log_loss(y,p,labels=np.arange(6)),
            brier=np.mean(np.sum((p-np.eye(6)[y])**2,axis=1)),**group_intervals(y,p,bz['test_groups'],True)))
        table(out,'E9_EfficientNet_same_official_cohort.csv',align);put(out/'E9_paired_accuracy.json',paired(y,bz[model+'_raw'],raw,bz['test_groups']))
        np.savez_compressed(out/'E9_group_predictions.npz',test_groups=bz['test_groups'],test_labels=y,new_raw=raw,new_scaled=scale(raw,fit.x),
                      calibration_groups=g['val'].index.to_numpy(str),calibration_labels=g['val'].label.to_numpy(int),calibration=prob(g['val']))
        m='beit_base';folder=folders['diagnostic_beit_base_2_a_repeat'];new=group_file(folder,m,2,'test').set_index('group')
        old=data(pkg/f'baseline/a/{m}/2/{m}_fold2_test_images.npz');sp=[]
        for group,y0 in zip(new.index,new.label):
            inds=old['patient_clusters']==group;assert inds.any() and (old['labels'][inds]==y0).all()
            imgs=normalized(old['probs'][inds]);sp.append(np.average(imgs,axis=0,weights=imgs.max(1)))
        new_p=prob(new);new_y=new.label.to_numpy(int)
        table(out,'E6_BEiT_fold2_comparison.csv',[dict(run=k,**point(new_y,p)) for k,p in [('primary',np.array(sp)),('supplementary_repeat',new_p)]])
        table(out,'E6_BEiT_fold2_instrumentation.csv',read(folder/'history.json'))
        br=[]
        for f in range(1,6):
            h=read(folders[f'alignment_efficientnetv2_s_{f}_a_recipe']/'history.json')
            br.extend(dict(fold=f,**r) for r in h)
        table(out,'E9_EfficientNet_training_diagnostics.csv',br)
        table(out,'job_registry.csv',received)
        (out/'READ_ME.md').write_text('All results are supplementary. Release-pool full and original-only test cohorts differ between arms. Only their shared clinical originals support paired item comparisons. Confidence intervals resample known groups and condition on each fitted run. The ten-seed sign tests describe holdout-level directions on the same underlying corpus; overlapping holdouts limit independence. The recipe comparison changes multiple factors jointly. The instrumented BEiT repeat cannot establish the historical cause of the weak fold. Stored primary estimates retain their original analysis units.\\n'.replace('\\n','\n'),encoding='utf-8')
        put(out/'report_status.json',dict(complete=True,verified_jobs=len(jobs),manifest=manifest,sources=inputs,
            scope=read(pkg/'plan.json')['gates'],outputs={p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file()}))
def final(pkg,search,out,manifest):
    out.parent.mkdir(exist_ok=True,parents=True)
    if out.exists():
        prior=read(out/'report_status.json')
        if not prior.get('complete') or prior.get('manifest')!=manifest:raise ValueError('Existing report is incomplete or belongs to another design.')
        for rel,h in prior['outputs'].items():
            if sha(out/rel)!=h:raise ValueError('Existing final report file has changed: '+rel)
    else:
        with tempfile.TemporaryDirectory(dir=out.parent) as scratch:
            staged=Path(scratch)/out.name
            assemble(pkg,search,staged,manifest)
            staged.rename(out)
    target=out.parent/'REV_Final_Supplement_RESULTS_ONLY.zip'
    partial=target.with_suffix('.partial')
    with ZipFile(partial,'w',ZIP_DEFLATED) as z:
        for p in sorted(out.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(out.parent))
    partial.replace(target)
    print('FINAL REPORT COMPLETE: 46/46. Output:',target,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ['package','search','out']:p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--manifest',required=True);x=p.parse_args();final(x.package,x.search,x.out,x.manifest)

