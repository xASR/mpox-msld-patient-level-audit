"""Saved-prediction analyses. No network, neural-network evaluation, or fitting of decision rules."""
import argparse,json,hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
from PIL import Image,ImageOps
from scipy.fft import dctn
from scipy.stats import binomtest
from revision_stats import group_intervals,temperature,normalized

def perceptual(im):
    coeff=dctn(np.asarray(im.convert('L').resize((32,32)),dtype=np.float64),type=2,norm='ortho')[:8,:8].ravel()[1:]
    return np.uint64(sum(int(bit)<<i for i,bit in enumerate(coeff>np.median(coeff))))
def variants(path):
    with Image.open(path) as raw:
        im=raw.convert('RGB')
    result=[]
    for frac in (1.,.85,.7):
        dx=int((1-frac)*im.width/2);dy=int((1-frac)*im.height/2)
        crop=im.crop((dx,dy,im.width-dx,im.height-dy))
        for mirrored in (False,True):
            v=ImageOps.mirror(crop) if mirrored else crop
            for angle in (0,90,180,270):result.append(perceptual(v.rotate(angle,expand=True)))
    return result
def table(out,name,rows):pd.DataFrame(rows).to_csv(out/name,index=False)
def run(root,dataset,out):
    out.mkdir(parents=True,exist_ok=True);plan=json.loads((root/'plan.json').read_text())
    ar=root/'baseline/a';br=root/'baseline/b';arep=json.loads((ar/'report.json').read_text());brep=json.loads((br/'report.json').read_text())
    pool=pd.read_csv(root/'design/pool.csv');orig=pool[pool.is_original].sort_values('source').reset_index(drop=True)
    names=orig.source.to_numpy(str);ni={n:i for i,n in enumerate(names)}
    paths=[dataset/p for p in orig.relative]
    for p,h in zip(paths,orig.sha256):
        if hashlib.sha256(p.read_bytes()).hexdigest()!=h:raise ValueError(f'Original image hash mismatch {p.name}')
    with ThreadPoolExecutor(max_workers=6) as ex: hv=np.asarray(list(ex.map(variants,paths)),dtype=np.uint64)
    lut=np.array([int(i).bit_count() for i in range(256)],dtype=np.uint8)
    distances=np.stack([lut[(hv^q).view(np.uint8).reshape(len(names),24,8)].sum(2).min(1) for q in hv[:,0]]).astype(np.uint8)
    np.savez_compressed(out/'E2_full_phash_distances.npz',filenames=names,distances=distances,variant_hashes=hv)
    main_group=pd.read_csv(ar/'per_patient_predictions.csv').set_index('patient_id')
    model_names=list(arep['calibration_by_fold']['1']['weights']);probs={n:[] for n in model_names+['ensemble']}
    labels=[];groups=[];file_order=[];folds=[];image_rows=[];group_rows=[];diag=[]
    for f in range(1,6):
        fold=pd.read_csv(root/f'design/a_fold{f}.csv'); test=fold[fold.role=='test'].copy()
        train=fold[fold.role=='train']; eligible=np.array([ni[x] for x in train.filename])
        sc=arep['calibration_by_fold'][str(f)];mix=None
        for m in model_names:
            job=ar/m/str(f);d=np.load(job/f'{m}_fold{f}_test_images.npz',allow_pickle=False)
            assert len(set(d['filenames']))==len(test) and set(d['filenames'])==set(test.filename)
            pos=pd.Index(d['filenames']).get_indexer(test.filename);assert (pos>=0).all()
            assert np.array_equal(d['labels'][pos],test.label) and np.array_equal(d['patient_clusters'][pos],test.patient_cluster)
            p=normalized(d['probs'][pos]);probs[m].append(p);term=sc['weights'][m]*temperature(p,sc['temperatures'][m])
            mix=term if mix is None else mix+term
            hist=json.loads((job/'epoch_history.json').read_text());complete=json.loads((job/'job_complete.json').read_text())
            from collections import Counter
            hits=[];pg=[];gy=[]
            for group,part in test.reset_index(drop=True).groupby('patient_cluster'):
                im=p[part.index];v=np.average(im,axis=0,weights=im.max(1));pg.append(v.argmax());gy.append(part.label.iloc[0]);hits.append(pg[-1]==gy[-1])
            diag.append(dict(fold=f,model=m,n=len(hits),correct=sum(hits),accuracy=np.mean(hits),epochs=len(hist),
                selected_epochs=','.join(str(q['epoch']) for q in complete['weights']),last_train_loss=hist[-1]['mean_training_loss'],
                last_selection_logloss=hist[-1]['selection_logloss'],predicted_class_counts=str(np.bincount(pg,minlength=6).tolist()),
                training_originals=len(train),calibration_groups=sc['calibration_clusters']))
        probs['ensemble'].append(mix);labels.extend(test.label);groups.extend(test.patient_cluster);file_order.extend(test.filename);folds.extend([f]*len(test))
        for row in test.itertuples():
            d=distances[ni[row.filename],eligible]; nearest=int(eligible[np.argmin(d)])
            image_rows.append(dict(fold=f,group=row.patient_cluster,filename=row.filename,nearest_training_source=names[nearest],
                                   nearest_distance=int(d.min()),ties_at_minimum=int((d==d.min()).sum())))
    assert len(set(file_order))==755 and len(set(groups))==520
    z={k:np.concatenate(v) for k,v in probs.items()};y=np.array(labels);g=np.array(groups,str)
    np.savez_compressed(out/'E1_image_predictions.npz',filename=np.array(file_order,str),label=y,group=g,fold=folds,**z)
    table(out,'E1_image_metrics.csv',[dict(model=k,**group_intervals(y,p,g,True)) for k,p in z.items()])
    table(out,'E6_primary_fold_diagnostics.csv',diag)
    comparison=pd.DataFrame(image_rows);table(out,'E2_nearest_train_image.csv',comparison)
    for (fold,key),part in comparison.groupby(['fold','group']):
        row=main_group.loc[key];md=part.nearest_distance.min()
        group_rows.append(dict(fold=fold,group=key,min_distance=md,candidate=bool(md<=plan['similarity_phash_max']),correct=int(row['prediction']==row['label'])))
    gr=pd.DataFrame(group_rows);assert len(gr)==520 and gr.correct.sum()==442;table(out,'E2_group_predictions.csv',gr)
    summ=[]
    for flag in [False,True]:
        q=gr[gr.candidate==flag];n=len(q);k=int(q.correct.sum())
        ci=binomtest(k,n).proportion_ci(method='wilson') if n else (None,None)
        summ.append(dict(candidate=flag,n=n,correct=k,accuracy=k/n if n else None,wilson_low=ci[0],wilson_high=ci[1]))
    table(out,'E2_candidate_strata.csv',summ)
    a=gr[gr.candidate].correct.to_numpy();b=gr[~gr.candidate].correct.to_numpy()
    delta=dict(comparison='candidate minus no candidate under the specified metric',cutoff=8)
    if len(a) and len(b):
        rng=np.random.default_rng(20261118);draw=rng.choice(a,(2000,len(a))).mean(1)-rng.choice(b,(2000,len(b))).mean(1)
        delta.update(estimate=float(a.mean()-b.mean()),bootstrap_95=list(np.percentile(draw,[2.5,97.5])))
    else:delta['status']='not estimable; empty stratum'
    (out/'E2_difference.json').write_text(json.dumps(delta,indent=2))
    pd.read_csv(br/'reliability_bin_data.csv').to_csv(out/'E7_frozen_reliability_bins.csv',index=False)
    bz=np.load(br/'analysis_probabilities.npz');by=bz['test_labels'];rows=[];bins=[]
    import matplotlib
    matplotlib.use('Agg');import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,4,figsize=(13,5),constrained_layout=True)
    for i,m in enumerate(model_names):
        for stage in ['raw','scaled']:
            p=normalized(bz[m+'_'+stage]);conf=p.max(1);hit=p.argmax(1)==by;inds=np.minimum((conf*10).astype(int),9)
            rows.append(dict(model=m,stage=stage,n=len(by),accuracy=hit.mean(),mean_confidence=conf.mean(),
                confidence_minus_accuracy=conf.mean()-hit.mean(),frozen_temperature=brep['temperatures'][m]))
            for j in range(10):
                q=inds==j;bins.append(dict(model=m,stage=stage,bin_left=j/10,bin_right=(j+1)/10,count=int(q.sum()),
                  confidence=conf[q].mean() if q.any() else None,accuracy=hit[q].mean() if q.any() else None))
            sub=pd.DataFrame(bins[-10:]);axes[0,i].stairs(sub['count'],np.linspace(0,1,11),label=stage)
            axes[1,i].plot(sub.confidence,sub.accuracy,'o-',ms=3,label=stage)
        axes[0,i].set_title(m);axes[1,i].plot([0,1],[0,1],'k:',lw=.7);axes[1,i].set(xlim=(0,1),ylim=(0,1),xlabel='Confidence')
    axes[0,0].set_ylabel('Known groups');axes[1,0].set_ylabel('Observed accuracy');axes[0,0].legend()
    for ext in ['png','pdf']:fig.savefig(out/('E7_confidence_reliability.'+ext),dpi=220)
    plt.close(fig);table(out,'E7_confidence_summary.csv',rows);table(out,'E7_confidence_bins.csv',bins)
    (out/'interpretation.txt').write_text('E1 applies archived group-fitted fold parameters to held-out image vectors. Confidence intervals resample known groups with their images. E2 measures retrieval proximity under the fixed metric. Clinical identities and true exposure are unverified. E6 training histories and predicted classes characterize weak-fold performance. Historic gradient norms and AMP skip counts are unavailable. E7 describes calibration on held-out groups. Training-recipe causation cannot be identified from these comparisons.\\n'.replace('\\n','\n'))
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('inputs',type=Path);ap.add_argument('dataset',type=Path);ap.add_argument('out',type=Path)
    x=ap.parse_args();run(x.inputs,x.dataset,x.out)

