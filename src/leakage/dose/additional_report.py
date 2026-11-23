"""Append verified extensions to the unchanged supplemental CPU analysis."""
from pathlib import Path
import argparse,itertools,json,shutil
import numpy as np
import pandas as pd
from scipy.stats import t,ttest_rel,binomtest
from supplemental_common import sha,digest,write_json,bundle,extract_results,unique_markers,export_tables
from supplemental_audit import holm
from final_analysis import numpy_metrics
from extension_common import verify_extension,baseline,dose_frames,check_files

def collect(inputs,ext,roots,expected):
    def key(v):return (v['job']['seed'],v['job']['dose']) if v.get('schema')=='dose384_extension_v1' else None
    found=unique_markers(roots,'dose_complete.json',key);records={};missing=[]
    for seed in range(5):
        for dose,condition in [(0,'clean'),(8,'contaminated')]:records[(seed,dose)]=baseline(ext,seed,condition)
        for dose in (2,4):
            if (seed,dose) not in found:missing.append({'seed':seed,'dose':dose});continue
            p,meta=found[(seed,dose)];check_files(p.parent,meta)
            ident=meta['identity'];mc,_=records[(seed,0)]
            if meta['status']!='complete' or meta['epochs']!=25 or meta['identity_sha256']!=digest(ident):raise ValueError('Incomplete dose job.')
            if ident['extension_manifest_sha256']!=expected or ident['script_sha256']!=sha(Path(ext)/'code/dose_extension.py'):raise ValueError('Dose inputs/code changed.')
            if ident['environment']!=mc['identity']['environment'] or ident['recipe']!=mc['identity']['recipe'] or meta['initial_state']!=mc['initial_state']:raise ValueError('Matched environment/recipe/initial state differs.')
            clean,frame,test=dose_frames(inputs,ext,seed,dose)
            if meta['training_entries']!=len(frame) or meta['steps_per_epoch']!=len(frame)//16 or meta['exposed_augmented_slots']!=dose*len(test):raise ValueError('Dose budget changed.')
            h=json.loads((p.parent/'history.json').read_text())
            if [r['epoch'] for r in h]!=list(range(1,26)) or any(r['phase']!=('head' if r['epoch']<=3 else 'full') or not np.isfinite(r['mean_training_loss']) for r in h):raise ValueError('Training history differs.')
            with np.load(p.parent/'test_predictions.npz',allow_pickle=False) as z:a={k:z[k] for k in z.files}
            for k,c in [('filenames','filename'),('labels','label'),('groups','patient_cluster')]:
                if not np.array_equal(a[k],test[c]):raise ValueError('Dose evaluation cohort differs.')
            if a['probs'].shape!=(len(test),6) or not np.isfinite(a['probs']).all() or (a['probs']<0).any() or not np.allclose(a['probs'].sum(1),1,atol=1e-5):raise ValueError('Invalid predictions.')
            records[(seed,dose)]=(meta,a)
    return records,missing

def dose_tables(records):
    rows=[]
    for (seed,dose),(meta,a) in sorted(records.items()):
        if 'job' in meta:nexposed=meta['exposed_augmented_slots']
        else:nexposed=dose*len(a['labels'])
        n=meta['training_entries']
        rows.append({'seed':seed,'relatives_per_test_image':dose,'exposed_training_fraction':nexposed/n,'unit':'image',**numpy_metrics(a['probs'],a['labels'])})
        groups=np.sort(np.unique(a['groups']));probs=np.stack([a['probs'][a['groups']==g].mean(0) for g in groups]);y=np.array([a['labels'][a['groups']==g][0] for g in groups])
        rows.append({'seed':seed,'relatives_per_test_image':dose,'exposed_training_fraction':nexposed/n,'unit':'known_group_mean',**numpy_metrics(probs,y)})
    metrics=pd.DataFrame(rows);summary=metrics.groupby(['unit','relatives_per_test_image']).agg(seeds=('seed','nunique'),accuracy_mean=('accuracy','mean'),accuracy_sd=('accuracy','std'),f1_mean=('macro_f1','mean'),auc_mean=('macro_auc','mean'),exposure_min=('exposed_training_fraction','min'),exposure_max=('exposed_training_fraction','max')).reset_index()
    tests=[]
    for lo,hi in itertools.combinations((0,2,4,8),2):
        if not all((s,d) in records for s in range(5) for d in (lo,hi)):continue
        d=np.array([100*((records[(s,hi)][1]['probs'].argmax(1)==records[(s,hi)][1]['labels']).mean()-(records[(s,lo)][1]['probs'].argmax(1)==records[(s,lo)][1]['labels']).mean()) for s in range(5)])
        ci=d.mean()+np.array([-1,1])*t.ppf(.975,4)*d.std(ddof=1)/np.sqrt(5)
        nz=d[d!=0];sign=float(binomtest(int((nz>0).sum()),len(nz),.5).pvalue) if len(nz) else 1.
        tests.append({'lower_dose':lo,'higher_dose':hi,'paired_seeds':5,'mean_gap_pp':float(d.mean()),'ci_low_pp':float(ci[0]),'ci_high_pp':float(ci[1]),'p_sign':sign,'p_t':float(ttest_rel(d,np.zeros(5)).pvalue) if d.std(ddof=1)>0 else None})
    if len(tests)==6:
        for r,p in zip(tests,holm([r['p_sign'] for r in tests])):r['p_sign_holm_6']=float(p)
    else:
        for r in tests:r['p_sign_holm_6']=None
    # Whole-known-group bootstrap per seed for each dose vs its same clean cohort.
    intervals=[]
    for seed in range(5):
        if (seed,0) not in records:continue
        clean=records[(seed,0)][1];groups=np.unique(clean['groups']);rng=np.random.default_rng(170117+seed)
        draws=rng.integers(0,len(groups),(2000,len(groups)));counts=np.array([(clean['groups']==g).sum() for g in groups])
        for dose in (2,4,8):
            if (seed,dose) not in records:continue
            exposed=records[(seed,dose)][1]
            if not np.array_equal(clean['filenames'],exposed['filenames']):raise ValueError('Within-seed cohort mismatch.')
            delta=(exposed['probs'].argmax(1)==exposed['labels']).astype(float)-(clean['probs'].argmax(1)==clean['labels'])
            sums=np.array([delta[clean['groups']==g].sum() for g in groups]);sample=100*sums[draws].sum(1)/counts[draws].sum(1);ci=np.percentile(sample,[2.5,97.5])
            intervals.append({'seed':seed,'dose':dose,'gap_pp':100*float(delta.mean()),'ci_low_pp':float(ci[0]),'ci_high_pp':float(ci[1]),'bootstrap_unit':'known group; all image rows retained'})
    return {'dose_per_seed_metrics':metrics,'dose_summary':summary,'dose_paired_seed_contrasts':tests,'dose_vs_clean_group_bootstrap':intervals}

def figures(tables,contrasts,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Serif','font.size':9,'pdf.fonttype':42,'svg.fonttype':'none'})
    frame=pd.DataFrame(tables['dose_per_seed_metrics']);frame=frame[frame.unit.eq('image')]
    fig,ax=plt.subplots(figsize=(5.5,3.8))
    for seed,part in frame.groupby('seed'):
        part=part.sort_values('relatives_per_test_image');ax.plot(part.relatives_per_test_image,100*part.accuracy,'o-',alpha=.6,lw=1,label=f'Seed {seed}')
    ax.set(xlabel='Augmented relatives per reserved test image',ylabel='Original-image accuracy (%)',xticks=[0,2,4,8]);ax.spines[['top','right']].set_visible(False);ax.legend(frameon=False,fontsize=8)
    for ext in ('pdf','svg','png'):fig.savefig(out/f'dose_response.{ext}',dpi=600,bbox_inches='tight')
    plt.close(fig)
    if contrasts is None:return
    a=contrasts[(contrasts.scope=='artifact_positive_known_groups')&contrasts.groups.gt(0)]
    if a.empty:return
    fig,ax=plt.subplots(figsize=(5.5,2.8));x=np.arange(len(a));y=a.mean_delta_vs_controls.to_numpy();lo=a.ci_low.to_numpy();hi=a.ci_high.to_numpy()
    ax.errorbar(x,y,yerr=np.maximum(0,np.stack([y-lo,hi-y])),fmt='o',color='#24577A',capsize=4);ax.axhline(0,color='#666',lw=.8,linestyle='--')
    ax.set(xticks=x,xticklabels=a['fill'],xlabel='Mask fill',ylabel='Reference-class score change\nrelative to matched controls');ax.spines[['top','right']].set_visible(False)
    for ext in ('pdf','svg','png'):fig.savefig(out/f'artifact_masking_contrasts.{ext}',dpi=600,bbox_inches='tight')
    plt.close(fig)

def main(args):
    verify_extension(args.extension,args.expected_extension_sha)
    report=Path(args.work)/'SUPP_Final_Report';report.mkdir(parents=True,exist_ok=True)
    out=report/'AdditionalChecks';out.mkdir(exist_ok=True)
    roots=extract_results(args.input_root,Path(args.work)/'additional_result_cache')
    records,missing=collect(args.inputs,args.extension,roots,args.expected_extension_sha)
    tables=dose_tables(records);export_tables(tables,out/'Dose_Response/tables');contrasts=None
    artifacts=unique_markers(roots,'artifact_complete.json',lambda v:v['identity']['extension_manifest_sha256'])
    artifact_status='pending'
    if args.expected_extension_sha in artifacts:
        p,meta=artifacts[args.expected_extension_sha];check_files(p.parent,meta)
        if meta['identity_sha256']!=digest(meta['identity']) or meta['identity']['script_sha256']!=sha(Path(args.extension)/'code/artifact_extension.py'):raise ValueError('Artifact code differs.')
        # Validate signed annotations and independently recompute the six reported contrasts.
        from artifact_extension import annotations,contrast_rows
        clinical=pd.read_csv(Path(args.inputs)/'a_design/seed_42/clinical_images.csv')
        ann,_=annotations(p.parent/'artifact_annotations_REVIEWED.csv',clinical)
        if sha(p.parent/'artifact_annotations_REVIEWED.csv')!=meta['identity']['annotations_sha256']:raise ValueError('Artifact annotation identity differs.')
        with np.load(p.parent/'group_predictions.npz',allow_pickle=False) as z:a={k:z[k] for k in z.files}
        values={f'{k}:{f}':a[f'{k}_{f}'] for f in ('mean','blur') for k in ('artifact','artifact_control1','artifact_control2')}
        recomputed=pd.DataFrame(contrast_rows(a['groups'],a['labels'],a['baseline'],values,a['affected']))
        contrasts=recomputed
        saved=pd.read_csv(p.parent/'tables/matched_artifact_contrasts.csv')
        pd.testing.assert_frame_equal(recomputed,saved,check_dtype=False,rtol=1e-10,atol=1e-12)
        dest=out/'Artifact_Reliance';dest.mkdir(exist_ok=True)
        for f in [p,p.parent/'artifact_status.json',p.parent/'artifact_annotations_REVIEWED.csv',p.parent/'baseline_reproduction.json']:
            shutil.copyfile(f,dest/f.name)
        shutil.copytree(p.parent/'tables',dest/'tables',dirs_exist_ok=True)
        artifact_status='complete'
    figures(tables,contrasts,out/'figures')
    status={'dose_new_conditions_complete':10-len(missing),'dose_new_conditions_required':10,'artifact_reliance':artifact_status,
        'extension_manifest_sha256':args.expected_extension_sha,'both_additional_checks_complete':not missing and artifact_status=='complete',
        'remaining_limitations':['No external clinical validation','No authoritative patient identities','No expert lesion-localisation validation','Unfinished duplicate adjudication','Historical MSLD design reuse','Dose conditions are controlled exposure, not literal published-protocol replication']}
    write_json(out/'ADDITIONAL_COMPLETENESS.json',status);pd.DataFrame(missing).to_csv(out/'MISSING_DOSE_JOBS.csv',index=False)
    (out/'INTERPRETATION.md').write_text('The additional 2/4-relative conditions reuse the same test cohort, initialization, slots, labels, updates and recipe as the saved 0/8-relative conditions. They are a supplemental dose-sensitivity analysis after earlier test exposure. Report all four levels and all five seeds, actual exposure fractions, both seed-level tests and whole-group conditional bootstrap intervals. Six sign-test contrasts are Holm adjusted; do not treat reused images/seeds as independent clinical cohorts.\n\nArtifact masks cover the union of reviewed eye bars and watermark/text/logo boxes; type-specific prevalence is descriptive. Counterfactual masking uses two deterministic equal-area controls disjoint from the artifact mask, mean/blur fills, and fixed original calibration. Controls can obscure lesions; masking changes the input distribution. Findings quantify inference sensitivity, not clinical reasoning or lesion localization.\n',encoding='utf-8')
    marker=report/'COMPLETENESS.json'
    if marker.exists():
        full=json.loads(marker.read_text());full['additional_checks']=status;write_json(marker,full)
    bundle(report);print(json.dumps(status,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--inputs',required=True);p.add_argument('--extension',required=True);p.add_argument('--expected-extension-sha',required=True)
    p.add_argument('--input-root',default='/kaggle/input');p.add_argument('--work',default='/kaggle/working');main(p.parse_args())
