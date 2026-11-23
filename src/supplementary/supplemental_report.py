"""Merge all prescribed results; report incomplete experiments explicitly."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from scipy.stats import t, ttest_rel, binomtest
from supplemental_common import (MODELS,SEEDS,LABELS,sha,digest,verify_inputs,write_json,bundle,
    extract_results,unique_markers,export_tables)
from supplemental_audit import variants,paired_comparisons,b_comparisons
from protocol_a_core import inspect_job,read_group
from final_analysis import numpy_metrics


def check_files(folder,meta,allow_missing_weights=True):
    for name,rec in meta['files'].items():
        p=(Path(folder)/name).resolve()
        if not p.is_relative_to(Path(folder).resolve()):raise ValueError('Result path escaped job.')
        if allow_missing_weights and p.suffix=='.pt' and not p.exists():continue
        if not p.is_file() or p.stat().st_size!=rec['bytes'] or sha(p)!=rec['sha256']:raise ValueError('Result file missing or changed: '+str(p))


def collect_repeats(inputs,roots):
    inputs=Path(inputs)
    mapping={sha(inputs/f'a_design/seed_{s}/protocol_a_manifest.json'):s for s in (42,*SEEDS)}
    def key(v):
        s=mapping.get(v.get('identity',{}).get('input_manifest_sha256'))
        return (s,v['model'],int(v['fold'])) if s is not None else None
    markers=unique_markers([inputs/'a_baseline',*roots],'job_complete.json',key)
    data={};missing=[]
    for s in (42,*SEEDS):
        for model in MODELS:
            for f in range(1,6):
                k=(s,model,f)
                if k not in markers:missing.append({'kind':'repeat','seed':s,'model':model,'fold':f});continue
                p,meta=markers[k];design=inputs/f'a_design/seed_{s}'
                inspect_job(p.parent,design,sha(design/'protocol_a_manifest.json'),require_weights=False)
                # A completed job may have run on another account; verify its recipe,
                # split and recorded code, while keeping its actual environment in metadata.
                manifest=json.loads((design/'protocol_a_manifest.json').read_text());ident=meta['identity']
                if ident['training']!=manifest['training'] or ident['pretrained_source']!=manifest['models'][model] or ident['seed']!=s+100*MODELS.index(model)+f:
                    raise ValueError('Completed repeated job has a different recipe.')
                if s!=42:
                    for name,recorded in ident['training_code'].items():
                        source=inputs/'code'/name
                        if not source.is_file() or sha(source)!=recorded:raise ValueError('Repeated run code differs from frozen package.')
                data[k]={split:read_group(p.parent,f'{model}_fold{f}_{split}') for split in ('val','test')}
    return data,missing


def evaluate_repeat(data,seed):
    parts={};ys=[];ids=[];folds=[];parameters={}
    for f in range(1,6):
        reference=data[(seed,MODELS[0],f)]
        for m in MODELS:
            for split in ('val','test'):
                other=data[(seed,m,f)][split]
                for key in ('labels','pids'):
                    if not np.array_equal(reference[split][key],other[key]):raise ValueError('Models have different evaluation rows.')
        if set(reference['val']['pids'])&set(reference['test']['pids']):raise ValueError('Calibration/test overlap.')
        cal=[data[(seed,m,f)]['val']['probs'] for m in MODELS];test=[data[(seed,m,f)]['test']['probs'] for m in MODELS]
        values,params=variants(cal,test,reference['val']['labels'])
        for name,p in values.items():parts.setdefault(name,[]).append(p)
        ys.append(reference['test']['labels']);ids.append(reference['test']['pids']);folds.extend([f]*len(ys[-1]));parameters[f]=params
    y=np.concatenate(ys);groups=np.concatenate(ids);order=np.argsort(groups)
    if len(groups)!=520 or len(set(groups))!=520:raise ValueError('Repeated Protocol A must test every known group exactly once.')
    probs={name:np.concatenate(p)[order] for name,p in parts.items()}
    return groups[order],y[order],probs,parameters,np.array(folds)[order]


def collect_ablation(inputs,roots):
    def key(v):return (v['seed'],v['condition']) if v.get('schema')=='supplemental_ablation384_v1' else None
    markers=unique_markers(roots,'ablation_complete.json',key);missing=[];records={}
    for seed in range(5):
        for condition in ('clean','contaminated'):
            k=(seed,condition)
            if k not in markers:missing.append({'kind':'ablation','seed':seed,'condition':condition});continue
            p,meta=markers[k];check_files(p.parent,meta)
            if meta['status']!='complete' or meta['epochs']!=25 or meta['identity_sha256']!=digest(meta['identity']):raise ValueError('Incomplete ablation or identity mismatch.')
            if meta['identity']['manifest_sha256']!=sha(Path(inputs)/'supplemental_manifest.json'):raise ValueError('Ablation used another input package.')
            plan=json.loads((Path(inputs)/'analysis_plan.json').read_text())
            if meta['identity']['recipe']!=plan['ablation384']:raise ValueError('Ablation recipe changed.')
            for name,value in meta['identity']['code'].items():
                if sha(Path(inputs)/'code'/name)!=value:raise ValueError('Ablation code changed.')
            expected=pd.read_csv(Path(inputs)/f'ablation/seed_{seed}/test.csv')
            with np.load(p.parent/'test_predictions.npz',allow_pickle=False) as z:a={n:z[n] for n in z.files}
            if not np.array_equal(a['filenames'],expected.filename) or not np.array_equal(a['labels'],expected.label) or not np.array_equal(a['groups'],expected.patient_cluster):raise ValueError('Ablation test cohort changed.')
            records[k]=(meta,a)
    return records,missing


def ablation_tables(records):
    rows=[];gaps=[]
    for seed in range(5):
        if any((seed,c) not in records for c in ('clean','contaminated')):continue
        mc,a=records[(seed,'clean')];ml,b=records[(seed,'contaminated')]
        if mc['initial_state']['parameter_sha256']!=ml['initial_state']['parameter_sha256']:raise ValueError('Within-pair initial weights differ.')
        if mc['steps_per_epoch']!=ml['steps_per_epoch'] or mc['training_entries']!=ml['training_entries']:raise ValueError('Within-pair training budget differs.')
        if mc['identity']['environment']!=ml['identity']['environment']:raise ValueError('Within-pair environments differ; run both conditions on the same account/GPU environment.')
        for condition,z in (('clean',a),('contaminated',b)):
            rows.append({'seed':seed,'condition':condition,'unit':'image',**numpy_metrics(z['probs'],z['labels'])})
            groups=np.sort(np.unique(z['groups']));p=np.stack([z['probs'][z['groups']==g].mean(0) for g in groups]);y=np.array([z['labels'][z['groups']==g][0] for g in groups])
            rows.append({'seed':seed,'condition':condition,'unit':'known_group_mean',**numpy_metrics(p,y)})
        ca=a['probs'].argmax(1)==a['labels'];cb=b['probs'].argmax(1)==b['labels']
        gaps.append({'seed':seed,'clean_accuracy':float(ca.mean()),'contaminated_accuracy':float(cb.mean()),'gap_pp':100*float(cb.mean()-ca.mean())})
    if len(gaps)!=5:return rows,gaps,{'status':'incomplete','paired_seeds':len(gaps)}
    d=np.array([r['gap_pp'] for r in gaps]);se=d.std(ddof=1)/np.sqrt(5);ci=d.mean()+np.array([-1,1])*t.ppf(.975,4)*se
    exact=float(binomtest(int((d>0).sum()),int((d!=0).sum()),.5).pvalue) if (d!=0).any() else 1.
    paired_p=float(ttest_rel([r['contaminated_accuracy'] for r in gaps],[r['clean_accuracy'] for r in gaps]).pvalue) if d.std(ddof=1)>0 else None
    return rows,gaps,{'status':'complete','pairs':5,'mean_gap_pp':float(d.mean()),'sd_gap_pp':float(d.std(ddof=1)),
        't_interval_95_pp':ci.tolist(),'paired_t_p':paired_p,
        'exact_sign_p':exact,'note':'Five seeds reuse the corpus; image appearances are not independent patients. This is a backbone/resolution/recipe sensitivity experiment, not a universal leakage penalty.'}


def figures(seed_frame,ablation,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out.mkdir(exist_ok=True)
    if not seed_frame.empty:
        fig,ax=plt.subplots(figsize=(6.8,4))
        for m in ('ensemble',*MODELS):
            a=seed_frame[seed_frame.method.eq(m)].sort_values('seed')
            ax.plot(a.seed.astype(str),100*a.accuracy,'o-',label=m)
        ax.set(xlabel='Outer split and training seed',ylabel='Known-group accuracy (%)');ax.legend(fontsize=7);ax.spines[['top','right']].set_visible(False)
        for ext in ('pdf','svg','png'):fig.savefig(out/f'repeated_protocol_a.{ext}',dpi=600,bbox_inches='tight')
        plt.close(fig)
    if ablation:
        fig,ax=plt.subplots(figsize=(4,4))
        for row in ablation:ax.plot([0,1],100*np.array([row['clean_accuracy'],row['contaminated_accuracy']]),'o-',alpha=.75,label=f'Seed {row["seed"]}')
        ax.set(xticks=[0,1],xticklabels=['Clean','Contaminated'],ylabel='Image accuracy (%)');ax.legend(fontsize=8);ax.spines[['top','right']].set_visible(False)
        for ext in ('pdf','svg','png'):fig.savefig(out/f'matched_384_ablation.{ext}',dpi=600,bbox_inches='tight')
        plt.close(fig)


def additional_checks(inputs,roots,out):
    plan_sha=sha(Path(inputs)/'analysis_plan.json');xai={};audits=[];reviewed=[]
    for root in roots:
        for p in Path(root).rglob('*_status.json'):
            if p.parent.name!='SUPP_XAI' and p.name!='audit_status.json':continue
            value=json.loads(p.read_text())
            if p.parent.name=='SUPP_XAI' and value.get('status')=='complete' and value.get('analysis_plan_sha256')==plan_sha:
                mode=value['mode']
                if mode in ('all','sensitivity'):xai['sensitivity']='complete'
                if mode in ('all','sanity'):xai['sanity']='complete'
                dest=out/'XAI'/mode;dest.mkdir(parents=True,exist_ok=True)
                import shutil
                for table in (p.parent/'tables').glob('*'):
                    if table.is_file():shutil.copyfile(table,dest/table.name)
                for table in p.parent.glob('*.csv'):shutil.copyfile(table,dest/table.name)
                write_json(dest/'status.json',value)
            elif p.name=='audit_status.json' and value.get('images_verified')==755:
                candidates=p.parent/'duplicate_candidates_REVIEW_REQUIRED.csv'
                if not candidates.is_file() or sha(candidates)!=value.get('candidates_sha256'):raise ValueError('Original duplicate candidate table changed.')
                audits.append(pd.read_csv(candidates).fillna(''))
        for p in Path(root).rglob('duplicate_candidates_REVIEWED.csv'):reviewed.append((p,pd.read_csv(p).fillna('')))
    review_status={'status':'pending','reason':'Review candidate table and supply duplicate_candidates_REVIEWED.csv.'}
    if reviewed:
        reference=audits[0] if audits else None
        if reference is None:raise ValueError('Attach original CPU audit output with the reviewed duplicate sheet.')
        _,table=reviewed[0]
        fixed=[c for c in reference.columns if c not in ('decision','reviewer','rationale')]
        pd.testing.assert_frame_equal(reference[fixed],table[fixed],check_dtype=False)
        if not table.decision.isin(['same_source','different_source','uncertain']).all() or table.reviewer.astype(str).str.strip().eq('').any():
            review_status={'status':'pending','reason':'Every candidate needs an allowed decision and reviewer.'}
        else:
            same_group=table.same_known_group.astype(str).str.lower().eq('true')
            cross=table.decision.eq('same_source')&~same_group
            uncertain=table.decision.eq('uncertain')
            review_status={'status':'new_cross_group_links_require_split_revision' if cross.any() else ('reviewed_with_uncertain_candidates' if uncertain.any() else 'reviewed'),
                'confirmed_new_cross_group_pairs':int(cross.sum()),'uncertain_pairs':int(uncertain.sum()),
                'scope':'Review of retrieved content candidates; true clinical identities and unseen relationships remain unverified.'}
            table.to_csv(out/'reviewed_duplicate_candidates.csv',index=False)
    return {'XAI_sensitivity':xai.get('sensitivity','pending'),'XAI_randomization':xai.get('sanity','pending'),'duplicate_content_review':review_status}


def main(args):
    inputs=Path(args.inputs);verify_inputs(inputs,args.expected_sha)
    out=Path(args.work)/'SUPP_Final_Report';out.mkdir(parents=True,exist_ok=True)
    roots=extract_results(args.input_root,out.parent/'supplemental_result_cache')
    data,missing_a=collect_repeats(inputs,roots);abl,missing_b=collect_ablation(inputs,roots)
    rows=[];paired=[];consistency=[];by_seed={}
    for s in (42,*SEEDS):
        if any(r['seed']==s for r in missing_a):continue
        ids,y,probs,parameters,folds=evaluate_repeat(data,s);by_seed[s]=(ids,y,probs)
        np.savez_compressed(out/f'A_seed_{s}_predictions.npz',groups=ids,labels=y,folds=folds,**probs)
        write_json(out/f'A_seed_{s}_calibration.json',parameters)
        rows += [{'seed':s,'method':m,**numpy_metrics(p,y)} for m,p in probs.items()]
        paired += [{'seed':s,**r} for r in paired_comparisons(y,probs)]
    frame=pd.DataFrame(rows);tables={'A_per_seed_metrics':rows,'A_per_seed_paired_comparisons':paired}
    if not frame.empty:
        fresh=frame[frame.seed.ne(42)]
        stability=fresh.groupby('method').agg(completed_seeds=('seed','nunique'),accuracy_mean=('accuracy','mean'),accuracy_sd=('accuracy','std'),accuracy_min=('accuracy','min'),accuracy_max=('accuracy','max'),f1_mean=('macro_f1','mean'),f1_sd=('macro_f1','std'),auc_mean=('macro_auc','mean'),auc_sd=('macro_auc','std'),ece_mean=('ece','mean')).reset_index()
        tables['A_five_new_seed_stability']=stability
    if all(s in by_seed for s in SEEDS):
        ref_ids,ref_y,_=by_seed[SEEDS[0]]
        for s in SEEDS:
            if not np.array_equal(by_seed[s][0],ref_ids) or not np.array_equal(by_seed[s][1],ref_y):raise ValueError('Repeat group labels differ.')
        predictions=np.stack([by_seed[s][2]['ensemble'].argmax(1) for s in SEEDS])
        correct=predictions==ref_y[None,:]
        tables['A_group_repeat_consistency']=[{'group':g,'label':int(ref_y[i]),'correct_runs':int(correct[:,i].sum()),'runs':5,'distinct_predictions':len(set(predictions[:,i]))} for i,g in enumerate(ref_ids)]
        # One cluster bootstrap unit is a known group, retaining its whole vector
        # across seeds. Never pool 5*520 rows as independent patients.
        mean_correct=correct.mean(0);rng=np.random.default_rng(615);draws=rng.integers(0,len(ref_ids),(2000,len(ref_ids)))
        ci=np.percentile(mean_correct[draws].mean(1),[2.5,97.5])
        write_json(out/'A_conditional_repeated_group_bootstrap.json',{'mean_accuracy':float(mean_correct.mean()),'ci_95':ci.tolist(),
            'unit':'one known group with all five correlated predictions kept together',
            'scope':'Conditional on these fitted models and these five seeded runs; not population variability of training or an external-validation interval.'})
    arows,gaps,stats=ablation_tables(abl);tables.update({'ablation384_per_seed':arows,'ablation384_paired_gaps':gaps})
    write_json(out/'ablation384_summary.json',stats)
    export_tables(tables,out/'tables');b_comparisons(inputs,out/'B_comparisons')
    figures(frame,gaps,out/'figures')
    missing=missing_a+missing_b;pd.DataFrame(missing).to_csv(out/'MISSING_TRAINING_JOBS.csv',index=False)
    extra=additional_checks(inputs,roots,out)
    status={'training_complete':not missing,'expected_new_A_jobs':100,'completed_new_A_jobs':100-len([r for r in missing_a if r['seed']!=42]),
        'expected_ablation_jobs':10,'completed_ablation_jobs':len(abl),'pending_external_validation':True,'pending_verified_patient_identity':True,
        'expert_localisation_validation':'pending',**extra,
        'claims':'Supplemental internal analyses. No promised superiority; report all seeds, all methods and unfavorable findings. No official manuscript result is replaced.'}
    write_json(out/'COMPLETENESS.json',status)
    (out/'INTERPRETATION.md').write_text('''# Supplemental interpretation\n\nReport Protocol A and Protocol B separately: their cohorts, training pools and aggregation differ. Protocol A covers every known group once per seeded outer evaluation; Protocol B preserves the official test sheets. Do not compare their accuracies as if their test cohorts were identical.\n\nThe five new runs vary split and training seeds together. Mean/SD/range describe observed internal sensitivity; the runs overlap in training data and do not create 2,600 independent patients. Report baseline seed 42 separately.\n\nThe 384px ConvNeXt ablation is a controlled recipe sensitivity analysis alongside the unchanged 128px ViT Table 5. It changes architecture, resolution, normalization and training schedule together; it cannot isolate a resolution effect. Report image-level and known-group summaries, the five paired gaps, and both the paired t interval/test and exact five-pair sign test. Do not convert this gap into a universal clinical accuracy penalty.\n\nEnsemble comparisons and subset fits use calibration data only. These additional analyses are supplemental after prior test-set exposure; no subset is selected using its test score. Holm adjustment covers the seven named accuracy contrasts within each protocol/seed, not all possible explorations in the project. No significant ensemble advantage is guaranteed or required.\n\nDuplicate retrieval is an aid to content adjudication, not verified identity. If additional cross-group related images are confirmed, record the evidence and revise affected splits and training before making revised generalisation claims. Even a completed review cannot establish that no undiscovered relationships exist.\n\nPerturbation and randomization results describe model sensitivity. Masking may hide lesions and creates a distribution shift. A passed saliency sanity check does not establish lesion localisation, clinical reasoning or clinical utility. Expert masks and independent clinical validation remain pending. Historical use of MSLD during development cannot be erased by resplitting it.\n''',encoding='utf-8')
    print(json.dumps(status,indent=2));bundle(out)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--inputs',required=True);p.add_argument('--expected-sha',required=True)
    p.add_argument('--input-root',default='/kaggle/input');p.add_argument('--work',default='/kaggle/working');main(p.parse_args())
