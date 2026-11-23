"""CPU-only merge of twenty completed corrected-fold results; no legacy mixing."""
from pathlib import Path
import argparse
import json
import platform
from zipfile import ZipFile, ZIP_DEFLATED
import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import binomtest
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                             roc_auc_score, log_loss, confusion_matrix, roc_curve)
from corrected_core import (MODELS, CLASSES, find_inputs, verify_inputs, sha256,
                            atomic_json, atomic_array, safe_extract, pool_predictions)


def scaled(prob,temp):
    logs=np.log(np.clip(prob,1e-12,1))/temp
    logs-=logs.max(1,keepdims=True)
    exp=np.exp(logs)
    return exp/exp.sum(1,keepdims=True)


def ece(prob,y):
    conf=prob.max(1);correct=prob.argmax(1)==y;value=0.
    bins=[]
    for low,high in zip(np.linspace(0,1,11)[:-1],np.linspace(0,1,11)[1:]):
        mask=(conf>low)&(conf<=high)
        if mask.any():
            confidence=float(conf[mask].mean());accuracy=float(correct[mask].mean())
            value+=mask.mean()*abs(confidence-accuracy)
            bins.append({'lower':float(low),'upper':float(high),'n':int(mask.sum()),'confidence':confidence,'accuracy':accuracy})
    return float(value),bins


def metrics(prob,y):
    prob=prob/prob.sum(1,keepdims=True)
    pred=prob.argmax(1)
    try:auc=float(roc_auc_score(y,prob,multi_class='ovr',average='macro',labels=list(range(6))))
    except ValueError:auc=None
    return {'n':len(y),'correct':int((pred==y).sum()),'accuracy':float(accuracy_score(y,pred)),
            'macro_precision':float(precision_score(y,pred,labels=list(range(6)),average='macro',zero_division=0)),
            'macro_recall':float(recall_score(y,pred,labels=list(range(6)),average='macro',zero_division=0)),
            'macro_f1':float(f1_score(y,pred,labels=list(range(6)),average='macro',zero_division=0)),
            'macro_auc':auc,'ece':ece(prob,y)[0],'log_loss':float(log_loss(y,prob,labels=list(range(6))))}


def load_completed(root,manifest_sha):
    records={}
    for marker in sorted(Path(root).rglob('fold_complete.json')):
        meta=json.loads(marker.read_text())
        if meta.get('schema')!='pb_known_duplicate_corrected_v1':continue
        if meta.get('input_manifest_sha256')!=manifest_sha:
            raise ValueError('A corrected result uses a different input manifest. Keep one frozen rerun version.')
        key=(meta['model'],int(meta['fold']))
        if key[0] not in MODELS or key[1] not in range(1,6) or meta.get('status')!='complete' or meta.get('epochs_completed')!=25:
            raise ValueError('An invalid completed-fold marker was found.')
        if key in records:
            if records[key]['meta']['identity_sha256']!=meta['identity_sha256'] or records[key]['meta']['prediction_files']!=meta['prediction_files']:
                raise ValueError(f'Conflicting completed results for {key}.')
            continue
        directory=marker.parent
        for name,info in meta['prediction_files'].items():
            if not (directory/name).is_file() or sha256(directory/name)!=info['sha256']:
                raise ValueError(f'Missing or changed corrected predictions: {directory/name}')
        history=directory/'epoch_history.json'
        if not history.is_file() or sha256(history)!=meta['history_sha256']:
            raise ValueError('Completed fold history is missing or changed.')
        if [r['epoch'] for r in json.loads(history.read_text())]!=list(range(1,26)):
            raise ValueError('Completed training history is not twenty-five consecutive epochs.')
        hist=json.loads(history.read_text())
        weights=meta.get('weights',[])
        retained=sorted(hist,key=lambda row:row['selection_logloss'])[:2]
        if len(weights)!=2 or [int(row['epoch']) for row in weights]!=[int(row['epoch']) for row in retained]:
            raise ValueError('Retained weight epochs differ from the two best selection losses.')
        records[key]={'meta':meta,'directory':directory}
    missing=[f'{m} fold {f}' for m in MODELS for f in range(1,6) if (m,f) not in records]
    if missing:raise ValueError('Still missing completed corrected jobs: '+', '.join(missing))
    return records


def prediction(record,model,fold,split):
    stem=f'pb_corrected_{model}_fold{fold}_{split}'
    result={kind:np.load(record['directory']/f'{stem}_{kind}.npy',allow_pickle=False)
            for kind in ['probs','labels','pids']}
    p,y,ids=result['probs'],result['labels'],result['pids']
    if p.shape!=(len(y),6) or len(ids)!=len(y) or len(set(ids))!=len(ids) or not np.isfinite(p).all() or (p<0).any() or not np.allclose(p.sum(1),1,atol=1e-5):
        raise ValueError('Invalid corrected prediction arrays.')
    return result


def wilson(correct,n):
    z=1.959963984540054;p=correct/n;den=1+z*z/n
    mid=(p+z*z/(2*n))/den
    radius=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [float(mid-radius),float(mid+radius)]


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root',type=Path,default=Path('/kaggle/input'))
    parser.add_argument('--output-root',type=Path,default=Path('/kaggle/working/PB_Corrected_Publication'))
    parser.add_argument('--expected-manifest-sha',required=True)
    parser.add_argument('--bootstrap',type=int,default=2000)
    args=parser.parse_args(argv)
    inputs=find_inputs(args.input_root,args.expected_manifest_sha)
    manifest=verify_inputs(inputs,args.expected_manifest_sha)
    cache=args.output_root/'extracted_results'
    for archive in sorted(args.input_root.rglob('*RESULTS_ONLY*.zip')):
        safe_extract(archive,cache/sha256(archive))
    # Small results packages and directly attached full notebook outputs are accepted.
    records=load_completed(args.input_root,args.expected_manifest_sha) if not cache.exists() else None
    if records is None:
        # A tiny index directory combines paths without copying weight files.
        roots=[args.input_root,cache]
        marker_sets=[]
        for root in roots:marker_sets.extend(root.rglob('fold_complete.json'))
        index=args.output_root/'result_index';index.mkdir(parents=True,exist_ok=True)
        import shutil
        for i,marker in enumerate(sorted(set(marker_sets))):
            meta=json.loads(marker.read_text())
            if meta.get('schema')!='pb_known_duplicate_corrected_v1':continue
            dest=index/f'item_{i}';dest.mkdir(exist_ok=True)
            for name in ['fold_complete.json','epoch_history.json',*meta.get('prediction_files',{})]:
                source=marker.parent/name
                if source.is_file():shutil.copyfile(source,dest/name)
        records=load_completed(index,args.expected_manifest_sha)
    args.output_root.mkdir(parents=True,exist_ok=True)
    pooled={};fold_rows=[];provenance=[]
    for model in MODELS:
        pooled[model]={}
        for split in ['test','val']:
            parts=[]
            for fold in range(1,6):
                part=prediction(records[(model,fold)],model,fold,split)
                assignment=pd.read_csv(inputs/f'splits/fold{fold}.csv')
                role='test' if split=='test' else 'val_calibration'
                expected=set(assignment.loc[assignment.role.eq(role),'patient_cluster'])
                if set(part['pids'])!=expected:raise ValueError('Result groups differ from corrected frozen assignments.')
                label_map=assignment.groupby('patient_cluster').label.first().to_dict()
                if not np.array_equal(part['labels'],np.asarray([label_map[v] for v in part['pids']])):
                    raise ValueError('Result labels differ from the frozen class order.')
                parts.append(part)
                if split=='test':fold_rows.append({'model':model,'fold':fold,**metrics(part['probs'],part['labels'])})
            pooled[model][split]=pool_predictions(parts)
        for fold in range(1,6):
            meta=records[(model,fold)]['meta']
            if meta['identity']['training']!=manifest['training'] or meta['identity']['pretrained_source']!=manifest['models'][model]:
                raise ValueError('Training recipe or pretrained source differs across the corrected jobs.')
            for name,digest in meta['identity']['code'].items():
                if manifest['files']['code/'+name]['sha256']!=digest:
                    raise ValueError('A result was trained with code outside the frozen input package.')
            expected_signature=__import__('hashlib').sha256((manifest['files'][f'splits/fold{fold}.csv']['sha256']+
                              manifest['files'][f'augmented/fold{fold}.csv']['sha256']).encode()).hexdigest()
            if meta['identity']['data_signature']!=expected_signature:
                raise ValueError('A result uses a different corrected split/augmentation signature.')
            provenance.append({'model':model,'fold':fold,'training_seed':meta['identity']['training_seed'],
                               'identity_sha256':meta['identity_sha256'],**meta['identity']['environment']})
    reference=pooled[MODELS[0]]
    for model in MODELS[1:]:
        for split in ['test','val']:
            if any(not np.array_equal(pooled[model][split][key],reference[split][key]) for key in ['labels','pids']):
                raise ValueError('Corrected models have different evaluated group cohorts.')
    ids=reference['test']['pids'];y=reference['test']['labels']
    calids=reference['val']['pids'];caly=reference['val']['labels']
    keep=~np.isin(calids,ids)
    if set(caly[keep])!=set(range(6)):raise ValueError('A class is absent after global calibration/test group filtering.')
    expected=manifest['cohort_counts']
    if len(ids)!=expected['unique_test_groups'] or int(keep.sum())!=expected['pooled_calibration_groups_after_test_exclusion']:
        raise ValueError('Pooled corrected cohorts differ from the frozen plan.')
    temperatures={};test_scaled=[];cal_scaled=[];rows=[]
    for model in MODELS:
        test=pooled[model]['test']['probs'];cal=pooled[model]['val']['probs'][keep]
        fit=minimize_scalar(lambda temp:log_loss(caly[keep],scaled(cal,temp),labels=list(range(6))),bounds=(.1,5),method='bounded')
        if not fit.success:raise ValueError('Temperature optimization failed.')
        temp=float(fit.x);temperatures[model]=temp
        transformed=scaled(test,temp);test_scaled.append(transformed);cal_scaled.append(scaled(cal,temp))
        rows.extend([{'model':model,'stage':'raw',**metrics(test,y)},
                     {'model':model,'stage':'temperature_scaled',**metrics(transformed,y)}])
    cal_stack=np.stack(cal_scaled,axis=0);test_stack=np.stack(test_scaled,axis=0)
    def softmax(x):
        exp=np.exp(x-np.max(x));return exp/exp.sum()
    fit=minimize(lambda x:log_loss(caly[keep],np.tensordot(softmax(x),cal_stack,axes=1),labels=list(range(6))),
                 np.zeros(4),method='L-BFGS-B')
    if not fit.success:raise ValueError('Ensemble weight optimization failed: '+str(fit.message))
    weights=softmax(fit.x);prob=np.tensordot(weights,test_stack,axes=1);point=metrics(prob,y)
    rows.append({'model':'ensemble','stage':'calibrated',**point})
    paired=[]
    for model in MODELS:
        single=pooled[model]['test']['probs'].argmax(1)==y;ens=prob.argmax(1)==y
        b=int((ens&~single).sum());c=int((single&~ens).sum())
        paired.append({'model':model,'only_ensemble_correct':b,'only_backbone_correct':c,
                       'exact_mcnemar_p':float(binomtest(min(b,c),b+c,.5).pvalue) if b+c else 1.0})
    last=0.
    for index,row in enumerate(sorted(paired,key=lambda r:r['exact_mcnemar_p'])):
        last=min(1.,max(last,(4-index)*row['exact_mcnemar_p']));row['holm_p']=last
    rng=np.random.default_rng(42);bootstrap=[]
    boot_diff={model:{'macro_f1':[],'macro_auc':[]} for model in MODELS}
    for _ in range(args.bootstrap):
        sample=rng.integers(0,len(y),len(y));entry=metrics(prob[sample],y[sample]);bootstrap.append(entry)
        for model in MODELS:
            single=metrics(pooled[model]['test']['probs'][sample],y[sample])
            for key in ['macro_f1','macro_auc']:
                if entry[key] is not None and single[key] is not None:
                    boot_diff[model][key].append(entry[key]-single[key])
    ci={key:[float(v) for v in np.quantile([r[key] for r in bootstrap if r[key] is not None],[.025,.975])]
        for key in ['macro_f1','macro_auc','ece']} if bootstrap else {}
    for row in paired:
        for key in ['macro_f1','macro_auc']:
            row[key+'_difference']=point[key]-metrics(pooled[row['model']]['test']['probs'],y)[key]
            values=boot_diff[row['model']][key]
            if values:
                low,high=np.quantile(values,[.025,.975]);row[key+'_difference_ci_low']=float(low);row[key+'_difference_ci_high']=float(high)
    cm=confusion_matrix(y,prob.argmax(1),labels=list(range(6)))
    perclass=[];roc={}
    for cls,name in enumerate(CLASSES):
        truth=y==cls;guess=prob.argmax(1)==cls
        fpr,tpr,threshold=roc_curve(truth,prob[:,cls])
        roc[name]={'fpr':fpr.tolist(),'tpr':tpr.tolist()}
        perclass.append({'class':'Mpox' if name=='Monkeypox' else name,'support':int(truth.sum()),
                         'precision':float(precision_score(truth,guess,zero_division=0)),
                         'recall':float(recall_score(truth,guess,zero_division=0)),
                         'f1':float(f1_score(truth,guess,zero_division=0)),
                         'auc':float(roc_auc_score(truth,prob[:,cls]))})
    pd.DataFrame(rows).to_csv(args.output_root/'model_metrics.csv',index=False)
    pd.DataFrame(perclass).to_csv(args.output_root/'per_class_metrics.csv',index=False)
    pd.DataFrame(cm,index=CLASSES,columns=CLASSES).to_csv(args.output_root/'confusion_matrix.csv')
    pd.DataFrame(fold_rows).to_csv(args.output_root/'raw_backbone_fold_metrics.csv',index=False)
    pd.DataFrame(paired).to_csv(args.output_root/'paired_accuracy_comparisons.csv',index=False)
    pd.DataFrame(provenance).to_csv(args.output_root/'training_session_provenance.csv',index=False)
    pd.DataFrame({'model':MODELS,'temperature':[temperatures[m] for m in MODELS],'ensemble_weight':weights}).to_csv(args.output_root/'calibration_parameters.csv',index=False)
    predictions=pd.DataFrame({'known_group':ids,'label':y,'prediction':prob.argmax(1),**{f'prob_{i}':prob[:,i] for i in range(6)}})
    predictions.to_csv(args.output_root/'per_group_predictions.csv',index=False)
    atomic_array(args.output_root/'ensemble_probs.npy',prob)
    atomic_array(args.output_root/'test_labels.npy',y)
    atomic_array(args.output_root/'test_groups.npy',ids.astype(str))
    report={'protocol':manifest['protocol'],'schema':manifest['schema'],'input_manifest_sha256':args.expected_manifest_sha,
            'completed_jobs':20,'test_groups':len(ids),'calibration_candidates':len(calids),
            'excluded_calibration_test_overlap':int((~keep).sum()),'calibration_groups_used':int(keep.sum()),
            'ensemble':point,'accuracy_wilson_95_ci':wilson(point['correct'],len(y)),
            'temperatures':temperatures,'weights':dict(zip(MODELS,weights.tolist())),
            'conditional_bootstrap_n':args.bootstrap,'conditional_bootstrap_95_ci':ci,
            'bootstrap_scope':'Resamples known groups with training, temperature and weights fixed; excludes training/design uncertainty.',
            'reliability_bins':ece(prob,y)[1],'per_class':perclass,'confusion_matrix':cm.tolist(),
            'historical_design_and_unknown_identity_limitations_remain':True}
    atomic_json(args.output_root/'report.json',report);atomic_json(args.output_root/'roc_points.json',roc)
    # Publication-ready vector and raster plots. The manuscript can select A/B figures later.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Serif','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    figures=args.output_root/'figures';figures.mkdir(exist_ok=True)
    fig,ax=plt.subplots(figsize=(6.6,5.5));im=ax.imshow(cm,cmap='Blues');fig.colorbar(im,ax=ax,label='Evaluated known groups')
    labels=['Mpox' if n=='Monkeypox' else n for n in CLASSES]
    ax.set(xticks=range(6),yticks=range(6),xticklabels=labels,yticklabels=labels,xlabel='Predicted class',ylabel='Reference class')
    plt.setp(ax.get_xticklabels(),rotation=35,ha='right')
    for i in range(6):
        for j in range(6):ax.text(j,i,str(cm[i,j]),ha='center',va='center',color='white' if cm[i,j]>cm.max()/2 else 'black')
    fig.tight_layout()
    for ext in ['png','pdf','svg']:fig.savefig(figures/f'confusion_matrix.{ext}',dpi=300)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(5.8,4.8))
    for name,entry in roc.items():ax.plot(entry['fpr'],entry['tpr'],label='Mpox' if name=='Monkeypox' else name)
    ax.plot([0,1],[0,1],'--',color='gray');ax.set(xlabel='False-positive rate',ylabel='True-positive rate');ax.legend(fontsize=8)
    fig.tight_layout()
    for ext in ['png','pdf','svg']:fig.savefig(figures/f'class_roc.{ext}',dpi=300)
    plt.close(fig)
    bins=report['reliability_bins'];fig,ax=plt.subplots(figsize=(5.5,4.5))
    ax.plot([0,1],[0,1],'--',color='gray');ax.plot([b['confidence'] for b in bins],[b['accuracy'] for b in bins],'o-')
    ax.set(xlabel='Mean confidence',ylabel='Observed accuracy',xlim=(0,1),ylim=(0,1));fig.tight_layout()
    for ext in ['png','pdf','svg']:fig.savefig(figures/f'reliability.{ext}',dpi=300)
    plt.close(fig)
    bundle=args.output_root.parent/'PB_Corrected_Publication_Outputs.zip'
    with ZipFile(bundle,'w',compression=ZIP_DEFLATED) as zipped:
        for path in sorted(args.output_root.rglob('*')):
            if path.is_file() and not any(part in ['extracted_results','result_index'] for part in path.relative_to(args.output_root).parts):
                zipped.write(path,'PB_Corrected_Publication/'+path.relative_to(args.output_root).as_posix())
    print(json.dumps(report,indent=2));print(f'Download: {bundle}')


if __name__=='__main__':main()
