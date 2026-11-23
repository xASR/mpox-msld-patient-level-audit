"""Quantitative ensemble sensitivity and parameter-randomization sanity checks.

These are internal sensitivity diagnostics, not clinical or causal validation.
All aggregation context is recomputed after changing images or parameters.
"""
from pathlib import Path
import argparse
import gc as python_gc
import json
import time
import numpy as np
import pandas as pd
from PIL import Image, ImageFilter
from scipy.stats import spearmanr
import torch
import torch.nn.functional as F
import timm
from supplemental_common import (MODELS, sha, digest, write_json, verify_inputs, bundle, export_tables)
import ensemble_gradcam as cam
from final_common import prepare_inputs, read_results
from final_analysis import numpy_metrics


def mask_for(name, filename, annotations=None):
    n=384;mask=np.zeros((n,n),dtype=bool)
    if name=='border':mask[:12]=True;mask[-12:]=True;mask[:,:12]=True;mask[:,-12:]=True
    elif name.startswith('control'):
        shift=(96,128)[int(name[-1])-1]
        for start in (shift,384-shift-12): mask[start:start+12]=True;mask[:,start:start+12]=True
    elif name=='artifact':
        for x0,y0,x1,y1 in annotations.get(filename,[]):mask[y0:y1,x0:x1]=True
    elif name.startswith('artifact_control'):
        original=mask_for('artifact',filename,annotations)
        mask=np.roll(original,(96 if name.endswith('1') else 192,128),axis=(0,1))
    elif name!='original':raise ValueError(name)
    return mask


def edit_image(image, filename, condition, annotations):
    kind,fill=condition.split(':') if ':' in condition else (condition,'mean')
    if kind=='original':return image
    image=cam.TRANSFORM.transforms[0](image)
    mask=mask_for(kind,filename,annotations)
    a=np.asarray(image).copy()
    if fill=='blur':replacement=np.asarray(image.filter(ImageFilter.GaussianBlur(12)))
    else:replacement=np.broadcast_to(np.round(np.array([.485,.456,.406])*255).astype(np.uint8),a.shape)
    a[mask]=replacement[mask]
    return Image.fromarray(a)


def load_annotations(path,clinical):
    if not path:return None
    frame=pd.read_csv(path).fillna('');expected=clinical.set_index('filename').sha256.to_dict();values={}
    if frame.filename.duplicated().any():raise ValueError('Duplicate artifact annotation rows.')
    for row in frame.itertuples():
        if row.filename not in expected or row.sha256!=expected[row.filename]:raise ValueError('Annotation image hash differs.')
        if str(row.annotation_blinded).lower()!='true' or not str(row.reviewer).strip():raise ValueError('Artifact review must be blinded and signed.')
        if str(row.artifact_present).lower() not in ('true','false'):raise ValueError('Complete artifact_present with true/false, not pending.')
        boxes=json.loads(row.boxes_xyxy_json)
        for box in boxes:
            if len(box)!=4 or any(not isinstance(v,int) for v in box):raise ValueError('Use integer 384px bounding boxes.')
            x0,y0,x1,y1=box
            if not (0<=x0<x1<=384 and 0<=y0<y1<=384):raise ValueError('Invalid artifact box.')
        if bool(boxes)!=(str(row.artifact_present).lower()=='true'):raise ValueError('Artifact flag/boxes differ.')
        values[row.filename]=boxes
    if set(values)!=set(expected):raise ValueError('Annotate all 755 images before targeted artifact analysis; do not select using results.')
    return values


def net_for(manifest,model,weight,device,stage,fold,rank):
    seed=40000+100*MODELS.index(model)+10*fold+rank
    with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if stage=='random_all':
            cfg=manifest['models'][model];kwargs={'pretrained':False,'num_classes':6}
            if model in ('swin_small','beit_base'):kwargs['img_size']=384
            net=timm.create_model(cfg['timm_name'],**kwargs).eval().to(device)
        else:
            net=cam.load_network(manifest,model,weight,device)
            if stage=='random_head':
                head=net.get_classifier()
                if not hasattr(head,'reset_parameters'):raise ValueError('Unrecognized classifier module.')
                head.reset_parameters()
    return net


def regroup(cache,ids):
    pooled={}
    for model in MODELS:
        sums={g:[] for g in ids}
        for fold in range(1,6):
            left=cache[(model,fold,1)];right=cache[(model,fold,2)]
            if not np.array_equal(left['groups'],right['groups']):raise ValueError('Rank groups differ.')
            for i,g in enumerate(left['groups']):sums[g].append((left['group_probs'][i]+right['group_probs'][i])/2)
        if any(not v for v in sums.values()):raise ValueError('Missing group inference.')
        pooled[model]=np.stack([np.mean(sums[g],axis=0) for g in ids])
    return pooled


def score(pooled,temps,weights):
    return sum(weights[m]*cam.scale_numpy(pooled[m],temps[m]) for m in MODELS)


def predict(manifest,splits,weights,images,out,device,stage,condition,annotations,budget_end,batch=8):
    out.mkdir(parents=True,exist_ok=True);cache={}
    for model in MODELS:
        for fold in range(1,6):
            frame=splits[fold].loc[splits[fold].role.eq('test')].reset_index(drop=True)
            for rank in (1,2):
                if time.monotonic()>budget_end:raise TimeoutError('Analysis paused. Save FULL notebook outputs and attach them to a new copy to reuse finished branches.')
                wp=weights[(model,fold,rank)];dest=out/f'{model}_f{fold}_r{rank}.npz';mark=dest.with_suffix('.json')
                identity={'stage':stage,'condition':condition,'weight':sha(wp),'filenames':frame.filename.tolist(),
                    'annotation':digest(annotations),'script':sha(__file__),'torch':str(torch.__version__),'timm':timm.__version__,
                    'Pillow':__import__('PIL').__version__,'gpu':torch.cuda.get_device_name(0),'batch':batch}
                signature=digest(identity)
                if dest.exists() and mark.exists() and json.loads(mark.read_text()).get('signature')==signature and json.loads(mark.read_text()).get('sha256')==sha(dest):
                    with np.load(dest,allow_pickle=False) as z:p=z['probs']
                else:
                    net=net_for(manifest,model,wp,device,stage,fold,rank);parts=[]
                    with torch.no_grad():
                        for start in range(0,len(frame),batch):
                            chunk=frame.iloc[start:start+batch]
                            x=torch.stack([cam.TRANSFORM(edit_image(Image.open(images[n.upper()]).convert('RGB'),n,condition,annotations)) for n in chunk.filename]).to(device)
                            pviews=[F.softmax(net(v),1) for v in (x,x.flip([3]),x.rot90(1,[2,3]),x.rot90(3,[2,3]))]
                            parts.append(torch.stack(pviews).mean(0).cpu().numpy())
                    p=np.concatenate(parts);np.savez_compressed(dest,probs=p)
                    write_json(mark,{'signature':signature,'identity':identity,'sha256':sha(dest)})
                    del net;python_gc.collect();torch.cuda.empty_cache()
                groups=np.sort(frame.patient_cluster.unique())
                gp=np.stack([cam.geometric_numpy(p[frame.patient_cluster.eq(g).to_numpy()]) for g in groups])
                cache[(model,fold,rank)]={'frame':frame,'groups':groups,'group_probs':gp,'mean_view_probs':p}
                print(stage,condition,model,fold,rank,'saved',flush=True)
    return cache


def fixed_cases(ids,y,splits):
    """Two groups per class, selected by a fixed hash, independent of confidence/errors."""
    rows=[]
    for c in range(6):
        groups=sorted(ids[y==c],key=lambda g:digest({'selection_seed':615,'group':str(g)}))[:2]
        for g in groups:
            filenames=sorted(set().union(*(set(a.loc[a.role.eq('test')&a.patient_cluster.eq(g),'filename']) for a in splits.values())))
            rows.append({'group':str(g),'filename':filenames[0],'target':c})
    return rows


def signed_maps(manifest,splits,weights,images,cache,pooled,temps,mixture,ensemble,ids,cases,out,device,stage,budget_end):
    out.mkdir(parents=True,exist_ok=True);allmaps=[np.zeros((384,384),np.float64) for _ in cases]
    index={g:i for i,g in enumerate(ids)}
    signature=digest({'stage':stage,'cases':cases,'temps':temps,'weights':mixture,'script':sha(__file__),
                      'torch':str(torch.__version__),'timm':timm.__version__,'pooled':{m:digest(pooled[m].tolist()) for m in MODELS}})
    for model in MODELS:
        for fold in range(1,6):
            for rank in (1,2):
                data=cache[(model,fold,rank)];frame=data['frame'];net=None
                for k,case in enumerate(cases):
                    if case['filename'] not in set(frame.filename):continue
                    p=out/f'{model}_f{fold}_r{rank}_case{k}.npy';mark=p.with_suffix('.json')
                    if p.exists() and mark.exists() and json.loads(mark.read_text()).get('signature')==signature and json.loads(mark.read_text()).get('sha256')==sha(p):
                        raw=np.load(p,allow_pickle=False)
                    else:
                        if time.monotonic()>budget_end:raise TimeoutError('Sanity checks paused; retain full analysis output to resume.')
                        if net is None:net=net_for(manifest,model,weights[(model,fold,rank)],device,stage,fold,rank)
                        layer,_=cam.target_layer(net,model);capture={}
                        def hook(module,args,output):capture['a']=output
                        handle=layer.register_forward_hook(hook)
                        try:
                            net.zero_grad(set_to_none=True)
                            x=cam.TRANSFORM(Image.open(images[case['filename'].upper()]).convert('RGB'))[None].to(device)
                            q=net(cam.views(x)).softmax(1).mean(0)
                            mask=frame.patient_cluster.eq(case['group']).to_numpy();qi=int(np.flatnonzero(frame.filename.eq(case['filename']))[0]);other=mask.copy();other[qi]=False
                            logs=np.log(data['mean_view_probs'][other]+1e-8).sum(0);gi=index[case['group']]
                            ranki=int(np.flatnonzero(data['groups']==case['group'])[0]);rawp=pooled[model][gi]
                            nf=sum(case['group'] in set(a.loc[a.role.eq('test'),'patient_cluster']) for a in splits.values())
                            fixed=float(ensemble[gi,case['target']]-mixture[model]*cam.scale_numpy(rawp,temps[model])[case['target']])
                            tensor=lambda a:torch.as_tensor(a,device=device,dtype=q.dtype)
                            s=cam.branch_score(q,tensor(logs),int(mask.sum()),tensor(data['group_probs'][ranki]),tensor(rawp),nf,
                                               temps[model],mixture[model],fixed,case['target'])
                            difference=float(abs(float(s.detach().cpu())-float(ensemble[gi,case['target']])))
                            if difference>1e-3:raise ValueError('Changed-model context does not reproduce its full ensemble score.')
                            a=capture['a'];grad=torch.autograd.grad(s,a)[0]
                            aa=cam.spatial(a,model);gg=cam.spatial(grad,model)
                            signed=(gg.mean((-2,-1),keepdim=True)*aa).sum(1)
                            signed=F.interpolate(signed[:,None],size=(384,384),mode='bilinear',align_corners=False)[:,0]
                            raw=sum(cam.unview(signed[v],v) for v in range(4)).detach().cpu().numpy()
                            if not np.isfinite(raw).all():raise ValueError('Nonfinite map.')
                            np.save(p,raw,allow_pickle=False);write_json(mark,{'signature':signature,'sha256':sha(p),'partial_score_error':difference})
                        finally:handle.remove()
                    allmaps[k]+=raw
                if net is not None:del net
                python_gc.collect();torch.cuda.empty_cache()
    result=np.stack(allmaps)
    np.save(out/'summed_signed_maps.npy',result,allow_pickle=False)
    return result


def map_comparison(base,changed,cases,stage):
    rows=[]
    for a,b,case in zip(base,changed,cases):
        # Coarsen before rank comparison so interpolated pixels do not create pseudo observations.
        aa=np.maximum(a,0).reshape(24,16,24,16).mean((1,3)).ravel();bb=np.maximum(b,0).reshape(24,16,24,16).mean((1,3)).ravel()
        k=max(1,int(np.ceil(.1*len(aa))));ia=set(np.argsort(aa,kind='stable')[-k:]);ib=set(np.argsort(bb,kind='stable')[-k:])
        constant=np.ptp(aa)==0 or np.ptp(bb)==0
        correlation=None if constant else float(spearmanr(aa,bb).statistic)
        rows.append({**case,'stage':stage,'spearman_24x24':correlation,'top10_jaccard':len(ia&ib)/len(ia|ib),
                     'constant_map':constant,'diagnostic_only':True})
    return rows


def sensitivity_rows(ids,y,base,changed,name):
    pred=base.argmax(1);correct=base.argmax(1)==y;cp=changed.argmax(1)==y
    return pd.DataFrame({'group':ids,'label':y,'condition':name,'baseline_prediction':pred,
        'changed_prediction':changed.argmax(1),'baseline_correct':correct,'changed_correct':cp,
        'reference_score_delta':changed[np.arange(len(y)),y]-base[np.arange(len(y)),y],
        'baseline_predicted_score_delta':changed[np.arange(len(y)),pred]-base[np.arange(len(y)),pred],
        'decision_changed':changed.argmax(1)!=pred})


def main(args):
    verify_inputs(args.inputs,args.expected_sha);out=Path(args.work)/'SUPP_XAI';out.mkdir(parents=True,exist_ok=True)
    # Attach this notebook's previous full output to a new copy after a pause.
    import shutil
    for previous in Path(args.input_root).rglob('SUPP_XAI'):
        if previous.is_dir() and previous.resolve()!=out.resolve():
            for source in previous.rglob('*'):
                if source.is_file() and source.suffix in ('.npz','.npy','.json'):
                    dest=out/source.relative_to(previous)
                    if not dest.exists():dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,dest)
    root=Path(args.inputs)/'b_inputs';marker=root/'pb_final_verified_manifest.json'
    verified=prepare_inputs(root,out,sha(marker));manifest,records,pooled,splits,keep=read_results(verified)
    clinical=pd.read_csv(verified/'frozen/clinical_images.csv');images=cam.find_originals(args.input_root,clinical)
    weights=cam.find_weights(args.input_root,records,out)
    temps,mixture,official=cam.fit_ensemble(pooled,keep)
    ids=pooled[MODELS[0]]['test']['pids'];y=pooled[MODELS[0]]['test']['labels']
    annotations=load_annotations(args.annotations,clinical)
    if args.verify_only:print('PREFLIGHT PASS: 40/40 weights and all 755 images verified.');return
    if not torch.cuda.is_available():raise RuntimeError('GPU T4 x2 required.')
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True,warn_only=True)
    device=torch.device('cuda:0');end=time.monotonic()+args.session_hours*3600
    cases=fixed_cases(ids,y,splits);pd.DataFrame(cases).to_csv(out/'fixed_explanation_cases.csv',index=False)
    try:
        cache=predict(manifest,splits,weights,images,out/'cache/trained_original',device,'trained','original',None,end)
        raw=regroup(cache,ids);base=score(raw,temps,mixture)
        if np.abs(base-official).max()>5e-4 or not np.array_equal(base.argmax(1),official.argmax(1)):
            raise ValueError('Saved ensemble reproduction failed; no subsequent explanation is accepted.')
        write_json(out/'baseline_reproduction.json',{'max_abs_diff':float(np.abs(base-official).max()),'same_predictions':True})
        rows=[];sensitivity=[]
        if args.mode in ('all','sensitivity'):
            conditions=[f'{kind}:{fill}' for fill in ('mean','blur') for kind in ('border','control1','control2')]
            if annotations:conditions += [f'{kind}:{fill}' for fill in ('mean','blur') for kind in ('artifact','artifact_control1','artifact_control2')]
            for condition in conditions:
                edited=predict(manifest,splits,weights,images,out/'cache'/condition.replace(':','_'),device,'trained',condition,annotations,end)
                prediction=score(regroup(edited,ids),temps,mixture)
                part=sensitivity_rows(ids,y,base,prediction,condition);sensitivity.append(part)
                rows.append({'condition':condition,**numpy_metrics(prediction,y),'decision_change_fraction':float((prediction.argmax(1)!=base.argmax(1)).mean())})
                pd.concat(sensitivity).to_csv(out/'group_sensitivity.csv',index=False)
            export_tables({'perturbation_summary':rows},out/'tables')
            # Matched within-group contrasts, with equal areas and both fill strategies.
            contrasts=[]
            for fill in ('mean','blur'):
                for prefix in (['border']+(['artifact'] if annotations else [])):
                    arrays={r.condition.iloc[0]:r.set_index('group') for r in sensitivity}
                    target=arrays[prefix+':'+fill]['reference_score_delta']
                    control_prefix='control' if prefix=='border' else 'artifact_control'
                    controls=(arrays[control_prefix+'1:'+fill]['reference_score_delta']+arrays[control_prefix+'2:'+fill]['reference_score_delta'])/2
                    diff=(target-controls).to_numpy();rng=np.random.default_rng(615)
                    draws=rng.integers(0,len(diff),(2000,len(diff)));ci=np.percentile(diff[draws].mean(1),[2.5,97.5])
                    contrasts.append({'intervention':prefix,'fill':fill,'mean_score_delta_vs_controls':float(diff.mean()),'ci_low':float(ci[0]),'ci_high':float(ci[1]),'groups':len(diff)})
            export_tables({'matched_perturbation_contrasts':contrasts},out/'tables')
        if args.mode in ('all','sanity'):
            baseline_maps=signed_maps(manifest,splits,weights,images,cache,raw,temps,mixture,base,ids,cases,out/'maps/trained',device,'trained',end)
            comparisons=[]
            for stage in ('random_head','random_all'):
                altered=predict(manifest,splits,weights,images,out/'cache'/stage,device,stage,'original',None,end)
                changed_raw=regroup(altered,ids);changed=score(changed_raw,temps,mixture)
                maps=signed_maps(manifest,splits,weights,images,altered,changed_raw,temps,mixture,changed,ids,cases,out/'maps'/stage,device,stage,end)
                comparisons+=map_comparison(baseline_maps,maps,cases,stage)
                pd.DataFrame(comparisons).to_csv(out/'parameter_randomization_maps.csv',index=False)
                write_json(out/(stage+'_scores.json'),{'temperatures_and_mixture':'fixed at original calibration-only fit',
                    'randomization_seed_rule':'40000 + 100*model_index + 10*fold + rank','all_group_context_recomputed':True,
                    'decision_change_fraction':float((base.argmax(1)!=changed.argmax(1)).mean())})
            export_tables({'parameter_randomization_summary':pd.DataFrame(comparisons).groupby('stage').agg(
                cases=('group','size'),median_spearman=('spearman_24x24','median'),median_top10_jaccard=('top10_jaccard','median'),constant_maps=('constant_map','sum')).reset_index()},out/'tables')
        write_json(out/f'{args.mode}_status.json',{'status':'complete','mode':args.mode,'analysis_plan_sha256':sha(Path(args.inputs)/'analysis_plan.json'),
            'external_validation':'pending','verified_patient_identity':'pending','expert_localisation_masks':'pending',
            'targeted_artifact_masks':'complete' if annotations and args.mode in ('all','sensitivity') else 'pending',
            'interpretation':'Masking changes the input distribution and can cover lesions. Randomization sensitivity does not prove clinical validity. No lesion-localisation accuracy is claimed.'})
    except TimeoutError as e:
        write_json(out/f'{args.mode}_status.json',{'status':'paused','mode':args.mode,'instruction':str(e)})
        print(str(e),flush=True)
    finally:bundle(out)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--inputs',required=True);p.add_argument('--expected-sha',required=True)
    p.add_argument('--input-root',default='/kaggle/input');p.add_argument('--work',default='/kaggle/working')
    p.add_argument('--mode',choices=['all','sensitivity','sanity'],default='all');p.add_argument('--annotations');p.add_argument('--session-hours',type=float,default=10.5);p.add_argument('--verify-only',action='store_true')
    main(p.parse_args())
