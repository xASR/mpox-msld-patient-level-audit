"""Qualitative Grad-CAM targeting the complete calibrated known-group ensemble.

No retraining. All twenty jobs and both selected checkpoints contribute. The
gradient follows the original four-view, image, checkpoint and fold aggregation.
The cross-backbone map is a coarse aggregate across model-specific feature layers,
not an exact pixel attribution, a lesion segmentation, or a clinical explanation.
"""
from pathlib import Path
import argparse
import gc as python_gc
import hashlib
import json
import math
import os
import platform
import shutil
import tempfile
import time
from zipfile import ZipFile,ZIP_DEFLATED
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F
import timm
from torchvision import transforms as T
from scipy.optimize import minimize,minimize_scalar
from sklearn.metrics import log_loss
from final_common import (MODELS,DISPLAY,LABELS,FROZEN_SHA,prepare_inputs,read_results,
                          sha,write_json)

TRANSFORM=T.Compose([T.Resize((384,384)),T.ToTensor(),T.Normalize([.485,.456,.406],[.229,.224,.225])])
VIEW_NAMES=['identity','horizontal_flip','rotate90','rotate270']

def views(x):return torch.cat([x,torch.flip(x,[3]),torch.rot90(x,1,[2,3]),torch.rot90(x,3,[2,3])],0)

def unview(x,index):
    if index==1:return torch.flip(x,[-1])
    if index==2:return torch.rot90(x,3,[-2,-1])
    if index==3:return torch.rot90(x,1,[-2,-1])
    return x

def geometric_torch(mean_image_probs):
    z=torch.log(mean_image_probs+1e-8).mean(0);z=z-z.max();e=z.exp()
    return e/(e.sum()+1e-8)

def geometric_numpy(mean_image_probs):
    z=np.log(mean_image_probs+1e-8).mean(0);z-=z.max();e=np.exp(z)
    return e/(e.sum()+1e-8)

def scale_numpy(p,t):
    z=np.log(np.clip(p,1e-12,1))/t;z-=z.max(-1,keepdims=True);e=np.exp(z)
    return e/e.sum(-1,keepdims=True)

def fit_ensemble(pooled,keep):
    y=pooled[MODELS[0]]['val']['labels'][keep];temps={};cal=[];test=[]
    for m in MODELS:
        p=pooled[m]['val']['probs'][keep]
        fit=minimize_scalar(lambda t:log_loss(y,scale_numpy(p,t),labels=list(range(6))),bounds=(.1,5),method='bounded')
        if not fit.success:raise ValueError('Temperature fit failed.')
        temps[m]=float(fit.x);cal.append(scale_numpy(p,fit.x));test.append(scale_numpy(pooled[m]['test']['probs'],fit.x))
    cal=np.stack(cal)
    def softmax(x):e=np.exp(x-x.max());return e/e.sum()
    fit=minimize(lambda x:log_loss(y,np.tensordot(softmax(x),cal,axes=1),labels=list(range(6))),np.zeros(4),method='L-BFGS-B')
    if not fit.success:raise ValueError('Mixture fit failed.')
    weights=dict(zip(MODELS,softmax(fit.x).tolist()))
    ensemble=sum(weights[m]*test[i] for i,m in enumerate(MODELS))
    return temps,weights,ensemble

def branch_score(query_probs,other_logsum,n_images,cached_rank_group,pooled_raw,
                 active_fold_count,temp,weight,other_model_score,target):
    """Exact partial score graph; other images/checkpoints/folds/models are fixed.

    query_probs is the mean of this query's four softmax views. Updating one rank
    changes its fold mean by 1/2 and its pooled group mean by 1/active_fold_count.
    No test labels or new calibration fitting occur inside this function.
    """
    mean_log=(other_logsum+torch.log(query_probs+1e-8))/n_images
    e=torch.exp(mean_log-mean_log.max());rank_group=e/(e.sum()+1e-8)
    updated=pooled_raw+(rank_group-cached_rank_group)/(2*active_fold_count)
    transformed=torch.softmax(torch.log(torch.clamp(updated,min=1e-12,max=1))/temp,dim=0)
    return other_model_score+weight*transformed[target]

def spatial(tensor,model):
    if model=='swin_small':return tensor.permute(0,3,1,2)
    if model=='beit_base':
        patch=tensor[:,1:,:];side=math.isqrt(patch.shape[1])
        if side*side!=patch.shape[1]:raise ValueError('Unexpected BEiT token grid.')
        return patch.reshape(patch.shape[0],side,side,patch.shape[2]).permute(0,3,1,2)
    return tensor

def target_layer(net,model):
    if model=='efficientnetv2_s':return net.conv_head,'conv_head'
    if model=='convnext_tiny':return net.stages[-1],'stages[-1]'
    if model=='swin_small':return net.layers[-1].blocks[-1].norm1,'layers[-1].blocks[-1].norm1'
    if model=='beit_base':return net.blocks[-1].norm1,'blocks[-1].norm1'
    raise ValueError(model)

def load_network(manifest,model,weight,device):
    cfg=manifest['models'][model];kwargs={'pretrained':False,'num_classes':6}
    if model in ['swin_small','beit_base']:kwargs['img_size']=384
    net=timm.create_model(cfg['timm_name'],**kwargs)
    net.load_state_dict(torch.load(weight,map_location='cpu',weights_only=True),strict=True)
    return net.eval().to(device)

def find_originals(search_root,clinical):
    expected={r.filename.upper():r.sha256 for r in clinical.itertuples()}
    candidates=[p for p in Path(search_root).rglob('*') if p.is_dir() and p.name.lower()=='original images']
    good=[]
    for folder in candidates:
        images={}
        for p in sorted(folder.rglob('*')):
            if p.is_file() and p.suffix.lower() in ['.jpg','.jpeg','.png']:
                key=p.name.upper()
                if key in images and sha(images[key])!=sha(p):raise ValueError('Conflicting original filenames.')
                images[key]=p
        if set(images)==set(expected):
            for name,digest in expected.items():
                if sha(images[name])!=digest:raise ValueError('Original image hash differs: '+name)
            good.append(images)
    if not good:raise FileNotFoundError('Attach the original joydippaul MSLD v2.0 dataset with all 755 Original Images.')
    return good[0]

def find_weights(search_root,records,out):
    weights={};rows=[];expected={};index={}
    # Traverse the large image input tree once, rather than once per weight.
    for p in Path(search_root).rglob('*.pt'):index.setdefault(p.name,[]).append(p)
    for (model,fold),rec in records.items():
        for info in rec['meta']['weights']:
            key=(model,fold,info['rank']);expected[key]=info
            candidates=sorted(index.get(info['file'],[]))
            valid=[p for p in candidates if p.is_file() and p.stat().st_size==info['bytes'] and sha(p)==info['sha256']]
            if valid:weights[key]=valid[0]
    # Kaggle sometimes leaves an uploaded ZIP intact. Stream only the selected,
    # hash-matching members into temporary storage; never extract arbitrary paths
    # or add these 7.3 GB of input weights to the saved analysis output.
    if len(weights)<len(expected):
        for archive in sorted(Path(search_root).rglob('PB_Ensemble_Selected_Weights.zip')):
            cache=Path(tempfile.gettempdir())/'pb_final_selected_weights'/hashlib.sha256(str(archive.resolve()).encode()).hexdigest()[:20]
            cache.mkdir(parents=True,exist_ok=True)
            with ZipFile(archive) as z:
                if sum(i.file_size for i in z.infolist())>10*2**30:raise ValueError('Expected the selected-weight upload ZIP, not full optimizer backups.')
                entries={}
                for entry in z.infolist():entries.setdefault(Path(entry.filename.replace('\\','/')).name,[]).append(entry)
                for key,info in expected.items():
                    if key in weights:continue
                    for entry in entries.get(info['file'],[]):
                        if entry.file_size!=info['bytes']:continue
                        dest=cache/info['file'];temporary=dest.with_suffix('.tmp');digest=hashlib.sha256()
                        with z.open(entry) as source,temporary.open('wb') as output:
                            for block in iter(lambda:source.read(2**20),b''):digest.update(block);output.write(block)
                        if digest.hexdigest()==info['sha256']:
                            temporary.replace(dest);weights[key]=dest;break
    missing=[info['file'] for key,info in expected.items() if key not in weights]
    if missing:raise FileNotFoundError('Missing matching selected weights: '+', '.join(missing)+'. Attach all 40 selected weights, not the small results ZIPs.')
    for (model,fold,rank),info in expected.items():rows.append({'model':model,'fold':fold,**info,'verified':True})
    pd.DataFrame(rows).to_csv(out/'selected_weight_verification.csv',index=False)
    return weights

def inference_cache(manifest,records,splits,weights,images,out,device,batch=8):
    """All 40 selected snapshots on official test images only; cache each branch."""
    cache=out/'verified_inference_cache';cache.mkdir(exist_ok=True)
    values={};verification=[]
    for model in MODELS:
        for fold in range(1,6):
            frame=splits[fold].loc[splits[fold].role.eq('test')].reset_index(drop=True)
            ranks=[]
            for rank in [1,2]:
                wp=weights[(model,fold,rank)];dest=cache/f'{model}_f{fold}_r{rank}.npz';mark=dest.with_suffix('.json')
                # Kaggle inputs are read-only. These forty file hashes were
                # already verified in find_weights; reuse the verified digest.
                weight_sha=next(w['sha256'] for w in records[(model,fold)]['meta']['weights'] if w['rank']==rank)
                identity={'weight_sha256':weight_sha,'filenames':frame.filename.tolist(),
                          'source':FROZEN_SHA,'timm':timm.__version__,'torch':str(torch.__version__),
                          'torchvision':__import__('torchvision').__version__,
                          'gpu':torch.cuda.get_device_name(0) if device.type=='cuda' else 'CPU',
                          'Pillow':__import__('PIL').__version__,'batch':batch,'script_sha256':sha(__file__)}
                identity_sha=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
                valid=dest.exists() and mark.exists() and json.loads(mark.read_text()).get('identity')==identity_sha and json.loads(mark.read_text()).get('file_sha256')==sha(dest)
                if valid:
                    with np.load(dest,allow_pickle=False) as arr:p=arr['mean_view_probs']
                else:
                    net=load_network(manifest,model,wp,device);chunks=[]
                    with torch.no_grad():
                        for start in range(0,len(frame),batch):
                            rows=frame.iloc[start:start+batch]
                            x=torch.stack([TRANSFORM(Image.open(images[n.upper()]).convert('RGB')) for n in rows.filename]).to(device)
                            # Compute four view tensors separately, as in original tta_predict.
                            probs=torch.stack([F.softmax(net(v),dim=1) for v in [x,torch.flip(x,[3]),torch.rot90(x,1,[2,3]),torch.rot90(x,3,[2,3])]],0)
                            chunks.append(probs.mean(0).cpu().numpy())
                    p=np.concatenate(chunks);del net;python_gc.collect()
                    if device.type=='cuda':torch.cuda.empty_cache()
                    temp=dest.with_suffix('.tmp')
                    with temp.open('wb') as stream:np.savez_compressed(stream,mean_view_probs=p)
                    temp.replace(dest);write_json(mark,{'identity':identity_sha,'configuration':identity,'file_sha256':sha(dest)})
                if p.shape!=(len(frame),6) or not np.isfinite(p).all():raise ValueError('Invalid inference cache.')
                groups=np.sort(frame.patient_cluster.unique())
                group_probs=np.stack([geometric_numpy(p[frame.patient_cluster.eq(g).to_numpy()]) for g in groups])
                values[(model,fold,rank)]={'mean_view_probs':p,'frame':frame,'groups':groups,'group_probs':group_probs}
                ranks.append(group_probs)
                print(f'Inference checked {model} fold{fold} rank{rank}',flush=True)
            reference=__import__('merge_corrected').prediction(records[(model,fold)],model,fold,'test')
            recreated=np.mean(ranks,axis=0);diff=float(np.abs(recreated-reference['probs']).max())
            same_ids=np.array_equal(groups,reference['pids']);same_pred=np.array_equal(recreated.argmax(1),reference['probs'].argmax(1))
            row={'model':model,'fold':fold,'test_groups':len(groups),'max_absolute_difference':diff,
                 'same_group_order':same_ids,'same_predictions':same_pred,'passed':same_ids and same_pred and diff<=5e-4}
            verification.append(row);pd.DataFrame(verification).to_csv(out/'test_prediction_reproduction.csv',index=False)
            if not row['passed']:raise ValueError(f'Weight reproduction failed for {model} fold{fold}. No heatmap should be trusted; inspect the CSV.')
    return values

def select_cases(ids,y,prob,max_groups=18):
    """Predefined display rule; deliberately error-enriched, never random cherry-picking."""
    frame=pd.DataFrame({'known_group':ids,'label':y,'prediction':prob.argmax(1),'confidence':prob.max(1)})
    selected={}
    for c in range(6):
        correct=frame[(frame.label==c)&(frame.prediction==c)].sort_values(['confidence','known_group'])
        if len(correct):
            selected[correct.iloc[0].known_group]='Lowest-confidence correct example in reference class'
            middle=correct.iloc[len(correct)//2];selected.setdefault(middle.known_group,'Median-confidence correct example in reference class')
        errors=frame[(frame.label==c)&(frame.prediction!=c)].sort_values(['confidence','known_group'],ascending=[False,True])
        if len(errors):selected[errors.iloc[0].known_group]='Highest-confidence error in reference class'
    # Preserve all mpox/chickenpox confusions in the gallery even if this exceeds the nominal group count.
    confusions=frame[((frame.label==2)&(frame.prediction==4))|((frame.label==4)&(frame.prediction==2))]
    ordered=list(selected)[:max_groups]
    for g in confusions.known_group:
        if g not in ordered:ordered.append(g)
        selected[g]='Mpox/chickenpox confusion'
    output=frame[frame.known_group.isin(ordered)].copy()
    output['selection_reason']=output.known_group.map(selected)
    return output.sort_values(['label','known_group']).reset_index(drop=True)

def compute_maps(manifest,records,splits,weights,images,cache,pooled,temps,mixture,ensemble,cases,out,device):
    ref=pooled[MODELS[0]]['test'];id_index={g:i for i,g in enumerate(ref['pids'])}
    targets=[]
    for row in cases.itertuples():
        filenames=sorted(set().union(*(set(a.loc[a.role.eq('test')&a.patient_cluster.eq(row.known_group),'filename']) for a in splits.values())))
        for filename in filenames:
            for target in sorted({int(row.label),int(row.prediction)}):
                targets.append({'known_group':row.known_group,'filename':filename,'target':target,
                                'reference':int(row.label),'prediction':int(row.prediction),
                                'confidence':float(row.confidence),'selection_reason':row.selection_reason})
    pd.DataFrame(targets).to_csv(out/'explanation_selection.csv',index=False)
    weight_digests={str((model,fold,w['rank'])):w['sha256'] for (model,fold),rec in records.items() for w in rec['meta']['weights']}
    signature={'weights':weight_digests,'temperatures':temps,'mixture':mixture,
               'targets':targets,'script_sha256':sha(__file__),'torch':str(torch.__version__),'timm':timm.__version__}
    signature_sha=hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest()
    component_root=out/'component_maps';component_root.mkdir(exist_ok=True)
    component_rows=[]
    # One architecture/snapshot in GPU memory at a time. Every completed component is restartable.
    for model in MODELS:
        for fold in range(1,6):
            for rank in [1,2]:
                relevant=[(i,t) for i,t in enumerate(targets) if t['filename'] in set(splits[fold].loc[splits[fold].role.eq('test'),'filename'])]
                if not relevant:continue
                folder=component_root/f'{model}_f{fold}_r{rank}';folder.mkdir(exist_ok=True)
                net=None
                for i,t in relevant:
                    dest=folder/f'case_{i:04d}.npy';mark=dest.with_suffix('.json')
                    if dest.exists() and mark.exists():
                        meta=json.loads(mark.read_text())
                        if meta.get('signature')==signature_sha and meta.get('sha256')==sha(dest):
                            component_rows.append(meta['record']);continue
                    if net is None:net=load_network(manifest,model,weights[(model,fold,rank)],device)
                    layer,layer_name=target_layer(net,model);capture={}
                    def hook(module,args,output):capture['activation']=output
                    handle=layer.register_forward_hook(hook)
                    try:
                        net.zero_grad(set_to_none=True)
                        x=TRANSFORM(Image.open(images[t['filename'].upper()]).convert('RGB'))[None].to(device)
                        # Four views share one batch; gradients already include the 1/4 view factor.
                        view_probs=F.softmax(net(views(x)),dim=1);query_mean=view_probs.mean(0)
                        data=cache[(model,fold,rank)];frame=data['frame'];g=t['known_group'];groupmask=frame.patient_cluster.eq(g).to_numpy()
                        query_index=int(np.flatnonzero(frame.filename.eq(t['filename']))[0]);othermask=groupmask.copy();othermask[query_index]=False
                        other_logs=np.log(data['mean_view_probs'][othermask]+1e-8).sum(0)
                        group_index=int(np.flatnonzero(data['groups']==g)[0]);cached_group=data['group_probs'][group_index]
                        gi=id_index[g];raw=pooled[model]['test']['probs'][gi]
                        nfold=sum(g in set(a.loc[a.role.eq('test'),'patient_cluster']) for a in splits.values())
                        other_score=float(ensemble[gi,t['target']]-mixture[model]*scale_numpy(raw,temps[model])[t['target']])
                        to_tensor=lambda v:torch.as_tensor(v,device=device,dtype=query_mean.dtype)
                        score=branch_score(query_mean,to_tensor(other_logs),int(groupmask.sum()),to_tensor(cached_group),
                            to_tensor(raw),nfold,temps[model],mixture[model],other_score,t['target'])
                        baseline=float(ensemble[gi,t['target']]);score_diff=abs(float(score.detach().cpu())-baseline)
                        if score_diff>1e-3:raise ValueError('Explanation score does not reproduce the frozen ensemble score.')
                        activation=capture['activation'];gradient=torch.autograd.grad(score,activation,retain_graph=False)[0]
                        a=spatial(activation,model);grad=spatial(gradient,model)
                        if not torch.isfinite(grad).all():raise ValueError('Nonfinite Grad-CAM gradients.')
                        signed=(grad.mean((-2,-1),keepdim=True)*a).sum(1)
                        signed=F.interpolate(signed[:,None],size=(384,384),mode='bilinear',align_corners=False)[:,0]
                        # Restore view coordinates before summing; no per-branch normalization.
                        raw_map=sum(unview(signed[v],v) for v in range(4)).detach().cpu().numpy().astype(np.float32)
                        if not np.isfinite(raw_map).all():raise ValueError('Nonfinite Grad-CAM map.')
                        temp=dest.with_suffix('.tmp')
                        with temp.open('wb') as stream:np.save(stream,raw_map,allow_pickle=False)
                        temp.replace(dest)
                        record={'case':i,'model':model,'fold':fold,'rank':rank,'layer':layer_name,
                                'ensemble_target_score':baseline,'partial_graph_score':float(score.detach().cpu()),
                                'score_abs_diff':score_diff,'gradient_l1':float(grad.detach().abs().sum().cpu()),
                                'signed_map_l1':float(np.abs(raw_map).sum()),'file':dest.relative_to(out).as_posix()}
                        write_json(mark,{'signature':signature_sha,'sha256':sha(dest),'record':record});component_rows.append(record)
                    finally:handle.remove()
                if net is not None:del net
                python_gc.collect()
                if device.type=='cuda':torch.cuda.empty_cache()
                print(f'Explanation components saved: {model} fold{fold} rank{rank}',flush=True)
    pd.DataFrame(component_rows).to_csv(out/'component_map_provenance.csv',index=False)
    # Aggregate all ensemble-score components before rectification and normalization.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Serif','font.size':10,'pdf.fonttype':42,'svg.fonttype':'none'})
    gallery=out/'gallery';gallery.mkdir(exist_ok=True);rawfolder=out/'ensemble_maps';rawfolder.mkdir(exist_ok=True)
    summary=[]
    for i,t in enumerate(targets):
        components=[r for r in component_rows if r['case']==i]
        aggregate=sum(np.load(out/r['file'],allow_pickle=False) for r in components)
        positive=np.maximum(aggregate,0);maximum=float(positive.max());normal=positive/maximum if maximum>0 else positive
        np.save(rawfolder/f'case_{i:04d}_signed.npy',aggregate,allow_pickle=False)
        np.save(rawfolder/f'case_{i:04d}_normalized.npy',normal.astype(np.float32),allow_pickle=False)
        original=Image.open(images[t['filename'].upper()]).convert('RGB').resize((384,384))
        fig,axes=plt.subplots(1,2,figsize=(6.8,3.6))
        axes[0].imshow(original);axes[0].set_title('Input image')
        axes[1].imshow(original);axes[1].imshow(normal,cmap='magma',alpha=.4,vmin=0,vmax=1);axes[1].set_title(f'Ensemble target: {LABELS[t["target"]]}')
        for ax in axes:ax.axis('off')
        fig.suptitle(f'Reference: {LABELS[t["reference"]]} | Prediction: {LABELS[t["prediction"]]} ({t["confidence"]:.3f})',fontsize=10)
        fig.tight_layout()
        for ext in ['png','pdf']:fig.savefig(gallery/f'case_{i:04d}.{ext}',dpi=600,bbox_inches='tight',facecolor='white')
        plt.close(fig)
        summary.append({**t,'case':i,'components':len(components),'positive_map_max':maximum,
                        'nonzero_map':maximum>0,'raw_l1':float(np.abs(aggregate).sum())})
    pd.DataFrame(summary).to_csv(out/'gallery_index.csv',index=False)
    # A multipage gallery is useful for review; each caption has a stable case identifier.
    from matplotlib.backends.backend_pdf import PdfPages
    with PdfPages(out/'Ensemble_GradCAM_Gallery.pdf') as pdf:
        for start in range(0,len(targets),6):
            fig,axes=plt.subplots(3,2,figsize=(8.3,11.7))
            for ax in axes.ravel():ax.axis('off')
            for ax,i in zip(axes.ravel(),range(start,min(start+6,len(targets)))):
                ax.imshow(plt.imread(gallery/f'case_{i:04d}.png'));ax.set_title(f'Case {i:04d}',fontsize=8)
            fig.tight_layout();pdf.savefig(fig);plt.close(fig)
    return targets,summary

def bundle(out):
    target=out.parent/'PB_Ensemble_GradCAM_Outputs.zip'
    with ZipFile(target,'w',ZIP_DEFLATED) as z:
        for p in sorted(out.rglob('*')):
            if p.is_file():z.write(p,'PB_Ensemble_GradCAM/'+p.relative_to(out).as_posix())
    print('Download:',target,flush=True)

def restore_analysis_cache(search_root,out):
    """Continue verified analysis components from an attached saved output."""
    copied=0
    for source in sorted(Path(search_root).rglob('PB_Ensemble_GradCAM')):
        if not source.is_dir() or source.resolve()==out.resolve():continue
        for folder in ['verified_inference_cache','component_maps']:
            for p in (source/folder).rglob('*'):
                if not p.is_file() or p.suffix not in ['.npy','.npz','.json']:continue
                dest=out/p.relative_to(source)
                if not dest.exists():dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest);copied+=1
    if copied:print(f'Restored {copied} analysis cache files; every reused component is checked again.',flush=True)

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-root',type=Path,default=Path('/kaggle/input'))
    p.add_argument('--work',type=Path,default=Path('/kaggle/working'))
    p.add_argument('--expected-final-sha',required=True)
    p.add_argument('--batch',type=int,default=8)
    p.add_argument('--max-groups',type=int,default=18)
    p.add_argument('--verify-only',action='store_true')
    args=p.parse_args(argv);out=args.work/'PB_Ensemble_GradCAM';out.mkdir(parents=True,exist_ok=True)
    os.environ['MPLCONFIGDIR']=str(args.work/'.plot_cache')
    restore_analysis_cache(args.input_root,out)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True,warn_only=True)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type!='cuda':raise RuntimeError('Select a GPU for this notebook. Use the separate CPU publication notebook for tables and figures.')
    if timm.__version__!='1.0.29':raise ValueError('Use the original timm 1.0.29 architecture definitions.')
    inputs=prepare_inputs(args.input_root,args.work,args.expected_final_sha)
    manifest,records,pooled,splits,keep=read_results(inputs)
    clinical=pd.read_csv(inputs/'frozen/clinical_images.csv')
    images=find_originals(args.input_root,clinical)
    weights=find_weights(args.input_root,records,out)
    temps,mixture,ensemble=fit_ensemble(pooled,keep)
    write_json(out/'ensemble_calibration.json',{'temperatures':temps,'weights':mixture,'calibration_groups':138,'test_groups':215})
    cache=inference_cache(manifest,records,splits,weights,images,out,device,args.batch)
    recreated={}
    ids=pooled[MODELS[0]]['test']['pids'];y=pooled[MODELS[0]]['test']['labels']
    for model in MODELS:
        probs=[]
        for group in ids:
            entries=[]
            for f in range(1,6):
                data=cache[(model,f,1)]
                if group in data['groups']:
                    j=int(np.flatnonzero(data['groups']==group)[0]);entries.append((cache[(model,f,1)]['group_probs'][j]+cache[(model,f,2)]['group_probs'][j])/2)
            probs.append(np.mean(entries,axis=0))
        recreated[model]=np.stack(probs)
    recreated_ensemble=sum(mixture[m]*scale_numpy(recreated[m],temps[m]) for m in MODELS)
    diff=float(np.abs(ensemble-recreated_ensemble).max());same=np.array_equal(ensemble.argmax(1),recreated_ensemble.argmax(1))
    write_json(out/'ensemble_prediction_reproduction.json',{'max_absolute_difference':diff,'same_all_test_predictions':same,'passed':same and diff<=1e-3})
    if not same or diff>1e-3:raise ValueError('Full ensemble prediction reproduction failed.')
    print('ALL 40 WEIGHTS AND THE FULL ENSEMBLE VERIFIED',flush=True)
    cases=select_cases(ids,y,ensemble,args.max_groups);cases.to_csv(out/'selected_groups.csv',index=False)
    if not args.verify_only:
        targets,summary=compute_maps(manifest,records,splits,weights,images,cache,pooled,temps,mixture,ensemble,cases,out,device)
        write_json(out/'explanation_status.json',{'status':'complete','selected_groups':len(cases),'image_target_maps':len(targets),
            'nonzero_maps':sum(v['nonzero_map'] for v in summary),'all_four_models':True,
            'gradient_target':'Full calibrated pooled known-group ensemble class probability',
            'aggregation':'Unnormalised signed model-specific Grad-CAM contributions summed after inverse TTA transforms; rectify and normalize once.',
            'scope':'Image contribution with all other group context fixed. Coarse cross-layer qualitative aggregate, not an exact or complete pixel decomposition.',
            'selection':'Deterministic confidence-stratified correct examples and errors, plus every mpox/chickenpox confusion; error-enriched.',
            'no_lesion_masks_or_localisation_validation':True,'clinical_identity_independently_verified':False})
        (out/'METHOD_AND_LIMITATIONS.md').write_text('''# Ensemble score Grad CAM

The gradient target is the class probability of the full calibrated ensemble, using all four backbones, two retained checkpoints, the original four view transforms, geometric image aggregation, and repeated-fold averaging. Other images in the known group are held fixed when explaining one image. Only snapshots for folds in which that image was an official test image are used for its map.

Model-specific layers are EfficientNetV2-S conv_head, the last ConvNeXt stage, the last Swin block norm1, and the last BEiT block norm1 with the class token removed from the spatial map. Swin and BEiT maps adapt Grad-CAM to spatial token features. The signed component maps are restored to input coordinates and summed before rectification and display normalization. Independently normalized backbone heatmaps are not treated as an ensemble explanation.

The aggregate remains a heuristic combination of coarse features at different layers. It is not an additive pixel decomposition, lesion segmentation or proof of clinically appropriate reasoning. Individual color scales cannot be compared as effect sizes. Zero maps are retained and flagged. The gallery deliberately includes errors and is not a prevalence estimate. No lesion-mask localization statistic, randomization-test success or causal claim is inferred from appearance alone.

Selection and per-component gradients, scores, identities and arrays are exported for inspection. Grad-CAM is based on Selvaraju et al., ICCV 2017, https://openaccess.thecvf.com/content_iccv_2017/html/Selvaraju_Grad-CAM_Visual_Explanations_ICCV_2017_paper.html . Saliency validation limitations are discussed by Adebayo et al., NeurIPS 2018, https://papers.nips.cc/paper_files/paper/2018/hash/294a8ed24b1ad22ec2e7efea049b8737-Abstract.html . No comprehensive saliency validation is claimed by this qualitative run.
''',encoding='utf-8')
    bundle(out)

if __name__=='__main__':main()
