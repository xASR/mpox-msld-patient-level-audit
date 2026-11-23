"""Recover matched-batch randomized-model maps without altering frozen modules.

The original prediction implementation/signatures remain unchanged. Only the
fully randomized map path changes: identical minibatches and separate TTA views,
with a two-pass VJP and a leaf at the target layer to bound GPU memory.
"""
from pathlib import Path,PurePosixPath
from zipfile import ZipFile
import argparse,json,shutil,stat,time,gc
import numpy as np
import torch
import torch.nn.functional as F
import timm
import PIL
from supplemental_common import sha,digest,write_json,bundle
import supplemental_xai as original
import ensemble_gradcam as cam

METHOD='random_all_matched_minibatch_v1'
ORIGINAL_SHA='588bdc3cb4b6784da6c62e5113cd6021e09e679b6f3fbd972a8d6b73d86a2e26'
LEGACY_SIGNED_MAPS=original.signed_maps

def exact_context(cache,pooled,temps,mixture,ensemble,ids,model,fold,rank,case,q):
    data=cache[(model,fold,rank)];frame=data['frame']
    qi=int(np.flatnonzero(frame.filename.eq(case['filename']))[0]);mask=frame.patient_cluster.eq(case['group']).to_numpy();other=mask.copy();other[qi]=False
    gi=int(np.flatnonzero(ids==case['group'])[0]);ri=int(np.flatnonzero(data['groups']==case['group'])[0])
    nf=sum(case['group'] in set(cache[(model,f,1)]['groups']) for f in range(1,6))
    fixed=float(ensemble[gi,case['target']]-mixture[model]*cam.scale_numpy(pooled[model][gi],temps[model])[case['target']])
    t=lambda x:torch.as_tensor(x,device=q.device,dtype=q.dtype)
    return cam.branch_score(q,t(np.log(data['mean_view_probs'][other]+1e-8).sum(0)),int(mask.sum()),
        t(data['group_probs'][ri]),t(pooled[model][gi]),nf,temps[model],mixture[model],fixed,case['target']),float(ensemble[gi,case['target']])

def matched_signed_maps(manifest,splits,weights,images,cache,pooled,temps,mixture,ensemble,ids,cases,out,device,stage,budget_end):
    if stage!='random_all':
        return LEGACY_SIGNED_MAPS(manifest,splits,weights,images,cache,pooled,temps,mixture,ensemble,ids,cases,out,device,stage,budget_end)
    out=Path(out).parent/METHOD;out.mkdir(parents=True,exist_ok=True)
    signature=digest({'method':METHOD,'implementation':sha(__file__),'parent':sha(original.__file__),
        'stage':stage,'cases':cases,'temps':temps,'mixture':mixture,'models':manifest['models'],
        'torch':str(torch.__version__),'timm':timm.__version__,'batch':8,
        'cache':{f'{m}_{f}_{r}':digest({'filenames':v['frame'].filename.tolist(),'probs':v['mean_view_probs'].tolist()}) for (m,f,r),v in cache.items()},
        'pooled':{m:digest(p.tolist()) for m,p in pooled.items()}})
    allmaps=[np.zeros((384,384),np.float64) for _ in cases]
    for model in original.MODELS:
        for fold in range(1,6):
            for rank in (1,2):
                data=cache[(model,fold,rank)];frame=data['frame'];net=None;handle=None;capture={}
                try:
                    for k,case in enumerate(cases):
                        if case['filename'] not in set(frame.filename):continue
                        p=out/f'{model}_f{fold}_r{rank}_case{k}.npy';mark=p.with_suffix('.json')
                        if p.exists() and mark.exists() and json.loads(mark.read_text()).get('signature')==signature and json.loads(mark.read_text()).get('sha256')==sha(p):
                            raw=np.load(p,allow_pickle=False);allmaps[k]+=raw;continue
                        if time.monotonic()>budget_end:raise TimeoutError('Matched-batch maps paused. Keep full output or its results ZIP and attach it to a new recovery notebook.')
                        if net is None:
                            net=original.net_for(manifest,model,weights[(model,fold,rank)],device,stage,fold,rank)
                            for param in net.parameters():param.requires_grad_(False)
                            layer,_=cam.target_layer(net,model)
                            def hook(module,args,value):
                                leaf=value.detach().requires_grad_(True);capture['a']=leaf;return leaf
                            handle=layer.register_forward_hook(hook)
                        qi=int(np.flatnonzero(frame.filename.eq(case['filename']))[0]);start=(qi//8)*8;chunk=frame.iloc[start:start+8];within=qi-start
                        x=torch.stack([cam.TRANSFORM(__import__('PIL.Image',fromlist=['Image']).open(images[n.upper()]).convert('RGB')) for n in chunk.filename]).to(device)
                        view=lambda i:(x if i==0 else x.flip([3]) if i==1 else x.rot90(1,[2,3]) if i==2 else x.rot90(3,[2,3]))
                        # Pass1 has the same minibatch/view layout as predict(). The
                        # target-layer leaf is active, so the repeated gradient pass
                        # follows the same forward path, not a no-grad fast path.
                        first=[]
                        for v in range(4):
                            with torch.enable_grad():qv=net(view(v)).softmax(1)[within]
                            first.append(qv.detach().cpu());del qv;capture.clear()
                        actual=torch.stack(first).mean(0).numpy();cached=data['mean_view_probs'][qi]
                        probability_error=float(np.abs(actual-cached).max())
                        proxy=torch.as_tensor(actual,dtype=torch.float64).requires_grad_(True)
                        s,expected=exact_context(cache,pooled,temps,mixture,ensemble,ids,model,fold,rank,case,proxy)
                        score_error=abs(float(s.detach())-expected)
                        diagnostic={'method':METHOD,'model':model,'fold':fold,'rank':rank,'case':case,
                            'batch_filenames':chunk.filename.tolist(),'query_index_in_batch':within,
                            'query_probability_error':probability_error,'partial_score_error':score_error,
                            'cached_query_probabilities':cached.tolist(),'matched_batch_query_probabilities':actual.tolist()}
                        if probability_error>5e-6 or score_error>1e-3:
                            write_json(out/'FAILURE_DIAGNOSTICS.json',diagnostic)
                            raise ValueError('Matched minibatch still differs from saved predictions. See FAILURE_DIAGNOSTICS.json; no mismatched map is accepted.')
                        dq=torch.autograd.grad(s,proxy)[0].detach().to(device=device,dtype=x.dtype)/4
                        raw=np.zeros((384,384),np.float64);repeat_error=0.
                        for v in range(4):
                            with torch.enable_grad():
                                qv=net(view(v)).softmax(1)[within];a=capture['a']
                                repeat_error=max(repeat_error,float(np.abs(qv.detach().cpu().numpy()-first[v].numpy()).max()))
                                grad=torch.autograd.grad(qv,a,grad_outputs=dq)[0]
                                aa=cam.spatial(a[within:within+1],model);gg=cam.spatial(grad[within:within+1],model)
                                signed=(gg.mean((-2,-1),keepdim=True)*aa).sum(1)
                                signed=F.interpolate(signed[:,None],size=(384,384),mode='bilinear',align_corners=False)[0,0]
                                raw+=cam.unview(signed,v).detach().cpu().numpy()
                            del qv,a,grad,aa,gg,signed;capture.clear()
                        diagnostic['repeated_view_probability_error']=repeat_error
                        if repeat_error>1e-7 or not np.isfinite(raw).all():
                            write_json(out/'FAILURE_DIAGNOSTICS.json',diagnostic)
                            raise ValueError('Gradient-pass reproduction/nonfinite-map check failed. No invalid map is accepted.')
                        np.save(p,raw,allow_pickle=False);write_json(mark,{'signature':signature,'sha256':sha(p),**diagnostic})
                        allmaps[k]+=raw
                        print('Recovered matched-batch map:',model,fold,rank,'case',k,'score error',score_error,flush=True)
                        del x,s,proxy,dq
                finally:
                    if handle is not None:handle.remove()
                    if net is not None:del net
                    capture.clear();gc.collect();torch.cuda.empty_cache()
    result=np.stack(allmaps);np.save(out/'summed_signed_maps.npy',result,allow_pickle=False)
    return result

def import_cache(search,out):
    search=Path(search);out=Path(out).resolve()
    # A full prior notebook output may also contain its old raw-ZIP import
    # staging directory. Import the cumulative SUPP_XAI, not that older duplicate.
    sources=[p for p in search.rglob('SUPP_XAI') if 'xai_recovery_import' not in p.parts]
    for archive in search.rglob('SUPP_XAI*RESULTS_ONLY.zip'):
        dest=out.parent/'xai_recovery_import'/sha(archive)
        with ZipFile(archive) as z:
            if z.testzip() is not None or sum(i.file_size for i in z.infolist())>500*2**20:raise ValueError('Invalid/oversized XAI results ZIP.')
            for i in z.infolist():
                rel=PurePosixPath(i.filename.replace('\\','/'));p=dest.joinpath(*rel.parts).resolve()
                if rel.is_absolute() or '..' in rel.parts or any(':' in t for t in rel.parts) or stat.S_ISLNK(i.external_attr>>16) or not p.is_relative_to(dest.resolve()):raise ValueError('Unsafe XAI archive.')
                if i.is_dir():p.mkdir(parents=True,exist_ok=True)
                else:
                    p.parent.mkdir(parents=True,exist_ok=True)
                    with z.open(i) as a,p.open('wb') as b:shutil.copyfileobj(a,b)
        sources.extend(dest.rglob('SUPP_XAI'))
    found=0
    for source in sources:
        if source.resolve()==out or not(source/'cache').is_dir():continue
        for part in ('cache','maps'):
            for p in (source/part).rglob('*'):
                if not p.is_file() or p.suffix not in ('.npz','.npy','.json'):continue
                target=out/p.relative_to(source);target.parent.mkdir(parents=True,exist_ok=True)
                if target.exists() and sha(target)!=sha(p):raise ValueError('Conflicting recovery caches. Attach only the intended final prior output.')
                if not target.exists():shutil.copyfile(p,target)
        found+=1
    if not found:raise FileNotFoundError('Attach failed SUPP_XAI_RESULTS_ONLY.zip as private msld-xai-recovery-cache, or attach the last FULL recovery output. This notebook refuses a silent fresh start.')
    verified=0;environment={'torch':str(torch.__version__),'timm':timm.__version__,'Pillow':PIL.__version__,'gpu':torch.cuda.get_device_name(0)}
    for folder in ('trained_original','border_mean','control1_mean','control2_mean','border_blur','control1_blur','control2_blur','random_head','random_all'):
        for model in original.MODELS:
            for fold in range(1,6):
                for rank in (1,2):
                    p=out/f'cache/{folder}/{model}_f{fold}_r{rank}.npz';mark=p.with_suffix('.json')
                    if not p.is_file() or not mark.is_file():raise ValueError('Expected all360 saved prediction branches. Missing '+str(p))
                    m=json.loads(mark.read_text());ident=m['identity']
                    if m['signature']!=digest(ident) or m['sha256']!=sha(p) or ident['script']!=ORIGINAL_SHA or ident['batch']!=8:raise ValueError('Changed prediction cache.')
                    if any(ident[k]!=v for k,v in environment.items()):raise ValueError('Use the same T4/PyTorch/timm/Pillow environment as the failed run. Cache reuse would otherwise be unsafe.')
                    verified+=1
    for p in (out/'maps').rglob('*.json'):
        if p.name=='FAILURE_DIAGNOSTICS.json':continue
        m=json.loads(p.read_text())
        if sha(p.with_suffix('.npy'))!=m['sha256']:raise ValueError('Changed cached map: '+str(p))
    return verified

def main(args):
    if sha(original.__file__)!=ORIGINAL_SHA:raise ValueError('Frozen supplemental_xai.py changed. Use original supplemental input package.')
    out=Path(args.work)/'SUPP_XAI';out.mkdir(parents=True,exist_ok=True)
    verified=import_cache(args.input_root,out)
    write_json(out/'recovery_method.json',{'method':METHOD,'implementation_sha256':sha(__file__),
        'original_script_sha256':ORIGINAL_SHA,'prediction_cache_branches_verified':verified,
        'trained_and_random_head_maps':'original implementation and compatible cache signatures retained',
        'random_all_maps':'recomputed with matching eight-image minibatches and separate four-view forwards; target-layer VJP bounds memory',
        'guards':'cached query probability error <=5e-6, full partial-score error <=1e-3, repeated view probability error <=1e-7, finite maps',
        'training_and_main_results_changed':False})
    print('RECOVERY CACHE PASS: 360/360 prediction branches verified. No training is performed.',flush=True)
    original.signed_maps=matched_signed_maps
    status='running'
    try:
        original.main(args)
        status='preflight_passed' if args.verify_only else json.loads((out/f'{args.mode}_status.json').read_text())['status']
        if not args.verify_only:
            marker=out/f'{args.mode}_status.json';value=json.loads(marker.read_text())
            value['implementation_recovery']={'method':METHOD,'implementation_sha256':sha(__file__),'prediction_cache_branches_verified':verified,
                'description':'Fully randomized maps use matching image minibatches/separate TTA views and target-layer VJP. Original score-consistency guards retained; no retraining or calibration change.'}
            write_json(marker,value)
    except Exception as error:
        status='failed';write_json(out/f'{args.mode}_status.json',{'status':'failed','mode':args.mode,'error':str(error),'recovery_method':METHOD});raise
    finally:
        write_json(out/'recovery_status.json',{'status':status,'method':METHOD})
        if not args.verify_only:bundle(out)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--inputs',required=True);p.add_argument('--expected-sha',required=True)
    p.add_argument('--input-root',default='/kaggle/input');p.add_argument('--work',default='/kaggle/working');p.add_argument('--mode',default='all',choices=['all'])
    p.add_argument('--annotations');p.add_argument('--session-hours',type=float,default=10.5);p.add_argument('--verify-only',action='store_true');main(p.parse_args())
