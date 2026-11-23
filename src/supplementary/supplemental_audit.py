"""CPU content-duplicate triage, annotation templates, and ensemble comparisons."""
from pathlib import Path
import argparse
import hashlib
import json
import itertools
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from scipy.fft import dctn
from scipy.optimize import minimize
from scipy.stats import binomtest
from supplemental_common import (MODELS, LABELS, sha, verify_inputs, write_json, bundle, export_tables)
from protocol_a_core import find_images, read_group, inspect_job
from study_statistics import fit_calibration, temperature
from final_analysis import numpy_metrics


def phash(image):
    a=np.asarray(image.convert('L').resize((32,32)),dtype=np.float64)
    z=dctn(a,type=2,norm='ortho')[:8,:8].ravel()[1:]
    bits=z>np.median(z)
    return np.uint64(sum(int(b)<<i for i,b in enumerate(bits)))


def crop_variants(image):
    w,h=image.size
    for fraction in (1., .85, .7):
        dx=int((1-fraction)*w/2);dy=int((1-fraction)*h/2)
        crop=image.crop((dx,dy,w-dx,h-dy))
        for flip in (False,True):
            v=ImageOps.mirror(crop) if flip else crop
            for rotation in (0,90,180,270): yield f'center{fraction}_flip{flip}_rot{rotation}',v.rotate(rotation,expand=True)


def local_match(left,right):
    """A review aid, never an automatic patient-identity decision."""
    import cv2
    def features(path):
        a=np.asarray(Image.open(path).convert('L'));scale=min(1,800/max(a.shape))
        if scale<1:a=cv2.resize(a,None,fx=scale,fy=scale)
        return cv2.SIFT_create(nfeatures=1200).detectAndCompute(a,None)
    a,da=features(left);b,db=features(right)
    if da is None or db is None or len(da)<4 or len(db)<4:return 0,0,0.
    pairs=cv2.BFMatcher().knnMatch(da,db,k=2)
    good=[x for pair in pairs if len(pair)==2 for x,y in [pair] if x.distance<.7*y.distance]
    if len(good)<4:return len(good),0,0.
    x=np.float32([a[p.queryIdx].pt for p in good]);y=np.float32([b[p.trainIdx].pt for p in good])
    _,mask=cv2.findHomography(x,y,cv2.RANSAC,5.)
    n=int(mask.sum()) if mask is not None else 0
    return len(good),n,n/len(good)


def duplicate_audit(inputs,input_root,out):
    clinical=pd.read_csv(Path(inputs)/'a_design/seed_42/clinical_images.csv')
    images=find_images(input_root,clinical,True);names=clinical.filename.tolist();variants=[];owners=[];tags=[];original=[];rgb=[]
    for i,name in enumerate(names):
        im=Image.open(images[name]).convert('RGB'); original.append(phash(im))
        rgb.append(hashlib.sha256(str(im.size).encode()+im.tobytes()).hexdigest())
        for tag,v in crop_variants(im):variants.append(phash(v));owners.append(i);tags.append(tag)
    variants=np.asarray(variants,dtype=np.uint64);owners=np.asarray(owners)
    pairs={}
    # Candidate search includes same-group pairs and every cross-label pair.
    for i,q in enumerate(original):
        counts=np.bitwise_count(np.bitwise_xor(variants,q))
        distances=np.full(len(names),64,dtype=int);np.minimum.at(distances,owners,counts)
        distances[i]=64
        for j in np.union1d(np.flatnonzero(distances<=8),np.argsort(distances)[:3]):
            key=tuple(sorted((i,int(j))))
            pos=np.flatnonzero(owners==j);best=pos[np.argmin(counts[pos])]
            rec=(int(distances[j]),tags[int(best)])
            if key not in pairs or rec[0]<pairs[key][0]:pairs[key]=rec
    for i,j in itertools.combinations(range(len(names)),2):
        if rgb[i]==rgb[j] or clinical.sha256.iloc[i]==clinical.sha256.iloc[j]:pairs[(i,j)]=(0,'exact_decoded_or_raw')
    rows=[]
    try:import cv2;cv_version=cv2.__version__
    except ImportError:cv_version=None
    for (i,j),(distance,tag) in sorted(pairs.items(),key=lambda x:(x[1][0],x[0])):
        matches,inliers,ratio=local_match(images[names[i]],images[names[j]]) if cv_version else (None,None,None)
        rows.append({'left':names[i],'right':names[j],'left_group':clinical.patient_cluster.iloc[i],
            'right_group':clinical.patient_cluster.iloc[j],'same_known_group':clinical.patient_cluster.iloc[i]==clinical.patient_cluster.iloc[j],
            'left_sha256':clinical.sha256.iloc[i],'right_sha256':clinical.sha256.iloc[j],
            'same_label':bool(clinical.label.iloc[i]==clinical.label.iloc[j]),'raw_identical':clinical.sha256.iloc[i]==clinical.sha256.iloc[j],
            'rgb_identical':rgb[i]==rgb[j],'phash_distance':distance,'retrieval_variant':tag,
            'sift_matches':matches,'ransac_inliers':inliers,'inlier_fraction':ratio,
            'decision':'pending','reviewer':'','rationale':''})
    frame=pd.DataFrame(rows);frame.to_csv(out/'duplicate_candidates_REVIEW_REQUIRED.csv',index=False)
    review=out/'duplicate_review.html';parts=['<meta charset="utf-8"><title>Private duplicate review</title>',
        '<p>Review content matches; do not infer clinical identity from resemblance. Do not publish these private review images.</p>']
    thumb=out/'private_duplicate_thumbnails';thumb.mkdir(exist_ok=True)
    for name in set(frame.left)|set(frame.right):
        im=Image.open(images[name]).convert('RGB');im.thumbnail((240,180));im.save(thumb/(name+'.png'))
    for r in frame.itertuples():
        parts.append(f'<p>{r.left} / {r.right}, pHash {r.phash_distance}, exact RGB {r.rgb_identical}<br><img src="private_duplicate_thumbnails/{r.left}.png"><img src="private_duplicate_thumbnails/{r.right}.png"></p>')
    review.write_text('\n'.join(parts),encoding='utf-8')
    # Annotation sheet contains no diagnoses, predictions, probabilities, or heatmaps.
    ann=clinical[['filename','sha256']].copy();ann['artifact_present']='pending';ann['boxes_xyxy_json']='[]';ann['reviewer']='';ann['annotation_blinded']='pending'
    ann['coordinates']='384x384 resized image, 0<=x0<x1<=384, 0<=y0<y1<=384'
    ann.to_csv(out/'artifact_annotations_TEMPLATE.csv',index=False)
    write_json(out/'audit_status.json',{'images_verified':755,'candidate_pairs':len(frame),'opencv':cv_version,
        'candidates_sha256':sha(out/'duplicate_candidates_REVIEW_REQUIRED.csv'),
        'scope':'Content-duplicate candidate retrieval, not verified clinical identity or an exhaustive guarantee.',
        'manual_adjudication':'pending','verified_patient_identity':'pending','external_validation':'pending',
        'action_if_new_cross_group_match':'Do not silently merge after testing. Document adjudication, rebuild affected grouped splits and rerun affected experiments.'})
    return frame


def mixfit(cal,y):
    def softmax(z):e=np.exp(z-z.max());return e/e.sum()
    result=minimize(lambda z: -np.log(np.clip(np.tensordot(softmax(z),cal,axes=1)[np.arange(len(y)),y],1e-15,1)).mean(),
                    np.zeros(len(cal)),method='L-BFGS-B')
    if not result.success:raise ValueError('Subset calibration fit failed: '+str(result.message))
    return softmax(result.x)


def variants(cal,test,y,scale=temperature,precomputed=None):
    if precomputed is None:temps,w,_=fit_calibration(cal,y)
    else:temps,w=precomputed
    sc=[scale(p,t) for p,t in zip(cal,temps)];st=[scale(p,t) for p,t in zip(test,temps)]
    result={m:test[i] for i,m in enumerate(MODELS)}
    result.update({m+'_scaled':st[i] for i,m in enumerate(MODELS)})
    result['ensemble']=sum(wi*pi for wi,pi in zip(w,st))
    result['uniform_raw']=np.mean(test,axis=0);result['uniform_scaled']=np.mean(st,axis=0)
    chosen=int(np.argmin([-np.log(np.clip(p[np.arange(len(y)),y],1e-15,1)).mean() for p in sc]))
    result['calibration_selected_single']=st[chosen]
    # Every nonempty subset is retained; never choose a subset from its test score.
    params={'temperatures':list(map(float,temps)),'weights':list(map(float,w)),'calibration_selected_single':MODELS[chosen],'subsets':{}}
    for size in (2,3):
        for subset in itertools.combinations(range(4),size):
            ws=mixfit(np.stack([sc[i] for i in subset]),y)
            name='subset_'+'+'.join(MODELS[i] for i in subset)
            result[name]=sum(wi*st[i] for wi,i in zip(ws,subset))
            params['subsets'][name]=list(map(float,ws))
    return result,params


def holm(p):
    p=np.asarray(p);order=np.argsort(p);adjusted=np.empty(len(p));values=np.maximum.accumulate(p[order]*(len(p)-np.arange(len(p))))
    adjusted[order]=np.minimum(1,values);return adjusted


def paired_comparisons(y,probabilities,bootstrap=2000):
    target=probabilities['ensemble'];a=target.argmax(1)==y;rows=[]
    comparisons=list(MODELS)+['uniform_raw','uniform_scaled','calibration_selected_single']
    rng=np.random.default_rng(615)
    draws=rng.integers(0,len(y),(bootstrap,len(y)))
    for name in comparisons:
        b=probabilities[name].argmax(1)==y;both=int((a&~b).sum());other=int((~a&b).sum());difference=a.astype(float)-b
        boot=difference[draws].mean(1);ci=np.percentile(boot,[2.5,97.5])
        p=float(binomtest(both,both+other,.5).pvalue) if both+other else 1.
        rows.append({'comparison':name,'ensemble_only_correct':both,'comparator_only_correct':other,
            'accuracy_delta_pp':100*float(difference.mean()),'ci_low_pp':100*float(ci[0]),'ci_high_pp':100*float(ci[1]),'p_exact':p})
    for r,p in zip(rows,holm([r['p_exact'] for r in rows])):r['p_holm_7']=float(p)
    return rows


def b_comparisons(inputs,out):
    from final_common import prepare_inputs,read_results
    from ensemble_gradcam import fit_ensemble,scale_numpy
    marker=Path(inputs)/'b_inputs/pb_final_verified_manifest.json'
    root=prepare_inputs(Path(inputs)/'b_inputs',out,sha(marker));manifest,records,pooled,splits,keep=read_results(root)
    temps,weights,official=fit_ensemble(pooled,keep)
    y=pooled[MODELS[0]]['test']['labels'];yc=pooled[MODELS[0]]['val']['labels'][keep]
    cal=[pooled[m]['val']['probs'][keep] for m in MODELS];test=[pooled[m]['test']['probs'] for m in MODELS]
    result,params=variants(cal,test,yc,scale_numpy,([temps[m] for m in MODELS],[weights[m] for m in MODELS]))
    if not np.allclose(result['ensemble'],official,atol=1e-12):raise ValueError('Official ensemble reconstruction differs.')
    metrics=[{'method':m,**numpy_metrics(p,y)} for m,p in result.items()]
    export_tables({'B_ensemble_sensitivity':metrics,'B_paired_comparisons':paired_comparisons(y,result)},out/'tables')
    write_json(out/'B_calibration_only_parameters.json',params)
    np.savez_compressed(out/'B_comparison_predictions.npz',labels=y,groups=pooled[MODELS[0]]['test']['pids'],**result)
    return metrics


def main(args):
    verify_inputs(args.inputs,args.expected_sha);out=Path(args.work)/'SUPP_CPU_Audit';out.mkdir(parents=True,exist_ok=True)
    b_comparisons(args.inputs,out)
    if not args.statistics_only:duplicate_audit(args.inputs,args.input_root,out)
    write_json(out/'PENDING_REQUIREMENTS.json',{'external_validation':'pending: no independent labelled cohort supplied',
        'verified_patient_identity':'pending: no authoritative patient linkage supplied',
        'localisation_validation':'pending: no independent expert lesion masks supplied',
        'historical_MSLD_reuse':'cannot be removed retrospectively; new analyses remain internal and supplemental',
        'manual_artifact_review':'pending: annotate blinded to predictions and heatmaps before targeted artifact masking',
        'MCVSLD_v1':'overlaps MSLD; audited Mendeley version 1 is not an independent external test'})
    bundle(out)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--inputs',required=True);p.add_argument('--expected-sha',required=True)
    p.add_argument('--input-root',default='/kaggle/input');p.add_argument('--work',default='/kaggle/working');p.add_argument('--statistics-only',action='store_true')
    main(p.parse_args())
