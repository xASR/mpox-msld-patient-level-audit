import numpy as np
from sklearn.metrics import accuracy_score,f1_score,roc_auc_score

def normalized(p):
    p=np.asarray(p,dtype=np.float64)
    if p.ndim!=2 or p.shape[1]!=6 or not np.isfinite(p).all() or (p<0).any() or not np.allclose(p.sum(1),1,atol=2e-5):
        raise ValueError("Invalid six-class probabilities")
    return p/p.sum(axis=1,keepdims=True)

def temperature(p,t):
    x=np.log(np.clip(normalized(p),1e-8,1))/float(t)
    x=np.exp(x-x.max(1,keepdims=True));return x/x.sum(1,keepdims=True)

def point(y,p,auc=False):
    y=np.asarray(y,dtype=int);p=normalized(p);pred=p.argmax(1)
    ans=dict(n=len(y),correct=int((pred==y).sum()),accuracy=float(accuracy_score(y,pred)),
             macro_f1=float(f1_score(y,pred,labels=np.arange(6),average='macro',zero_division=0)))
    if auc:ans['macro_auc']=float(roc_auc_score(y,p,multi_class="ovr",average="macro"))
    return ans

def group_intervals(y,p,groups,auc=False,seed=20261118,draws=2000):
    y=np.asarray(y,dtype=int);p=normalized(p);pred=p.argmax(1);keys,ind=np.unique(groups,return_inverse=True)
    ans=point(y,p,auc);ans['known_groups']=len(keys)
    rng=np.random.default_rng(seed);stats=[]
    for offset in range(0,draws,40):
        b=min(40,draws-offset)
        w=rng.multinomial(len(keys),np.full(len(keys),1/len(keys)),size=b)[:,ind]
        tp=[];den=[]
        for c in range(6):
            tp.append(w[:,(y==c)&(pred==c)].sum(1))
            den.append(w[:,y==c].sum(1)+w[:,pred==c].sum(1))
        tp=np.array(tp,dtype=float);den=np.array(den)
        scores=[w[:,y==pred].sum(1)/w.sum(1),np.divide(2*tp,den,out=np.zeros_like(tp),where=den!=0).mean(0)]
        if auc:
            per_class=[]
            for c in range(6):
                order=np.argsort(p[:,c],kind='stable');score=p[order,c];pos=(y[order]==c)
                begins=np.r_[0,np.flatnonzero(np.diff(score)!=0)+1]
                wp=np.add.reduceat(w[:,order]*pos,begins,axis=1)
                wn=np.add.reduceat(w[:,order]*(~pos),begins,axis=1)
                total=wp.sum(1)*wn.sum(1)
                num=(wp*(np.cumsum(wn,axis=1)-.5*wn)).sum(1)
                per_class.append(np.divide(num,total,out=np.full(b,np.nan),where=total>0))
            scores.append(np.mean(per_class,axis=0))
        stats.append(np.asarray(scores).T)
    boot=np.concatenate(stats)
    for k,name in enumerate(['accuracy','macro_f1']+(['macro_auc'] if auc else [])):
        valid=boot[:,k][np.isfinite(boot[:,k])]
        ans[name+'_boot_n']=len(valid)
        ans[name+'_lo'],ans[name+'_hi']=map(float,np.percentile(valid,[2.5,97.5]))
    return ans

