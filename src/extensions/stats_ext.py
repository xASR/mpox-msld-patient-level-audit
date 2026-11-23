"""Six-class summaries; bootstrap indices always sample whole known groups."""
import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

CLASSES = ['Healthy', 'Cowpox', 'Mpox', 'Measles', 'Chickenpox', 'HFMD']
METRICS = ['accuracy', 'macro_f1', 'macro_auc', 'ece']

def scale(p, t):
    z = np.log(np.clip(p, 1e-12, 1)) / t
    z = np.exp(z - z.max(axis=-1, keepdims=True))
    return z / z.sum(axis=-1, keepdims=True)

def weighted(y, p, weights):
    """Vectorized point or bootstrap measures, including ties in weighted OvR AUC."""
    y, p, w = np.asarray(y, int), np.asarray(p), np.atleast_2d(weights)
    pred = p.argmax(1)
    correct = pred == y
    acc = (w * correct).sum(1) / w.sum(1)
    f1, auc = [], []
    for c in range(6):
        pos, call = y == c, pred == c
        tp = (w * (pos & call)).sum(1)
        denom = (w * (pos.astype(int) + call)).sum(1)
        f1.append(np.divide(2*tp, denom, out=np.zeros(len(w)), where=denom>0))
        order = p[:,c].argsort(kind='stable')
        cuts = np.r_[0, np.flatnonzero(np.diff(p[order,c]))+1]
        wp = np.add.reduceat((w*pos)[:,order], cuts, axis=1)
        wn = np.add.reduceat((w*~pos)[:,order], cuts, axis=1)
        denom = wp.sum(1)*wn.sum(1)
        num = (wp*(wn.cumsum(1)-0.5*wn)).sum(1)
        auc.append(np.divide(num, denom, out=np.full(len(w),np.nan), where=denom>0))
    conf = p.max(1)
    bins = np.clip(np.ceil(conf*10).astype(int)-1,0,9)
    ece = sum(abs((w*((conf-correct)*(bins==i))).sum(1)) for i in range(10))/w.sum(1)
    return np.stack([acc,np.mean(f1,axis=0),np.mean(auc,axis=0),ece],axis=1)

def estimates(y,p,groups,other=None,draws=2000):
    _, ix = np.unique(groups,return_inverse=True)
    n = ix.max()+1
    def measure(w):
        a=weighted(y,p,w)
        return a if other is None else a-weighted(y,other,w)
    point=measure(np.ones((1,len(y)),int))[0]
    rng=np.random.default_rng(615)
    batches=[]
    for k in range(0,draws,64):
        w=rng.multinomial(n,np.full(n,1/n),size=min(64,draws-k))[:,ix]
        batches.append(measure(w))
    boot=np.concatenate(batches)
    return {key:float(val) for i,m in enumerate(METRICS) for key,val in
            zip((m,m+'_lo',m+'_hi'),(point[i],*np.nanquantile(boot[:,i],[.025,.975])))}

def mcnemar(y, new, reference):
    a,b=new.argmax(1)==y,reference.argmax(1)==y
    win,loss=int(sum(a&~b)),int(sum(~a&b))
    return dict(gain=win,loss=loss,p_exact=binomtest(win,win+loss,.5).pvalue if win+loss else 1.)

def wilson(x,n):
    if n==0:return np.nan,np.nan,np.nan
    z=1.959963984540054
    p=x/n
    mid=(p+z*z/(2*n))/(1+z*z/n)
    half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return p,max(0.,mid-half),min(1.,mid+half)

