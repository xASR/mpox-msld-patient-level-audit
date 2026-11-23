"""Independent workers share a disk-backed cache of resized RGB tensors."""
import hashlib,os,shutil,time
from pathlib import Path
from filelock import FileLock
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset,Sampler
from torchvision import transforms as T
from PIL import Image
from revision_io import read,put

class BudgetSampler(Sampler):
    def __init__(self,n,budget,generator):
        self.n=n;self.budget=budget;self.generator=generator;self.last=None
        if not 0<budget<=n:raise ValueError('Update budget exceeds the eligible pool.')
    def __len__(self):return self.budget
    def __iter__(self):
        self.last=torch.randperm(self.n,generator=self.generator)[:self.budget].numpy()
        return iter(self.last.tolist())

def design(package,j):
    seed=j['index'];m=j['model'];plan=read(package/'plan.json')
    pool=pd.read_csv(package/'design/pool.csv');orig=pool[pool.is_original].set_index('source')
    if j['family']=='release':
        s=pd.read_csv(package/f'design/release_s{seed}.csv');assert np.array_equal(s.item,pool.item)
        is_test=s[j['arm']+'_test'].eq(1).to_numpy()
        rows=pool.copy();rows['key']=rows.item;rows['variant']=0; rows['augmentation_seed']=0
        parts={'train':rows[~is_test].copy(),'test':rows[is_test].copy()}
        clean_reference=rows[rows.is_original & s.group_test.eq(0)]
        budget=plan['release_epoch_slots'][str(seed)]
        if j['arm']=='group' and set(parts['train'].group)&set(parts['test'].group):raise ValueError('Group split overlap')
        return rows,parts,budget,clean_reference.label
    parts={}
    for arm in ('clean','contaminated','test','diagnostic'):
        df=pd.read_csv(package/f'design/matched_s{seed}_{arm}.csv')
        df['relative']=df.source.map(orig.relative);df['rgb']=df.source.map(orig.rgb)
        df['is_original']=df.variant.eq(0);df['key']=df.source+'|'+df.variant.astype(str)
        parts[arm]=df
    cache=pd.concat(parts.values()).drop_duplicates('key').reset_index(drop=True)
    clean,dirty,test=parts['clean'],parts['contaminated'],parts['test']
    if not np.array_equal(clean.label,dirty.label) or set(clean.group)&set(test.group):raise ValueError('Invalid matched roles')
    if len(set(test.group))!=104 or (dirty.source.isin(set(test.source))).sum()!=8*len(test):raise ValueError('Invalid contamination dose')
    return cache,dict(train=parts[j['arm']],test=test,diagnostic=parts['diagnostic']),len(clean),clean.label

class PixelStore:
    def __init__(self,frame,paths,size,folder,tag,deadline):
        self.frame=frame.reset_index(drop=True)
        self.pos=dict(zip(self.frame.key,range(len(frame))))
        self.paths=[paths[r] for r in self.frame.relative]
        self.v=self.frame.variant.to_numpy();self.seeds=self.frame.augmentation_seed.to_numpy()
        self.resize=T.Resize((size,size));self.to_pixels=T.PILToTensor()
        self.augment=T.Compose([T.RandomRotation(30),T.RandomAffine(0,translate=(.1,.1),scale=(.8,1.2),shear=15),T.ColorJitter(.15,.2,.15,.1)])
        self.array=None;self.file=folder/(tag+'.npy');folder.mkdir(parents=True,exist_ok=True)
        self.tag=tag
        # Writers publish only fully materialized arrays; killed writers leave no valid cache.
        with FileLock(str(folder/(tag+'.lock'))):
            shape=(len(frame),3,size,size);need=int(np.prod(shape))
            if not self.file.exists():
                if shutil.disk_usage(folder).free < need+3*2**30:
                    if tag.startswith('release'):
                        print('Using direct release image reads because scratch space is limited.',flush=True);return
                    raise OSError('Insufficient scratch space for matched-pixel verification. Resume in a fresh session.')
                tmp=self.file.with_suffix('.partial.npy')
                mm=np.lib.format.open_memmap(tmp,mode='w+',shape=shape,dtype=np.uint8);hashes=[]
                for i in range(len(frame)):
                    if i%200==0 and time.time()>deadline-360:raise TimeoutError('Stopping during scratch preparation; training state is preserved.')
                    mm[i]=self.render(i).numpy()
                    if tag.startswith('matched'):hashes.append(hashlib.sha256(mm[i].tobytes()).hexdigest())
                mm.flush();del mm
                if hashes:put(folder/(tag+'.hashes.json'),hashes)
                os.replace(tmp,self.file)
            self.array=np.load(self.file,mmap_mode='r')
            if self.array.shape!=shape or self.array.dtype!=np.uint8:raise ValueError('Scratch shape mismatch')
    def render(self,i):
        with Image.open(self.paths[i]) as raw:im=raw.convert('RGB')
        if int(self.v[i])>0:
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(int(self.seeds[i]));im=self.augment(im)
        return self.to_pixels(self.resize(im))
    def get(self,i):return torch.from_numpy(self.array[i].copy()) if self.array is not None else self.render(i)

class ImageRows(Dataset):
    def __init__(self,frame,store,vit):
        self.indices=[store.pos[k] for k in frame.key];self.store=store;self.vit=vit
        self.labels=frame['vit_label' if vit else 'label'].to_numpy();self.groups=frame.group.tolist()
        self.norm=T.Normalize([.485,.456,.406],[.229,.224,.225])
    def __len__(self):return len(self.indices)
    def __getitem__(self,i):
        x=self.store.get(self.indices[i]);x=x if self.vit else self.norm(x.float().div_(255.))
        return x,int(self.labels[i]),self.groups[i]

def verify_matched_pixels(store,parts,folder):
    hashes=read(folder/(store.tag+'.hashes.json'))
    subsets={k:{hashes[store.pos[a]] for a in df.key} for k,df in parts.items()}
    if subsets['train']&subsets['test'] or subsets['train']&subsets['diagnostic']:
        raise ValueError('Exact transformed test or reserved diagnostic pixels in training.')

