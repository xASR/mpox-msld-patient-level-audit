from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
from PIL import Image

META=['width','height','aspect','area','file_bytes','bytes_per_pixel']

def extract(path,expected):
    path=Path(path)
    if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
        raise ValueError(f'Image bytes changed: {path}')
    with Image.open(path) as img:
        w,h=img.size
        size=path.stat().st_size
        row=dict(zip(META,[w,h,w/h,w*h,size,size/(w*h)]))
        rgb=np.asarray(img.convert('RGB'))
    border=np.ones((h,w),bool)
    border[12:-12,12:-12]=False
    regions={'perimeter':rgb[border], 'upper_left':rgb[:12,:12], 'upper_right':rgb[:12,-12:],
             'lower_left':rgb[-12:,:12], 'lower_right':rgb[-12:,-12:]}
    for region,arr in regions.items():
        pixels=arr.reshape(-1,3).astype(np.float64)
        for i,color in enumerate('RGB'):
            for stat,func in [('mean',np.mean),('sd',np.std),('variance',np.var)]:
                row[f'{region}_{color}_{stat}']=func(pixels[:,i])
        row[region+'_near_black']=np.all(pixels<=8,axis=1).mean()
        row[region+'_near_white']=np.all(pixels>=247,axis=1).mean()
    return row

