"""Optional native-file reproduction of both frozen tabular feature sets."""
from pathlib import Path
import os,sys
import numpy as np
import pandas as pd
from features import extract

def check(package,search):
    source=pd.read_csv(package/'originals.csv')
    ref=pd.read_csv(package/'features.csv').set_index('filename')
    local=Path('F:/Mpox_MSLD_Research_Archive/03_Datasets/MSLD_v2_0/extracted/Original Images/Original Images/FOLDS')
    if os.getenv('MSLD_FOLDS'):
        dirs=[Path(os.environ['MSLD_FOLDS'])]
    elif local.is_dir():dirs=[local]
    else:dirs=[p for p in search.rglob('FOLDS') if p.is_dir()]
    if not dirs:
        print('Using the verified native-file feature table in the input package.');return
    for r in source.itertuples():
        parts=Path(r.relative).parts;tail=parts[parts.index('FOLDS')+1:]
        paths=[d.joinpath(*tail) for d in dirs if d.joinpath(*tail).is_file()]
        if not paths:raise FileNotFoundError(r.source)
        feat=extract(paths[0],r.sha256)
        assert set(feat)==set(ref.columns)
        assert np.allclose([feat[k] for k in ref.columns],ref.loc[r.source],rtol=1e-13,atol=1e-12)
    print('Reproduced all six metadata and 55 border/corner features on 755 verified originals.')
if __name__=='__main__':check(*map(Path,sys.argv[1:3]))

