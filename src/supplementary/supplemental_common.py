"""Common integrity checks for the prespecified supplemental MSLD analyses."""
from pathlib import Path, PurePosixPath
from zipfile import ZipFile, ZIP_DEFLATED
import hashlib
import json
import os
import shutil
import stat
import numpy as np
import pandas as pd

MODELS = ('efficientnetv2_s', 'convnext_tiny', 'swin_small', 'beit_base')
SEEDS = (17, 73, 101, 137, 211)
LABELS = ('Healthy', 'Cowpox', 'Mpox', 'Measles', 'Chickenpox', 'HFMD')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(2**20), b''): h.update(b)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + '.tmp')
    def convert(v):
        if isinstance(v,np.generic):return v.item()
        if isinstance(v,np.ndarray):return v.tolist()
        raise TypeError('Unsupported JSON value: '+type(v).__name__)
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False, default=convert), encoding='utf-8')
    tmp.replace(p)


def verify_inputs(root, expected):
    root = Path(root)
    marker = root / 'supplemental_manifest.json'
    if sha(marker) != expected: raise ValueError('Supplemental manifest differs.')
    index = json.loads(marker.read_text())
    for rel, rec in index['files'].items():
        p = (root / rel).resolve()
        if not p.is_relative_to(root.resolve()) or not p.is_file() or p.stat().st_size != rec['bytes'] or sha(p) != rec['sha256']:
            raise ValueError('Input missing or changed: ' + rel)
    return index


def bundle(folder, filename=None, weights=False):
    folder = Path(folder)
    target = folder.parent / (filename or (folder.name + '_RESULTS_ONLY.zip'))
    with ZipFile(target, 'w', ZIP_DEFLATED) as z:
        for p in sorted(folder.rglob('*')):
            if not p.is_file() or '__pycache__' in p.parts or p.name.endswith('.tmp'): continue
            if not weights and p.suffix == '.pt': continue
            z.write(p, folder.name + '/' + p.relative_to(folder).as_posix())
    print('Download small results:', target, flush=True)
    return target


def extract_results(search, destination):
    """Accept either Kaggle-expanded results or untouched small result ZIPs."""
    destination = Path(destination).resolve(); destination.mkdir(parents=True, exist_ok=True)
    roots = [Path(search)]
    for archive in sorted(Path(search).rglob('SUPP_*RESULTS_ONLY.zip')):
        cache = destination / sha(archive)
        with ZipFile(archive) as z:
            if sum(i.file_size for i in z.infolist()) > 2 * 2**30: raise ValueError('Expected small supplemental results, not all weights.')
            for i in z.infolist():
                rel = PurePosixPath(i.filename.replace('\\', '/'))
                p = cache.joinpath(*rel.parts).resolve()
                if rel.is_absolute() or '..' in rel.parts or any(':' in a for a in rel.parts) or stat.S_ISLNK(i.external_attr >> 16) or not p.is_relative_to(cache.resolve()):
                    raise ValueError('Unsafe result ZIP path.')
                if i.is_dir(): p.mkdir(parents=True, exist_ok=True)
                else:
                    p.parent.mkdir(parents=True, exist_ok=True)
                    with z.open(i) as src, p.open('wb') as dst: shutil.copyfileobj(src, dst)
        roots.append(cache)
    return roots


def unique_markers(roots, name, key_function):
    seen = {}
    for root in roots:
        for p in sorted(Path(root).rglob(name)):
            value = json.loads(p.read_text()); key = key_function(value)
            if key is None: continue
            if key in seen and seen[key][1] != value: raise ValueError('Conflicting completed results for ' + str(key))
            seen[key] = (p, value)
    return seen


def export_tables(tables, out):
    """Editable CSV/Markdown, and compact horizontal-rule PDF tables."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Serif', 'font.size': 9, 'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    for name, data in tables.items():
        df = pd.DataFrame(data); df.to_csv(out / (name + '.csv'), index=False)
        if df.empty: continue
        display = df.copy()
        for c in display:
            def formatted(v):
                if isinstance(v,(float,np.floating)) and np.isfinite(v):
                    return f'{v:.3g}' if str(c).startswith('p_') else f'{v:.4f}'
                return str(v)
            display[c] = display[c].map(formatted)
        lines = ['| ' + ' | '.join(map(str, display.columns)) + ' |', '| ' + ' | '.join(['---']*len(display.columns)) + ' |']
        lines += ['| ' + ' | '.join(map(str, row)) + ' |' for row in display.itertuples(index=False, name=None)]
        (out / (name + '.md')).write_text('\n'.join(lines), encoding='utf-8')
        # Per-group audit records remain editable data, not dozens of paper pages.
        if len(df)>200:continue
        # Paginate both rows and wide column sets. Repeat the identifying column.
        column_sets=[list(display.columns)] if len(display.columns)<=8 else [[display.columns[0],*display.columns[1+i:1+i+7]] for i in range(0,len(display.columns)-1,7)]
        for page, start in enumerate(range(0, len(df), 18), 1):
          for panel,columns in enumerate(column_sets,1):
            part = display.loc[:,columns].iloc[start:start+18].copy()
            abbreviations={'efficientnetv2_s':'E','convnext_tiny':'C','swin_small':'S','beit_base':'B'}
            for c in ('method','comparison'):
                if c in part:
                    for full,short in abbreviations.items():part[c]=part[c].str.replace(full,short,regex=False)
                    part[c]=part[c].str.replace('calibration_selected_single','Cal. selected single',regex=False).str.replace('uniform_raw','Uniform (raw)',regex=False).str.replace('uniform_scaled','Uniform (cal.)',regex=False).str.replace('_scaled',' (cal.)',regex=False).str.replace('subset_','',regex=False)
            width = max(7.2, len(part.columns)*1.12)
            fig, ax = plt.subplots(figsize=(width, max(2.0, .25*(len(part)+3)+.45)))
            fig.subplots_adjust(left=.02,right=.98,top=.98,bottom=.15)
            ax.axis('off')
            labels={'method':'Method','comparison':'Comparator','p_holm_7':'Holm p\n(7 tests)','p_exact':'Exact p',
                'ensemble_only_correct':'Ensemble\nonly correct','comparator_only_correct':'Comparator\nonly correct',
                'accuracy_delta_pp':'Accuracy\nchange (pp)','ci_low_pp':'95% CI\nlower (pp)','ci_high_pp':'95% CI\nupper (pp)',
                'macro_precision':'Macro\nprecision','macro_recall':'Macro\nrecall','macro_f1':'Macro F1','macro_auc':'Macro AUC',
                'brier_multiclass':'Multiclass\nBrier','log_loss':'Log loss','ece':'ECE','accuracy':'Accuracy'}
            tab = ax.table(cellText=part.values, colLabels=[labels.get(str(c),str(c).replace('_','\n')) for c in part.columns], cellLoc='center',bbox=[0,.02,1,.96])
            tab.auto_set_font_size(False); tab.set_fontsize(8)
            tab.auto_set_column_width(list(range(len(part.columns))))
            for (r, c), cell in tab.get_celld().items():
                cell.set_facecolor('white'); cell.set_edgecolor('black'); cell.set_linewidth(.5)
                cell.visible_edges = 'TB' if r == 0 else ('B' if r == len(part) else '')
                if r == 0: cell.set_text_props(weight='bold');cell.set_height(cell.get_height()*1.9)
            if any(c in part for c in ('method','comparison')):
                fig.text(.02,.025,'E: EfficientNetV2-S; C: ConvNeXt-Tiny; S: Swin-Small; B: BEiT-Base.\nFull precision values and identifiers are retained in the accompanying CSV.',fontsize=7)
            for ext in ('pdf', 'png'):
                suffix='' if len(column_sets)==1 else f'_c{panel}'
                fig.savefig(out / f'{name}_p{page}{suffix}.{ext}', dpi=600, bbox_inches='tight', facecolor='white')
            plt.close(fig)


def jobs_for(account):
    families = {'alve': MODELS[:2], 'friend1': (MODELS[2],), 'friend2': (MODELS[3],)}
    jobs = [{'kind': 'repeat', 'seed': s, 'model': m, 'fold': f}
            for s in SEEDS for m in families[account] for f in range(1, 6)]
    # Alternating conditions are a matched pair; both must finish before analysis.
    abl_seeds = {'alve': (), 'friend1': (0, 1, 2), 'friend2': (3, 4)}[account]
    jobs += [{'kind': 'ablation', 'seed': s, 'model': 'convnext_tiny', 'condition': c}
             for s in abl_seeds for c in ('clean', 'contaminated')]
    return jobs


def job_name(job):
    if job['kind'] == 'repeat': return f"A_s{job['seed']}_{job['model']}_f{job['fold']}"
    return f"ABL_s{job['seed']}_{job['condition']}"
