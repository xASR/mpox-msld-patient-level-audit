"""Reproduce the frozen ensemble and export journal-neutral tables and figures."""
from pathlib import Path
import argparse
import json
import os
import platform
import sys
from zipfile import ZipFile, ZIP_DEFLATED
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import (confusion_matrix, precision_recall_curve, average_precision_score,
                             roc_curve, roc_auc_score, cohen_kappa_score)
from final_common import (MODELS, DISPLAY, CLASSES, LABELS, FROZEN_SHA, prepare_inputs,
                          read_results, sha, write_json)

WORKFLOW=[
    'Verify 755 clinical images and merge four confirmed cross-ID duplicate links into 520 known groups.',
    'Preserve official test images and exclude lower-priority role conflicts using test > calibration > selection > training.',
    'Train four backbones on five frozen folds for 25 epochs and retain two checkpoints by selection-group log loss.',
    'Average four view probabilities, geometrically aggregate images within each group and average the two checkpoints.',
    'Average repeated group appearances across official test or calibration sheets.',
    'Exclude all pooled test groups from calibration candidates, leaving 138 eligible calibration groups.',
    'Fit four scalar temperatures and positive mixture weights using calibration log loss only.',
    'Evaluate the fixed ensemble on 215 known groups and export discrimination, calibration and paired uncertainty analyses.',
    'Verify all selected weights reproduce saved test predictions before producing qualitative ensemble-score Grad-CAM maps.'
]

def numpy_metrics(p,y):
    """Independent count/rank implementation for numerical and bootstrap checks."""
    p=p/p.sum(1,keepdims=True);pred=p.argmax(1);n=len(y)
    cm=np.bincount(6*y+pred,minlength=36).reshape(6,6).astype(float)
    tp=np.diag(cm);support=cm.sum(1);guesses=cm.sum(0)
    precision=np.divide(tp,guesses,out=np.zeros(6),where=guesses>0)
    recall=np.divide(tp,support,out=np.zeros(6),where=support>0)
    f1=np.divide(2*tp,support+guesses,out=np.zeros(6),where=(support+guesses)>0)
    auc=[]
    for c in range(6):
        positive=y==c;a=int(positive.sum());b=n-a
        if a==0 or b==0:auc.append(np.nan)
        else:auc.append((rankdata(p[:,c])[positive].sum()-a*(a+1)/2)/(a*b))
    conf=p.max(1);correct=pred==y;ece=0.
    for low,high in zip(np.linspace(0,1,11)[:-1],np.linspace(0,1,11)[1:]):
        mask=(conf>low)&(conf<=high)
        if mask.any():ece+=mask.mean()*abs(correct[mask].mean()-conf[mask].mean())
    return {'n':n,'correct':int(correct.sum()),'accuracy':float(correct.mean()),
            'macro_precision':float(precision.mean()),'macro_recall':float(recall.mean()),
            'macro_f1':float(f1.mean()),'macro_auc':float(np.mean(auc)),
            'ece':float(ece),'log_loss':float(-np.log(np.clip(p[np.arange(n),y],np.finfo(float).eps,1)).mean()),
            'brier_multiclass':float(((p-np.eye(6)[y])**2).sum(1).mean())}

def savefig(fig,root,name):
    fig.tight_layout(pad=1.0)
    for ext in ['png','pdf','svg']:
        fig.savefig(root/f'{name}.{ext}',dpi=600,bbox_inches='tight',facecolor='white')
    import matplotlib.pyplot as plt
    plt.close(fig)

def export_tables(out,tables):
    """One compact, editable table per page; manuscript-style horizontal rules."""
    from docx import Document
    from docx.shared import Inches,Pt,RGBColor
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.enum.table import WD_TABLE_ALIGNMENT,WD_CELL_VERTICAL_ALIGNMENT
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from reportlab.platypus import SimpleDocTemplate,Paragraph,Table,TableStyle,PageBreak,Spacer
    from reportlab.lib import colors
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.pagesizes import letter
    from xml.sax.saxutils import escape
    folder=out/'tables';folder.mkdir(exist_ok=True)
    doc=Document();section=doc.sections[0]
    section.page_width=Inches(8.5);section.page_height=Inches(11)
    section.top_margin=section.bottom_margin=Inches(.65)
    section.left_margin=section.right_margin=Inches(.6)
    doc.core_properties.author='';doc.core_properties.last_modified_by=''
    for style in ['Normal','Title','Heading 1','Heading 2']:
        doc.styles[style].font.name='Times New Roman';doc.styles[style].font.color.rgb=RGBColor(0,0,0)
        fonts=doc.styles[style].element.get_or_add_rPr().rFonts
        if fonts is not None:
            for attr in ['asciiTheme','hAnsiTheme','eastAsiaTheme','cstheme']:
                fonts.attrib.pop(qn('w:'+attr),None)
    # The bundled Word template has a blue bottom border on Title. Remove it
    # from all paragraph styles and paragraphs, including inherited residue.
    for node in doc.styles.element.xpath('.//w:pBdr'):
        node.getparent().remove(node)
    doc.styles['Normal'].font.size=Pt(10.5)
    doc.styles['Normal'].paragraph_format.space_after=Pt(7)
    doc.styles['Title'].font.size=Pt(15)
    body=ParagraphStyle('body',fontName='Times-Roman',fontSize=10,leading=13,spaceAfter=8)
    cell=ParagraphStyle('cell',parent=body,fontSize=9,leading=11,spaceAfter=0)
    head=ParagraphStyle('head',parent=cell,fontName='Times-Bold')
    number_cell=ParagraphStyle('numbercell',parent=cell,alignment=1)
    number_head=ParagraphStyle('numberhead',parent=head,alignment=1)
    title=ParagraphStyle('title',parent=body,fontName='Times-Bold',fontSize=15,leading=18,spaceAfter=12)
    pdf=[];md=['# Corrected Protocol B publication tables','',
        'All results are conditional on the recorded models and calibration fit. The evaluation unit is a known duplicate-aware group, not independently verified clinical identity.','']
    latex=[]
    for k,(caption,headers,rows,note,widths) in enumerate(tables,1):
        if k>1:doc.add_page_break();pdf.append(PageBreak())
        title_text=f'Table B{k} {caption}'
        doc.add_paragraph(title_text,'Title');pdf.append(Paragraph(escape(title_text),title))
        if k==1:
            intro='Corrected official-fold evaluation using 755 clinical images. This pack records the final ensemble, supporting comparisons, calibration, coverage and reproducibility checks.'
            doc.add_paragraph(intro);pdf.append(Paragraph(intro,body))
        md.extend([f'## {title_text}','','| '+' | '.join(h.replace('\n',' ') for h in headers)+' |','| '+' | '.join(['---']*len(headers))+' |'])
        md.extend('| '+' | '.join(str(v).replace('\n',' ') for v in row)+' |' for row in rows)
        md.extend(['',note,''])
        pd.DataFrame(rows,columns=headers).to_csv(folder/f'table_{k:02d}.csv',index=False)
        t=doc.add_table(rows=1,cols=len(headers));t.alignment=WD_TABLE_ALIGNMENT.CENTER;t.autofit=False
        repeat=OxmlElement('w:tblHeader');t.rows[0]._tr.get_or_add_trPr().append(repeat)
        borders=OxmlElement('w:tblBorders')
        for edge in ['top','bottom','left','right','insideH','insideV']:
            e=OxmlElement('w:'+edge);e.set(qn('w:val'),'single' if edge in ['top','bottom'] else 'nil')
            e.set(qn('w:sz'),'8');e.set(qn('w:color'),'333333');borders.append(e)
        t._tbl.tblPr.append(borders)
        for values in rows:t.add_row()
        for ri,values in enumerate([headers]+rows):
            r=t.rows[ri];no_split=OxmlElement('w:cantSplit');r._tr.get_or_add_trPr().append(no_split)
            for ci,(value,w) in enumerate(zip(values,widths)):
                c=r.cells[ci];c.width=Inches(w);c.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
                pr=c._tc.get_or_add_tcPr();margins=OxmlElement('w:tcMar')
                for edge in ['top','bottom','left','right']:
                    e=OxmlElement('w:'+edge);e.set(qn('w:w'),'90');e.set(qn('w:type'),'dxa');margins.append(e)
                pr.append(margins)
                if ri==0:
                    cb=OxmlElement('w:tcBorders');edge=OxmlElement('w:bottom');edge.set(qn('w:val'),'single');edge.set(qn('w:sz'),'6');cb.append(edge);pr.append(cb)
                p=c.paragraphs[0];p.paragraph_format.space_after=Pt(0);p.paragraph_format.line_spacing=1.05
                p.alignment=WD_ALIGN_PARAGRAPH.LEFT if ci==0 else WD_ALIGN_PARAGRAPH.CENTER
                run=p.add_run(str(value));run.font.size=Pt(9.5);run.bold=ri==0
        doc.add_paragraph(note)
        data=[[Paragraph(escape(str(v)).replace('\n','<br/>'),(head if ci==0 else number_head) if ri==0 else (cell if ci==0 else number_cell)) for ci,v in enumerate(values)]
              for ri,values in enumerate([headers]+rows)]
        pt=Table(data,colWidths=[w*72 for w in widths],repeatRows=1,hAlign='CENTER')
        pt.setStyle(TableStyle([('LINEABOVE',(0,0),(-1,0),.8,colors.black),('LINEBELOW',(0,0),(-1,0),.5,colors.black),
            ('LINEBELOW',(0,-1),(-1,-1),.8,colors.black),('VALIGN',(0,0),(-1,-1),'MIDDLE'),
            ('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5),
            ('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6)]))
        pdf.extend([pt,Spacer(1,12),Paragraph(escape(note),body)])
        def tex(v):
            s=str(v).replace('\n',' ')
            for a,b in [('&',r'\&'),('%',r'\%'),('_',r'\_')]:s=s.replace(a,b)
            return s
        latex.extend([r'\begin{table}[htbp]',r'\centering',r'\caption{'+caption+'}',r'\begin{tabular}{l'+'c'*(len(headers)-1)+'}',r'\toprule',
                      ' & '.join(map(tex,headers))+r' \\',r'\midrule'])
        latex.extend(' & '.join(map(tex,row))+r' \\' for row in rows)
        latex.extend([r'\bottomrule',r'\end{tabular}',r'\par\smallskip\footnotesize '+tex(note),r'\end{table}',''])
    doc.add_page_break();doc.add_paragraph('Algorithm B1 Corrected official fold workflow','Title')
    pdf.extend([PageBreak(),Paragraph('Algorithm B1 Corrected official fold workflow',title)])
    md.extend(['## Algorithm B1 Corrected official fold workflow',''])
    for k,line in enumerate(WORKFLOW,1):
        doc.add_paragraph(f'{k}. {line}');pdf.append(Paragraph(f'{k}. {escape(line)}',body));md.append(f'{k}. {line}')
    for node in doc.element.xpath('.//w:pBdr'):
        node.getparent().remove(node)
    doc.save(folder/'PB_Corrected_Publication_Tables.docx')
    SimpleDocTemplate(str(folder/'PB_Corrected_Publication_Tables.pdf'),pagesize=letter,
        leftMargin=.6*72,rightMargin=.6*72,topMargin=.65*72,bottomMargin=.65*72).build(pdf)
    (folder/'PB_Corrected_Publication_Tables.md').write_text('\n'.join(md),encoding='utf-8')
    (folder/'PB_Corrected_Publication_Tables.tex').write_text('\n'.join(latex),encoding='utf-8')

def extend_outputs(inputs,out,bootstrap=2000):
    from merge_corrected import scaled,metrics,wilson,ece
    manifest,records,pooled,splits,keep=read_results(inputs)
    report=json.loads((out/'report.json').read_text())
    ref=pooled[MODELS[0]];y=ref['test']['labels'];ids=ref['test']['pids']
    probabilities={m:pooled[m]['test']['probs'] for m in MODELS}
    probability_scaled={m:scaled(probabilities[m],report['temperatures'][m]) for m in MODELS}
    ensemble=sum(report['weights'][m]*probability_scaled[m] for m in MODELS)
    if not np.allclose(ensemble,np.load(out/'ensemble_probs.npy'),atol=1e-12,rtol=0):raise ValueError('Reconstruction differs from frozen merge.')
    probabilities['ensemble']=ensemble
    verified=[]
    for m,p in probabilities.items():
        a=numpy_metrics(p,y);b=metrics(p,y)
        for key in b:
            if not np.isclose(a[key],b[key],atol=2e-12,rtol=0):raise ValueError(f'Independent metric mismatch {m} {key}')
        verified.append({'model':m,**a,'accuracy_ci_low':wilson(a['correct'],len(y))[0],
                         'accuracy_ci_high':wilson(a['correct'],len(y))[1]})
    pd.DataFrame(verified).to_csv(out/'independently_verified_model_metrics.csv',index=False)
    np.savez_compressed(out/'analysis_probabilities.npz',test_groups=ids,test_labels=y,
        calibration_groups=ref['val']['pids'][keep],calibration_labels=ref['val']['labels'][keep],
        **{m+'_raw':p for m,p in probabilities.items()},
        **{m+'_scaled':p for m,p in probability_scaled.items()})
    rng=np.random.default_rng(42);boot={m:[] for m in probabilities}
    for _ in range(bootstrap):
        sample=rng.integers(0,len(y),len(y))
        for m,p in probabilities.items():boot[m].append(numpy_metrics(p[sample],y[sample]))
    ci_rows=[]
    for m,draws in boot.items():
        for key in ['accuracy','macro_precision','macro_recall','macro_f1','macro_auc','ece','log_loss','brier_multiclass']:
            values=np.array([v[key] for v in draws]);values=values[np.isfinite(values)]
            lo,hi=np.quantile(values,[.025,.975])
            ci_rows.append({'model':m,'metric':key,'estimate':numpy_metrics(probabilities[m],y)[key],
                            'low':lo,'high':hi,'valid_draws':len(values),'bootstrap_n':bootstrap})
    pd.DataFrame(ci_rows).to_csv(out/'conditional_group_bootstrap_intervals.csv',index=False)
    np.savez_compressed(out/'conditional_group_bootstrap_draws.npz',
        **{m:np.array([[d[k] for k in ['accuracy','macro_f1','macro_auc','ece','log_loss','brier_multiclass']] for d in values]) for m,values in boot.items()})
    write_json(out/'bootstrap_columns.json',['accuracy','macro_f1','macro_auc','ece','log_loss','brier_multiclass'])
    cm=confusion_matrix(y,ensemble.argmax(1),labels=range(6));n=len(y)
    perclass=[];curve_rows=[];reliability=[]
    for c,label in enumerate(LABELS):
        truth=y==c;pred=ensemble.argmax(1)==c;tp=int(cm[c,c]);fn=int(truth.sum()-tp)
        fp=int(pred.sum()-tp);tn=n-tp-fn-fp
        recall_ci=wilson(tp,tp+fn);spec_ci=wilson(tn,tn+fp)
        precision_ci=wilson(tp,tp+fp) if tp+fp else [None,None]
        perclass.append({**report['per_class'][c],'specificity':tn/(tn+fp),'tp':tp,'fn':fn,'fp':fp,'tn':tn,
            'precision_ci_low':precision_ci[0],'precision_ci_high':precision_ci[1],
            'recall_ci_low':recall_ci[0],'recall_ci_high':recall_ci[1],
            'specificity_ci_low':spec_ci[0],'specificity_ci_high':spec_ci[1],
            'average_precision':float(average_precision_score(truth,ensemble[:,c]))})
        fpr,tpr,thr=roc_curve(truth,ensemble[:,c]);precision,recall,prth=precision_recall_curve(truth,ensemble[:,c])
        for x,v,t in zip(fpr,tpr,thr):curve_rows.append({'class':label,'curve':'ROC','x':x,'y':v,'threshold':t})
        for i,(x,v) in enumerate(zip(recall,precision)):curve_rows.append({'class':label,'curve':'PR','x':x,'y':v,'threshold':prth[i] if i<len(prth) else np.nan})
    pd.DataFrame(perclass).to_csv(out/'per_class_metrics_with_intervals.csv',index=False)
    pd.DataFrame(curve_rows).to_csv(out/'curve_coordinates.csv',index=False)
    calibrations=[]
    for m in MODELS:
        for stage,p in [('raw',probabilities[m]),('temperature_scaled',probability_scaled[m])]:
            calibrations.append({'model':m,'stage':stage,**numpy_metrics(p,y)})
            for b in ece(p/p.sum(1,keepdims=True),y)[1]:reliability.append({'model':m,'stage':stage,**b})
    calibrations.append({'model':'ensemble','stage':'calibrated',**numpy_metrics(ensemble,y)})
    for b in ece(ensemble,y)[1]:reliability.append({'model':'ensemble','stage':'calibrated',**b})
    pd.DataFrame(calibrations).to_csv(out/'calibration_summary.csv',index=False)
    pd.DataFrame(reliability).to_csv(out/'reliability_bin_data.csv',index=False)
    predictions=pd.DataFrame({'known_group':ids,'reference_class':[LABELS[v] for v in y],
        'ensemble_class':[LABELS[v] for v in ensemble.argmax(1)],'ensemble_confidence':ensemble.max(1),
        'test_fold_appearances':[sum(g in set(a.loc[a.role.eq('test'),'patient_cluster']) for a in splits.values()) for g in ids]})
    for m in MODELS:
        predictions[m+'_prediction']=[LABELS[v] for v in probabilities[m].argmax(1)]
    predictions.to_csv(out/'prediction_agreement_and_errors.csv',index=False)
    names=list(probabilities);agreement=np.array([[(probabilities[a].argmax(1)==probabilities[b].argmax(1)).mean() for b in names] for a in names])
    kappas=np.array([[cohen_kappa_score(probabilities[a].argmax(1),probabilities[b].argmax(1)) for b in names] for a in names])
    pd.DataFrame(agreement,index=names,columns=names).to_csv(out/'prediction_agreement.csv')
    pd.DataFrame(kappas,index=names,columns=names).to_csv(out/'prediction_kappa.csv')
    covered=set().union(*(set(a.loc[a.role.eq('test'),'filename']) for a in splits.values()))
    cal_cohort=pd.DataFrame({'known_group':ref['val']['pids'],'label':ref['val']['labels'],'eligible_for_calibration':keep})
    cal_cohort.to_csv(out/'calibration_cohort.csv',index=False)
    training=[]
    for (m,f),record in records.items():
        for h in json.loads((record['directory']/'epoch_history.json').read_text()):training.append({'model':m,'fold':f,**h})
    pd.DataFrame(training).to_csv(out/'training_histories.csv',index=False)
    fit_provenance={'input_manifest_sha256':FROZEN_SHA,'final_input_manifest_sha256':sha(inputs/'pb_final_verified_manifest.json'),
        'calibration_groups':138,'test_groups':215,'calibration_test_overlap_after_filter':0,
        'temperature_fit':'Bounded scalar calibration log-loss minimization on [0.1,5].',
        'mixture_fit':'Positive softmax weights, uniform initialization, L-BFGS-B on calibration log-loss.',
        'calibration_cohort_sha256':sha(out/'calibration_cohort.csv'),
        'test_labels_used_for_parameter_fitting':False,'ensemble_prediction_sha256':sha(out/'ensemble_probs.npy'),
        'numpy_independent_count_rank_check':'PASS','software':{'python':platform.python_version(),'numpy':np.__version__},
        'bootstrap_scope':'Known-group sampling only; training, checkpoint selection and calibration remain fixed. Not training/design uncertainty.',
        'clinical_identity_verified':False,'historical_design_and_same_corpus_development_limitations_remain':True}
    write_json(out/'analysis_provenance.json',fit_provenance)
    # Journal-neutral serif figures; figure titles stay in the separate caption file.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Serif','font.size':10,'axes.spines.top':False,
                         'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none'})
    colors=['#0072B2','#E69F00','#009E73','#D55E00','#CC79A7','#56B4E9']
    figs=out/'figures';figs.mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,2,figsize=(12,5.2))
    for ax,values,normal in [(axes[0],cm,False),(axes[1],cm/cm.sum(1,keepdims=True),True)]:
        im=ax.imshow(values,cmap='Blues',vmin=0,vmax=1 if normal else cm.max())
        for i in range(6):
            for j in range(6):ax.text(j,i,f'{values[i,j]*100:.1f}%' if normal else str(values[i,j]),ha='center',va='center',fontsize=9,color='white' if values[i,j]>(.5 if normal else cm.max()/2) else 'black')
        ax.set(xticks=range(6),yticks=range(6),xticklabels=LABELS,yticklabels=LABELS,xlabel='Predicted class',ylabel='Reference class')
        plt.setp(ax.get_xticklabels(),rotation=35,ha='right');fig.colorbar(im,ax=ax,fraction=.046,pad=.04,label='Proportion' if normal else 'Known groups')
    savefig(fig,figs,'ensemble_confusion_matrices')
    fig,axes=plt.subplots(1,2,figsize=(10.8,4.6))
    for c,color in enumerate(colors):
        fpr,tpr,_=roc_curve(y==c,ensemble[:,c]);p,r,_=precision_recall_curve(y==c,ensemble[:,c])
        axes[0].plot(fpr,tpr,color=color,lw=1.5,label=f'{LABELS[c]} ({perclass[c]["auc"]:.3f})')
        axes[1].plot(r,p,color=color,lw=1.5,label=f'{LABELS[c]} ({perclass[c]["average_precision"]:.3f})')
    axes[0].plot([0,1],[0,1],ls='--',color='.5',lw=.8)
    axes[0].set(xlabel='False-positive rate',ylabel='True-positive rate',xlim=(0,1),ylim=(0,1.02))
    axes[1].set(xlabel='Recall',ylabel='Precision',xlim=(0,1),ylim=(0,1.02))
    for ax in axes:ax.legend(fontsize=8,loc='lower left',frameon=False)
    savefig(fig,figs,'ensemble_roc_and_precision_recall')
    fig,ax=plt.subplots(figsize=(6.2,4.8))
    onehot=np.eye(6)[y]
    for m,color in zip(names,colors):
        p=probabilities[m];fpr,tpr,_=roc_curve(onehot.ravel(),p.ravel())
        ax.plot(fpr,tpr,color=color,label=f'{DISPLAY[m]} ({roc_auc_score(onehot,p,average="micro"):.3f})')
    ax.plot([0,1],[0,1],'--',color='.5',lw=.8);ax.set(xlabel='Micro-average false-positive rate',ylabel='Micro-average true-positive rate',xlim=(0,1),ylim=(0,1.02));ax.legend(fontsize=9,frameon=False)
    savefig(fig,figs,'model_roc_comparison_micro')
    fig,axes=plt.subplots(2,2,figsize=(9.4,8))
    for ax,m in zip(axes.ravel(),MODELS):
        ax.plot([0,1],[0,1],'--',color='.5',lw=.8)
        for label,p,color in [('Raw',probabilities[m],colors[0]),('Scaled',probability_scaled[m],colors[1]),('Ensemble',ensemble,colors[2])]:
            _,bins=ece(p/p.sum(1,keepdims=True),y);ax.plot([b['confidence'] for b in bins],[b['accuracy'] for b in bins],'o-',ms=4,label=label,color=color)
        ax.set(title=DISPLAY[m],xlabel='Mean confidence',ylabel='Observed accuracy',xlim=(0,1),ylim=(0,1));ax.legend(fontsize=8,frameon=False)
    savefig(fig,figs,'calibration_reliability_comparison')
    fig,ax=plt.subplots(figsize=(6.5,5.3));im=ax.imshow(agreement,cmap='Blues',vmin=.5,vmax=1)
    for i in range(5):
        for j in range(5):ax.text(j,i,f'{agreement[i,j]*100:.1f}%',ha='center',va='center',color='white' if agreement[i,j]>.8 else 'black')
    ax.set(xticks=range(5),yticks=range(5),xticklabels=[DISPLAY[m] for m in names],yticklabels=[DISPLAY[m] for m in names]);plt.setp(ax.get_xticklabels(),rotation=35,ha='right');fig.colorbar(im,ax=ax,label='Prediction agreement')
    savefig(fig,figs,'model_prediction_agreement')
    fig,axes=plt.subplots(2,2,figsize=(9.5,7.2));training=pd.DataFrame(training)
    for ax,m in zip(axes.ravel(),MODELS):
        for f in range(1,6):
            h=training[(training.model==m)&(training.fold==f)];ax.plot(h.epoch,h.selection_logloss,lw=1,label=f'Fold {f}')
        ax.axvline(3.5,color='.6',ls='--',lw=.8);ax.set(title=DISPLAY[m],xlabel='Epoch',ylabel='Selection log loss');ax.legend(fontsize=7,frameon=False)
    savefig(fig,figs,'selection_learning_curves')
    summary=pd.DataFrame(verified).set_index('model');tables=[];pct=lambda v:f'{100*v:.2f}';num=lambda v:f'{v:.4f}'
    clinical=pd.read_csv(inputs/'frozen/clinical_images.csv')
    rows=[]
    for c,label in enumerate(LABELS):
        a=clinical[clinical.label.eq(c)];rows.append([label,str(len(a)),str(a.patient_id.nunique()),str(a.patient_cluster.nunique())])
    rows.append(['Total','755','524','520'])
    tables.append(('Dataset composition',['Class','Clinical images','Filename IDs','Known groups'],rows,
        'Images are counted once in the clinical inventory. Known groups merge the four confirmed cross-ID duplicate links. Neither filename identifiers nor known groups establish independently verified clinical identity.',[2.2,1.7,1.7,1.7]))
    split=pd.read_csv(inputs/'frozen/split_summary.csv')
    rows=[[str(int(r.fold)),str(int(r.train_images)),str(int(r.augmented_training_entries)),str(int(r.selection_groups)),str(int(r.calibration_groups)),str(int(r.test_images)),str(int(r.test_groups))] for r in split.itertuples()]
    tables.append(('Corrected data partitions',['Fold','Training images','Training augments','Selection groups','Calib. groups','Test images','Test groups'],rows,
        'The official test image sets are preserved. Lower-priority conflicting roles are excluded using test > calibration > selection > training. Released augmentation entries retain the frozen sampling multiplicities. Test groups recur across sheets.',[.5,1.1,1.3,1.1,1.1,1.1,1.1]))
    rows=[]
    for m in names:
        a=summary.loc[m];rows.append([DISPLAY[m],f'{int(a.correct)}/215',f'{pct(a.accuracy)}\n({pct(a.accuracy_ci_low)}-{pct(a.accuracy_ci_high)})',num(a.macro_f1),num(a.macro_auc)])
    tables.append(('Pooled test performance',['Model','Correct / n','Accuracy %\n95% CI','Macro F1','Macro AUC'],rows,
        f'Backbones use raw probabilities; the ensemble uses calibration-only temperatures and weights. Accuracy CIs are Wilson intervals. AUC is macro one-versus-rest. All models use the same 215 known groups. Ensemble F1 bootstrap CI {num(report["conditional_bootstrap_95_ci"]["macro_f1"][0])}-{num(report["conditional_bootstrap_95_ci"]["macro_f1"][1])}; AUC CI {num(report["conditional_bootstrap_95_ci"]["macro_auc"][0])}-{num(report["conditional_bootstrap_95_ci"]["macro_auc"][1])}.',[2,1,1.8,1.25,1.25]))
    rows=[[r['class'],str(r['support']),num(r['precision']),num(r['recall']),num(r['specificity']),num(r['f1']),num(r['auc'])] for r in perclass]
    tables.append(('Ensemble performance by class',['Class','n','Precision','Recall','Specificity','F1','AUC'],rows,
        'One-versus-rest class metrics. Detailed Wilson intervals for precision, recall and specificity are in the companion CSV. Macro averages weight the six classes equally.',[1.5,.5,1.06,1.06,1.06,1.06,1.06]))
    rows=[]
    for a in calibrations:rows.append([DISPLAY[a['model']],a['stage'].replace('temperature_scaled','Scaled').replace('calibrated','Ensemble').capitalize(),pct(a['ece']),num(a['log_loss']),num(a['brier_multiclass'])])
    tables.append(('Test calibration summary',['Model','Stage','ECE %','Log loss','Brier score'],rows,
        'ECE uses ten equal-width top-confidence bins. The multiclass Brier score is the mean sum of squared probability errors across six classes and is not divided by six. Test metrics assess parameters fitted on calibration groups. Smaller values are better.',[2.1,1.5,1.1,1.3,1.3]))
    rows=[[DISPLAY[m],f'{report["temperatures"][m]:.6f}',f'{report["weights"][m]:.9g}'] for m in MODELS]
    tables.append(('Calibration parameters',['Backbone','Temperature','Mixture weight'],rows,
        'Temperatures and positive mixture weights are fitted only on 138 calibration groups after excluding all 79 groups overlapping the pooled test cohort. Weights sum to one and describe the calibration fit; they do not independently establish backbone importance or added benefit.',[3.1,2.1,2.1]))
    paired=pd.read_csv(out/'paired_accuracy_comparisons.csv')
    rows=[[DISPLAY[r.model],str(r.only_ensemble_correct),str(r.only_backbone_correct),f'{r.exact_mcnemar_p:.5g}',f'{r.holm_p:.5g}'] for r in paired.itertuples()]
    tables.append(('Paired accuracy comparisons',['Raw backbone','Ensemble only correct','Backbone only correct','Exact p','Holm p'],rows,
        'Exact two-sided McNemar tests use paired group correctness. Holm adjustment covers the four ensemble-versus-backbone comparisons. The same held-out groups are evaluated in every comparison.',[2.1,1.45,1.45,1.15,1.15]))
    rows=[[DISPLAY[r.model],num(r.macro_f1_difference),f'{r.macro_f1_difference_ci_low:.4f} to {r.macro_f1_difference_ci_high:.4f}',num(r.macro_auc_difference),f'{r.macro_auc_difference_ci_low:.4f} to {r.macro_auc_difference_ci_high:.4f}'] for r in paired.itertuples()]
    tables.append(('Paired discrimination differences',['Raw backbone','F1 difference','F1 difference\n95% CI','AUC difference','AUC difference\n95% CI'],rows,
        'Ensemble minus raw backbone. Percentile intervals use 2000 paired known-group bootstrap samples with seed 42. Fits remain fixed; intervals exclude training, design and calibration-estimation uncertainty and are not multiplicity-adjusted.',[1.7,1.1,1.8,1.05,1.65]))
    rows=[['Clinical inventory','755 images'],['Official test appearances','364 images'],['Distinct official test images','297 images'],['Distinct-image coverage',f'{len(covered)}/755 ({100*len(covered)/755:.2f}%)'],['Pooled test cohort','215 known groups'],['Calibration candidates','217 known groups'],['Calibration groups excluded','79 known groups'],['Calibration fit cohort','138 known groups']]
    tables.append(('Coverage and calibration cohorts',['Quantity','Count'],rows,
        '364/755 is an appearance ratio, not unique coverage. The corrected unique-image coverage is 297/755 (39.34%). Known-group training and evaluation separation is checked within each official fold. These results retain same-corpus development and unknown-identity limitations.',[4.5,2.8]))
    write_json(out/'publication_tables.json',tables)
    export_tables(out,tables)
    algorithm='''Algorithm 1 Corrected official fold ensemble workflow
1. Verify the 755-image inventory and merge the four confirmed cross-ID duplicate links into 520 known groups.
2. Preserve each official test image set and exclude lower-priority role conflicts using test > calibration > selection > training.
3. Train each of four backbones for 25 epochs on five frozen folds and retain two checkpoints by selection-group log loss.
4. Average four view probabilities, geometrically aggregate images within known groups and average the two retained checkpoints.
5. Arithmetic-average repeated known-group appearances across official test or calibration sheets.
6. Remove every pooled test group from the pooled calibration candidates, leaving 138 eligible calibration groups.
7. Fit four scalar temperatures and positive mixture weights using only calibration log loss.
8. Evaluate the fixed ensemble on 215 known groups and export discrimination, calibration, paired comparisons and uncertainty.
9. Verify selected weights reproduce saved test predictions before producing qualitative ensemble-score Grad-CAM maps.
'''
    (out/'Algorithm_1.txt').write_text(algorithm,encoding='utf-8')
    captions='''# Figure captions

1. Ensemble confusion matrices. Left: counts; right: row percentages. Support comprises 215 known groups.
2. Ensemble ROC and precision-recall curves. Legends give class AUC and average precision, respectively.
3. Micro-average ROC comparison. This pooled binary construction differs from the headline macro one-versus-rest AUC.
4. Reliability of raw, scaled and ensemble predictions. Ten equal-width confidence bins are used; empty bins are omitted.
5. Prediction agreement among backbones and ensemble. Percentages use the same 215 known groups.
6. Checkpoint selection learning curves. The dashed line marks the transition after three head-only epochs.

The ensemble Grad-CAM notebook adds a separately labelled qualitative gallery after weight verification.
'''
    (out/'Figure_captions.md').write_text(captions,encoding='utf-8')
    best=summary.loc[list(MODELS)].accuracy.idxmax()
    note=f'''# Verified corrected Protocol B results

All 20 jobs completed 25 epochs. The fixed ensemble classified {report['ensemble']['correct']}/215 groups correctly ({100*report['ensemble']['accuracy']:.2f}%), with macro F1 {report['ensemble']['macro_f1']:.4f}, macro AUC {report['ensemble']['macro_auc']:.4f}, ECE {100*report['ensemble']['ece']:.2f}% and log loss {report['ensemble']['log_loss']:.4f}.

The raw backbone with the highest observed accuracy is {DISPLAY[best]} ({int(summary.loc[best,'correct'])}/215). Use the paired tests and difference intervals before claiming ensemble superiority.

The outputs verify recorded known-group separation and remove pooled calibration/test overlaps. They do not establish independently verified clinical identities, external clinical validity, or absence of unknown duplicates. Protocol A remains the separate 755-image evaluation; these official-fold results must be labelled Protocol B. The existing controlled 128-pixel Table 5 ablation remains a separate experiment.

The numerical pipeline does not resolve author declarations, external validation or a manuscript-wide rewrite. No old incomplete-fold result or old 216-identifier Protocol B score is used here.

Ensemble Grad-CAM requires all 40 selected weights. Maps describe the fixed ensemble score with the rest of the known-group context held fixed. Heatmap intensities are not lesion measurements or clinical explanations.
'''
    (out/'RESULTS_READ_ME.md').write_text(note,encoding='utf-8')

def make_bundle(out):
    target=out.parent/'PB_Final_Publication_Outputs.zip'
    with ZipFile(target,'w',ZIP_DEFLATED) as z:
        for p in sorted(out.rglob('*')):
            if p.is_file() and not any(x in {'extracted_results','result_index','.plot_cache'} for x in p.relative_to(out).parts):
                z.write(p,'PB_Final_Publication/'+p.relative_to(out).as_posix())
    return target

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root',type=Path,default=Path('/kaggle/input'))
    parser.add_argument('--work',type=Path,default=Path('/kaggle/working'))
    parser.add_argument('--expected-final-sha',required=True)
    parser.add_argument('--bootstrap',type=int,default=2000)
    parser.add_argument('--extend-existing',action='store_true',help='Local use after a verified frozen merge only.')
    args=parser.parse_args(argv);out=args.work/'PB_Final_Publication';out.mkdir(parents=True,exist_ok=True)
    os.environ['MPLCONFIGDIR']=str(args.work/'.plot_cache')
    inputs=prepare_inputs(args.input_root,args.work,args.expected_final_sha)
    if not args.extend_existing:
        from merge_corrected import main as merge
        merge(['--input-root',str(inputs),'--output-root',str(out),'--expected-manifest-sha',FROZEN_SHA,'--bootstrap',str(args.bootstrap)])
    if json.loads((out/'report.json').read_text())['input_manifest_sha256']!=FROZEN_SHA:raise ValueError('Existing report belongs to another protocol.')
    extend_outputs(inputs,out,args.bootstrap)
    print('FINAL ANALYSIS PASS',flush=True)
    print((out/'RESULTS_READ_ME.md').read_text(),flush=True)
    print('Download:',make_bundle(out),flush=True)

if __name__=='__main__':main()
