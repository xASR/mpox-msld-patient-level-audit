"""Editable journal-neutral tables and vector publication figures from saved results."""
from pathlib import Path
import os
import csv
import json
import hashlib
import numpy as np
import pandas as pd
_cache = Path(os.environ.get('MPLCONFIGDIR', Path.cwd() / '.publication_plot_cache'))
_cache.mkdir(parents=True, exist_ok=True)
os.environ['MPLCONFIGDIR'] = str(_cache)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.section import WD_ORIENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, PageBreak, Spacer
from study_statistics import DISPLAY, NAMES, wilson


def pct(value):
    return f'{100 * float(value):.2f}'


def number(value):
    return f'{float(value):.4f}'


def tables_from_results(root, context, report):
    root = Path(root)
    clinical = pd.read_csv(root / 'clinical_images.csv')
    tables = []
    has_clusters = 'patient_cluster' in clinical
    rows = []
    for cls, name in enumerate(NAMES):
        group = clinical.loc[clinical.label == cls]
        row = [name, str(len(group)), str(group.patient_id.nunique())]
        if has_clusters:
            row.append(str(group.patient_cluster.nunique()))
        rows.append(row)
    total = ['Total', str(len(clinical)), str(clinical.patient_id.nunique())]
    if has_clusters:
        total.append(str(clinical.patient_cluster.nunique()))
    rows.append(total)
    tables.append(('Dataset composition', ['Class', 'Clinical images', 'Filename IDs'] + (['Patient clusters'] if has_clusters else []), rows,
                   'Keywords.jpg is excluded. Fold copies are counted once per original image. Filename IDs are not independently verified clinical identities. Known duplicate pairs are grouped for Protocol A.'))
    split = pd.read_csv(root / 'split_summary.csv')
    rows = [[str(int(r.fold)), str(int(r.train_images)), str(int(r.selection_clusters)), str(int(r.calibration_clusters)),
             str(int(r.test_images)), str(int(r.test_clusters))] for r in split.itertuples(index=False)]
    footnote = ('Training uses original images with online augmentation. Selection chooses two checkpoints; calibration fits temperatures and ensemble weights. Test groups are used once.' if context['protocol'] == 'A' else
                'Training counts refer to original images; released training augmentations are additional. Official test groups can recur across sheets. The pooled test set has 216 identifiers; 79 calibration/test overlaps are removed, leaving 138 calibration identifiers.')
    tables.append(('Data partitions', ['Fold', 'Training images', 'Selection groups', 'Calibration groups', 'Test images', 'Test groups'], rows, footnote))
    for variant, key in (('Raw backbone comparison', 'singles_raw'), ('Temperature scaled backbone comparison', 'singles_tempscaled')):
        rows = []
        for model, label in [*DISPLAY.items(), ('ensemble', 'Calibrated ensemble')]:
            m = report['ens_fixed'] if model == 'ensemble' else report[key][model]
            correct = report['correct'][model]
            lo, hi = wilson(correct, report['n_test'])
            rows.append([label, f"{correct}/{report['n_test']}", f'{pct(m["acc"])} ({pct(lo)}-{pct(hi)})',
                         number(m['f1']), number(m['auc']), pct(m['ece']), number(m['ll'])])
        ci = report['boot_ci']
        footnote = ('Accuracy intervals are Wilson 95% CIs. F1 and AUC are macro averages; AUC uses one-versus-rest. ECE uses 10 equal-width bins. '
                    f"The ensemble bootstrap 95% CIs are {number(ci['f1'][0])}-{number(ci['f1'][1])} for F1 and {number(ci['auc'][0])}-{number(ci['auc'][1])} for AUC. "
                    f"Bootstrap resamples: {report['boot_n']}; fitted models and calibration parameters remain fixed. Groups denote {context['evaluation_unit']}.")
        tables.append((variant, ['Model', 'Correct / n', 'Accuracy % (95% CI)', 'Macro F1', 'Macro AUC', 'ECE %', 'Log loss'], rows, footnote))
    rows = [[name, str(int(report['per_class'][name]['support'])), number(report['per_class'][name]['precision']),
             number(report['per_class'][name]['recall']), number(report['per_class'][name]['f1-score']),
             number(report['per_class_auc'][name])] for name in NAMES]
    tables.append(('Ensemble performance by class', ['Class', 'Support', 'Precision', 'Recall', 'F1', 'AUC'], rows,
                   'Support is the number of evaluated groups. The class order is Healthy, Cowpox, Mpox, Measles, Chickenpox and HFMD.'))
    ordered = sorted(DISPLAY, key=lambda m: report['mcnemar'][m]['p'])
    adjusted, running = {}, 0.0
    for rank, model in enumerate(ordered):
        running = max(running, (len(ordered) - rank) * report['mcnemar'][model]['p'])
        adjusted[model] = min(1.0, running)
    rows = []
    for model, label in DISPLAY.items():
        mm = report['mcnemar'][model]
        f1_ci, auc_ci = report['boot_df1'][model], report['boot_dauc'][model]
        rows.append([label, str(mm['ens_only']), str(mm['single_only']), number(mm['p']), number(adjusted[model]),
                     f'{number(f1_ci[1])} to {number(f1_ci[2])}', f'{number(auc_ci[1])} to {number(auc_ci[2])}'])
    tables.append(('Paired comparisons with the ensemble', ['Backbone', 'Ensemble only correct', 'Backbone only correct', 'Exact p', 'Holm p', 'F1 difference 95% CI', 'AUC difference 95% CI'], rows,
                   'Exact two-sided McNemar tests compare correctness on the same groups. Holm adjustment covers the four comparisons. Difference intervals are paired bootstrap ensemble-minus-raw-backbone intervals; they are not adjusted for multiplicity.'))
    if context['protocol'] == 'B':
        rows = [[label, f'{report["temps"][m][0]:.6f}', f'{report["weights"][m]:.9g}'] for m, label in DISPLAY.items()]
        footnote = 'Parameters are fitted on 138 overlap-filtered calibration identifiers. Positive transformer weights are near zero; the results do not establish useful transformer contributions or superiority over ConvNeXt-Tiny.'
        header = ['Backbone', 'Temperature', 'Ensemble weight']
    else:
        rows = [[str(fold), DISPLAY[m], f'{r["temperatures"][m]:.6f}', f'{r["weights"][m]:.9g}']
                for fold, r in report['calibration_by_fold'].items() for m in DISPLAY]
        footnote = 'Each outer fold has its own calibration patients, temperatures and ensemble weights. No parameter is fitted on the pooled outer-test predictions.'
        header = ['Outer fold', 'Backbone', 'Temperature', 'Ensemble weight']
    tables.append(('Calibration parameters', header, rows, footnote))
    sensitivity = root / context['analysis_folder'] / 'duplicate_sensitivity.json'
    if sensitivity.is_file():
        secondary = json.loads(sensitivity.read_text())
        rows = []
        for label, n, correct, m in [('Primary', report['n_test'], report['correct']['ensemble'], report['ens_fixed']),
                                     ('Duplicate sensitivity', secondary['remaining_patients'], secondary['correct'], secondary['metrics'])]:
            rows.append([label, str(n), str(correct), pct(m['acc']), number(m['f1']), number(m['auc']), pct(m['ece']), number(m['ll'])])
        tables.append(('Known duplicate sensitivity', ['Analysis', 'n', 'Correct', 'Accuracy %', 'Macro F1', 'Macro AUC', 'ECE %', 'Log loss'], rows,
                       'The sensitivity analysis excludes CHP_26, HFMD_130, HEALTHY_21 and HEALTHY_77 from evaluation. Temperatures and weights remain fixed.'))
    return tables


def export_tables(root, context, tables):
    table_dir = Path(root) / 'publication/tables'
    table_dir.mkdir(parents=True, exist_ok=True)
    title = f"Protocol {context['protocol']} publication tables"
    doc = Document()
    doc.core_properties.author = ''
    doc.core_properties.last_modified_by = ''
    section = doc.sections[0]
    section.orientation = WD_ORIENT.LANDSCAPE
    section.page_width, section.page_height = Inches(11.69), Inches(8.27)
    section.top_margin = section.bottom_margin = Inches(.55)
    section.left_margin = section.right_margin = Inches(.6)
    normal = doc.styles['Normal']
    normal.font.name, normal.font.size = 'Times New Roman', Pt(10)
    normal.paragraph_format.space_after = Pt(4)
    doc.styles['Title'].font.name = 'Times New Roman'
    doc.styles['Title'].font.size = Pt(15)
    doc.styles['Title'].font.color.rgb = RGBColor(0, 0, 0)
    pdf_items = []
    pdf_title = ParagraphStyle('Caption', fontName='Times-Bold', fontSize=13, leading=16, spaceAfter=10)
    pdf_body = ParagraphStyle('Body', fontName='Times-Roman', fontSize=9, leading=11)
    pdf_head = ParagraphStyle('Header', parent=pdf_body, fontName='Times-Bold')
    captions = []
    for i, (caption, header, rows, note) in enumerate(tables, 1):
        if i > 1:
            doc.add_page_break()
            pdf_items.append(PageBreak())
        caption = f'Table {i} {caption}'
        doc.add_paragraph(caption, 'Title')
        table = doc.add_table(rows=1, cols=len(header))
        table.autofit = False
        widths = [1.8] + [(10.35 - 1.8) / (len(header) - 1)] * (len(header) - 1)
        for column, width in zip(table.columns, widths):
            column.width = Inches(width)
        for cell, text, width in zip(table.rows[0].cells, header, widths):
            cell.width = Inches(width)
            cell.text = str(text)
            for run in cell.paragraphs[0].runs:
                run.bold = True
        repeat = OxmlElement('w:tblHeader')
        table.rows[0]._tr.get_or_add_trPr().append(repeat)
        borders = OxmlElement('w:tblBorders')
        for edge in ('top', 'bottom', 'left', 'right', 'insideH', 'insideV'):
            element = OxmlElement('w:' + edge)
            element.set(qn('w:val'), 'single' if edge in ('top', 'bottom') else 'nil')
            element.set(qn('w:sz'), '8')
            borders.append(element)
        table._tbl.tblPr.append(borders)
        for values in rows:
            cells = table.add_row().cells
            for cell, text, width in zip(cells, values, widths):
                cell.width = Inches(width)
                cell.text = str(text)
        for row_index, row in enumerate(table.rows):
            no_split = OxmlElement('w:cantSplit')
            row._tr.get_or_add_trPr().append(no_split)
            for cell in row.cells:
                for p in cell.paragraphs:
                    p.paragraph_format.space_after = Pt(2)
                    for run in p.runs:
                        run.font.size = Pt(9 if len(rows) > 16 else 10)
                if row_index == 0:
                    cell_borders = OxmlElement('w:tcBorders')
                    edge = OxmlElement('w:bottom')
                    edge.set(qn('w:val'), 'single')
                    edge.set(qn('w:sz'), '6')
                    cell_borders.append(edge)
                    cell._tc.get_or_add_tcPr().append(cell_borders)
        doc.add_paragraph(note)
        width = landscape(A4)[0] - 72
        proportions = np.array([1.65] + [1.0] * (len(header) - 1))
        col_widths = (proportions / proportions.sum() * width).tolist()
        data = [[Paragraph(str(v), pdf_head) for v in header]] + [[Paragraph(str(v), pdf_body) for v in row] for row in rows]
        pdf_table = Table(data, colWidths=col_widths, repeatRows=1, hAlign='LEFT')
        pdf_table.setStyle(TableStyle([('LINEABOVE', (0, 0), (-1, 0), .8, colors.black),
                                      ('LINEBELOW', (0, 0), (-1, 0), .6, colors.black),
                                      ('LINEBELOW', (0, -1), (-1, -1), .8, colors.black),
                                      ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                                      ('TOPPADDING', (0, 0), (-1, -1), 4),
                                      ('BOTTOMPADDING', (0, 0), (-1, -1), 4)]))
        pdf_items += [Paragraph(caption, pdf_title), pdf_table, Spacer(1, 10), Paragraph(note, pdf_body)]
        name = f'Table_{i:02}.csv'
        with (table_dir / name).open('w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)
        captions.append(f'{caption}\n\n{note}')
    doc_path = table_dir / f"Protocol_{context['protocol']}_Publication_Tables.docx"
    pdf_path = doc_path.with_suffix('.pdf')
    doc.save(doc_path)
    SimpleDocTemplate(str(pdf_path), pagesize=landscape(A4), leftMargin=36, rightMargin=36,
                      topMargin=32, bottomMargin=32, title=title, author='').build(pdf_items)
    (table_dir / 'Table_notes.md').write_text('\n\n'.join(captions) + '\n', encoding='utf-8')
    return doc_path, pdf_path


def export_figures(root, context, report):
    root = Path(root)
    destination = root / 'publication/figures'
    destination.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    captions = []

    def save(fig, name, caption):
        for extension in ('pdf', 'svg', 'png'):
            fig.savefig(destination / f'{name}.{extension}', dpi=600, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        captions.append(f'{name}\n\n{caption}')

    matrix = np.asarray(report['confusion'])
    fig, ax = plt.subplots(figsize=(7.2, 6.2), layout='constrained')
    image = ax.imshow(matrix, cmap='Blues', vmin=0)
    for (i, j), value in np.ndenumerate(matrix):
        ax.text(j, i, str(value), ha='center', va='center', color='white' if value > matrix.max() * .55 else 'black')
    ax.set(xticks=range(6), yticks=range(6), xticklabels=NAMES, yticklabels=NAMES,
           xlabel='Predicted class', ylabel='Reference class')
    plt.setp(ax.get_xticklabels(), rotation=35, ha='right')
    fig.colorbar(image, ax=ax, shrink=.8, label='Number of evaluated groups')
    save(fig, 'Figure_01_Confusion_Matrix', f"Ensemble confusion counts for {report['n_test']} evaluation units ({context['evaluation_unit']}). Rows are reference classes; columns are predicted classes.")
    roc = json.loads((root / context['analysis_folder'] / 'roc_points.json').read_text())
    fig, ax = plt.subplots(figsize=(6.6, 5.7), layout='constrained')
    for name in NAMES:
        ax.plot(roc[name]['fpr'], roc[name]['tpr'], lw=1.6, label=f"{name} ({report['per_class_auc'][name]:.3f})")
    ax.plot([0, 1], [0, 1], '--', color='.6', lw=1)
    ax.set(xlabel='False positive rate', ylabel='True positive rate', xlim=(0, 1), ylim=(0, 1.01))
    ax.legend(title='Class (AUC)', loc='lower right', fontsize=9, frameon=False)
    save(fig, 'Figure_02_Class_ROC', 'One-versus-rest ROC curves of the calibrated ensemble; legend values are class AUCs.')
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.3), layout='constrained')
    for ax, key, label in zip(axes, ('reliability_ens', 'reliability_convnext_raw'), ('Calibrated ensemble', 'ConvNeXt-Tiny raw')):
        bins = report[key]
        ax.plot([0, 1], [0, 1], '--', color='.6', lw=1)
        ax.plot([r['conf'] for r in bins], [r['acc'] for r in bins], 'o-', color='#24537b', lw=1.5)
        ax.set(title=label, xlabel='Mean confidence', ylabel='Observed accuracy', xlim=(0, 1), ylim=(0, 1.02))
    save(fig, 'Figure_03_Calibration', 'Reliability diagrams use ten equal-width confidence bins; empty bins are omitted. ECE and numerical bin counts are provided with the analysis.')
    labels, values, errors = [], [], []
    for model, name in [*DISPLAY.items(), ('ensemble', 'Ensemble')]:
        correct = report['correct'][model]
        value = correct / report['n_test']
        lo, hi = wilson(correct, report['n_test'])
        labels.append(name)
        values.append(100 * value)
        errors.append([100 * (value - lo), 100 * (hi - value)])
    fig, ax = plt.subplots(figsize=(8.1, 5.1), layout='constrained')
    ax.bar(range(5), values, color=['#809bb0'] * 4 + ['#24537b'], yerr=np.array(errors).T, capsize=4)
    ax.set(xticks=range(5), xticklabels=labels, ylabel='Accuracy (%)', ylim=(0, 105))
    plt.setp(ax.get_xticklabels(), rotation=18, ha='right')
    for i, value in enumerate(values):
        ax.text(i, 3, f'{value:.2f}', ha='center', color='white', fontweight='bold')
    save(fig, 'Figure_04_Model_Accuracy', 'Raw backbone and calibrated-ensemble accuracies on the same evaluation groups. Error bars are Wilson 95% confidence intervals; paired comparisons are reported separately.')
    if context['protocol'] == 'B':
        fig, ax = plt.subplots(figsize=(7.5, 4.1), layout='constrained')
        w = [report['weights'][m] * 100 for m in DISPLAY]
        ax.barh(range(4), w, color='#24537b')
        ax.set(yticks=range(4), yticklabels=list(DISPLAY.values()), xlabel='Ensemble weight (%)', xlim=(0, 82))
        ax.invert_yaxis()
        for i, value in enumerate(w):
            ax.text(value + 1, i, f'{value:.4g}%', va='center')
        caption = 'Weights fitted on 138 calibration identifiers. Near-zero transformer weights are retained and shown numerically.'
    else:
        values = np.array([[row['weights'][m] for m in DISPLAY] for row in report['calibration_by_fold'].values()])
        fig, ax = plt.subplots(figsize=(7.5, 4.8), layout='constrained')
        image = ax.imshow(values, cmap='Blues', vmin=0, vmax=1, aspect='auto')
        for (i, j), value in np.ndenumerate(values):
            ax.text(j, i, f'{value:.3f}', ha='center', va='center', color='white' if value > .55 else 'black')
        ax.set(xticks=range(4), xticklabels=list(DISPLAY.values()), yticks=range(5), yticklabels=range(1, 6), ylabel='Outer fold')
        plt.setp(ax.get_xticklabels(), rotation=18, ha='right')
        fig.colorbar(image, ax=ax, label='Ensemble weight')
        caption = 'Fold-specific weights fitted only on each outer fold calibration groups.'
    save(fig, 'Figure_05_Ensemble_Weights', caption)
    fig, ax = plt.subplots(figsize=(9, 4.5), layout='constrained')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')
    if context['protocol'] == 'B':
        boxes = [(1, 4.8, '755 clinical images\n524 filename IDs'), (5.8, 4.8, 'Five official data sheets\nFour completed backbones'),
                 (1, 2.7, '217 calibration IDs\nRemove 79 test overlaps'), (5.8, 2.7, '216 test IDs\nHeld-out model predictions'),
                 (1, .6, '138 calibration IDs\nFit temperatures and weights'),
                 (5.8, .6, '216 test IDs\n190 correct (87.96%)')]
        arrows = [(3.8, 5.15, 5.7, 5.15), (6, 4.7, 3.6, 3.6), (7.2, 4.7, 7.2, 3.6),
                  (2.4, 2.6, 2.4, 1.5), (3.8, 1.05, 5.7, 1.05), (7.2, 2.6, 7.2, 1.5)]
        caption = 'Protocol B data flow. Clinical-image counts, candidate calibration identifiers, overlap exclusions and the final evaluated test cohort are distinguished.'
    else:
        boxes = [(1, 4.8, '755 clinical images\n524 filename IDs'), (5.8, 4.8, '520 groups after linking\nknown duplicate identifiers'),
                 (1, 2.7, 'Inner training, selection\nand calibration partitions'), (5.8, 2.7, 'Five outer test folds\nEach group tested once'),
                 (1, .6, 'Fit models and calibration\nWithin each outer fold'),
                 (5.8, .6, '520 held-out group predictions\nNo global test-based fitting')]
        arrows = [(3.8, 5.15, 5.7, 5.15), (6, 4.7, 3.6, 3.6), (7.2, 4.7, 7.2, 3.6),
                  (2.4, 2.6, 2.4, 1.5), (3.8, 1.05, 5.7, 1.05), (7.2, 2.6, 7.2, 1.5)]
        caption = 'Revised Protocol A data flow. Known duplicate-aware groups remain disjoint within each fold; model selection and calibration use only the inner development partitions.'
    for x, y, text in boxes:
        ax.add_patch(FancyBboxPatch((x, y), 2.8, .9, boxstyle='round,pad=.08', edgecolor='#24537b', facecolor='#f1f5f8'))
        ax.text(x + 1.4, y + .45, text, ha='center', va='center', fontsize=9)
    for x1, y1, x2, y2 in arrows:
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1), arrowprops={'arrowstyle': '->', 'color': '#24537b'})
    save(fig, 'Figure_06_Data_Flow', caption)
    (destination / 'Figure_captions.md').write_text('\n\n'.join(captions) + '\n', encoding='utf-8')


def generate_publication(root):
    root = Path(root)
    context = json.loads((root / 'publication_context.json').read_text())
    report = json.loads((root / context['analysis_folder'] / 'report.json').read_text())
    tables = tables_from_results(root, context, report)
    doc, pdf = export_tables(root, context, tables)
    export_figures(root, context, report)
    publication = root / 'publication'
    records = {p.relative_to(publication).as_posix(): {'bytes': p.stat().st_size, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
               for p in sorted(publication.rglob('*')) if p.is_file() and p.name != 'publication_manifest.json'}
    (publication / 'publication_manifest.json').write_text(json.dumps({'protocol': context['protocol'],
        'tables': len(tables), 'figures': 6, 'style': 'Journal-neutral editable tables with three horizontal rules; vector PDF/SVG and 600 dpi PNG figures.',
        'files': records}, indent=2), encoding='utf-8')
    print(f'Publication exports ready: {len(tables)} tables, editable Word table document, table PDF, six figures in PDF/SVG/600 dpi PNG.', flush=True)
    return {'tables': len(tables), 'figures': 6, 'docx': str(doc), 'pdf': str(pdf)}
