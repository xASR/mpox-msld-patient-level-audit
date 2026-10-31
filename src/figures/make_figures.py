"""Publication figures from the merge analysis. Usage: python make_figures.py results/protocol_b_merge figures"""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

A, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
r = json.load(open(os.path.join(A, 'report.json')))
roc = json.load(open(os.path.join(A, 'roc_points.json')))
NAMES = ['Healthy', 'Cowpox', 'Mpox', 'Measles', 'Chickenpox', 'HFMD']
BLUE, GREY, DARK = '#2a6fdb', '#9aa0a6', '#202124'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.edgecolor': '#5f6368', 'axes.labelcolor': DARK,
                     'xtick.color': '#5f6368', 'ytick.color': '#5f6368'})

# Confusion matrix
cm = np.array(r['confusion'])
fig, ax = plt.subplots(figsize=(5.6, 4.8), dpi=300)
norm = cm / cm.sum(1, keepdims=True)
ax.imshow(norm, cmap='Blues', vmin=0, vmax=1)
for i in range(6):
    for j in range(6):
        if cm[i, j]:
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=10,
                    color='white' if norm[i, j] > 0.6 else DARK)
ax.set_xticks(range(6), NAMES, rotation=35, ha='right'); ax.set_yticks(range(6), NAMES)
ax.set_xlabel('Predicted class'); ax.set_ylabel('True class')
ax.set_title('Most errors are chickenpox predicted as mpox', fontsize=11, loc='left', color=DARK)
for s in ax.spines.values(): s.set_visible(False)
fig.tight_layout(); fig.savefig(os.path.join(OUT, 'fig3_confusion_matrix.png')); plt.close(fig)

# ROC per class
fig, ax = plt.subplots(figsize=(5.6, 4.8), dpi=300)
ax.plot([0, 1], [0, 1], ls='--', lw=0.8, color=GREY)
auc = r['per_class_auc']
order = sorted(NAMES, key=lambda n: auc[n])
for n in order:
    hl = n in ('Mpox', 'Chickenpox')
    ax.step(roc[n]['fpr'], roc[n]['tpr'], where='post', lw=1.8 if hl else 1.1,
            color=BLUE if n == 'Mpox' else ('#e37400' if n == 'Chickenpox' else GREY),
            alpha=1 if hl else 0.8, label=f'{n} (AUC {auc[n]:.3f})')
ax.set_xlim(-0.01, 1); ax.set_ylim(0, 1.01)
ax.set_xlabel('False positive rate'); ax.set_ylabel('True positive rate')
ax.set_title('Mpox and chickenpox have the lowest one-vs-rest AUC', fontsize=11, loc='left', color=DARK)
ax.legend(frameon=False, fontsize=8.5, loc='lower right')
fig.tight_layout(); fig.savefig(os.path.join(OUT, 'fig4_roc_per_class.png')); plt.close(fig)

# Reliability diagram: ensemble vs raw ConvNeXt
fig, ax = plt.subplots(figsize=(5.6, 4.8), dpi=300)
ax.plot([0, 1], [0, 1], ls='--', lw=0.8, color=GREY, label='Perfect calibration')
for key, col, lab in [('reliability_convnext_raw', GREY, f"ConvNeXt-Tiny before scaling (ECE {r['singles_raw']['convnext_tiny']['ece']:.3f})"),
                      ('reliability_ens', BLUE, f"Ensemble after scaling (ECE {r['ens_fixed']['ece']:.3f})")]:
    rows = r[key]
    x = [b['conf'] for b in rows]; y = [b['acc'] for b in rows]; s = [b['n'] for b in rows]
    ax.scatter(x, y, s=[12 + 2.2 * v for v in s], color=col, alpha=0.85, edgecolor='white', lw=0.6, label=lab, zorder=3)
ax.set_xlim(0.2, 1.0); ax.set_ylim(0, 1.03)
ax.set_xlabel('Mean confidence in bin'); ax.set_ylabel('Accuracy in bin')
ax.set_title('Temperature scaling removes most of the underconfidence', fontsize=11, loc='left', color=DARK)
ax.text(0.99, 0.03, 'Marker area = patients in bin (186 test patients).\nBins with fewer than 10 patients are noisy.', fontsize=7.5, color='#5f6368', ha='right')
ax.legend(frameon=False, fontsize=8.5, loc='upper left')
fig.tight_layout(); fig.savefig(os.path.join(OUT, 'fig5_reliability.png')); plt.close(fig)
print('ok')
