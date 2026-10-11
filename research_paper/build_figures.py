"""Rebuild manuscript plots from saved reports: uv run --with matplotlib build_figures.py."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
EVIDENCE = ROOT.parent / 'docs' / 'evidence'
OUT = ROOT / 'figures'
OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.family': 'DejaVu Serif', 'font.size': 8,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'savefig.dpi': 450})
COLORS = ['#355C7D', '#B76B2B']


def rows(module, filename):
    with (EVIDENCE / f'module-{module}-candidate' / filename).open() as file:
        return list(csv.DictReader(file))


def save(fig, name):
    fig.tight_layout(pad=0.6)
    fig.savefig(OUT / f'{name}.png', facecolor='white')
    fig.savefig(OUT / f'{name}.pdf', facecolor='white')
    plt.close(fig)


classes = json.loads((EVIDENCE / 'module-1-candidate/config.json').read_text())['class_order']
counts = {}
for row in rows(1, 'class_distribution.csv'):
    crop = classes[int(row['label'])].split('___')[0].split('_(')[0].replace('_', ' ').title()
    counts[crop] = counts.get(crop, 0) + int(row['pv_train']) + int(row['pv_validation'])
assert sum(counts.values()) == 54305 and len(counts) == 14
ordered = sorted(counts.items(), key=lambda item: item[1])
fig, ax = plt.subplots(figsize=(3.4, 3.15))
ax.barh([c for c, _ in ordered], [n for _, n in ordered], color=COLORS[0])
ax.set(xlabel='Images in consumed snapshot', xlim=(0, max(counts.values()) * 1.32))
ax.tick_params(axis='y', length=0, labelsize=7)
for i, (_, count) in enumerate(ordered):
    ax.text(count + 160, i, f'{count:,}', va='center', fontsize=7)
save(fig, 'fig1_crop_distribution')

disease = {(r['model'], r['split']): r for r in rows(1, 'results_summary.csv')}
fig, ax = plt.subplots(figsize=(3.4, 2.55))
for i, (model, label) in enumerate([('original_checkpoint', 'Original'), ('adapted_checkpoint', 'Adapted')]):
    values = [100 * float(disease[model, split]['accuracy']) for split in ['pv_validation', 'field_test']]
    bars = ax.bar(np.arange(2) + (i - 0.5) * 0.33, values, width=0.33, label=label, color=COLORS[i])
    ax.bar_label(bars, labels=[f'{v:.2f}' for v in values], fontsize=7, padding=2)
assert round(float(disease['adapted_checkpoint', 'field_test']['accuracy']) * 252) == 141
ax.set(xticks=[0, 1], xticklabels=['PlantVillage validation\nn = 8,145', 'PlantDoc test\nn = 252'],
       ylabel='Accuracy (%)', ylim=(0, 112))
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.18), ncol=2, frameon=False)
save(fig, 'fig2_disease_comparison')


def confusion(module, filename, labels, name):
    data = rows(module, filename)
    matrix = np.array([[int(row[key]) for key in list(row)[1:]] for row in data])
    assert matrix.shape == (3, 3)
    if module == 2:
        assert matrix.sum() == 1549 and matrix[2, 0] == 74 and matrix[0, 0] == 23
    else:
        assert matrix.sum() == 19 and np.trace(matrix) == 10 and matrix[1, 1] == 0
    fig, ax = plt.subplots(figsize=(3.4, 2.8))
    ax.imshow(matrix / matrix.sum(axis=1, keepdims=True), cmap='Blues', vmin=0, vmax=1)
    for r in range(3):
        for c in range(3):
            ax.text(c, r, str(matrix[r, c]), ha='center', va='center', fontsize=10,
                    color='white' if matrix[r, c] / matrix[r].sum() > 0.6 else 'black')
    ax.set(xticks=range(3), yticks=range(3), xticklabels=labels, yticklabels=labels,
           xlabel='Predicted category', ylabel='True category')
    ax.tick_params(length=0)
    save(fig, name)


confusion(2, 'rf_baseline__test_confusion_counts.csv', ['Healthy', 'Moderate', 'Severe'], 'fig3_rf_confusion')
pest = {r['model']: r for r in rows(3, 'results_summary.csv') if r['split'] == 'test'}
names = ['majority', 'persistence', 'lstm_u16_d0.3', 'logistic']
assert all(int(pest[n]['n']) == 19 for n in names)
fig, ax = plt.subplots(figsize=(3.4, 2.65))
for i, (metric, label) in enumerate([('accuracy', 'Accuracy'), ('macro_f1_all_classes', 'Macro F1')]):
    values = [float(pest[name][metric]) for name in names]
    bars = ax.barh(np.arange(4) + (i - 0.5) * 0.34, values, height=0.34, color=COLORS[i], label=label)
    ax.bar_label(bars, labels=[f'{v:.3f}' for v in values], fontsize=7, padding=2)
ax.set(yticks=range(4), yticklabels=['Majority', 'Persistence', 'LSTM', 'Logistic'],
       xlim=(0, 1), xlabel='Score on 19 observed-target weeks')
ax.invert_yaxis()
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=2, frameon=False)
save(fig, 'fig4_pest_comparison')
confusion(3, 'lstm_u16_d0.3__test_confusion_counts.csv', ['Low', 'Medium', 'High'], 'fig5_lstm_confusion')
assert len(list(OUT.glob('fig*.png'))) == 5
print('Five figures rebuilt; source counts and key results checked.')
