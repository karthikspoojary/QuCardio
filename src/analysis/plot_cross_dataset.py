"""
ST-5F — Consolidated Cross-Dataset Comparison Plot
===================================================
Reads results from ST-5A through ST-5E and produces a single publication-quality
figure + summary table combining all dataset/strategy experiments.

Requires the following results files (runs what exists, skips missing):
  results/full_dataset_metrics.json     (ST-5A — scaling study)
  results/qsvc_scaling_study.json       (ST-5A — QSVC scaling)
  results/ecg_data_frozen/metrics.json  (ST-5B — ECG_DATA frozen eval)
  results/newecg_4class/metrics.json    (ST-5C — new-ecg-data 2-class)
  results/newecg_8class/metrics.json    (ST-5D — 8-class transferability)
  results/augmentation_robustness.json  (ST-5E — augmentation robustness)
  results/metrics_report.json           (baseline — 742-sample train/test)

Outputs:
  results/cross_dataset_comparison.png — grouped bar chart (4 strategies × 3 models)
  results/cross_dataset_summary.json   — unified table for the paper
"""

import sys
import json
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
import config

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_json(path):
    p = Path(path)
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return None


def run():
    print("=" * 65)
    print("ST-5F — Consolidated Cross-Dataset Comparison")
    print("=" * 65)

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Collect results ───────────────────────────────────────────────────────
    baseline    = load_json(config.RESULTS_DIR / 'metrics_report.json')
    ecg_frozen  = load_json(config.RESULTS_DIR / 'ecg_data_frozen' / 'metrics.json')
    newecg_4cls = load_json(config.RESULTS_DIR / 'newecg_4class' / 'metrics.json')
    newecg_8cls = load_json(config.RESULTS_DIR / 'newecg_8class' / 'metrics.json')
    aug_robust  = load_json(config.RESULTS_DIR / 'augmentation_robustness.json')
    full_ds     = load_json(config.RESULTS_DIR / 'full_dataset_metrics.json')
    qsvc_scale  = load_json(config.RESULTS_DIR / 'qsvc_scaling_study.json')

    summary = {}

    # ── Strategy S1: Baseline (742 train / 186 test) ─────────────────────────
    if baseline:
        models = baseline.get('models', {})
        summary['S1 — Baseline\n(n=742 train, n=186 test)'] = {
            'SVM':     models.get('Classical SVM',  {}).get('accuracy', None),
            'QSVC':    models.get('QSVC',           {}).get('accuracy', None),
            'Pegasos': models.get('Pegasos QSVC',   {}).get('accuracy', None),
        }
        print("S1 Baseline loaded.")

    # ── Strategy S2: Full dataset best (largest N from scaling study) ─────────
    svm_best, peg_best, qsvc_best = None, None, None

    if full_ds:
        scaling = full_ds.get('scaling_results', [])
        if scaling:
            last = max(scaling, key=lambda x: x.get('n_train', 0))
            svm_best = last.get('svm_accuracy')
            peg_best = last.get('pegasos_accuracy')
        # Also check top-level fields
        if svm_best is None:
            svm_best = full_ds.get('svm_accuracy') or full_ds.get('svm_acc')
        if peg_best is None:
            peg_best = full_ds.get('pegasos_accuracy') or full_ds.get('pegasos_acc')

    if qsvc_scale:
        # Use the largest N result
        max_n = max(int(k) for k in qsvc_scale.keys())
        qsvc_best = qsvc_scale[str(max_n)].get('accuracy')

    if any(v is not None for v in [svm_best, peg_best, qsvc_best]):
        summary['S2 — Full Dataset\n(n=3023 train, n=928 test)'] = {
            'SVM':     svm_best,
            'QSVC':    qsvc_best,
            'Pegasos': peg_best,
        }
        print("S2 Full dataset loaded.")

    # ── Strategy S3: Cross-dataset frozen (ECG_DATA/test) ────────────────────
    if ecg_frozen:
        summary['S3 — Cross-Dataset\n(ECG_DATA/test, frozen models)'] = {
            'SVM':     ecg_frozen.get('SVM',     {}).get('accuracy'),
            'QSVC':    ecg_frozen.get('QSVC',    {}).get('accuracy'),
            'Pegasos': ecg_frozen.get('Pegasos', {}).get('accuracy'),
        }
        print("S3 ECG_DATA frozen loaded.")

    # ── Strategy S4: new-ecg-data 4-class ────────────────────────────────────
    if newecg_4cls:
        summary['S4 — New ECG Data\n(n=48, 2-class, frozen)'] = {
            'SVM':     newecg_4cls.get('SVM',     {}).get('accuracy_2class'),
            'QSVC':    newecg_4cls.get('QSVC',    {}).get('accuracy_2class'),
            'Pegasos': newecg_4cls.get('Pegasos', {}).get('accuracy_2class'),
        }
        print("S4 new-ecg-data 4-class loaded.")

    # ── Save summary JSON ─────────────────────────────────────────────────────
    out_json = config.RESULTS_DIR / 'cross_dataset_summary.json'
    with open(out_json, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved → {out_json}")

    # ── Augmentation robustness sub-plot ──────────────────────────────────────
    _plot_main(summary)
    if aug_robust:
        _plot_augmentation_drop(aug_robust)
    if newecg_8cls:
        print(f"\nST-5D 8-class result: acc={newecg_8cls.get('accuracy',0)*100:.1f}%  "
              f"F1={newecg_8cls.get('f1_macro',0):.3f}  n={newecg_8cls.get('n_total')}")

    print("\n✅ ST-5F complete.")
    return summary


def _plot_main(summary):
    """Grouped bar chart: strategies × models."""
    strategies = list(summary.keys())
    if not strategies:
        print("No data to plot.")
        return

    models = ['SVM', 'QSVC', 'Pegasos']
    colors = {'SVM': '#3b82f6', 'QSVC': '#7c3aed', 'Pegasos': '#10b981'}
    x   = np.arange(len(strategies))
    w   = 0.22

    fig, ax = plt.subplots(figsize=(max(10, len(strategies) * 3), 5))

    for i, model in enumerate(models):
        vals = []
        for strat in strategies:
            v = summary[strat].get(model)
            vals.append((v * 100) if v is not None else 0)
        offset = (i - 1) * w
        bars = ax.bar(x + offset, vals, w, label=model,
                      color=colors[model], edgecolor='white')
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f'{val:.1f}', ha='center', va='bottom', fontsize=7.5)

    ax.set_xticks(x)
    ax.set_xticklabels(strategies, fontsize=9)
    ax.set_ylabel('Accuracy (%)', fontsize=11)
    ax.set_ylim(0, 112)
    ax.set_title('Cross-Dataset Comparison — All Strategies (QuCardio)',
                 fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    out_png = config.RESULTS_DIR / 'cross_dataset_comparison.png'
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Cross-dataset comparison plot saved → {out_png}")


def _plot_augmentation_drop(aug_robust):
    """Accuracy-drop plot relative to 1_origin baseline."""
    if '1_origin' not in aug_robust:
        return

    baseline_svm  = aug_robust['1_origin'].get('svm_acc',     0)
    baseline_qsvc = aug_robust['1_origin'].get('qsvc_acc',    0)
    baseline_peg  = aug_robust['1_origin'].get('pegasos_acc', 0)

    folders = [f for f in aug_robust if f != '1_origin']
    folder_labels = {
        '2_photo': 'Photo', '3_add_brightness': '+Bright',
        '4_reduce_brightness': '-Bright', '5_add_contrast': '+Contrast',
        '6_reduce_contrast': '-Contrast', '7_affine': 'Affine',
        '8_angle': 'Rotation', '9_ratio': 'Aspect',
    }

    x = np.arange(len(folders))
    w = 0.28

    svm_drop  = [(baseline_svm  - aug_robust[f].get('svm_acc',  baseline_svm))  * 100 for f in folders]
    qsvc_drop = [(baseline_qsvc - aug_robust[f].get('qsvc_acc', baseline_qsvc)) * 100 for f in folders]
    peg_drop  = [(baseline_peg  - aug_robust[f].get('pegasos_acc', baseline_peg)) * 100 for f in folders]

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.bar(x - w, svm_drop,  w, label='SVM',    color='#3b82f6')
    ax.bar(x,     qsvc_drop, w, label='QSVC',   color='#7c3aed')
    ax.bar(x + w, peg_drop,  w, label='Pegasos',color='#10b981')

    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([folder_labels.get(f, f) for f in folders],
                       rotation=15, ha='right', fontsize=9)
    ax.set_ylabel('Accuracy Drop vs Clean Baseline (pp)', fontsize=10)
    ax.set_title('Robustness: Accuracy Drop Under Augmentation\n'
                 '(negative = degraded vs 1_origin baseline)', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    out_png = config.RESULTS_DIR / 'augmentation_drop.png'
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Augmentation drop plot saved → {out_png}")


if __name__ == '__main__':
    run()
