"""
ST-5E — Augmentation Robustness Experiment
==========================================
Evaluates all three frozen models across all 9 augmentation variant folders
of data/new ecg data/ without any retraining.

Folders tested:
  1_origin        — clean baseline
  2_photo         — photographed on screen
  3_add_brightness — increased brightness
  4_reduce_brightness — reduced brightness
  5_add_contrast  — increased contrast
  6_reduce_contrast — reduced contrast
  7_affine        — affine distortion
  8_angle         — rotation
  9_ratio         — aspect-ratio change

Each folder has 48 .JPG files (same images, different transforms).
2-class label: NSR→0, all others→1 (Normal vs Arrhythmia).

Shows which real-world capture degradations most affect model performance,
and whether the quantum kernel is more robust than the classical SVM.

Outputs:
  results/augmentation_robustness.json
  results/augmentation_robustness.png  — grouped bar chart (9×3), 300 DPI
"""

import sys
import json
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
import config

from sklearn.metrics import accuracy_score
from src.analysis._frozen_eval_utils import (
    extract_features, project_features,
    predict_svm, predict_qsvc, predict_pegasos,
)

PREFIX_TO_2CLASS = {
    'NSR':     0,    # Normal
    'AF':      1, 'RBBB': 1, 'LBBB_AF': 1, 'LBBB': 1,
    'SNT':     1, 'SNB':  1, 'AVBI':    1,
}

AUG_FOLDERS = [
    '1_origin', '2_photo', '3_add_brightness', '4_reduce_brightness',
    '5_add_contrast', '6_reduce_contrast', '7_affine', '8_angle', '9_ratio',
]

FOLDER_LABELS = {
    '1_origin':           'Original',
    '2_photo':            'Photo',
    '3_add_brightness':   '+Brightness',
    '4_reduce_brightness':'-Brightness',
    '5_add_contrast':     '+Contrast',
    '6_reduce_contrast':  '-Contrast',
    '7_affine':           'Affine',
    '8_angle':            'Rotation',
    '9_ratio':            'Aspect Ratio',
}


def label_from_filename(name):
    stem = Path(name).stem
    for prefix in sorted(PREFIX_TO_2CLASS.keys(), key=len, reverse=True):
        if stem.startswith(prefix + '_') or stem == prefix:
            return PREFIX_TO_2CLASS[prefix]
    return None


def load_folder(folder_path):
    folder = Path(folder_path)
    paths, labels = [], []
    for p in sorted(folder.iterdir()):
        if p.suffix.upper() != '.JPG':
            continue
        if 'Zone.Identifier' in p.name:
            continue
        lbl = label_from_filename(p.name)
        if lbl is None:
            continue
        paths.append(p)
        labels.append(lbl)
    return paths, np.array(labels)


def run():
    print("=" * 65)
    print("ST-5E — Augmentation Robustness Experiment")
    print("=" * 65)

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results = {}
    base_dir = Path('data/new ecg data')

    for folder_name in AUG_FOLDERS:
        folder_path = base_dir / folder_name
        if not folder_path.exists():
            print(f"\n  WARN: {folder_path} not found — skipping")
            continue

        print(f"\n── {folder_name} ({FOLDER_LABELS[folder_name]}) ──────────────────")
        paths, y_true = load_folder(folder_path)
        if len(paths) == 0:
            print("  No valid images found — skipping")
            continue

        # Extract and project — shared across all 3 models for this folder
        print(f"  Extracting features ({len(paths)} images) …")
        raw_feats, valid_mask = extract_features(paths, batch_size=8)
        y_true = y_true[valid_mask]
        X_9d   = project_features(raw_feats)

        folder_results = {'n': int(len(y_true))}

        # SVM
        y_pred_svm = predict_svm(X_9d)
        acc_svm = accuracy_score(y_true, y_pred_svm)
        folder_results['svm_acc'] = round(float(acc_svm), 6)
        print(f"  SVM  : {acc_svm*100:.1f}%")

        # QSVC
        y_pred_qsvc = predict_qsvc(X_9d)
        acc_qsvc = accuracy_score(y_true, y_pred_qsvc)
        folder_results['qsvc_acc'] = round(float(acc_qsvc), 6)
        print(f"  QSVC : {acc_qsvc*100:.1f}%")

        # Pegasos
        y_pred_peg = predict_pegasos(X_9d)
        if y_pred_peg is not None:
            acc_peg = accuracy_score(y_true, y_pred_peg)
            folder_results['pegasos_acc'] = round(float(acc_peg), 6)
            print(f"  Peg  : {acc_peg*100:.1f}%")

        results[folder_name] = folder_results

    # ── Save JSON ─────────────────────────────────────────────────────────────
    out_json = config.RESULTS_DIR / 'augmentation_robustness.json'
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {out_json}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    _plot(results)
    print("✅ ST-5E complete.")
    return results


def _plot(results):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    folders   = [f for f in AUG_FOLDERS if f in results]
    x_labels  = [FOLDER_LABELS[f] for f in folders]
    svm_vals  = [results[f].get('svm_acc',     0) * 100 for f in folders]
    qsvc_vals = [results[f].get('qsvc_acc',    0) * 100 for f in folders]
    peg_vals  = [results[f].get('pegasos_acc', 0) * 100 for f in folders]

    x   = np.arange(len(folders))
    w   = 0.25

    fig, ax = plt.subplots(figsize=(13, 5))
    b1 = ax.bar(x - w,   svm_vals,  w, label='SVM',    color='#3b82f6')
    b2 = ax.bar(x,       qsvc_vals, w, label='QSVC',   color='#7c3aed')
    b3 = ax.bar(x + w,   peg_vals,  w, label='Pegasos',color='#10b981')

    # Star on 1_origin baseline bars
    if '1_origin' in results:
        origin_idx = folders.index('1_origin')
        for bar_group, vals in [(b1, svm_vals), (b2, qsvc_vals), (b3, peg_vals)]:
            bars = list(bar_group)
            ax.text(bars[origin_idx].get_x() + bars[origin_idx].get_width()/2,
                    vals[origin_idx] + 1.5, '★', ha='center', fontsize=10, color='#ef4444')

    # Value labels
    for bars in [b1, b2, b3]:
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width()/2, h + 0.3,
                        f'{h:.0f}', ha='center', va='bottom', fontsize=6.5)

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=20, ha='right', fontsize=9)
    ax.set_ylabel('Accuracy (%)', fontsize=11)
    ax.set_ylim(0, 110)
    ax.set_title('ST-5E: Augmentation Robustness — Frozen Models on new-ecg-data\n'
                 '(★ = clean baseline, 2-class: Normal vs Arrhythmia)', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    out_png = config.RESULTS_DIR / 'augmentation_robustness.png'
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved → {out_png}")


if __name__ == '__main__':
    run()
