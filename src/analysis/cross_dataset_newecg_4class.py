"""
ST-5C — Cross-Dataset Generalization: new-ecg-data (4-class mapped, 2 active classes)
=======================================================================================
Evaluates frozen models on data/new ecg data/1_origin/ (48 images).
This dataset uses a completely different ECG rendering style (CPSC 2018 / PhysioNet),
so good performance here is strong clinical generalization evidence.

Clinical label mapping (corrected — AVBI → Arrhythmia, not History_of_MI):
  NSR                → 0 (Normal)
  AF, RBBB, LBBB,    → 1 (Arrhythmia)
  SNT, SNB, AVBI,
  LBBB_AF (compound) → 1 (Arrhythmia)

Only classes 0 and 1 are present — MI and History_of_MI are absent from this dataset.
The 4-class models are evaluated as-is; classes 2 and 3 will produce zero predictions
on this subset, which is reported and explained in the paper.

Outputs:
  results/newecg_4class/metrics.json
  results/newecg_4class/cm_svm.png
  results/newecg_4class/cm_qsvc.png
  results/newecg_4class/cm_pegasos.png
"""

import sys
import json
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
import config

from sklearn.metrics import accuracy_score, f1_score, classification_report
from src.analysis._frozen_eval_utils import (
    extract_features, project_features,
    predict_svm, predict_qsvc, predict_pegasos,
    save_confusion_matrix,
)

# Prefix → 4-class label
PREFIX_TO_LABEL = {
    'NSR':     0,    # Normal Sinus Rhythm → Normal
    'AF':      1,    # Atrial Fibrillation → Arrhythmia
    'RBBB':    1,    # Right Bundle Branch Block → Arrhythmia
    'LBBB_AF': 1,    # Compound LBBB+AF → Arrhythmia (must check before LBBB)
    'LBBB':    1,    # Left Bundle Branch Block → Arrhythmia
    'SNT':     1,    # Sinus Tachycardia → Arrhythmia
    'SNB':     1,    # Sinus Bradycardia → Arrhythmia
    'AVBI':    1,    # AV Block I → Arrhythmia (conduction delay, NOT MI)
}

# 2-class names for the CM (only 0 and 1 are present)
ACTIVE_CLASSES = ['Normal', 'Arrhythmia']

OUT_DIR = config.RESULTS_DIR / 'newecg_4class'


def label_from_filename(name):
    """Extract label from filename prefix (e.g. 'AF_A1182_0.JPG' → 1)."""
    stem = Path(name).stem   # 'AF_A1182_0'
    for prefix in sorted(PREFIX_TO_LABEL.keys(), key=len, reverse=True):
        if stem.startswith(prefix + '_') or stem.startswith(prefix):
            return PREFIX_TO_LABEL[prefix]
    return None


def load_new_ecg_origin():
    """Load all .JPG files from 1_origin/, skip .JPGZone.Identifier files."""
    folder = Path('data/new ecg data/1_origin')
    if not folder.exists():
        raise FileNotFoundError(f"{folder} not found")

    paths, labels = [], []
    skipped = []
    for p in sorted(folder.iterdir()):
        # Only real image files — skip Zone.Identifier metadata
        if not p.suffix.upper() == '.JPG':
            continue
        if 'Zone.Identifier' in p.name:
            continue
        lbl = label_from_filename(p.name)
        if lbl is None:
            skipped.append(p.name)
            continue
        paths.append(p)
        labels.append(lbl)

    if skipped:
        print(f"  WARN: no label for {len(skipped)} files: {skipped}")

    counts = {0: labels.count(0), 1: labels.count(1)}
    print(f"  Loaded {len(paths)} images: Normal={counts[0]}, Arrhythmia={counts[1]}")
    return paths, np.array(labels)


def run():
    print("=" * 65)
    print("ST-5C — Cross-Dataset: new-ecg-data 4-class mapped (2 active)")
    print("=" * 65)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    paths, y_true = load_new_ecg_origin()

    print("\nExtracting ResNet50 pool1_pool features …")
    raw_features, valid_mask = extract_features(paths, batch_size=8)
    y_true = y_true[valid_mask]
    print(f"  Valid samples: {len(raw_features)}")

    print("Projecting through frozen svd_reducer + minmax_scaler …")
    X_9d = project_features(raw_features)

    results = {}

    for model_name, pred_fn, cmap in [
        ('SVM',     predict_svm,     'Blues'),
        ('QSVC',    predict_qsvc,    'Purples'),
        ('Pegasos', predict_pegasos, 'Greens'),
    ]:
        print(f"\n{model_name} …")
        y_pred = pred_fn(X_9d)
        if y_pred is None:
            print(f"  {model_name} skipped.")
            continue

        acc = accuracy_score(y_true, y_pred)
        f1  = f1_score(y_true, y_pred, average='macro', zero_division=0)
        # Binary accuracy on the 2 active classes only
        mask_active = (y_true == 0) | (y_true == 1)
        acc_2class = accuracy_score(y_true[mask_active], y_pred[mask_active])

        print(f"  {model_name} accuracy (4-class): {acc*100:.2f}%  "
              f"2-class subset: {acc_2class*100:.2f}%  F1: {f1:.4f}")
        present = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
        present_names = [config.CLASS_NAMES[i] for i in present]
        print(classification_report(y_true, y_pred,
                                    labels=present,
                                    target_names=present_names, zero_division=0))
        # 2-class CM (rows/cols = Normal, Arrhythmia only)
        from sklearn.metrics import confusion_matrix
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import seaborn as sns
        cm = confusion_matrix(y_true[mask_active], y_pred[mask_active],
                              labels=[0, 1])
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.heatmap(cm, annot=True, fmt='d', cmap=cmap,
                    xticklabels=ACTIVE_CLASSES, yticklabels=ACTIVE_CLASSES, ax=ax)
        ax.set_title(f'{model_name} — new-ecg-data (frozen)\n2-class acc: {acc_2class*100:.1f}%',
                     fontsize=10)
        ax.set_xlabel('Predicted'); ax.set_ylabel('True')
        plt.tight_layout()
        plt.savefig(str(OUT_DIR / f'cm_{model_name.lower()}.png'), dpi=150)
        plt.close()

        results[model_name] = {
            'accuracy_4class':  round(float(acc),       6),
            'accuracy_2class':  round(float(acc_2class), 6),
            'f1_macro_4class':  round(float(f1),         6),
        }

    results['meta'] = {
        'n_images': int(len(y_true)),
        'source': 'data/new ecg data/1_origin/',
        'active_classes': 'Normal (0), Arrhythmia (1) — MI/History_MI absent',
        'label_mapping': 'NSR→0; AF/RBBB/LBBB/SNT/SNB/AVBI/LBBB_AF→1',
        'models_trained_on': 'data/raw/ (742-sample baseline)',
        'preprocessing': 'frozen — old fixed-threshold pipeline (thresh=200)',
    }

    out_json = OUT_DIR / 'metrics.json'
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {out_json}")
    print("✅ ST-5C complete.")
    return results


if __name__ == '__main__':
    run()
