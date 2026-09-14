"""
ST-5B — Cross-Dataset Generalization: Frozen Eval on ECG_DATA/test/ (929 images)
==================================================================================
Evaluates the three models trained on data/raw/ (742-sample baseline) directly
on data/ECG_DATA/test/ using frozen SVD + scaler — no retraining.

Tests whether the quantum kernel learned on 742 images generalises to the full
ECG_DATA test split (929 images, same 4 classes, same source hospital).

Pipeline (FROZEN — no refitting):
  image → preprocess_for_inference() → ResNet50 pool1_pool (462400-D)
        → frozen svd_reducer.pkl (9-D) → frozen minmax_scaler.pkl ([0,1])
        → SVM / QSVC / Pegasos

Outputs:
  results/ecg_data_frozen/metrics.json
  results/ecg_data_frozen/cm_svm.png
  results/ecg_data_frozen/cm_qsvc.png
  results/ecg_data_frozen/cm_pegasos.png
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

# ── Label mapping for ECG_DATA folder names ───────────────────────────────────
# Match by folder name PREFIX — the (NNNxNN=NNNN) suffix varies
FOLDER_LABEL_MAP = {
    'ECG Images of Patient that have abnormal heartbeat': 1,   # Arrhythmia
    'ECG Images of Myocardial Infarction Patients':       2,   # MI
    'Normal Person ECG Images':                           0,   # Normal
    'ECG Images of Patient that have History of MI':      3,   # History_of_MI
}

OUT_DIR = config.RESULTS_DIR / 'ecg_data_frozen'


def load_ecg_data_test():
    """Glob all .jpg images from data/ECG_DATA/test/ with labels."""
    root = Path('data/ECG_DATA/test')
    if not root.exists():
        raise FileNotFoundError(f"{root} not found")

    paths, labels = [], []
    for folder in sorted(root.iterdir()):
        if not folder.is_dir():
            continue
        # Match by prefix
        label = None
        for prefix, lbl in FOLDER_LABEL_MAP.items():
            if folder.name.startswith(prefix):
                label = lbl
                break
        if label is None:
            print(f"  WARN: no label mapping for folder '{folder.name}' — skipping")
            continue
        imgs = sorted(folder.glob('*.jpg')) + sorted(folder.glob('*.JPG'))
        print(f"  {folder.name[:55]}: {len(imgs)} images → label {label}")
        for p in imgs:
            paths.append(p)
            labels.append(label)

    print(f"  Total: {len(paths)} images")
    return paths, np.array(labels)


def run():
    print("=" * 65)
    print("ST-5B — Cross-Dataset: Frozen Eval on ECG_DATA/test/")
    print("=" * 65)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load images ───────────────────────────────────────────────────────────
    paths, y_true = load_ecg_data_test()

    # ── Extract ResNet50 features (batch=8) ───────────────────────────────────
    print("\nExtracting ResNet50 pool1_pool features …")
    raw_features, valid_mask = extract_features(paths, batch_size=8)
    y_true = y_true[valid_mask]
    print(f"  Valid samples: {len(raw_features)}")

    # ── Project through frozen SVD + scaler ───────────────────────────────────
    print("Projecting through frozen svd_reducer + minmax_scaler …")
    X_9d = project_features(raw_features)
    print(f"  9D features: {X_9d.shape}")

    results = {}

    # ── Classical SVM ─────────────────────────────────────────────────────────
    print("\nClassical SVM …")
    y_pred_svm = predict_svm(X_9d)
    acc_svm = accuracy_score(y_true, y_pred_svm)
    f1_svm  = f1_score(y_true, y_pred_svm, average='macro', zero_division=0)
    print(f"  SVM  accuracy: {acc_svm*100:.2f}%  F1-macro: {f1_svm:.4f}")
    print(classification_report(y_true, y_pred_svm,
                                target_names=config.CLASS_NAMES, zero_division=0))
    save_confusion_matrix(y_true, y_pred_svm, config.CLASS_NAMES,
                          'SVM — ECG_DATA/test (frozen)',
                          OUT_DIR / 'cm_svm.png', cmap='Blues')
    results['SVM'] = {'accuracy': round(float(acc_svm), 6),
                      'f1_macro': round(float(f1_svm),  6)}

    # ── QSVC ──────────────────────────────────────────────────────────────────
    print("\nQSVC …")
    y_pred_qsvc = predict_qsvc(X_9d)
    acc_qsvc = accuracy_score(y_true, y_pred_qsvc)
    f1_qsvc  = f1_score(y_true, y_pred_qsvc, average='macro', zero_division=0)
    print(f"  QSVC accuracy: {acc_qsvc*100:.2f}%  F1-macro: {f1_qsvc:.4f}")
    print(classification_report(y_true, y_pred_qsvc,
                                target_names=config.CLASS_NAMES, zero_division=0))
    save_confusion_matrix(y_true, y_pred_qsvc, config.CLASS_NAMES,
                          'QSVC — ECG_DATA/test (frozen)',
                          OUT_DIR / 'cm_qsvc.png', cmap='Purples')
    results['QSVC'] = {'accuracy': round(float(acc_qsvc), 6),
                       'f1_macro': round(float(f1_qsvc),  6)}

    # ── Pegasos ───────────────────────────────────────────────────────────────
    print("\nPegasos …")
    y_pred_peg = predict_pegasos(X_9d)
    if y_pred_peg is not None:
        acc_peg = accuracy_score(y_true, y_pred_peg)
        f1_peg  = f1_score(y_true, y_pred_peg, average='macro', zero_division=0)
        print(f"  Pegasos accuracy: {acc_peg*100:.2f}%  F1-macro: {f1_peg:.4f}")
        print(classification_report(y_true, y_pred_peg,
                                    target_names=config.CLASS_NAMES, zero_division=0))
        save_confusion_matrix(y_true, y_pred_peg, config.CLASS_NAMES,
                              'Pegasos — ECG_DATA/test (frozen)',
                              OUT_DIR / 'cm_pegasos.png', cmap='Greens')
        results['Pegasos'] = {'accuracy': round(float(acc_peg), 6),
                              'f1_macro': round(float(f1_peg),  6)}

    results['meta'] = {
        'n_images': int(len(y_true)),
        'source': 'data/ECG_DATA/test/',
        'models_trained_on': 'data/raw/ (742-sample baseline)',
        'preprocessing': 'frozen — old fixed-threshold pipeline (thresh=200)',
        'svd_scaler': 'frozen — backend/models/svd_reducer.pkl + minmax_scaler.pkl',
    }

    out_json = OUT_DIR / 'metrics.json'
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {out_json}")
    print("✅ ST-5B complete.")
    return results


if __name__ == '__main__':
    run()
