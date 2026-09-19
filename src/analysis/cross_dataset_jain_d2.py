"""
Cross-Dataset Evaluation — Jain et al. Dataset 2
=================================================
Evaluates models trained on Dataset 1 (data/raw/, 929 images, 4 classes)
directly on Dataset 2 (707 images, 3 classes — same hospital, same Mendeley
source as Jain et al. 2025, Journal of Supercomputing).

Dataset 2 class structure:
  Normal Person/       → label 0  (Normal)
  Abnormal heartbeat/  → label 1  (Arrhythmia)
  History of MI/       → label 3  (History_of_MI)
  ** No MI class **

Since Dataset 2 has no Myocardial Infarction class, we run two evaluations:
  (A) 3-class restricted: only predictions for classes {0,1,3} are scored.
      Samples predicted as class 2 (MI) count as errors.
  (B) Binary (Normal vs Abnormal): collapse Arrhythmia + History_of_MI → 1.
      Matches how Jain et al. reported their cross-dataset result.

Pipeline (FROZEN — no retraining):
  image → preprocess_for_inference() → ResNet50 pool1_pool (462,400-D)
        → frozen svd_reducer.pkl → frozen minmax_scaler.pkl
        → [×π] → QSVC / SVM / Pegasos

Outputs:
  results/paper/cross_dataset/jain_dataset2/metrics.json
  results/paper/cross_dataset/jain_dataset2/cm_svm.png
  results/paper/cross_dataset/jain_dataset2/cm_qsvc.png
  results/paper/cross_dataset/jain_dataset2/cm_pegasos.png
  results/paper/cross_dataset/jain_dataset2/classification_reports.txt
"""

import sys
import json
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
import config

from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix
)
from src.analysis._frozen_eval_utils import (
    extract_and_project_streaming,
    predict_svm, predict_qsvc, predict_pegasos,
    save_confusion_matrix,
)

# ── Dataset 2 path & label mapping ────────────────────────────────────────────
DATASET2_ROOT = Path(
    'data/ECG Dataset for Heart Condition Classification/ECG Dataset/ECG Dataset'
)

# Folder name → integer label (same scheme as Dataset 1)
FOLDER_LABEL_MAP = {
    'Normal Person':       0,   # Normal
    'Abnormal heartbeat':  1,   # Arrhythmia
    'History of MI':       3,   # History_of_MI  (no class 2 / MI in this dataset)
}

# 3-class names for confusion matrix axes (class 2 absent)
CLASS_NAMES_3 = ['Normal', 'Arrhythmia', 'History_of_MI']
LABEL_ORDER_3 = [0, 1, 3]

OUT_DIR = Path('results/paper/cross_dataset/jain_dataset2')


# ── Image loader ──────────────────────────────────────────────────────────────

def load_dataset2():
    """Return (image_paths, labels) for all images in Dataset 2."""
    if not DATASET2_ROOT.exists():
        raise FileNotFoundError(
            f"Dataset 2 not found at: {DATASET2_ROOT}\n"
            "Extract the RAR file and ensure the path is correct."
        )

    paths, labels = [], []
    for folder in sorted(DATASET2_ROOT.iterdir()):
        if not folder.is_dir():
            continue
        # Match folder name by prefix (robust to minor variations)
        label = None
        for prefix, lbl in FOLDER_LABEL_MAP.items():
            if folder.name.startswith(prefix):
                label = lbl
                break
        if label is None:
            print(f"  WARN: unrecognised folder '{folder.name}' — skipping")
            continue
        imgs = sorted(folder.glob('*.jpg')) + sorted(folder.glob('*.JPG')) \
             + sorted(folder.glob('*.png')) + sorted(folder.glob('*.PNG'))
        print(f"  {folder.name}: {len(imgs)} images → label {label} "
              f"({CLASS_NAMES_3[LABEL_ORDER_3.index(label)]})")
        for p in imgs:
            paths.append(p)
            labels.append(label)

    print(f"  Total: {len(paths)} images across {len(FOLDER_LABEL_MAP)} classes")
    return paths, np.array(labels)


# ── Scoring helpers ───────────────────────────────────────────────────────────

def score_3class(y_true, y_pred, model_name, report_lines):
    """
    Score on the 3-class subset {0,1,3}.
    Predictions of class 2 (MI) are treated as misclassifications.
    Returns (accuracy, f1_macro).
    """
    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, average='macro',
                   labels=LABEL_ORDER_3, zero_division=0)
    report = classification_report(
        y_true, y_pred,
        labels=LABEL_ORDER_3,
        target_names=CLASS_NAMES_3,
        zero_division=0
    )
    print(f"  {model_name} — 3-class accuracy: {acc*100:.2f}%  F1-macro: {f1:.4f}")
    print(report)
    report_lines.append(f"\n{'='*60}\n{model_name} — 3-class\n{'='*60}\n{report}")
    return acc, f1


def score_binary(y_true, y_pred, model_name, report_lines):
    """
    Binary Normal (0) vs Abnormal (1+3 → 1).
    """
    y_true_b = (y_true != 0).astype(int)
    y_pred_b = (y_pred != 0).astype(int)
    acc = accuracy_score(y_true_b, y_pred_b)
    f1  = f1_score(y_true_b, y_pred_b, average='macro', zero_division=0)
    report = classification_report(
        y_true_b, y_pred_b,
        target_names=['Normal', 'Abnormal'],
        zero_division=0
    )
    print(f"  {model_name} — binary accuracy: {acc*100:.2f}%  F1-macro: {f1:.4f}")
    report_lines.append(f"\n{'='*60}\n{model_name} — binary\n{'='*60}\n{report}")
    return acc, f1


# ── Main ──────────────────────────────────────────────────────────────────────

def run():
    print("=" * 65)
    print("Cross-Dataset Evaluation — Jain Dataset 2 (707 images, 3 classes)")
    print("=" * 65)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report_lines = []
    results = {}

    # ── Load images ───────────────────────────────────────────────────────────
    paths, y_true = load_dataset2()

    # ── Extract & project features (streaming to avoid 7 GB RAM spike) ────────
    print("\nExtracting ResNet50 pool1_pool features + projecting (frozen SVD/scaler)...")
    X_9d_01, valid_mask = extract_and_project_streaming(paths, batch_size=16)
    y_true = y_true[valid_mask]
    print(f"  Valid samples: {len(X_9d_01)}")
    print(f"  Class distribution: {dict(zip(*np.unique(y_true, return_counts=True)))}")

    # ── Classical SVM ─────────────────────────────────────────────────────────
    print("\n── Classical SVM ──")
    y_pred_svm = predict_svm(X_9d_01)

    acc_svm_3, f1_svm_3 = score_3class(y_true, y_pred_svm, 'SVM', report_lines)
    acc_svm_b, f1_svm_b = score_binary(y_true, y_pred_svm, 'SVM', report_lines)

    save_confusion_matrix(
        y_true, y_pred_svm, CLASS_NAMES_3,
        f'SVM — Jain Dataset 2 (3-class, frozen)\nAcc: {acc_svm_3*100:.2f}%',
        OUT_DIR / 'cm_svm.png', cmap='Blues'
    )
    results['SVM'] = {
        'accuracy_3class':    round(float(acc_svm_3), 6),
        'f1_macro_3class':    round(float(f1_svm_3),  6),
        'accuracy_binary':    round(float(acc_svm_b), 6),
        'f1_macro_binary':    round(float(f1_svm_b),  6),
    }

    # ── QSVC ──────────────────────────────────────────────────────────────────
    print("\n── QSVC ──")
    y_pred_qsvc = predict_qsvc(X_9d_01)

    acc_qsvc_3, f1_qsvc_3 = score_3class(y_true, y_pred_qsvc, 'QSVC', report_lines)
    acc_qsvc_b, f1_qsvc_b = score_binary(y_true, y_pred_qsvc, 'QSVC', report_lines)

    save_confusion_matrix(
        y_true, y_pred_qsvc, CLASS_NAMES_3,
        f'QSVC — Jain Dataset 2 (3-class, frozen)\nAcc: {acc_qsvc_3*100:.2f}%',
        OUT_DIR / 'cm_qsvc.png', cmap='Purples'
    )
    results['QSVC'] = {
        'accuracy_3class':    round(float(acc_qsvc_3), 6),
        'f1_macro_3class':    round(float(f1_qsvc_3),  6),
        'accuracy_binary':    round(float(acc_qsvc_b), 6),
        'f1_macro_binary':    round(float(f1_qsvc_b),  6),
    }

    # ── Pegasos ───────────────────────────────────────────────────────────────
    print("\n── Pegasos ──")
    y_pred_peg = predict_pegasos(X_9d_01)
    if y_pred_peg is not None:
        acc_peg_3, f1_peg_3 = score_3class(y_true, y_pred_peg, 'Pegasos', report_lines)
        acc_peg_b, f1_peg_b = score_binary(y_true, y_pred_peg, 'Pegasos', report_lines)

        save_confusion_matrix(
            y_true, y_pred_peg, CLASS_NAMES_3,
            f'Pegasos — Jain Dataset 2 (3-class, frozen)\nAcc: {acc_peg_3*100:.2f}%',
            OUT_DIR / 'cm_pegasos.png', cmap='Greens'
        )
        results['Pegasos'] = {
            'accuracy_3class':    round(float(acc_peg_3), 6),
            'f1_macro_3class':    round(float(f1_peg_3),  6),
            'accuracy_binary':    round(float(acc_peg_b), 6),
            'f1_macro_binary':    round(float(f1_peg_b),  6),
        }
    else:
        print("  Pegasos models unavailable — skipping.")

    # ── Meta ──────────────────────────────────────────────────────────────────
    results['meta'] = {
        'dataset':            'Jain et al. Dataset 2 — ECG Dataset for Heart Condition Classification',
        'source_path':        str(DATASET2_ROOT),
        'n_images':           int(len(y_true)),
        'classes_present':    ['Normal (0)', 'Arrhythmia (1)', 'History_of_MI (3)'],
        'class_absent':       'Myocardial_Infarction (2)',
        'models_trained_on':  'Dataset 1 — data/raw/ (929 images, 4 classes)',
        'pipeline':           'frozen svd_reducer.pkl + minmax_scaler.pkl + [x pi]',
        'note': (
            'Cross-dataset test: trained on Kaggle/Mendeley Dataset 1 (4-class), '
            'evaluated on Mendeley Dataset 2 (3-class, same hospital). '
            'MI predictions on Dataset 2 samples count as errors in 3-class score. '
            'Binary score collapses Arrhythmia + History_of_MI as Abnormal.'
        ),
    }

    # ── Save outputs ──────────────────────────────────────────────────────────
    out_json = OUT_DIR / 'metrics.json'
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nMetrics saved → {out_json}")

    out_txt = OUT_DIR / 'classification_reports.txt'
    with open(out_txt, 'w') as f:
        f.write('\n'.join(report_lines))
    print(f"Reports saved → {out_txt}")

    print("\n Summary")
    print(f"  {'Model':<10} {'3-class Acc':>12} {'3-class F1':>11} {'Binary Acc':>11}")
    print(f"  {'-'*47}")
    for model in ['SVM', 'QSVC', 'Pegasos']:
        if model in results and 'accuracy_3class' in results[model]:
            r = results[model]
            print(f"  {model:<10} {r['accuracy_3class']*100:>11.2f}% "
                  f"{r['f1_macro_3class']:>11.4f} "
                  f"{r['accuracy_binary']*100:>10.2f}%")

    print("\nDone.")
    return results


if __name__ == '__main__':
    run()
