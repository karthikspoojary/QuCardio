"""
cross_dataset_ptbxl.py — ST-5G: Cross-Dataset Evaluation on PTB-XL (frozen models)
======================================================================================
Evaluates the three QuCardio models (SVM, QSVC, Pegasos) that were TRAINED on
data/raw/ (742-sample split) against PTB-XL images rendered by
convert_ptbxl_matplotlib.py (folds 9+10, ~2,178 records, 4 classes).

WHY THIS IS THE DEFINITIVE CROSS-DATASET RESULT
-------------------------------------------------
• PTB-XL is the largest publicly available ECG dataset (21,799 records, 12-lead, 71 institutions)
• Folds 9+10 are PTB-XL's official held-out test split — never used in any training
• Completely different acquisition hardware, patient population, and rendering style
• All three models are FROZEN: svd_reducer.pkl + minmax_scaler.pkl + classifiers
  trained on the original 742-sample Kaggle data — ZERO re-training
• Good performance here proves genuine generalisation, not dataset artefact

PREPROCESSING NOTE
------------------
PTB-XL images are clean matplotlib renders (white bg, black waveform).
They go through preprocess_for_inference() just like all other cross-dataset
images — this is the SAME fixed-threshold pipeline used when the training
images in data/processed_340/ were created.  The frozen SVD/scaler are thus
applied consistently.

LABEL MAPPING (same as convert_ptbxl_matplotlib.py):
  Normal                → 0
  Myocardial_Infarction → 1   (folder name — maps to class index 2 via CLASS_MAP)
  Arrhythmia            → 2   (folder name — maps to class index 1 via CLASS_MAP)
  History_of_MI         → 3

Wait — CLASS_MAP in config.py:
  Normal: 0, Arrhythmia: 1, Myocardial_Infarction: 2, History_of_MI: 3

This script uses config.CLASS_MAP to look up label indices so folder→index is
always consistent with what the models were trained on.

OUTPUTS
-------
  results/ptbxl_cross_dataset/
    metrics.json          — accuracy, F1-macro, per-class F1 for all 3 models
    cm_svm.png
    cm_qsvc.png
    cm_pegasos.png
    classification_reports.txt

USAGE
-----
  # Step 1: render images (one-time, ~18 min)
  python src/analysis/convert_ptbxl_matplotlib.py

  # Step 2: evaluate frozen models (this script, ~25–40 min on CPU)
  python src/analysis/cross_dataset_ptbxl.py
"""

import sys
import json
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
import config

from sklearn.metrics import (accuracy_score, f1_score,
                             classification_report, confusion_matrix)

from src.analysis._frozen_eval_utils import (
    extract_features, project_features,
    extract_and_project_streaming,
    predict_svm, predict_qsvc, predict_pegasos,
    save_confusion_matrix,
)

# ── Constants ────────────────────────────────────────────────────────────────
PTBXL_IMAGES_DIR = Path('data/ptb-xl-images')
OUT_DIR = config.RESULTS_DIR / 'ptbxl_cross_dataset'

# Folder names exactly as written by convert_ptbxl_matplotlib.py
FOLDER_NAMES = ['Normal', 'Arrhythmia', 'Myocardial_Infarction', 'History_of_MI']

# Map folder name → integer label (must match training labels in config.CLASS_MAP)
FOLDER_TO_LABEL = {
    'Normal':                config.CLASS_MAP['Normal'],                # 0
    'Arrhythmia':            config.CLASS_MAP['Arrhythmia'],            # 1
    'Myocardial_Infarction': config.CLASS_MAP['Myocardial_Infarction'], # 2
    'History_of_MI':         config.CLASS_MAP['History_of_MI'],         # 3
}

# Human-readable class names for confusion matrices (indexed 0–3)
CLASS_LABELS = config.CLASS_NAMES   # ['Normal', 'Arrhythmia', 'Myocardial_Infarction', 'History_of_MI']


def load_ptbxl_images(images_dir: Path) -> tuple:
    """
    Walk images_dir/{Normal,Arrhythmia,Myocardial_Infarction,History_of_MI}/*.png
    and collect (paths, labels).

    Returns
    -------
    paths  : list of Path objects
    labels : np.ndarray of int (0–3), same length as paths
    """
    paths, labels = [], []
    missing_folders = []
    images_dir = Path(images_dir)

    for folder in FOLDER_NAMES:
        folder_dir = images_dir / folder
        if not folder_dir.exists():
            missing_folders.append(folder)
            continue
        imgs = sorted(folder_dir.glob('*.png'))
        label = FOLDER_TO_LABEL[folder]
        for p in imgs:
            paths.append(p)
            labels.append(label)

    if missing_folders:
        print(f"  WARN: these class folders are missing: {missing_folders}")
        print(f"  Run convert_ptbxl_matplotlib.py first.")

    counts = {f: labels.count(FOLDER_TO_LABEL[f]) for f in FOLDER_NAMES if f not in missing_folders}
    total = len(paths)
    print(f"  Loaded {total} images:")
    for folder, cnt in counts.items():
        print(f"    {folder}: {cnt}")

    return paths, np.array(labels)


def run():
    print("=" * 65)
    print("ST-5G — Cross-Dataset Evaluation: PTB-XL (frozen models)")
    print("=" * 65)
    print(f"  Source : {PTBXL_IMAGES_DIR}")
    print(f"  Models : trained on data/raw/ (742 samples, Kaggle dataset)")
    print(f"  Folds  : PTB-XL test split (strat_fold 9 & 10)")
    print()

    if not PTBXL_IMAGES_DIR.exists():
        print(f"ERROR: {PTBXL_IMAGES_DIR} does not exist.")
        print("Run this first:")
        print("  python src/analysis/convert_ptbxl_matplotlib.py")
        sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load images ──────────────────────────────────────────────────────────
    print("Loading PTB-XL image paths …")
    paths, y_true = load_ptbxl_images(PTBXL_IMAGES_DIR)

    if len(paths) == 0:
        print("ERROR: no images found. Run convert_ptbxl_matplotlib.py first.")
        sys.exit(1)

    # ── Extract ResNet50 features & Project to 9-D (Streaming + Caching) ─────
    cache_path = OUT_DIR / 'ptbxl_features_9d.npz'
    if cache_path.exists():
        print(f"\nLoading cached 9-D features from {cache_path} …")
        cached = np.load(cache_path)
        X_9d   = cached['X_9d']
        y_true = cached['y_true']
        print(f"  Loaded {len(X_9d)} samples.")
    else:
        print(f"\nExtracting ResNet50 features & projecting to 9-D ({len(paths)} images) …")
        print("  This uses streaming projection (ResNet50 pool1_pool → SVD → MinMax) to keep RAM < 200MB.")
        X_9d, valid_mask = extract_and_project_streaming(paths, batch_size=16)
        y_true = y_true[valid_mask]
        print(f"  Valid samples: {len(X_9d)}, shape: {X_9d.shape}")
        np.savez(cache_path, X_9d=X_9d, y_true=y_true)
        print(f"  Saved 9-D features to {cache_path}")

    # ── Evaluate each model ───────────────────────────────────────────────────
    results = {}
    report_lines = []

    for model_name, pred_fn, cmap in [
        ('SVM',     predict_svm,     'Blues'),
        ('QSVC',    predict_qsvc,    'Purples'),
        ('Pegasos', predict_pegasos, 'Greens'),
    ]:
        print(f"\n{'─'*50}")
        print(f"  Model: {model_name}")
        print(f"{'─'*50}")

        y_pred = pred_fn(X_9d)
        if y_pred is None:
            print(f"  {model_name} skipped (model files not found).")
            continue

        # ── Metrics ──────────────────────────────────────────────────────────
        acc    = accuracy_score(y_true, y_pred)
        f1_mac = f1_score(y_true, y_pred, average='macro',    zero_division=0)
        f1_wt  = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        f1_per = f1_score(y_true, y_pred, average=None,       zero_division=0)

        print(f"  Accuracy     : {acc*100:.2f}%")
        print(f"  F1-macro     : {f1_mac:.4f}")
        print(f"  F1-weighted  : {f1_wt:.4f}")

        # Per-class F1
        present_classes = sorted(set(y_true.tolist()))
        print(f"  Per-class F1 :")
        for ci in present_classes:
            if ci < len(f1_per):
                print(f"    {CLASS_LABELS[ci]:30s}: {f1_per[ci]:.4f}")

        # Classification report
        present_names = [CLASS_LABELS[i] for i in present_classes]
        rpt = classification_report(y_true, y_pred,
                                    labels=present_classes,
                                    target_names=present_names,
                                    zero_division=0)
        print(rpt)
        report_lines.append(f"\n{'='*60}\n{model_name}\n{'='*60}\n{rpt}")

        # ── Confusion matrix ──────────────────────────────────────────────────
        present_label_names = [CLASS_LABELS[i] for i in range(len(CLASS_LABELS))
                               if i in present_classes]
        save_confusion_matrix(
            y_true=y_true,
            y_pred=y_pred,
            labels=CLASS_LABELS,
            title=f'{model_name} — PTB-XL cross-dataset (frozen)',
            out_path=OUT_DIR / f'cm_{model_name.lower()}.png',
            cmap=cmap,
        )
        print(f"  CM saved → {OUT_DIR / f'cm_{model_name.lower()}.png'}")

        # ── Store results ─────────────────────────────────────────────────────
        per_class_f1 = {}
        for ci in range(len(CLASS_LABELS)):
            if ci < len(f1_per):
                per_class_f1[CLASS_LABELS[ci]] = round(float(f1_per[ci]), 6)

        results[model_name] = {
            'accuracy':    round(float(acc),    6),
            'f1_macro':    round(float(f1_mac), 6),
            'f1_weighted': round(float(f1_wt),  6),
            'f1_per_class': per_class_f1,
            'n_samples':   int(len(y_true)),
        }

    # ── Save classification reports text ──────────────────────────────────────
    rpt_path = OUT_DIR / 'classification_reports.txt'
    with open(rpt_path, 'w') as f:
        f.write('\n'.join(report_lines))
    print(f"\nClassification reports → {rpt_path}")

    # ── Class distribution ────────────────────────────────────────────────────
    class_dist = {}
    for ci, name in enumerate(CLASS_LABELS):
        class_dist[name] = int(np.sum(y_true == ci))

    # ── Save metrics JSON ─────────────────────────────────────────────────────
    results['meta'] = {
        'dataset':          'PTB-XL folds 9+10 (strat_fold in {9,10})',
        'n_images':         int(len(y_true)),
        'class_distribution': class_dist,
        'source_images':    str(PTBXL_IMAGES_DIR),
        'image_renderer':   'matplotlib (convert_ptbxl_matplotlib.py)',
        'models_trained_on': 'data/raw/ (742-sample baseline, Kaggle ECG dataset)',
        'preprocessing':    'frozen — old fixed-threshold pipeline (thresh=200)',
        'svd_reducer':      'backend/models/svd_reducer.pkl (TruncatedSVD, n=9)',
        'minmax_scaler':    'backend/models/minmax_scaler.pkl',
        'label_mapping':    'NORM→Normal(0), MI→Myocardial_Infarction(2), '
                            'CD→Arrhythmia(1), STTC→History_of_MI(3)',
        'note': (
            'Cross-dataset test: models trained on Kaggle ECG paper scans, '
            'evaluated on PTB-XL clinical 12-lead signals rendered as images. '
            'No retraining, no fine-tuning. '
            'This measures zero-shot visual generalization across acquisition domains.'
        ),
    }

    out_json = OUT_DIR / 'metrics.json'
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2)

    # ── Final summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("ST-5G COMPLETE — Cross-Dataset PTB-XL Summary")
    print("=" * 65)
    print(f"  {'Model':<12}  {'Accuracy':>10}  {'F1-macro':>10}")
    print(f"  {'─'*12}  {'─'*10}  {'─'*10}")
    for m in ['SVM', 'QSVC', 'Pegasos']:
        if m in results:
            r = results[m]
            print(f"  {m:<12}  {r['accuracy']*100:>9.2f}%  {r['f1_macro']:>10.4f}")
    print(f"\n  Results saved → {OUT_DIR}/")
    print(f"  metrics.json  → {out_json}")
    print("=" * 65)

    return results


if __name__ == '__main__':
    run()
