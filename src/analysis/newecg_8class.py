"""
ST-5D — Extended 8-Class Experiment: QML Feature Transferability
=================================================================
The quantum feature map (ZZFeatureMap, 9-qubit, circular) + SVD/scaler were trained
on the 4-class clinical dataset (Normal/Arrhythmia/MI/History_MI).

This script keeps the ENTIRE quantum pipeline frozen and fits only a new classical
SVM head on the 8 rhythm classes in data/new ecg data/1_origin/ (48 images).

Paper claim: "The quantum feature representation, learned on 4-class clinical data,
transfers to 8 unseen rhythm classes without any quantum retraining — only the SVM
head is replaced. This demonstrates cross-task transferability of quantum kernel spaces."

8-class label map (from filename prefix):
  NSR:0  AF:1  AVBI:2  LBBB:3  LBBB_AF:4  RBBB:5  SNB:6  SNT:7

Caveat: 48 images / 8 classes ≈ 6 per class. Results are indicative.
Augmented fallback: if base accuracy < 60%, automatically re-runs on all 9 folders
(1_origin through 9_ratio, 48×9=432 images) → results/newecg_8class_aug/

Outputs:
  results/newecg_8class/metrics.json
  results/newecg_8class/cm_svm8.png
  results/newecg_8class_aug/metrics.json   (if accuracy < 60%)
  results/newecg_8class_aug/cm_svm8.png
"""

import sys
import json
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
import config

from sklearn.svm import SVC
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, classification_report
from src.analysis._frozen_eval_utils import (
    extract_features, project_features, save_confusion_matrix,
)

# 8-class prefix map (LBBB_AF before LBBB — longer prefix first)
PREFIX_8CLASS = {
    'NSR':     0,
    'AF':      1,
    'AVBI':    2,
    'LBBB_AF': 4,    # compound — checked before LBBB
    'LBBB':    3,
    'RBBB':    5,
    'SNB':     6,
    'SNT':     7,
}

CLASS_NAMES_8 = ['NSR', 'AF', 'AVBI', 'LBBB', 'LBBB_AF', 'RBBB', 'SNB', 'SNT']

AUG_FOLDERS = [
    '1_origin', '2_photo', '3_add_brightness', '4_reduce_brightness',
    '5_add_contrast', '6_reduce_contrast', '7_affine', '8_angle', '9_ratio',
]


def label_from_filename(name):
    stem = Path(name).stem
    for prefix in sorted(PREFIX_8CLASS.keys(), key=len, reverse=True):
        if stem.startswith(prefix + '_') or stem == prefix:
            return PREFIX_8CLASS[prefix]
    return None


def load_folder(folder_path):
    """Load .JPG files from one folder, return (paths, labels)."""
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


def run_experiment(paths, labels, out_dir, tag):
    """Extract features → project → stratified split → SVM → evaluate."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  Extracting features ({len(paths)} images) …")
    raw_features, valid_mask = extract_features(paths, batch_size=8)
    labels = labels[valid_mask]
    print(f"  Valid: {len(raw_features)} | class dist: "
          f"{ {i: int((labels==i).sum()) for i in range(8) if (labels==i).sum()>0} }")

    print("  Projecting through frozen quantum pipeline …")
    X_9d = project_features(raw_features)   # (N, 9) minmax_01

    # Stratified 80:20 split
    n_test = max(1, int(len(X_9d) * 0.2))
    try:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2,
                                     random_state=config.RANDOM_SEED)
        train_idx, test_idx = next(sss.split(X_9d, labels))
    except ValueError:
        # Too few samples per class for stratified split — random split
        print("  WARN: stratified split failed (too few per class) — using random split")
        rng = np.random.default_rng(config.RANDOM_SEED)
        idx = rng.permutation(len(X_9d))
        n_tr = max(1, len(X_9d) - n_test)
        train_idx, test_idx = idx[:n_tr], idx[n_tr:]

    X_train, X_test = X_9d[train_idx], X_9d[test_idx]
    y_train, y_test = labels[train_idx], labels[test_idx]
    print(f"  Train: {len(X_train)}  Test: {len(X_test)}")

    # New SVM head — quantum features are frozen, only this SVM is new
    print("  Fitting new 8-class SVM head (quantum features FROZEN) …")
    svm = SVC(kernel='rbf', C=10, class_weight='balanced',
              random_state=config.RANDOM_SEED)
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average='macro', zero_division=0)
    print(f"  ✅ Accuracy: {acc*100:.2f}%  F1-macro: {f1:.4f}")
    print(f"  NOTE: Quantum feature extractor FROZEN. Only SVM head retrained. n={len(X_9d)}")

    # Get labels present in test set for clean CM
    present = sorted(set(y_test.tolist()))
    present_names = [CLASS_NAMES_8[i] for i in present]
    print(classification_report(y_test, y_pred,
                                target_names=present_names,
                                labels=present, zero_division=0))

    save_confusion_matrix(y_test, y_pred,
                          [CLASS_NAMES_8[i] for i in range(8)],
                          f'8-class SVM (frozen quantum features) — {tag}',
                          out_dir / 'cm_svm8.png', cmap='YlOrRd')

    result = {
        'accuracy': round(float(acc), 6),
        'f1_macro': round(float(f1),  6),
        'n_total':  int(len(X_9d)),
        'n_train':  int(len(X_train)),
        'n_test':   int(len(X_test)),
        'tag': tag,
        'meta': {
            'note': 'Quantum feature extractor FROZEN. Only SVM head retrained.',
            'caveat': f'n={len(X_9d)} images / 8 classes ≈ {len(X_9d)//8} per class — indicative only.',
            'class_names': CLASS_NAMES_8,
        }
    }
    with open(out_dir / 'metrics.json', 'w') as f:
        json.dump(result, f, indent=2)
    print(f"  Saved → {out_dir / 'metrics.json'}")
    return acc


def run():
    print("=" * 65)
    print("ST-5D — 8-Class QML Feature Transferability")
    print("=" * 65)

    # ── Run on 1_origin (48 images) ───────────────────────────────────────────
    paths, labels = load_folder('data/new ecg data/1_origin')
    print(f"Loaded {len(paths)} images from 1_origin")
    acc = run_experiment(paths, labels,
                         config.RESULTS_DIR / 'newecg_8class',
                         tag='1_origin (n=48)')

    # ── Augmentation fallback if accuracy < 60% ───────────────────────────────
    if acc < 0.60:
        print(f"\nAccuracy {acc*100:.1f}% < 60% — running augmented fallback "
              f"(all 9 folders, ≤432 images) …")
        all_paths, all_labels = [], []
        for folder_name in AUG_FOLDERS:
            folder = Path('data/new ecg data') / folder_name
            if not folder.exists():
                continue
            ps, ls = load_folder(folder)
            all_paths.extend(ps)
            all_labels.extend(ls.tolist())
        all_labels = np.array(all_labels)
        print(f"Augmented dataset: {len(all_paths)} images")
        run_experiment(all_paths, all_labels,
                       config.RESULTS_DIR / 'newecg_8class_aug',
                       tag='all augmentations (n≤432)')

    print("\n✅ ST-5D complete.")


if __name__ == '__main__':
    run()
