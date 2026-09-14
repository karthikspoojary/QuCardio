"""
ST-5A — Full Dataset (6,046 images) Training + Scaling Study
=============================================================
Retrains all three models on the full ECG_DATA/train/ split (6,046 images)
and evaluates on ECG_DATA/test/ (929 images).

Also produces a scaling curve: accuracy vs training-set size at
[742, 1500, 3000, 6046] for Classical SVM.

Pipeline (fresh — NOT using frozen models):
  ECG_DATA/train/ images
  → OTSU preprocessing (preprocess_ecg.py)
  → ResNet50 pool1_pool features (462,400-D)
  → Fresh TruncatedSVD(9) + MinMaxScaler
  → Classical SVM (GridSearchCV, all 6,046)
  → Pegasos QSVC (6 binary models, all 6,046)
  → QSVC (stratified sample 742 for kernel fitting)

Outputs:
  results/full_dataset_metrics.json   — accuracy, F1-macro, per-class metrics
  results/scaling_study.png           — accuracy vs n_train curve
  results/full_dataset_cm.png         — confusion matrices (SVM + Pegasos + QSVC)
  data/resnet50_features_full.npz     — 6,046-image ResNet50 features (cached)
  data/features_9d_full.npz           — 9-D scaled features (cached)

Usage:
  python src/analysis/full_dataset_retrain.py

  # Skip ResNet50 extraction if already cached:
  python src/analysis/full_dataset_retrain.py --skip-extraction

  # Scaling study only (requires cached features_9d_full.npz):
  python src/analysis/full_dataset_retrain.py --scaling-only

  # Skip QSVC (fastest path — SVM + Pegasos only):
  python src/analysis/full_dataset_retrain.py --no-qsvc
"""

import sys
import gc
import json
import time
import argparse
import numpy as np
import cv2
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm

# ── Path setup ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.preprocessing.preprocess_ecg import preprocess_ecg

import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import Model

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                              recall_score, confusion_matrix, classification_report)

from qiskit.circuit.library import ZZFeatureMap
from qiskit.quantum_info import Statevector

from src.quantum.train_pegasos import (
    PegasosSVMKernel, remap_labels_to_binary, get_binary_indices,
    compute_statevectors, build_kernel_from_statevectors,
)

# ── Constants ─────────────────────────────────────────────────────────────────
CLASS_NAMES = ['Normal', 'Arrhythmia', 'Myocardial_Infarction', 'History_of_MI']
CLASS_PAIRS = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]

# ECG_DATA folder-name prefix → integer label (verified from filesystem)
ECGDATA_LABEL_MAP = {
    'ECG Images of Patient that have abnormal heartbeat': 1,  # Arrhythmia
    'ECG Images of Myocardial Infarction Patients':       2,  # MI
    'Normal Person ECG Images':                           0,  # Normal
    'ECG Images of Patient that have History of MI':      3,  # History_of_MI
}

# Tuned Pegasos hyperparameters per class pair (from tune_pegasos_pass2.py)
PEGASOS_BEST_PARAMS = {
    (0, 1): {'C': 5.0,  'num_steps': 3000},
    (0, 2): {'C': 10.0, 'num_steps': 5000},
    (0, 3): {'C': 20.0, 'num_steps': 10000},
    (1, 2): {'C': 10.0, 'num_steps': 2000},
    (1, 3): {'C': 10.0, 'num_steps': 2000},
    (2, 3): {'C': 2.0,  'num_steps': 3000},
}

RESULTS_DIR   = PROJECT_ROOT / 'results'
DATA_DIR      = PROJECT_ROOT / 'data'
FEATURES_FULL = DATA_DIR / 'resnet50_features_full.npz'
FEATURES_9D   = DATA_DIR / 'features_9d_full.npz'

RESULTS_DIR.mkdir(exist_ok=True)

# Per-pair training indices stored after Pegasos training (for prediction)
_TRAIN_IDX = {}


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Load ECG_DATA images
# ─────────────────────────────────────────────────────────────────────────────

def load_ecgdata(split: str):
    """
    Load and OTSU-preprocess images from data/ECG_DATA/<split>/.
    split: 'train' (6,046) or 'test' (929)
    Returns: X float32 [0–255] shape (N,340,340), y int64 labels.
    """
    base = DATA_DIR / 'ECG_DATA' / split
    if not base.exists():
        raise FileNotFoundError(f"ECG_DATA {split} not found: {base}")

    X, y, skipped = [], [], 0
    for folder in sorted(base.iterdir()):
        if not folder.is_dir():
            continue
        label = next(
            (lbl for pfx, lbl in ECGDATA_LABEL_MAP.items()
             if folder.name.startswith(pfx)),
            None
        )
        if label is None:
            print(f"  [WARN] Unmapped folder: {folder.name!r} — skipping")
            continue

        images = sorted(
            p for p in folder.iterdir()
            if p.suffix.lower() in ('.jpg', '.jpeg', '.png')
            and 'Zone.Identifier' not in p.name
        )
        cls_name = CLASS_NAMES[label]
        print(f"  {cls_name}: {len(images)} images ...")

        for img_path in tqdm(images, desc=f"  {cls_name}", leave=False):
            try:
                arr = preprocess_ecg(str(img_path), output_size=(340, 340))
                # preprocess_ecg returns float32 [0,1]; convert to [0,255]
                # to match extract_resnet50_features convention
                X.append((arr * 255.0).astype(np.float32))
                y.append(label)
            except Exception as e:
                skipped += 1
                if skipped <= 5:
                    print(f"    [SKIP] {img_path.name}: {e}")

    print(f"\n  Loaded {len(X)} images  ({skipped} skipped)")
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — ResNet50 pool1_pool feature extraction
# ─────────────────────────────────────────────────────────────────────────────

def build_resnet_extractor():
    base = ResNet50(weights='imagenet', include_top=False)
    try:
        out = base.get_layer('pool1_pool').output
    except ValueError:
        out = base.layers[4].output
    return Model(inputs=base.input, outputs=out)


def extract_features(X: np.ndarray, model, batch_size: int = 8) -> np.ndarray:
    """X: float32 [0-255] (N,340,340). Returns (N, 462400) float32."""
    feat_dim = 85 * 85 * 64
    N   = len(X)
    out = np.empty((N, feat_dim), dtype=np.float32)
    for i in tqdm(range(0, N, batch_size), desc="  ResNet50"):
        end        = min(i + batch_size, N)
        batch      = X[i:end]
        batch_rgb  = np.repeat(batch[..., np.newaxis], 3, axis=-1)
        batch_prep = preprocess_input(batch_rgb)
        feats      = model.predict_on_batch(batch_prep)
        out[i:end] = feats.reshape(feats.shape[0], -1)
        del batch, batch_rgb, batch_prep, feats
        gc.collect()
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — SVD + MinMax scaling
# ─────────────────────────────────────────────────────────────────────────────

def reduce_and_scale(train_feats, test_feats, n_components=9):
    svd    = TruncatedSVD(n_components=n_components, random_state=42)
    scaler = MinMaxScaler()
    Xtr = scaler.fit_transform(svd.fit_transform(train_feats))
    Xte = scaler.transform(svd.transform(test_feats))
    print(f"  Explained variance: {svd.explained_variance_ratio_.sum():.2%}")
    return Xtr, Xte, svd, scaler


# ─────────────────────────────────────────────────────────────────────────────
# Step 4A — Classical SVM
# ─────────────────────────────────────────────────────────────────────────────

def train_svm(X_tr, y_tr, X_te, y_te):
    print("\n" + "="*55)
    print("Classical SVM — GridSearchCV on 6,046 training images")
    print("="*55)

    param_grid = {
        'C':            [0.01, 0.1, 1, 5, 10],
        'gamma':        ['scale', 'auto', 0.1, 0.01, 0.001],
        'kernel':       ['rbf'],
        'class_weight': ['balanced'],
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    gs = GridSearchCV(
        SVC(probability=False, random_state=42),
        param_grid, cv=cv, scoring='f1_macro', verbose=1, n_jobs=-1
    )
    t0 = time.time()
    gs.fit(X_tr, y_tr)
    train_time = time.time() - t0
    best = gs.best_params_
    print(f"\n  Best params: {best}  ({train_time:.1f}s)")

    svm_cal = CalibratedClassifierCV(
        SVC(**best, random_state=42, probability=False),
        cv=3, method='isotonic'
    )
    svm_cal.fit(X_tr, y_tr)

    y_pred  = svm_cal.predict(X_te)
    metrics = _compute_metrics(y_te, y_pred, 'Classical SVM (full dataset)')
    metrics['best_params']  = best
    metrics['train_time_s'] = round(train_time, 1)
    return svm_cal, metrics, y_pred


# ─────────────────────────────────────────────────────────────────────────────
# Step 4B — Pegasos QSVC
# ─────────────────────────────────────────────────────────────────────────────

def train_pegasos_full(X_tr, y_tr, X_te, y_te):
    print("\n" + "="*55)
    print("Pegasos QSVC — 6 binary models on 6,046 training images")
    print("="*55)

    # Best config from tuning: circular + minmax_0pi
    X_tr_pi = X_tr * np.pi
    X_te_pi = X_te * np.pi
    fmap    = ZZFeatureMap(feature_dimension=9, reps=2, entanglement='circular')

    print("  Computing train statevectors ...")
    t0 = time.time()
    sv_train = compute_statevectors(X_tr_pi, fmap)
    print(f"  Train SVs: {sv_train.shape}  ({time.time()-t0:.1f}s)")

    print("  Computing test statevectors ...")
    t0 = time.time()
    sv_test  = compute_statevectors(X_te_pi, fmap)
    print(f"  Test  SVs: {sv_test.shape}  ({time.time()-t0:.1f}s)")

    K_train = build_kernel_from_statevectors(sv_train, sv_train).astype(np.float64)
    K_test  = build_kernel_from_statevectors(sv_test,  sv_train).astype(np.float64)

    models = {}
    for c1, c2 in CLASS_PAIRS:
        idx              = get_binary_indices(y_tr, c1, c2)
        _TRAIN_IDX[(c1, c2)] = idx
        K_bin = K_train[np.ix_(idx, idx)]
        y_bin = remap_labels_to_binary(y_tr[idx], c1, c2)
        p     = PEGASOS_BEST_PARAMS[(c1, c2)]
        m     = PegasosSVMKernel(C=p['C'], num_steps=p['num_steps'], seed=42)
        m.fit(K_bin, y_bin)
        models[(c1, c2)] = m
        print(f"  [{c1},{c2}] n={len(idx)}  C={p['C']}  τ={p['num_steps']}")

    # Multiclass prediction — Algorithm 1
    y_pred = []
    for i in range(len(y_te)):
        row = K_test[i:i+1]   # (1, N_train)
        # Per-pair kernel rows
        def krow(c1, c2):
            return row[:, _TRAIN_IDX[(c1, c2)]]

        pred_01 = models[(0, 1)].predict(krow(0, 1))[0]
        pred_23 = models[(2, 3)].predict(krow(2, 3))[0]
        if pred_01 == -1:   # predicted class 0
            pair = (0, 2) if pred_23 == -1 else (0, 3)
        else:               # predicted class 1
            pair = (1, 2) if pred_23 == -1 else (1, 3)
        raw = models[pair].predict(krow(*pair))[0]
        # binary -1/+1 back to class label
        y_pred.append(pair[0] if raw == -1 else pair[1])

    y_pred  = np.array(y_pred, dtype=np.int64)
    metrics = _compute_metrics(y_te, y_pred, 'Pegasos QSVC (full dataset)')
    return models, metrics, y_pred


# ─────────────────────────────────────────────────────────────────────────────
# Step 4C — QSVC (stratified sample 742, full-dataset SVD)
# ─────────────────────────────────────────────────────────────────────────────

def train_qsvc_sampled(X_tr, y_tr, X_te, y_te, n_sample=742):
    print("\n" + "="*55)
    print(f"QSVC — stratified sample {n_sample} for kernel (full SVD)")
    print("="*55)

    _, X_s, _, y_s = train_test_split(
        X_tr, y_tr,
        test_size=n_sample / len(X_tr),
        stratify=y_tr, random_state=42
    )
    print(f"  Kernel sample: {X_s.shape}  dist={np.bincount(y_s)}")

    fmap = ZZFeatureMap(feature_dimension=9, reps=2, entanglement='circular')

    print("  Computing sample statevectors ...")
    sv_s = compute_statevectors(X_s * np.pi, fmap)
    print("  Computing test statevectors ...")
    sv_t = compute_statevectors(X_te * np.pi, fmap)

    K_tr = build_kernel_from_statevectors(sv_s, sv_s).astype(np.float32)
    K_te = build_kernel_from_statevectors(sv_t, sv_s).astype(np.float32)

    qsvc = SVC(kernel='precomputed', C=5.0, random_state=42)
    qsvc.fit(K_tr, y_s)
    y_pred  = qsvc.predict(K_te)
    metrics = _compute_metrics(y_te, y_pred, f'QSVC (sample={n_sample}, full SVD)')
    return qsvc, metrics, y_pred


# ─────────────────────────────────────────────────────────────────────────────
# Step 5 — Scaling study
# ─────────────────────────────────────────────────────────────────────────────

def scaling_study_9d(X_tr, y_tr, X_te, y_te, sizes=(300, 742, 1500, 2200, 3023)):
    """
    Evaluates both Classical SVM and Pegasos QSVC across different training dataset sizes N.
    Uses precomputed 9-D features and computes quantum kernels on the fly.
    """
    print("\n" + "="*55)
    print("Scaling Study — Accuracy & F1 vs Training Set Size N")
    print("="*55)

    fmap = ZZFeatureMap(feature_dimension=9, reps=2, entanglement='circular')
    sv_te = compute_statevectors(X_te * np.pi, fmap)

    results = {}
    for n in sizes:
        if n > len(y_tr):
            continue
        print(f"\n── Evaluating n_train = {n} ──")
        if n == len(y_tr):
            Xs, ys = X_tr, y_tr
        else:
            Xs, _, ys, _ = train_test_split(
                X_tr, y_tr, train_size=n,
                stratify=y_tr, random_state=42
            )

        # 1. Classical SVM
        svm = CalibratedClassifierCV(
            SVC(C=10, gamma='scale', kernel='rbf', class_weight='balanced', random_state=42),
            cv=3, method='isotonic'
        )
        svm.fit(Xs, ys)
        svm_preds = svm.predict(X_te)
        svm_acc = float(accuracy_score(y_te, svm_preds))
        svm_f1  = float(f1_score(y_te, svm_preds, average='macro', zero_division=0))

        # 2. Pegasos QSVC
        sv_s = compute_statevectors(Xs * np.pi, fmap)
        K_tr_s = build_kernel_from_statevectors(sv_s, sv_s).astype(np.float64)
        K_te_s = build_kernel_from_statevectors(sv_te, sv_s).astype(np.float64)

        models = {}
        pair_idx = {}
        for c1, c2 in CLASS_PAIRS:
            idx = get_binary_indices(ys, c1, c2)
            pair_idx[(c1, c2)] = idx
            K_bin = K_tr_s[np.ix_(idx, idx)]
            y_bin = remap_labels_to_binary(ys[idx], c1, c2)
            p = PEGASOS_BEST_PARAMS[(c1, c2)]
            m = PegasosSVMKernel(C=p['C'], num_steps=min(p['num_steps'], 3000), seed=42)
            m.fit(K_bin, y_bin)
            models[(c1, c2)] = m

        peg_preds = []
        for i in range(len(y_te)):
            row = K_te_s[i:i+1]
            def krow(c1, c2): return row[:, pair_idx[(c1, c2)]]
            pred_01 = models[(0, 1)].predict(krow(0, 1))[0]
            pred_23 = models[(2, 3)].predict(krow(2, 3))[0]
            if pred_01 == -1: pair = (0, 2) if pred_23 == -1 else (0, 3)
            else:             pair = (1, 2) if pred_23 == -1 else (1, 3)
            raw = models[pair].predict(krow(*pair))[0]
            peg_preds.append(pair[0] if raw == -1 else pair[1])

        peg_preds = np.array(peg_preds, dtype=np.int64)
        peg_acc = float(accuracy_score(y_te, peg_preds))
        peg_f1  = float(f1_score(y_te, peg_preds, average='macro', zero_division=0))

        results[n] = {
            'svm_acc': round(svm_acc, 4), 'svm_f1': round(svm_f1, 4),
            'pegasos_acc': round(peg_acc, 4), 'pegasos_f1': round(peg_f1, 4),
        }
        print(f"  n={n:<5} | Classical SVM: {svm_acc*100:.2f}% (F1={svm_f1*100:.2f}%) | Pegasos QSVC: {peg_acc*100:.2f}% (F1={peg_f1*100:.2f}%)")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _compute_metrics(y_true, y_pred, label=''):
    acc  = float(accuracy_score(y_true, y_pred))
    f1   = float(f1_score(y_true, y_pred, average='macro', zero_division=0))
    prec = float(precision_score(y_true, y_pred, average='macro', zero_division=0))
    rec  = float(recall_score(y_true, y_pred, average='macro', zero_division=0))
    print(f"\n  {label}")
    print(f"  Accuracy : {acc*100:.2f}%   F1-macro: {f1*100:.2f}%")
    print(f"  Precision: {prec*100:.2f}%  Recall:   {rec*100:.2f}%")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, zero_division=0))
    return {
        'accuracy': round(acc, 4), 'f1_macro': round(f1, 4),
        'precision_macro': round(prec, 4), 'recall_macro': round(rec, 4),
        'classification_report': classification_report(
            y_true, y_pred, target_names=CLASS_NAMES,
            zero_division=0, output_dict=True
        )
    }


def plot_confusion_matrices(y_te, preds_dict, save_path):
    n    = len(preds_dict)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 6))
    if n == 1:
        axes = [axes]
    for ax, (label, y_pred) in zip(axes, preds_dict.items()):
        cm  = confusion_matrix(y_te, y_pred)
        acc = accuracy_score(y_te, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
        ax.set_title(f'{label}\nAcc: {acc*100:.2f}%', fontsize=10)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.tick_params(axis='x', rotation=30)
    plt.suptitle('Full Dataset (6,046 train / 929 test) — Confusion Matrices', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_scaling_curve(scaling_results, save_path):
    sizes = sorted(int(k) for k in scaling_results.keys())
    svm_acc = [scaling_results[n]['svm_acc'] * 100 for n in sizes]
    peg_acc = [scaling_results[n]['pegasos_acc'] * 100 for n in sizes]

    plt.figure(figsize=(9, 6))
    plt.plot(sizes, svm_acc, 'b-o', linewidth=2.5, markersize=8, label='Classical SVM (RBF, C=10)')
    plt.plot(sizes, peg_acc, 'r-s', linewidth=2.5, markersize=8, label='Pegasos QSVC (Quantum 9-Qubit)')

    for x, yv in zip(sizes, svm_acc):
        plt.annotate(f'{yv:.1f}%', (x, yv), textcoords='offset points',
                     xytext=(0, -15), ha='center', fontsize=9, color='blue', fontweight='bold')
    for x, yv in zip(sizes, peg_acc):
        plt.annotate(f'{yv:.1f}%', (x, yv), textcoords='offset points',
                     xytext=(0, 10), ha='center', fontsize=9, color='red', fontweight='bold')

    plt.axvline(x=742, color='gray', linestyle='--', alpha=0.7, label='Paper Baseline Size (742)')
    plt.xlabel('Training Set Size (N)', fontsize=12, fontweight='bold')
    plt.ylabel('Test Accuracy (%)', fontsize=12, fontweight='bold')
    plt.title('Empirical Scaling Study: Classical SVM vs Pegasos QSVC\nEvaluated on 928 Unseen Test Images', fontsize=13, fontweight='bold', pad=12)
    plt.legend(fontsize=10, loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved scaling plot: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='ST-5A Full Dataset Retrain')
    parser.add_argument('--skip-extraction', action='store_true',
                        help='Use cached resnet50_features_full.npz and features_9d_full.npz')
    parser.add_argument('--scaling-only', action='store_true',
                        help='Run only the scaling study (needs cached features)')
    parser.add_argument('--no-qsvc', action='store_true',
                        help='Skip QSVC to save time')
    parser.add_argument('--no-pegasos', action='store_true',
                        help='Skip Pegasos to save time')
    args = parser.parse_args()

    t_total     = time.time()
    all_metrics = {}

    if FEATURES_9D.exists() and args.skip_extraction:
        print("\n" + "="*55)
        print("STEP 1, 2, 3 — Loading cached 9-D features (features_9d_full.npz)")
        print("="*55)
        print(f"  Loading {FEATURES_9D} ...")
        c9      = np.load(FEATURES_9D)
        X_train = c9['train_features']
        X_test  = c9['test_features']
        y_train = c9['y_train']
        y_test  = c9['y_test']
        print(f"  Train 9D: {X_train.shape}  dist={np.bincount(y_train)}")
        print(f"  Test  9D: {X_test.shape}   dist={np.bincount(y_test)}")
    elif args.skip_extraction and FEATURES_FULL.exists():
        print("\n" + "="*55)
        print("STEP 1 & 2 — Loading cached ResNet50 features")
        print("="*55)
        print(f"  Loading {FEATURES_FULL} ...")
        cache       = np.load(FEATURES_FULL)
        train_feats = cache['train_features']
        test_feats  = cache['test_features']
        y_train     = cache['y_train']
        y_test      = cache['y_test']
        print(f"  Train features loaded: {train_feats.shape}  y={np.bincount(y_train)}")
        print(f"  Test  features loaded: {test_feats.shape}   y={np.bincount(y_test)}")

        print("\n" + "="*55)
        print("STEP 3 — TruncatedSVD(9) + MinMaxScaler")
        print("="*55)
        X_train, X_test, svd_new, scaler_new = reduce_and_scale(train_feats, test_feats)
        np.savez_compressed(
            FEATURES_9D,
            train_features=X_train, test_features=X_test,
            y_train=y_train, y_test=y_test
        )
        print(f"  Cached 9D features to {FEATURES_9D}")
        del train_feats, test_feats; gc.collect()
    else:
        # ── Load images ───────────────────────────────────────────────────────
        print("\n" + "="*55)
        print("STEP 1 — Load ECG_DATA images (OTSU preprocessing)")
        print("="*55)
        print("Loading ECG_DATA/train/ (6,046 images) ...")
        X_tr_img, y_train = load_ecgdata('train')
        print(f"Train loaded: {X_tr_img.shape}  dist={np.bincount(y_train)}")

        print("\nLoading ECG_DATA/test/ (929 images) ...")
        X_te_img, y_test = load_ecgdata('test')
        print(f"Test  loaded: {X_te_img.shape}   dist={np.bincount(y_test)}")

        # ── ResNet50 features ─────────────────────────────────────────────────
        print("\n" + "="*55)
        print("STEP 2 — ResNet50 pool1_pool feature extraction")
        print("="*55)

        resnet = build_resnet_extractor()
        print(f"  Extracting train features ...")
        train_feats = extract_features(X_tr_img, resnet)
        del X_tr_img; gc.collect()

        print(f"  Extracting test features ...")
        test_feats = extract_features(X_te_img, resnet)
        del X_te_img, resnet; gc.collect()

        np.savez_compressed(
            FEATURES_FULL,
            train_features=train_feats, test_features=test_feats,
            y_train=y_train, y_test=y_test
        )
        print(f"  Cached to {FEATURES_FULL}")

        print("\n" + "="*55)
        print("STEP 3 — TruncatedSVD(9) + MinMaxScaler")
        print("="*55)
        X_train, X_test, svd_new, scaler_new = reduce_and_scale(train_feats, test_feats)
        np.savez_compressed(
            FEATURES_9D,
            train_features=X_train, test_features=X_test,
            y_train=y_train, y_test=y_test
        )
        print(f"  Cached 9D features to {FEATURES_9D}")
        del train_feats, test_feats; gc.collect()

    if not args.scaling_only:
        # ── Classical SVM ─────────────────────────────────────────────────────
        svm_model, svm_metrics, svm_pred = train_svm(
            X_train, y_train, X_test, y_test
        )
        all_metrics['classical_svm_full'] = svm_metrics

        peg_pred  = None
        qsvc_pred = None

        # ── Pegasos ────────────────────────────────────────────────────────────
        if not args.no_pegasos:
            _, peg_metrics, peg_pred = train_pegasos_full(
                X_train, y_train, X_test, y_test
            )
            all_metrics['pegasos_full'] = peg_metrics

        # ── QSVC ──────────────────────────────────────────────────────────────
        if not args.no_qsvc:
            _, qsvc_metrics, qsvc_pred = train_qsvc_sampled(
                X_train, y_train, X_test, y_test, n_sample=742
            )
            all_metrics['qsvc_full_svd'] = qsvc_metrics

        # ── Confusion matrices ─────────────────────────────────────────────────
        preds = {'Classical SVM': svm_pred}
        if peg_pred  is not None: preds['Pegasos QSVC']       = peg_pred
        if qsvc_pred is not None: preds['QSVC (sample=742)']  = qsvc_pred
        plot_confusion_matrices(y_test, preds, RESULTS_DIR / 'full_dataset_cm.png')

    # ── Scaling study ─────────────────────────────────────────────────────────
    scaling = scaling_study_9d(X_train, y_train, X_test, y_test,
                               sizes=(300, 742, 1500, 2200, 3023))
    all_metrics['scaling_study'] = scaling
    plot_scaling_curve(scaling, RESULTS_DIR / 'scaling_study.png')

    # ── Save all metrics ──────────────────────────────────────────────────────
    out = RESULTS_DIR / 'full_dataset_metrics.json'
    with open(out, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\n  Metrics saved: {out}")

    # ── Final summary ─────────────────────────────────────────────────────────
    print("\n" + "="*55)
    print("SUMMARY — Full Dataset (6,046 train) vs Baseline (742 train)")
    print("="*55)
    print(f"{'Model':<40} {'Baseline':>10} {'Full DS':>10}")
    print("-"*62)
    baselines = {'Classical SVM': 84.95, 'Pegasos QSVC': 91.94, 'QSVC': 94.62}
    key_map   = {
        'classical_svm_full': 'Classical SVM',
        'pegasos_full':        'Pegasos QSVC',
        'qsvc_full_svd':       'QSVC',
    }
    for k, label in key_map.items():
        if k in all_metrics:
            new_acc = all_metrics[k]['accuracy'] * 100
            baseline = baselines[label]
            delta    = new_acc - baseline
            sign     = '+' if delta >= 0 else ''
            print(f"  {label:<38} {baseline:>8.2f}%  {new_acc:>8.2f}%  ({sign}{delta:.2f} pp)")
    if 'scaling_study' in all_metrics:
        print(f"\n  Scaling curve SVM accuracy:")
        for n, v in sorted((int(k), v) for k, v in all_metrics['scaling_study'].items()):
            print(f"    n={n:<6}  acc={v['svm_acc']*100:.2f}%  F1={v['svm_f1']*100:.2f}%")

    print(f"\n  Total wall time: {(time.time()-t_total)/60:.1f} minutes")
    print("="*55)


if __name__ == '__main__':
    main()
