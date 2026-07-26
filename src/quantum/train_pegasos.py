"""
Multiclass Pegasos QSVC Training Script (Statevector Precomputed Kernel)
======================================================================
Based on the QuCardio paper (Section IV-B):
  - PegasosQSVC is binary-only, so we train 6 binary models (one per class pair)
  - We then use a 3-step decision tree (Algorithm 1 from paper) to get final 4-class prediction

Design note:
  The paper's Pegasos QSVC computes the quantum kernel on-the-fly during SGD.
  On a classical simulator, we instead precompute all statevectors once and use
  fast matrix-multiply for all kernel lookups. This is mathematically equivalent
  (same K(x,z) = |<ψ(x)|ψ(z)>|²), but drastically faster:
    - Paper approach: N×num_steps individual circuit evaluations (~minutes/hours)
    - Our approach:   N statevectors once (~3s), then instant matrix multiply
  The SVC solver here is sklearn's exact QP solver (equivalent to Pegasos at full
  convergence) with C=5.0 (tuned) vs paper's C=1.0 with Pegasos approximation.

Class pair mapping:
  (0,1) = Normal vs Arrhythmia
  (0,2) = Normal vs Myocardial_Infarction
  (0,3) = Normal vs History_of_MI
  (1,2) = Arrhythmia vs Myocardial_Infarction
  (1,3) = Arrhythmia vs History_of_MI
  (2,3) = Myocardial_Infarction vs History_of_MI

Decision Tree Algorithm (per paper Algorithm 1):
  Step 1: Test image through model(0,1) → pred_01  and model(2,3) → pred_23
  Step 2: If pred_01==0 and pred_23==2  → run model(0,2) → final
          If pred_01==0 and pred_23==3  → run model(0,3) → final
          If pred_01==1 and pred_23==2  → run model(1,2) → final
          If pred_01==1 and pred_23==3  → run model(1,3) → final
"""

import sys
import time
import numpy as np
import joblib
import json
from pathlib import Path
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
import config

from qiskit.circuit.library import ZZFeatureMap
from qiskit.quantum_info import Statevector

# ── All 6 binary class pairs (C(4,2) = 6 combinations) ─────────────────────
CLASS_PAIRS = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]


def get_binary_indices(y, c1, c2):
    """Extract indices corresponding to two specific classes."""
    return np.where((y == c1) | (y == c2))[0]


def compute_statevectors(X, feature_map):
    """Compute statevector representation for a set of samples."""
    statevectors = []
    for x in tqdm(X, desc="Computing statevectors"):
        bound = feature_map.assign_parameters(x)
        sv = Statevector(bound)
        statevectors.append(sv.data)
    return np.array(statevectors)


def build_kernel_from_statevectors(sv1, sv2):
    """Compute kernel matrix via complex matrix multiplication.
    K[i,j] = |<ψ(x_i)|ψ(x_j)>|² — same formula as FidelityQuantumKernel."""
    return np.abs(np.dot(sv1, sv2.conj().T)) ** 2


def train_pegasos():
    print("=" * 65)
    print("MULTICLASS PEGASOS QSVC  (Statevector Precomputed Kernel)")
    print("=" * 65)

    # ── 1. Load 9D features ─────────────────────────────────────────────────
    features_path = config.FEATURES_DIR / 'features_9d.npz'
    if not features_path.exists():
        print(f"ERROR: {features_path} not found. Run reduce_dimensions.py first.")
        return None

    print("Loading 9D features …")
    data      = np.load(features_path)
    X_train   = data['train_features']
    X_test    = data['test_features']
    y_train   = data['y_train']
    y_test    = data['y_test']
    print(f"  Train: {X_train.shape}   Test: {X_test.shape}")

    # Setup ZZFeatureMap
    feature_map = ZZFeatureMap(
        feature_dimension=config.FEATURE_DIMENSION,
        reps=config.REPS,
        entanglement='linear'
    )

    # ── 2. Load or Compute Statevectors ──────────────────────────────────────
    config.FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    sv_train_path = config.FEATURES_DIR / 'sv_train.npz'
    sv_test_path  = config.FEATURES_DIR / 'sv_test.npz'

    if sv_train_path.exists():
        print("Found existing train statevectors, loading …")
        sv_train = np.load(sv_train_path, allow_pickle=True)['sv_train']
    else:
        print("Computing train statevectors …")
        sv_train = compute_statevectors(X_train, feature_map)
        np.savez_compressed(sv_train_path, sv_train=sv_train)

    if sv_test_path.exists():
        print("Found existing test statevectors, loading …")
        sv_test = np.load(sv_test_path, allow_pickle=True)['sv_test']
    else:
        print("Computing test statevectors …")
        sv_test = compute_statevectors(X_test, feature_map)
        np.savez_compressed(sv_test_path, sv_test=sv_test)

    # ── 3. Compute Full Kernel Matrices (instant matrix multiply) ─────────────
    print("Building kernel matrices …")
    K_train_full = build_kernel_from_statevectors(sv_train, sv_train).astype(np.float32)
    K_test_full  = build_kernel_from_statevectors(sv_test, sv_train).astype(np.float32)

    # ── 4. Train 6 Binary Models ─────────────────────────────────────────────
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    models = {}
    total_start = time.time()

    for c1, c2 in CLASS_PAIRS:
        model_path = config.MODELS_DIR / f"pegasos_{c1}_{c2}.pkl"
        name = f"pegasos_{c1}_{c2}"

        if model_path.exists():
            print(f"[{name}] Found saved model – loading …")
            models[(c1, c2)] = joblib.load(model_path)
        else:
            idx = get_binary_indices(y_train, c1, c2)
            K_bin = K_train_full[np.ix_(idx, idx)]
            y_bin = y_train[idx]

            print(f"[{name}] Training on {len(idx)} samples …")
            svc = SVC(kernel='precomputed', C=config.C_PARAM, random_state=config.RANDOM_SEED)
            svc.fit(K_bin, y_bin)
            joblib.dump(svc, model_path)
            models[(c1, c2)] = svc
            print(f"  Saved → {model_path.name}")

    print(f"\nAll 6 binary models ready. Total time: {time.time() - total_start:.2f} s")

    # ── 5. Predict using Decision Tree (Algorithm 1) ─────────────────────────
    print("\nRunning multiclass decision-tree inference on test set …")
    y_pred = []

    for i in tqdm(range(len(X_test)), desc="Predicting"):
        def predict_node(c1, c2):
            idx = get_binary_indices(y_train, c1, c2)
            k_row = K_test_full[i, idx].reshape(1, -1)
            return models[(c1, c2)].predict(k_row)[0]

        pred_01 = predict_node(0, 1)
        pred_23 = predict_node(2, 3)

        if pred_01 == 0:
            final = predict_node(0, 2) if pred_23 == 2 else predict_node(0, 3)
        else:
            final = predict_node(1, 2) if pred_23 == 2 else predict_node(1, 3)

        y_pred.append(final)

    y_pred = np.array(y_pred)

    # ── 6. Evaluate ───────────────────────────────────────────────────────────
    acc = accuracy_score(y_test, y_pred)
    print(f"\n✅ Pegasos QSVC Accuracy: {acc * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=config.CLASS_NAMES,
                                zero_division=0))

    # ── 7. Confusion Matrix ───────────────────────────────────────────────────
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples',
                xticklabels=config.CLASS_NAMES, yticklabels=config.CLASS_NAMES)
    plt.title(f'Multiclass Pegasos QSVC\nAccuracy: {acc*100:.1f}%')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    cm_path = config.RESULTS_DIR / 'confusion_matrix_pegasos.png'
    plt.savefig(cm_path, dpi=150)
    print(f"Confusion matrix saved → {cm_path}")

    # ── 8. Save results JSON ──────────────────────────────────────────────────
    results = {
        'model': 'Multiclass Pegasos QSVC (Statevector precomputed, 6 binary models)',
        'accuracy': float(acc),
        'feature_dimension': config.FEATURE_DIMENSION,
        'reps': config.REPS,
        'C': config.C_PARAM,
    }
    results_path = config.RESULTS_DIR / 'pegasos_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved → {results_path}")

    return results


if __name__ == "__main__":
    train_pegasos()
