"""
Multiclass Pegasos QSVC Training Script
========================================
Based on the QuCardio paper (Section IV-B):
  - PegasosQSVC is binary-only → train 6 binary models (one per class pair)
  - Use paper's Algorithm 1 (3-step decision tree) for multiclass prediction

Implementation — Custom Pegasos SGD with precomputed statevector kernel:
  Qiskit's native PegasosQSVC + FidelityQuantumKernel has two problems on a
  classical CPU simulator:
    1. Speed: on-the-fly circuit evaluation during SGD is very slow.
    2. Stability: 9-qubit ZZFeatureMap produces avg kernel value ≈ 1/2⁹ ≈ 0.002,
       which is too small for the standard SGD learning rate schedule η_t = 1/(λt)
       to generate meaningful weight updates → model collapses.

  Solution:
    1. Compute all statevectors ONCE (N circuits, ~3 seconds).
    2. Build full kernel matrix via matrix multiply (instant).
    3. Run our own Pegasos SGD loop — at each step, kernel access is an O(1)
       row lookup from the precomputed matrix instead of a circuit execution.
    This gives TRUE Pegasos SGD (stochastic, approximate, different from QP/SVC)
    while running in seconds. The result differs from QSVC because SGD ≠ QP solver.

Class pairs:
  (0,1) Normal vs Arrhythmia        (0,2) Normal vs Myocardial_Infarction
  (0,3) Normal vs History_of_MI     (1,2) Arrhythmia vs Myocardial_Infarction
  (1,3) Arrhythmia vs History_of_MI (2,3) Myocardial_Infarction vs History_of_MI

Decision Tree (Algorithm 1 from paper):
  pred_01 ← model(0,1);  pred_23 ← model(2,3)
  if pred_01==0: final ← model(0,2) if pred_23==2 else model(0,3)
  else:          final ← model(1,2) if pred_23==2 else model(1,3)
"""

import sys
import time
import numpy as np
import joblib
import json
from pathlib import Path
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


# ─────────────────────────────────────────────────────────────────────────────
# Custom Pegasos SGD Solver
# ─────────────────────────────────────────────────────────────────────────────

class PegasosSVMKernel:
    """
    True Pegasos sub-gradient SVM using a precomputed kernel matrix.

    Algorithm (Shalev-Shwartz et al., 2007):
      λ = 1 / (N * C)
      for t = 1 … T:
        i     ← random sample
        score ← (1/λt) * Σ_j α_j * y_j * K(x_i, x_j)
        if y_i * score < 1:  α_i += 1   (hinge loss active → update)

    Prediction:  sign( Σ_j α_j * y_j * K(x_test, x_j) )

    Labels must be in {-1, +1}. Use remap_labels_to_binary() before fit().
    """

    def __init__(self, C=1.0, num_steps=1000, seed=42):
        self.C = C
        self.num_steps = num_steps
        self.seed = seed
        self.alpha_ = None
        self.y_train_ = None
        self.support_indices_ = None

    def fit(self, K_train, y_binary):
        """
        K_train  : (N, N) precomputed kernel matrix (float64)
        y_binary : (N,)   labels in {-1, +1}
        """
        np.random.seed(self.seed)
        N = len(y_binary)
        lambda_param = 1.0 / (N * self.C)
        alpha = np.zeros(N, dtype=np.float64)

        for t in range(1, self.num_steps + 1):
            i = np.random.randint(0, N)
            # w·φ(x_i) = (1/λt) * Σ_j α_j * y_j * K(x_i, x_j)
            score = (1.0 / (lambda_param * t)) * np.dot(alpha * y_binary, K_train[i])
            if y_binary[i] * score < 1.0:   # hinge loss active
                alpha[i] += 1.0

        self.alpha_ = alpha
        self.y_train_ = y_binary.copy()
        self.support_indices_ = np.where(alpha > 0)[0]
        return self

    def predict(self, K_test_rows):
        """
        K_test_rows : (M, N) kernel rows — test samples vs all train
        Returns     : (M,) predicted labels in {-1, +1}
        """
        decision = np.dot(K_test_rows, self.alpha_ * self.y_train_)
        return np.where(decision >= 0, 1, -1).astype(np.int64)

    def decision_function(self, K_test_rows):
        return np.dot(K_test_rows, self.alpha_ * self.y_train_)


def remap_labels_to_binary(y, c1, c2):
    """Map class c1 → -1 and c2 → +1 for Pegasos SGD."""
    return np.where(y == c1, -1, 1).astype(np.float64)


def get_binary_indices(y, c1, c2):
    """Extract indices of samples belonging to classes c1 or c2."""
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
    print("MULTICLASS PEGASOS QSVC  (Custom SGD + Statevector Kernel)")
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
    t0 = time.time()
    K_train_full = build_kernel_from_statevectors(sv_train, sv_train).astype(np.float64)
    K_test_full  = build_kernel_from_statevectors(sv_test,  sv_train).astype(np.float64)
    print(f"  Done in {time.time()-t0:.4f}s")

    # ── 4. Train 6 Binary Pegasos SGD Models ────────────────────────────────
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    models = {}
    total_start = time.time()

    print(f"\nTraining 6 binary Pegasos SGD models "
          f"(C={config.PEGASOS_C}, steps={config.N_STEPS}) …")

    for c1, c2 in CLASS_PAIRS:
        model_path = config.MODELS_DIR / f"pegasos_{c1}_{c2}.pkl"
        name = f"pegasos_{c1}_{c2}"

        if model_path.exists():
            print(f"  [{name}] Found saved model – loading …")
            models[(c1, c2)] = joblib.load(model_path)
        else:
            idx   = get_binary_indices(y_train, c1, c2)
            K_bin = K_train_full[np.ix_(idx, idx)]
            y_bin = remap_labels_to_binary(y_train[idx], c1, c2)  # {-1, +1}

            print(f"  [{name}] Training on {len(idx)} samples …")
            t0 = time.time()
            model = PegasosSVMKernel(
                C=config.PEGASOS_C,
                num_steps=config.N_STEPS,
                seed=config.RANDOM_SEED
            )
            model.fit(K_bin, y_bin)
            # Store training indices and class labels so predict_node can slice K_test
            model.train_indices_ = idx
            model.c1_ = c1
            model.c2_ = c2
            joblib.dump(model, model_path)
            models[(c1, c2)] = model
            n_sv = len(model.support_indices_)
            print(f"    Done in {time.time()-t0:.3f}s | support vectors: {n_sv}/{len(idx)}")

    print(f"\nAll 6 models ready. Total train time: {time.time() - total_start:.2f}s")

    # ── 5. Decision Tree Inference (Algorithm 1 from paper) ─────────────────
    print("\nRunning multiclass decision-tree inference …")

    def predict_node(i_test, c1, c2):
        """Predict original class label for test sample i using model (c1, c2).
        Returns c1 if Pegasos predicts -1, c2 if it predicts +1."""
        m     = models[(c1, c2)]
        idx   = m.train_indices_
        k_row = K_test_full[i_test, idx].reshape(1, -1)
        pred  = m.predict(k_row)[0]   # -1 or +1
        return c1 if pred == -1 else c2

    y_pred = []
    for i in tqdm(range(len(X_test)), desc="Predicting"):
        pred_01 = predict_node(i, 0, 1)
        pred_23 = predict_node(i, 2, 3)

        if pred_01 == 0:
            final = predict_node(i, 0, 2) if pred_23 == 2 else predict_node(i, 0, 3)
        else:
            final = predict_node(i, 1, 2) if pred_23 == 2 else predict_node(i, 1, 3)

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
    plt.close()
    print(f"Confusion matrix saved → {cm_path}")

    # ── 8. Save results JSON ──────────────────────────────────────────────────
    results = {
        'model': 'Multiclass Pegasos QSVC (Custom SGD, Statevector kernel)',
        'accuracy': float(acc),
        'feature_dimension': config.FEATURE_DIMENSION,
        'reps': config.REPS,
        'C': config.PEGASOS_C,
        'num_steps': config.N_STEPS,
        'entanglement': 'linear',
        'kernel': 'ZZFeatureMap statevector fidelity',
    }
    results_path = config.RESULTS_DIR / 'pegasos_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved → {results_path}")

    return results


if __name__ == "__main__":
    train_pegasos()
