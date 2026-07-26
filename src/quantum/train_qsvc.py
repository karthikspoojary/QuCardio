import os
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


def compute_statevectors(X, feature_map):
    """
    Compute one quantum statevector per sample.
    This is the fast approach: N circuits (not N×N circuits).
    """
    statevectors = []
    for i, x in enumerate(tqdm(X, desc="Computing statevectors")):
        bound = feature_map.assign_parameters(x)
        sv = Statevector(bound)
        statevectors.append(sv.data)
    return np.array(statevectors)  # shape (N, 2^n_qubits)


def build_kernel_from_statevectors(sv1, sv2):
    """
    Compute kernel matrix K[i,j] = |<psi_i | psi_j>|^2 via matrix multiplication.
    This replaces N×N separate circuit simulations with a single matrix multiply.
    sv1: (M, D) complex array
    sv2: (N, D) complex array
    Returns: (M, N) real kernel matrix
    """
    return np.abs(np.dot(sv1, sv2.conj().T)) ** 2


def train_qsvc():
    print("=" * 60)
    print("QUANTUM SUPPORT VECTOR CLASSIFIER (QSVC)")
    print("=" * 60)
    
    # 1. Load Data
    features_path = config.FEATURES_DIR / 'features_9d.npz'
    if not features_path.exists():
        print(f"Error: {features_path} not found. Run reduce_dimensions.py first.")
        return
        
    print("Loading 9D features...")
    data = np.load(features_path)
    X_train = data['train_features']
    X_test = data['test_features']
    y_train = data['y_train']
    y_test = data['y_test']
    
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    
    # Paths for caching statevectors and kernel matrices
    config.FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    sv_train_path = config.FEATURES_DIR / 'sv_train.npz'
    k_train_path  = config.FEATURES_DIR / 'K_train.npz'
    k_test_path   = config.FEATURES_DIR / 'K_test.npz'

    # 2. Setup ZZFeatureMap (no sampler needed for Statevector method)
    feature_map = ZZFeatureMap(
        feature_dimension=config.FEATURE_DIMENSION,
        reps=config.REPS,
        entanglement='linear'
    )

    # 3. Compute or Load training statevectors  (only 742 circuits, not 742×742!)
    if sv_train_path.exists():
        print(f"Found existing {sv_train_path}, loading …")
        sv_train = np.load(sv_train_path, allow_pickle=True)['sv_train']
    else:
        print("Computing training statevectors (N circuits, very fast) …")
        start = time.time()
        sv_train = compute_statevectors(X_train, feature_map)
        np.savez_compressed(sv_train_path, sv_train=sv_train)
        print(f"Done in {time.time()-start:.2f} s  →  saved to {sv_train_path}")

    # 4. Compute or Load K_train  (matrix multiply, instant)
    if k_train_path.exists():
        print(f"Found existing {k_train_path}, loading …")
        K_train = np.load(k_train_path)['K_train']
    else:
        print("Building K_train via matrix multiplication …")
        start = time.time()
        K_train = build_kernel_from_statevectors(sv_train, sv_train).astype(np.float32)
        np.savez_compressed(k_train_path, K_train=K_train)
        print(f"Done in {time.time()-start:.4f} s  →  saved to {k_train_path}")

    # 5. Compute or Load K_test  (186 circuits + matrix multiply)
    if k_test_path.exists():
        print(f"Found existing {k_test_path}, loading …")
        K_test = np.load(k_test_path)['K_test']
    else:
        print("Computing test statevectors …")
        start = time.time()
        sv_test = compute_statevectors(X_test, feature_map)
        print("Building K_test via matrix multiplication …")
        K_test = build_kernel_from_statevectors(sv_test, sv_train).astype(np.float32)
        np.savez_compressed(k_test_path, K_test=K_test)
        print(f"Done in {time.time()-start:.2f} s  →  saved to {k_test_path}")

    # 6. Train SVC with precomputed kernel
    print("\nTraining QSVC (precomputed kernel)...")
    qsvc = SVC(kernel='precomputed', C=config.C_PARAM, random_state=config.RANDOM_SEED)
    start = time.time()
    qsvc.fit(K_train, y_train)
    train_time = time.time() - start
    print(f"QSVC fit completed in {train_time:.2f} seconds.")

    # 7. Evaluate
    print("\nEvaluating QSVC...")
    y_pred = qsvc.predict(K_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"\n✅ QSVC Accuracy: {acc * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=config.CLASS_NAMES))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=config.CLASS_NAMES, yticklabels=config.CLASS_NAMES)
    plt.title(f'Quantum SVC\nAccuracy: {acc*100:.1f}%')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    cm_path = config.RESULTS_DIR / 'confusion_matrix_qsvc.png'
    plt.savefig(cm_path, dpi=150)
    print(f"Confusion matrix saved to {cm_path}")

    # Save Model + results
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = config.MODELS_DIR / 'qsvc_model.pkl'
    joblib.dump(qsvc, model_path)
    print(f"Model saved to {model_path}")

    results = {
        'model': 'QSVC (ZZFeatureMap, Statevector kernel)',
        'accuracy': float(acc),
        'feature_dimension': config.FEATURE_DIMENSION,
        'reps': config.REPS,
        'C': config.C_PARAM,
        'train_time_seconds': train_time
    }
    results_path = config.RESULTS_DIR / 'qsvc_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_path}")

    return results

if __name__ == "__main__":
    train_qsvc()
