"""
time_training_pipeline.py
Measures wall-clock time for each stage of the offline training pipeline.
Run from the project root: python src/time_training_pipeline.py
All models must already have their cached statevectors/features on disk.
"""

import time
import sys
import numpy as np
import joblib
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config

from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import TruncatedSVD
from qiskit.circuit.library import ZZFeatureMap
from qiskit.quantum_info import Statevector

print("=" * 60)
print("FULL OFFLINE TRAINING PIPELINE TIMING")
print("=" * 60)

# ── Stage 1: Load 9-D features (already extracted by ResNet50 + SVD offline) ──
t0 = time.perf_counter()
features_path = config.FEATURES_DIR / "features_9d.npz"
data = np.load(features_path)
X_train = data["train_features"]
X_test  = data["test_features"]
y_train = data["y_train"]
y_test  = data["y_test"]
t1 = time.perf_counter()
load_time = t1 - t0
print(f"[1] Load 9-D features ({X_train.shape[0]} train, {X_test.shape[0]} test): {load_time*1000:.1f} ms")

# ── Stage 2: MinMaxScaler fit (re-fit from scratch to time it) ──
t0 = time.perf_counter()
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled  = scaler.transform(X_test)
t1 = time.perf_counter()
scaler_time = t1 - t0
print(f"[2] MinMaxScaler fit + transform:                         {scaler_time*1000:.1f} ms")

# ── Stage 3: Classical SVM fit ──
t0 = time.perf_counter()
svm = SVC(kernel="rbf", C=10, class_weight="balanced", probability=True, random_state=42)
svm.fit(X_train_scaled, y_train)
t1 = time.perf_counter()
svm_fit_time = t1 - t0
print(f"[3] Classical SVM fit (742 samples, RBF, C=10):          {svm_fit_time:.2f} s")

# ── Stage 4: QSVC — statevectors + kernel matrix + SVC fit ──
feature_map = ZZFeatureMap(
    feature_dimension=config.FEATURE_DIMENSION,
    reps=config.REPS,
    entanglement="circular"
)
X_train_pi = X_train_scaled * np.pi

sv_train_path = config.FEATURES_DIR / "sv_train_qsvc_circular.npz"
if sv_train_path.exists():
    # Just time the kernel build + SVC fit (statevectors already cached)
    t0 = time.perf_counter()
    sv_train = np.load(sv_train_path, allow_pickle=True)["sv_train"]
    t_sv_load = time.perf_counter() - t0
    print(f"[4a] Load cached training statevectors (742):            {t_sv_load*1000:.1f} ms")

    t0 = time.perf_counter()
    K_train = np.abs(np.dot(sv_train, sv_train.conj().T)) ** 2
    t_kernel = time.perf_counter() - t0
    print(f"[4b] Build 742x742 kernel matrix (matrix multiply):     {t_kernel:.2f} s")

    t0 = time.perf_counter()
    qsvc = SVC(kernel="precomputed", C=5.0, random_state=42)
    qsvc.fit(K_train.astype(np.float32), y_train)
    t_qsvc_fit = time.perf_counter() - t0
    print(f"[4c] QSVC SVC fit (precomputed kernel):                 {t_qsvc_fit:.2f} s")
else:
    # Time full statevector computation
    from tqdm import tqdm
    t0 = time.perf_counter()
    statevectors = []
    for x in tqdm(X_train_pi, desc="Computing statevectors"):
        bound = feature_map.assign_parameters(x)
        sv = Statevector(bound)
        statevectors.append(sv.data)
    sv_train = np.array(statevectors)
    t_sv = time.perf_counter() - t0
    print(f"[4a] Compute 742 training statevectors:                 {t_sv:.2f} s")

    t0 = time.perf_counter()
    K_train = np.abs(np.dot(sv_train, sv_train.conj().T)) ** 2
    t_kernel = time.perf_counter() - t0
    print(f"[4b] Build 742x742 kernel matrix:                       {t_kernel:.2f} s")

    t0 = time.perf_counter()
    qsvc = SVC(kernel="precomputed", C=5.0, random_state=42)
    qsvc.fit(K_train.astype(np.float32), y_train)
    t_qsvc_fit = time.perf_counter() - t0
    print(f"[4c] QSVC SVC fit (precomputed kernel):                 {t_qsvc_fit:.2f} s")

print()
print("=" * 60)
print("SUMMARY")
print("=" * 60)
if sv_train_path.exists():
    total_from_features = load_time + scaler_time + svm_fit_time + t_sv_load + t_kernel + t_qsvc_fit
    print(f"Total (from 9-D features, statevectors cached): {total_from_features:.2f} s")
    print(f"  of which kernel matrix build: {t_kernel:.2f} s")
    print(f"  of which QSVC SVC.fit():      {t_qsvc_fit:.2f} s")
    print(f"  of which classical SVM fit:   {svm_fit_time:.2f} s")
else:
    total = load_time + scaler_time + svm_fit_time + t_sv + t_kernel + t_qsvc_fit
    print(f"Total (from 9-D features, fresh statevectors): {total:.2f} s")
    print(f"  of which statevector computation: {t_sv:.2f} s")
    print(f"  of which kernel matrix build:     {t_kernel:.2f} s")
    print(f"  of which QSVC SVC.fit():          {t_qsvc_fit:.2f} s")
print()
print("Note: ResNet50 feature extraction (all 928 images) and TruncatedSVD")
print("fit are one-time offline steps run separately by extract_resnet_features.py")
print("and reduce_dimensions.py. Timing those requires re-running from scratch.")
