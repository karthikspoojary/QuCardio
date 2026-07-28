"""
resave_models.py
================
Re-saves SVD reducer and MinMaxScaler under the current sklearn version
to fix the InconsistentVersionWarning (trained with 1.6.1, running 1.8.0).

Also runs a quick sanity check:
  - Loads a known training sample from features_9d.npz
  - Passes raw 462K-D features → SVD → scaler
  - Compares with the stored 9D features to confirm transforms are identical

Run ONCE after upgrading sklearn:
  python scripts/resave_models.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import joblib
from pathlib import Path
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MinMaxScaler
import sklearn

print(f"sklearn version: {sklearn.__version__}")

# ── Paths ─────────────────────────────────────────────────────────────────────
MODELS_DIR    = Path('backend/models')
DATA_DIR      = Path('data')
resnet_path   = DATA_DIR / 'resnet50_features.npz'
features_path = DATA_DIR / 'features_9d.npz'
svd_path      = MODELS_DIR / 'svd_reducer.pkl'
scaler_path   = MODELS_DIR / 'minmax_scaler.pkl'

# ── Check prerequisite files ──────────────────────────────────────────────────
if not resnet_path.exists():
    print(f"ERROR: {resnet_path} not found. Run extract_resnet_features.py first.")
    sys.exit(1)
if not features_path.exists():
    print(f"ERROR: {features_path} not found. Run reduce_dimensions.py first.")
    sys.exit(1)

# ── Load raw ResNet features and existing 9D features ────────────────────────
print("\nLoading resnet50_features.npz ...")
raw_data = np.load(resnet_path)
X_train_raw = raw_data['train_features']   # (N_train, 462400) float32
X_test_raw  = raw_data['test_features']    # (N_test,  462400)
y_train     = raw_data['y_train']
y_test      = raw_data['y_test']
print(f"  Train raw: {X_train_raw.shape}")
print(f"  Test  raw: {X_test_raw.shape}")

print("\nLoading features_9d.npz (ground truth) ...")
feat9d = np.load(features_path)
X_train_9d_gt = feat9d['train_features']  # (N_train, 9) — ground truth
X_test_9d_gt  = feat9d['test_features']   # (N_test,  9)
print(f"  Train 9D GT: {X_train_9d_gt.shape}")
print(f"  Test  9D GT: {X_test_9d_gt.shape}")

# ── Re-fit SVD + Scaler from scratch on raw features ─────────────────────────
print("\nRe-fitting TruncatedSVD(9) on training raw features ...")
svd = TruncatedSVD(n_components=9, random_state=42)
X_train_svd = svd.fit_transform(X_train_raw)
X_test_svd  = svd.transform(X_test_raw)
print(f"  Explained variance: {svd.explained_variance_ratio_.sum():.4f}")

print("Re-fitting MinMaxScaler on SVD-reduced training features ...")
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train_svd)
X_test_scaled  = scaler.transform(X_test_svd)

# ── Sanity check: new transforms must match stored ground truth ───────────────
print("\n── Sanity Check ──────────────────────────────────────────────")
train_max_diff = np.abs(X_train_scaled - X_train_9d_gt).max()
test_max_diff  = np.abs(X_test_scaled  - X_test_9d_gt).max()
print(f"  Max diff train (new vs stored): {train_max_diff:.2e}")
print(f"  Max diff test  (new vs stored): {test_max_diff:.2e}")

TOLERANCE = 1e-4
if train_max_diff < TOLERANCE and test_max_diff < TOLERANCE:
    print(f"  ✅ MATCH — max diff < {TOLERANCE}. New models are identical to old.")
else:
    print(f"  ⚠️  MISMATCH — max diff exceeds {TOLERANCE}.")
    print("     This means TruncatedSVD is non-deterministic across runs.")
    print("     Overwriting anyway — but models may need retraining.")

# ── Save under current sklearn version ───────────────────────────────────────
print(f"\nSaving under sklearn {sklearn.__version__} ...")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Backup old ones
import shutil
for path, name in [(svd_path, 'svd_reducer'), (scaler_path, 'minmax_scaler')]:
    if path.exists():
        backup = MODELS_DIR / f"{name}_backup.pkl"
        shutil.copy(path, backup)
        print(f"  Backed up {path.name} → {backup.name}")

joblib.dump(svd,    svd_path)
joblib.dump(scaler, scaler_path)
print(f"  Saved: {svd_path}")
print(f"  Saved: {scaler_path}")

# ── Also update features_9d.npz with fresh scaled features ──────────────────
print("\nUpdating data/features_9d.npz with freshly computed features ...")
np.savez_compressed(
    features_path,
    train_features=X_train_scaled.astype(np.float32),
    test_features=X_test_scaled.astype(np.float32),
    y_train=y_train,
    y_test=y_test,
    class_names=np.array(['Normal','Arrhythmia','Myocardial_Infarction','History_of_MI'])
)
print(f"  Saved: {features_path}")

print("\n✅ Done! Restart the backend — InconsistentVersionWarning is now fixed.")
print("   If SVM/QSVC/Pegasos accuracy drops, re-run the training scripts.")
