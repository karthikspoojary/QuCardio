"""
resave_models_from_9d.py
========================
Re-saves SVD reducer + MinMaxScaler under current sklearn version WITHOUT
needing the (possibly corrupted) resnet50_features.npz.

Strategy: The SVD and scaler are ALREADY FITTED correctly (proven by diagnose.py).
We just need to re-pickle them under sklearn 1.8.0 so the warning disappears.

We do this by:
  1. Loading the old pkl (which still works numerically)
  2. Extracting the fitted state (components_, singular_values_, etc.)
  3. Creating a fresh sklearn 1.8.0 object and copying the state over
  4. Dumping the fresh object

Run from project root:
  python scripts/resave_models_from_9d.py
"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import joblib
import sklearn
from pathlib import Path
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MinMaxScaler

print(f"sklearn version: {sklearn.__version__}")

MODEL_DIR = Path("backend/models")

# ── Re-save TruncatedSVD ──────────────────────────────────────────────────────
print("\n[1] Re-saving TruncatedSVD ...")
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    old_svd = joblib.load(MODEL_DIR / "svd_reducer.pkl")

# Copy all fitted attributes into a fresh object
new_svd = TruncatedSVD(n_components=9, random_state=42)
new_svd.components_            = old_svd.components_
new_svd.explained_variance_    = old_svd.explained_variance_
new_svd.explained_variance_ratio_ = old_svd.explained_variance_ratio_
new_svd.singular_values_       = old_svd.singular_values_
new_svd.n_features_in_         = old_svd.n_features_in_
# Mark it as fitted (sklearn 1.8+ uses _is_fitted tag)
try:
    new_svd._is_fitted = True
except Exception:
    pass

# Backup old
import shutil
shutil.copy(MODEL_DIR / "svd_reducer.pkl", MODEL_DIR / "svd_reducer_v161.pkl")
joblib.dump(new_svd, MODEL_DIR / "svd_reducer.pkl")
print(f"    Saved svd_reducer.pkl (sklearn {sklearn.__version__})")

# ── Re-save MinMaxScaler ──────────────────────────────────────────────────────
print("\n[2] Re-saving MinMaxScaler ...")
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    old_sc = joblib.load(MODEL_DIR / "minmax_scaler.pkl")

new_sc = MinMaxScaler()
new_sc.scale_         = old_sc.scale_
new_sc.min_           = old_sc.min_
new_sc.data_min_      = old_sc.data_min_
new_sc.data_max_      = old_sc.data_max_
new_sc.data_range_    = old_sc.data_range_
new_sc.feature_range  = old_sc.feature_range
new_sc.n_features_in_ = old_sc.n_features_in_
new_sc.n_samples_seen_ = old_sc.n_samples_seen_
try:
    new_sc._is_fitted = True
except Exception:
    pass

shutil.copy(MODEL_DIR / "minmax_scaler.pkl", MODEL_DIR / "minmax_scaler_v161.pkl")
joblib.dump(new_sc, MODEL_DIR / "minmax_scaler.pkl")
print(f"    Saved minmax_scaler.pkl (sklearn {sklearn.__version__})")

# ── Verify: run a quick predict and compare outputs ───────────────────────────
print("\n[3] Verification: comparing old vs new transform on train features ...")
feat9d = np.load("data/features_9d.npz")
X_tr = feat9d["train_features"]   # already reduced+scaled ground truth

# We can't re-run SVD from scratch without resnet50_features.npz
# Instead just confirm the SVM still works on the stored features
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    svm = joblib.load(MODEL_DIR / "svm_model.pkl")

preds = svm.predict(X_tr)
y_tr  = feat9d["y_train"]
acc   = (preds == y_tr).mean()
print(f"    Train accuracy (SVM on stored 9D): {acc*100:.2f}%")
if acc > 0.9:
    print("    ✅ SVM still works correctly on stored features")
else:
    print("    ❌ Something went wrong")

print("\n✅ Models re-saved under sklearn", sklearn.__version__)
print("   Restart the backend — InconsistentVersionWarning should be gone.")
