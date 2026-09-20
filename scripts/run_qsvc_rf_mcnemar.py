"""
A2 — QSVC vs Random Forest McNemar test
========================================
Both sit at similar accuracy (QSVC=94.62%, RF=91.94%).
This script rebuilds both prediction vectors from saved models,
computes the 2x2 contingency table, and runs McNemar's test.

Output: results/paper/statistics/mcnemar_qsvc_rf.json
"""
import numpy as np
import json
import joblib
import sys
from pathlib import Path
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

sys.path.append(str(Path(__file__).resolve().parent))
import config

OUT = Path("results/paper/statistics/mcnemar_qsvc_rf.json")

# ── Load 9D features ──────────────────────────────────────────────────────────
feat_path = config.FEATURES_DIR / 'features_9d.npz'
d         = np.load(feat_path)
X_train   = d['train_features']
X_test    = d['test_features']
y_train   = d['y_train']
y_test    = d['y_test']
print(f"Features: train={X_train.shape}  test={X_test.shape}")

# ── RF predictions ─────────────────────────────────────────────────────────────
print("\n=== Random Forest (n_estimators=500, balanced) ===")
rf = RandomForestClassifier(
    n_estimators=500,
    max_depth=None,
    class_weight='balanced',
    random_state=config.RANDOM_SEED,
    n_jobs=-1,
)
rf.fit(X_train, y_train)
y_pred_rf  = rf.predict(X_test)
acc_rf     = accuracy_score(y_test, y_pred_rf)
print(f"  Accuracy: {acc_rf*100:.4f}%  ({int(round(acc_rf*len(y_test)))}/{len(y_test)})")

# ── QSVC predictions ───────────────────────────────────────────────────────────
print("\n=== QSVC (precomputed kernel, C=5.0) ===")
qsvc_path  = config.MODELS_DIR / 'qsvc_model.pkl'
sv_tr_path = config.FEATURES_DIR / 'sv_train_qsvc.npz'
sv_te_path = config.FEATURES_DIR / 'sv_test_qsvc.npz'

if not (qsvc_path.exists() and sv_tr_path.exists() and sv_te_path.exists()):
    print("ERROR: QSVC model or statevector files not found.")
    print(f"  qsvc_model: {qsvc_path.exists()}")
    print(f"  sv_train:   {sv_tr_path.exists()}")
    print(f"  sv_test:    {sv_te_path.exists()}")
    sys.exit(1)

qsvc   = joblib.load(qsvc_path)
sv_tr  = np.load(sv_tr_path, allow_pickle=True)['sv_train']
sv_te  = np.load(sv_te_path, allow_pickle=True)['sv_test']
K_te   = (np.abs(np.dot(sv_te, sv_tr.conj().T)) ** 2).astype(np.float32)

y_pred_qsvc = qsvc.predict(K_te)
acc_qsvc    = accuracy_score(y_test, y_pred_qsvc)
print(f"  Accuracy: {acc_qsvc*100:.4f}%  ({int(round(acc_qsvc*len(y_test)))}/{len(y_test)})")

# ── McNemar contingency table ─────────────────────────────────────────────────
correct_qsvc = (y_pred_qsvc == y_test)
correct_rf   = (y_pred_rf   == y_test)

b = int(np.sum(~correct_qsvc & correct_rf))   # RF right, QSVC wrong
c = int(np.sum( correct_qsvc & ~correct_rf))  # QSVC right, RF wrong

# Continuity-corrected McNemar
b_plus_c = b + c
if b_plus_c > 0:
    chi2 = (abs(b - c) - 1.0) ** 2 / b_plus_c
else:
    chi2 = 0.0

# p-value from chi-squared with df=1
from scipy.stats import chi2 as chi2_dist
p_value     = float(chi2_dist.sf(chi2, df=1))
significant = p_value < 0.05

print(f"\n=== McNemar: QSVC vs Random Forest ===")
print(f"  b (RF right, QSVC wrong): {b}")
print(f"  c (QSVC right, RF wrong): {c}")
print(f"  chi2 (CC):                {chi2:.4f}")
print(f"  p-value:                  {p_value:.6f}")
print(f"  Significant (p<0.05):     {significant}")

result = {
    "model_a":     "QSVC",
    "model_b":     "Random Forest",
    "accuracy_a":  round(float(acc_qsvc), 6),
    "accuracy_b":  round(float(acc_rf),   6),
    "n_test":      int(len(y_test)),
    "contingency": {"b_b_right_a_wrong": b, "c_a_right_b_wrong": c},
    "mcnemar_chi2_cc": round(float(chi2), 4),
    "p_value":     round(p_value, 6),
    "significant": significant,
    "conclusion":  (
        f"QSVC ({acc_qsvc*100:.2f}%) vs Random Forest ({acc_rf*100:.2f}%): "
        f"b={b}, c={c}, chi2={chi2:.3f}, p={p_value:.3f}. "
        f"{'Significant' if significant else 'NOT significant'} at alpha=0.05."
    ),
}

OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, 'w') as f:
    json.dump(result, f, indent=2)
print(f"\nSaved → {OUT}")
