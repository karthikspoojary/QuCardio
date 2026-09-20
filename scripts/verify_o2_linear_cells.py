"""
O2 Verification: Are the two linear cells genuinely different?

Both [0,1]+linear and [0,pi]+linear give 89.78% (167/186).
This script:
  1. Rebuilds QSVC predictions for both configurations from scratch
  2. Saves prediction arrays
  3. Runs McNemar between them
  4. Reports whether b=0, c=0 (identical vectors) or b+c > 0 (different errors)

Output: results/paper/ablation/o2_linear_mcnemar.json
"""
import numpy as np
import json
from pathlib import Path
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from qiskit.circuit.library import ZZFeatureMap
from qiskit.quantum_info import Statevector
from tqdm import tqdm

OUT = Path("results/paper/ablation/o2_linear_mcnemar.json")

# ── Load features ─────────────────────────────────────────────────────────────
d = np.load("data/features_9d.npz")
X_train_01 = d["train_features"]   # already [0,1]
X_test_01  = d["test_features"]
y_train    = d["y_train"]
y_test     = d["y_test"]

X_train_0pi = X_train_01 * np.pi   # [0,pi]
X_test_0pi  = X_test_01  * np.pi

# ── Build feature map: linear, reps=2 ─────────────────────────────────────────
fm = ZZFeatureMap(feature_dimension=9, reps=2, entanglement="linear")

def compute_svs(X, label):
    svs = []
    for x in tqdm(X, desc=f"  SV {label}", leave=False):
        bound = fm.assign_parameters(x)
        svs.append(Statevector(bound).data)
    return np.array(svs)

def build_kernel(sv1, sv2):
    return (np.abs(np.dot(sv1, sv2.conj().T)) ** 2).astype(np.float32)

# ── Configuration A: [0,1] + linear ──────────────────────────────────────────
print("\n=== Configuration A: [0,1] + linear ===")
sv_tr_A = compute_svs(X_train_01, "[0,1] train")
sv_te_A = compute_svs(X_test_01,  "[0,1] test")
K_tr_A  = build_kernel(sv_tr_A, sv_tr_A)
K_te_A  = build_kernel(sv_te_A, sv_tr_A)
clf_A   = SVC(kernel="precomputed", C=5.0, random_state=42)
clf_A.fit(K_tr_A, y_train)
y_pred_A = clf_A.predict(K_te_A)
acc_A    = accuracy_score(y_test, y_pred_A)
print(f"  Accuracy: {acc_A*100:.2f}%  ({int(acc_A*186)}/186)")

# ── Configuration B: [0,pi] + linear ─────────────────────────────────────────
print("\n=== Configuration B: [0,pi] + linear ===")
sv_tr_B = compute_svs(X_train_0pi, "[0,pi] train")
sv_te_B = compute_svs(X_test_0pi,  "[0,pi] test")
K_tr_B  = build_kernel(sv_tr_B, sv_tr_B)
K_te_B  = build_kernel(sv_te_B, sv_tr_B)
clf_B   = SVC(kernel="precomputed", C=5.0, random_state=42)
clf_B.fit(K_tr_B, y_train)
y_pred_B = clf_B.predict(K_te_B)
acc_B    = accuracy_score(y_test, y_pred_B)
print(f"  Accuracy: {acc_B*100:.2f}%  ({int(acc_B*186)}/186)")

# ── Diff the prediction vectors ────────────────────────────────────────────────
n_identical = int(np.sum(y_pred_A == y_pred_B))
n_differ    = 186 - n_identical
# McNemar counts for A vs B
# b = B correct, A wrong
# c = A correct, B wrong
correct_A = (y_pred_A == y_test)
correct_B = (y_pred_B == y_test)
b = int(np.sum(~correct_A & correct_B))   # B right, A wrong
c = int(np.sum( correct_A & ~correct_B))  # A right, B wrong
b_plus_c = b + c

# Continuity-corrected McNemar chi2
if b_plus_c > 0:
    chi2 = (abs(b - c) - 1.0) ** 2 / (b + c)
else:
    chi2 = 0.0

print(f"\n=== O2 Summary ===")
print(f"  Positions where predictions DIFFER : {n_differ}")
print(f"  Positions where predictions AGREE  : {n_identical}")
print(f"  b (B right, A wrong)               : {b}")
print(f"  c (A right, B wrong)               : {c}")
print(f"  McNemar chi2 (continuity corrected): {chi2:.3f}")

if n_differ == 0:
    verdict = "IDENTICAL: encoding change never reached the circuit — BUG"
elif b == 0 and c == 0:
    verdict = "SAME_ERRORS: predictions differ at non-test-label positions only"
else:
    verdict = f"DIFFERENT: b={b}, c={c} — genuine discordance, models behave differently"

print(f"  Verdict: {verdict}")

# ── Save result ────────────────────────────────────────────────────────────────
result = {
    "config_A": {"encoding": "[0,1]",  "entanglement": "linear", "accuracy": round(float(acc_A), 6)},
    "config_B": {"encoding": "[0,pi]", "entanglement": "linear", "accuracy": round(float(acc_B), 6)},
    "n_test": 186,
    "positions_differ": n_differ,
    "positions_agree":  n_identical,
    "b_B_right_A_wrong": b,
    "c_A_right_B_wrong": c,
    "mcnemar_chi2_cc":   round(float(chi2), 4),
    "verdict": verdict,
}

with open(OUT, "w") as f:
    json.dump(result, f, indent=2)
print(f"\nSaved → {OUT}")
