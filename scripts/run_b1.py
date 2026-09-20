"""
B1 experiment: QSVC with [0,1] encoding + linear entanglement
This is the Prabhu et al. [5] configuration — the one missing cell in Table VII-A.
Features are already MinMax-scaled to [0,1] in features_9d.npz.
We do NOT multiply by pi. We use ZZFeatureMap with linear entanglement, reps=2, C=5.0.
"""
import numpy as np
from qiskit.circuit.library import ZZFeatureMap
from qiskit.quantum_info import Statevector
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
import time

print("=" * 60)
print("B1: [0,1] encoding + linear entanglement (Prabhu config)")
print("=" * 60)

# Load [0,1] features (no × pi)
d = np.load('data/features_9d.npz')
X_train = d['train_features']   # already MinMax [0,1]
X_test  = d['test_features']
y_train = d['y_train']
y_test  = d['y_test']
print(f"Train: {X_train.shape}  Test: {X_test.shape}")
print(f"Encoding range — train min: {X_train.min():.4f}  max: {X_train.max():.4f}")

# Build ZZFeatureMap with linear entanglement, reps=2
fm = ZZFeatureMap(feature_dimension=9, reps=2, entanglement='linear')

# Compute training statevectors
print("\nComputing train statevectors (742 samples)...")
t0 = time.time()
sv_train = []
for x in tqdm(X_train, desc="  train SVs"):
    bound = fm.assign_parameters(x)
    sv_train.append(Statevector(bound).data)
sv_train = np.array(sv_train)
print(f"  Done in {time.time()-t0:.1f}s")

# Compute test statevectors
print("Computing test statevectors (186 samples)...")
t0 = time.time()
sv_test = []
for x in tqdm(X_test, desc="  test SVs"):
    bound = fm.assign_parameters(x)
    sv_test.append(Statevector(bound).data)
sv_test = np.array(sv_test)
print(f"  Done in {time.time()-t0:.1f}s")

# Build kernel matrices
print("Building kernel matrices...")
K_train = (np.abs(np.dot(sv_train, sv_train.conj().T)) ** 2).astype(np.float32)
K_test  = (np.abs(np.dot(sv_test,  sv_train.conj().T)) ** 2).astype(np.float32)

# Train QSVC with C=5.0 (same as all other ablation rows)
print("Training QSVC (C=5.0, precomputed kernel)...")
clf = SVC(kernel='precomputed', C=5.0, random_state=42)
clf.fit(K_train, y_train)
y_pred = clf.predict(K_test)

acc = accuracy_score(y_test, y_pred)
f1  = f1_score(y_test, y_pred, average='macro', zero_division=0)

print("\n" + "=" * 60)
print(f"  [0,1] + linear  →  QSVC accuracy: {acc*100:.2f}%  F1-macro: {f1:.4f}")
print("=" * 60)

# Cross-reference with the other three cells for context
print("\nTable VII-A context:")
print(f"  [0,1]  + linear  : {acc*100:.2f}%  ← B1 (just measured)")
print(f"  [0,1]  + circular: 92.47%  (from ablation_encoding.json)")
print(f"  [0,pi] + linear  : 89.78%  (from ablation_entanglement.json)")
print(f"  [0,pi] + circular: 94.62%  (tuned model, metrics_report.json)")

# Interpret the result
print("\nInterpretation:")
if abs(acc - 0.8978) < 0.02:
    print("  → Close to 89.78%: the two factors INTERACT.")
    print("     Encoding contributes nothing under linear; only works once chain is closed.")
elif abs(acc - 0.9247) < 0.02:
    print("  → Close to 92.47%: topology contributes nothing under [0,1] scaling.")
elif acc < 0.88:
    print("  → Well below both: effects are approximately ADDITIVE.")
    combined = acc + (0.9462 - acc)
    print(f"     Combined gain from this cell to 94.62% = {(0.9462-acc)*100:.2f} pp")
else:
    print(f"  → Intermediate result: {acc*100:.2f}%. Check additive vs interactive manually.")

# Save result
import json
result = {
    "encoding": "[0,1]",
    "entanglement": "linear",
    "reps": 2,
    "C": 5.0,
    "qsvc_accuracy": round(float(acc), 6),
    "qsvc_f1_macro": round(float(f1), 6),
    "n_test": int(len(y_test))
}
with open('results/paper/ablation/b1_01_linear.json', 'w') as f:
    json.dump(result, f, indent=2)
print(f"\nResult saved to results/paper/ablation/b1_01_linear.json")
