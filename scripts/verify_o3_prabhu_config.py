"""
O3 Verification: [0,1] + full entanglement (Prabhu et al. actual default config)
This is the configuration Prabhu likely used (ZZFeatureMap default = full entanglement).
"""
import numpy as np
import json
from pathlib import Path
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from qiskit.circuit.library import ZZFeatureMap
from qiskit.quantum_info import Statevector
from tqdm import tqdm

OUT = Path("results/paper/ablation/o3_01_full.json")

d = np.load("data/features_9d.npz")
X_train = d["train_features"]   # [0,1]
X_test  = d["test_features"]
y_train = d["y_train"]
y_test  = d["y_test"]

fm = ZZFeatureMap(feature_dimension=9, reps=2, entanglement="full")
print("Entanglement:", fm.entanglement)

def compute_svs(X, label):
    svs = []
    for x in tqdm(X, desc=f"  SV {label}", leave=False):
        bound = fm.assign_parameters(x)
        svs.append(Statevector(bound).data)
    return np.array(svs)

print("\n=== [0,1] + full entanglement (Prabhu default config) ===")
sv_tr = compute_svs(X_train, "[0,1]+full train")
sv_te = compute_svs(X_test,  "[0,1]+full test")

K_tr = (np.abs(np.dot(sv_tr, sv_tr.conj().T)) ** 2).astype(np.float32)
K_te = (np.abs(np.dot(sv_te, sv_tr.conj().T)) ** 2).astype(np.float32)

clf = SVC(kernel="precomputed", C=5.0, random_state=42)
clf.fit(K_tr, y_train)
y_pred = clf.predict(K_te)
acc = accuracy_score(y_test, y_pred)
n_correct = int(acc * 186)
print(f"\n[0,1] + full  →  {acc*100:.2f}%  ({n_correct}/186)")
print(f"Prabhu reported: 94.09% (this pipeline, same config)")
print(f"Our reproduction gap: {(94.09 - acc*100):.2f} pp")

result = {
    "encoding": "[0,1]",
    "entanglement": "full",
    "reps": 2,
    "C": 5.0,
    "accuracy": round(float(acc), 6),
    "n_correct": n_correct,
    "n_test": 186,
    "prabhu_reported": 94.09,
    "reproduction_gap_pp": round(94.09 - acc*100, 2),
    "note": "Prabhu et al. did not state entanglement; Qiskit ZZFeatureMap default is full"
}

with open(OUT, "w") as f:
    json.dump(result, f, indent=2)
print(f"\nSaved → {OUT}")
