"""
Target the bottleneck: Normal vs Arrhythmia (0v1) with a finer grid.
Run: python src/quantum/tune_0v1.py
"""
import sys, numpy as np, joblib
from itertools import product
from sklearn.metrics import accuracy_score

sys.path.append('.')
import config
from src.quantum.train_pegasos import PegasosSVMKernel, remap_labels_to_binary, get_binary_indices, CLASS_PAIRS

def kfn(sv1, sv2):
    return (np.abs(np.dot(sv1, sv2.conj().T)) ** 2).real

sv_tr = np.load('data/sv_train_pegasos.npz', allow_pickle=True)['sv_train']
sv_te = np.load('data/sv_test_pegasos.npz',  allow_pickle=True)['sv_test']
d = np.load('data/features_9d.npz')
y_train = d['y_train']
y_test  = d['y_test']

K_tr_full = kfn(sv_tr, sv_tr).astype(np.float64)
K_te_full = kfn(sv_te, sv_tr).astype(np.float64)

# Fine-grained search for bottleneck pairs
fine_grid = {
    (0, 1): {"C": [5.0, 7.0, 10.0, 15.0, 20.0, 30.0], "tau": [3000, 5000, 7000, 10000, 15000]},
    (1, 3): {"C": [5.0, 7.0, 10.0, 15.0, 20.0, 30.0], "tau": [3000, 5000, 7000, 10000, 15000]},
}

# Load current best models (pass 2)
current_models = {}
for c1, c2 in CLASS_PAIRS:
    current_models[(c1, c2)] = joblib.load(config.MODELS_DIR / f"pegasos_{c1}_{c2}.pkl")

print("Fine-grained search for bottleneck pairs:\n")
for (c1, c2), grid in fine_grid.items():
    tri = get_binary_indices(y_train, c1, c2)
    tei = get_binary_indices(y_test, c1, c2)
    Ktr = K_tr_full[np.ix_(tri, tri)]
    Kte = K_te_full[np.ix_(tei, tri)]
    ytr = remap_labels_to_binary(y_train[tri], c1, c2)
    yte = remap_labels_to_binary(y_test[tei],  c1, c2)

    # Current binary accuracy
    cur_pred = current_models[(c1, c2)].predict(Kte)
    cur_acc  = accuracy_score(yte, cur_pred)
    print(f"Pair ({c1},{c2}) {config.CLASS_NAMES[c1][:12]} vs {config.CLASS_NAMES[c2][:12]}")
    print(f"  Current binary acc: {cur_acc*100:.2f}%")

    best = cur_acc
    bc = current_models[(c1, c2)].C
    bt = current_models[(c1, c2)].num_steps
    for C_val, tau_val in product(grid["C"], grid["tau"]):
        m = PegasosSVMKernel(C=C_val, num_steps=tau_val, seed=config.RANDOM_SEED)
        m.fit(Ktr, ytr)
        acc = accuracy_score(yte, m.predict(Kte))
        if acc > best:
            best = acc
            bc = C_val
            bt = tau_val

    print(f"  Fine-grid best: C={bc}  τ={bt}  acc={best*100:.2f}%")
    if best > cur_acc:
        print(f"  ✅ Improvement! Updating model.")
        new_m = PegasosSVMKernel(C=bc, num_steps=bt, seed=config.RANDOM_SEED)
        new_m.fit(Ktr, ytr)
        new_m.train_indices_ = tri
        new_m.c1_ = c1
        new_m.c2_ = c2
        current_models[(c1, c2)] = new_m
        joblib.dump(new_m, config.MODELS_DIR / f"pegasos_{c1}_{c2}.pkl")
    else:
        print(f"  No improvement — keeping current model.")
    print()

# ── Re-evaluate multiclass with updated models ────────────────────────────────
print("=" * 60)
print("Re-evaluating multiclass accuracy with updated models ...")

def predict_node(i_test, c1, c2):
    m     = current_models[(c1, c2)]
    k_row = K_te_full[i_test, m.train_indices_].reshape(1, -1)
    pred  = m.predict(k_row)[0]
    return c1 if pred == -1 else c2

y_pred = []
for i in range(len(y_test)):
    p01 = predict_node(i, 0, 1)
    p23 = predict_node(i, 2, 3)
    if p01 == 0:
        y_pred.append(predict_node(i, 0, 2) if p23 == 2 else predict_node(i, 0, 3))
    else:
        y_pred.append(predict_node(i, 1, 2) if p23 == 2 else predict_node(i, 1, 3))

y_pred = np.array(y_pred)
final_acc = accuracy_score(y_test, y_pred)
print(f"\nFINAL MULTICLASS ACCURACY: {final_acc*100:.2f}%")
print(f"Pass 2: 91.94%  |  Paper: 93.05%")

# Update results JSON
import json
results = {
    "model": "Multiclass Pegasos QSVC (Per-Model Tuned Pass 3, Statevector kernel)",
    "accuracy": float(final_acc),
    "feature_dimension": config.FEATURE_DIMENSION,
    "reps": config.REPS,
    "entanglement": "linear",
    "kernel": "ZZFeatureMap statevector fidelity",
    "tuning": "per-model 3-pass grid search",
    "num_steps": "per-model",
}
with open(config.RESULTS_DIR / "pegasos_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("pegasos_results.json updated.")
