"""
Multi-seed variance: 4 factorial cells over 5 seeds
=====================================================
Cells: [0,1]+linear, [0,1]+circular, [0,pi]+linear, [0,pi]+circular
Seeds: 0, 7, 21, 42, 99

Uses a fast precomputed-kernel SVC (C=5.0) for each cell.
Statevectors are recomputed fresh per seed (split changes per seed).

Output: results/paper/ablation/factorial_seed_variance.json
"""
import numpy as np
import json
from pathlib import Path
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from qiskit.circuit.library import ZZFeatureMap
from qiskit.quantum_info import Statevector
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

OUT   = Path("results/paper/ablation/factorial_seed_variance.json")
SEEDS = [0, 7, 21, 42, 99]
CELLS = [
    ("[0,1]",  "linear"),
    ("[0,1]",  "circular"),
    ("[0,pi]", "linear"),
    ("[0,pi]", "circular"),
]

# Load base 9D features (minmax_01 on full dataset)
d_full = np.load("data/features_9d.npz")
# We need to re-split per seed; use the full pool
# features_9d.npz stores train+test; reconstruct full dataset
X_full = np.vstack([d_full['train_features'], d_full['test_features']])
y_full = np.concatenate([d_full['y_train'], d_full['y_test']])
print(f"Full dataset: {X_full.shape}")

def compute_svs_fast(X, fm):
    svs = []
    for x in X:
        bound = fm.assign_parameters(x)
        svs.append(Statevector(bound).data)
    return np.array(svs)

def build_kernel(sv1, sv2):
    return (np.abs(np.dot(sv1, sv2.conj().T)) ** 2).astype(np.float32)

results = {}

for encoding, topology in CELLS:
    cell_key = f"{encoding}+{topology}"
    print(f"\n{'='*60}")
    print(f"Cell: {cell_key}")
    print(f"{'='*60}")
    
    fm = ZZFeatureMap(feature_dimension=9, reps=2, entanglement=topology)
    accs = []
    
    for seed in SEEDS:
        print(f"  seed={seed} ...", end=" ", flush=True)
        
        # Resplit with this seed (same 80:20 stratified ratio)
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_full, y_full,
            test_size=0.2,
            stratify=y_full,
            random_state=seed,
        )
        
        # Scale to encoding range
        if encoding == "[0,pi]":
            X_tr_enc = X_tr * np.pi
            X_te_enc = X_te * np.pi
        else:
            X_tr_enc = X_tr
            X_te_enc = X_te
        
        # Statevectors
        sv_tr = compute_svs_fast(X_tr_enc, fm)
        sv_te = compute_svs_fast(X_te_enc, fm)
        
        # Kernel
        K_tr  = build_kernel(sv_tr, sv_tr)
        K_te  = build_kernel(sv_te, sv_tr)
        
        # Fit and predict
        clf   = SVC(kernel="precomputed", C=5.0, random_state=seed)
        clf.fit(K_tr, y_tr)
        y_pred = clf.predict(K_te)
        acc    = float(accuracy_score(y_te, y_pred))
        accs.append(acc)
        print(f"{acc*100:.2f}%")
    
    mean_acc = float(np.mean(accs))
    std_acc  = float(np.std(accs, ddof=1))
    print(f"  >> Mean: {mean_acc*100:.2f}%  Std: {std_acc*100:.2f}%")
    
    results[cell_key] = {
        "encoding":    encoding,
        "entanglement": topology,
        "seeds":       SEEDS,
        "accuracies":  [round(a, 6) for a in accs],
        "mean":        round(mean_acc, 6),
        "std":         round(std_acc, 6),
        "mean_pct":    round(mean_acc * 100, 2),
        "std_pct":     round(std_acc * 100, 2),
    }

OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\n✅ Saved → {OUT}")

# Print summary table
print("\n=== SUMMARY ===")
print(f"{'Cell':<25} {'Mean':>8} {'Std':>7} {'Accuracies'}")
for k, v in results.items():
    accs_str = "  ".join(f"{a*100:.1f}%" for a in v['accuracies'])
    print(f"{k:<25} {v['mean_pct']:>7.2f}%  {v['std_pct']:>6.2f}%  {accs_str}")
