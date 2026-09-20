"""
KTA sweep across all 320 ablation configurations.

For each (encoding, entanglement, reps, C) combination, we need the
training kernel matrix K_train.  Rather than recomputing all 320 × 742
statevector sets from scratch, we cache statevectors per
(encoding, entanglement, reps) and reuse across C values (C doesn't
affect the kernel, only the SVC margin — KTA is purely about K and y).

This reduces the compute to 4 × 5 × 4 = 80 statevector sets.
Each set: 742 SV computations ~ 3–5 seconds → ~6–7 minutes total.

Saves: results/paper/ablation/kta_all_configs.json
"""
import sys, json, itertools
import numpy as np
from pathlib import Path
from tqdm import tqdm

sys.path.append('.')
import config
from qiskit.circuit.library import ZZFeatureMap
from qiskit.quantum_info import Statevector

OUT = Path('results/paper/ablation/kta_all_configs.json')

# ── Load data ──────────────────────────────────────────────────────────────────
d        = np.load(config.FEATURES_DIR / 'features_9d.npz')
X_01     = d['train_features']   # (742, 9) MinMax [0,1]
y_train  = d['y_train']
N        = len(y_train)

# ── Target kernel matrix Y (centered multiclass) ──────────────────────────────
n_cls = 4
Y_raw = np.where(y_train[:, None] == y_train[None, :], 1.0, -1.0/(n_cls-1))
H     = np.eye(N) - np.ones((N,N))/N
Y_c   = H @ Y_raw @ H

def kta(K: np.ndarray, Y_c: np.ndarray) -> float:
    K_c = H @ K @ H
    num = np.trace(K_c.T @ Y_c)
    den = np.sqrt(np.trace(K_c.T @ K_c) * np.trace(Y_c.T @ Y_c)) + 1e-12
    return float(num / den)

# ── Encoding transforms ────────────────────────────────────────────────────────
def encode(X_01, name):
    if name == 'raw':
        return X_01.copy()
    if name == 'l2_norm':
        norms = np.linalg.norm(X_01, axis=1, keepdims=True) + 1e-9
        return X_01 / norms
    if name == 'minmax_01':
        return X_01.copy()
    if name == 'minmax_0pi':
        return X_01 * np.pi
    if name == 'minmax_02pi':
        return X_01 * 2 * np.pi
    raise ValueError(f"Unknown encoding {name}")

ENCODINGS     = ['raw', 'l2_norm', 'minmax_01', 'minmax_0pi', 'minmax_02pi']
ENTANGLEMENTS = ['linear', 'full', 'circular', 'pairwise', 'sca']
REPS_LIST     = [1, 2, 3, 4]
C_VALUES      = [0.1, 1.0, 5.0, 10.0]   # C doesn't affect KTA, but stored for completeness

# Qiskit entanglement name map
ENT_MAP = {
    'linear':   'linear',
    'full':     'full',
    'circular': 'circular',
    'pairwise': 'pairwise',
    'sca':      'circular_alternating',
}

results = {}
total_sv_sets = len(ENCODINGS) * len(ENTANGLEMENTS) * len(REPS_LIST)
done = 0

for enc_name in ENCODINGS:
    X_enc = encode(X_01, enc_name)
    for ent_name in ENTANGLEMENTS:
        ent_qiskit = ENT_MAP[ent_name]
        for reps in REPS_LIST:
            done += 1
            key_sv = f"{enc_name}|{ent_name}|reps{reps}"
            print(f"[{done}/{total_sv_sets}] {key_sv} — computing SVs ...")

            try:
                fm  = ZZFeatureMap(feature_dimension=9, reps=reps,
                                   entanglement=ent_qiskit)
                svs = np.array([
                    Statevector(fm.assign_parameters(x)).data
                    for x in tqdm(X_enc, desc='  SVs', leave=False)
                ])
                K   = (np.abs(np.dot(svs, svs.conj().T))**2).astype(np.float64)
                score = kta(K, Y_c)
                print(f"  KTA = {score:.6f}")
            except Exception as e:
                print(f"  ERROR: {e}")
                score = None

            # C doesn't change KTA — store same score for all C values
            for C in C_VALUES:
                cfg_key = f"{enc_name}|{ent_name}|reps{reps}|C{C}"
                results[cfg_key] = {
                    'encoding':     enc_name,
                    'entanglement': ent_name,
                    'reps':         reps,
                    'C':            C,
                    'kta':          round(score, 8) if score is not None else None,
                }

            # Save incrementally
            OUT.parent.mkdir(parents=True, exist_ok=True)
            OUT.write_text(json.dumps(results, indent=2))

print(f"\n✅ KTA sweep complete — {len(results)} configurations")
print(f"Saved → {OUT}")

# ── Summary: top-10 by KTA ──────────────────────────────────────────────────
valid = [(k, v) for k, v in results.items() if v['kta'] is not None and v['C'] == 5.0]
valid.sort(key=lambda x: x[1]['kta'], reverse=True)
print("\nTop 10 configurations by KTA (C=5.0):")
print(f"{'Config':<45} {'KTA':>10}")
for k, v in valid[:10]:
    print(f"  {k:<45} {v['kta']:>10.6f}")
