"""
Fine sweep of encoding range near π.
Tests [0, 0.80π], [0, 0.85π], [0, 0.90π], [0, 0.95π], [0, π], [0, 1.00π]
all with circular entanglement, reps=2, C=5.0.

Answers: is [0,π] truly optimal, or does the optimum sit slightly below
the point where both endpoints collide modulo 2π?

Output: results/paper/ablation/encoding_fine_sweep.json
"""
import sys, json
import numpy as np
from pathlib import Path
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from qiskit.circuit.library import ZZFeatureMap
from qiskit.quantum_info import Statevector
from tqdm import tqdm

sys.path.append('.')
import config

OUT = Path('results/paper/ablation/encoding_fine_sweep.json')

d       = np.load(config.FEATURES_DIR / 'features_9d.npz')
X_01_tr = d['train_features']
X_01_te = d['test_features']
y_train = d['y_train']
y_test  = d['y_test']

fm = ZZFeatureMap(feature_dimension=9, reps=2, entanglement='circular')

def svs(X, desc=''):
    return np.array([
        Statevector(fm.assign_parameters(x)).data
        for x in tqdm(X, desc=f'  SV {desc}', leave=False)
    ])

def kernel(sv1, sv2):
    return (np.abs(np.dot(sv1, sv2.conj().T))**2).astype(np.float32)

# Multipliers relative to [0,1] features
MULTIPLIERS = {
    '[0, 0.80π]': 0.80 * np.pi,
    '[0, 0.85π]': 0.85 * np.pi,
    '[0, 0.90π]': 0.90 * np.pi,
    '[0, 0.95π]': 0.95 * np.pi,
    '[0, π]':     1.00 * np.pi,
    '[0, 1.05π]': 1.05 * np.pi,
}

results = {}
print(f"{'Range':<15} {'Accuracy':>10}  ({'n'}/186)")
print("-"*40)

for label, mult in MULTIPLIERS.items():
    X_tr = X_01_tr * mult
    X_te = X_01_te * mult

    sv_tr = svs(X_tr, f'{label} train')
    sv_te = svs(X_te, f'{label} test')
    K_tr  = kernel(sv_tr, sv_tr)
    K_te  = kernel(sv_te, sv_tr)

    clf   = SVC(kernel='precomputed', C=5.0, random_state=42)
    clf.fit(K_tr, y_train)
    y_pr  = clf.predict(K_te)
    acc   = float(accuracy_score(y_test, y_pr))
    n     = int(round(acc * 186))

    print(f"  {label:<13}  {acc*100:>7.2f}%  ({n}/186)")
    results[label] = {
        'multiplier': round(float(mult), 6),
        'accuracy':   round(acc, 6),
        'acc_pct':    round(acc*100, 2),
        'n_correct':  n,
        'n_test':     186,
    }

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(results, indent=2))
print(f"\n✅ Saved → {OUT}")

best = max(results, key=lambda k: results[k]['accuracy'])
print(f"Best encoding range: {best}  ({results[best]['acc_pct']}%)")
