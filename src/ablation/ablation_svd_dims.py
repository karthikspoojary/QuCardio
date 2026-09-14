"""
ST-2A — SVD Dimensionality Ablation (Replicates + Extends Paper's Fig. 8)
=========================================================================
Sweeps n_components ∈ {2,4,6,7,8,9,10,12,16,32} and reports QSVC accuracy
at each point. Extends paper's Fig.8 with n=10,12,16,32.

Key: ZZFeatureMap qubit count ALWAYS equals n (dynamic allocation).
  n=2  → 2-qubit circuit, statevector dim 4
  n=9  → 9-qubit circuit, statevector dim 512  (reuses cached SVs if present)
  n=16 → 16-qubit circuit, statevector dim 65536 (slow; skipped if >SKIP_LARGE)

Outputs:
  results/ablation_svd_dims.json
  results/ablation_svd_dims.png  (line plot with dashed n=9 marker, 300 DPI)
"""

import sys
import json
import time
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
import config

from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

from qiskit.circuit.library import ZZFeatureMap
from qiskit.quantum_info import Statevector

# Large n values (16, 32) produce statevectors of dim 65536 / 4B — skip if True
SKIP_LARGE = True           # set False if you have >16 GB RAM + time to spare
SKIP_THRESHOLD = 13         # skip n >= this if SKIP_LARGE is True

SVD_DIMS = [2, 4, 6, 7, 8, 9, 10, 12, 16, 32]


def compute_statevectors(X, feature_map):
    svs = []
    for x in tqdm(X, desc=f"  SV (n={feature_map.num_qubits})", leave=False):
        bound = feature_map.assign_parameters(x)
        svs.append(Statevector(bound).data)
    return np.array(svs)


def build_kernel(sv1, sv2):
    return np.abs(np.dot(sv1, sv2.conj().T)) ** 2


def run():
    print("=" * 65)
    print("ST-2A — SVD Dimensionality Ablation")
    print("=" * 65)

    # ── Load raw ResNet50 features ────────────────────────────────────────────
    feat_path = config.FEATURES_DIR / 'resnet50_features.npz'
    if not feat_path.exists():
        raise FileNotFoundError(f"{feat_path} not found. Run extract_resnet_features.py first.")

    print("Loading ResNet50 features …")
    d = np.load(feat_path)
    X_train_raw, X_test_raw = d['train_features'], d['test_features']
    y_train, y_test = d['y_train'], d['y_test']
    print(f"  Train: {X_train_raw.shape}  Test: {X_test_raw.shape}")

    # Cached n=9 statevectors (avoid recomputing the baseline point)
    sv_train_cache_path = config.FEATURES_DIR / 'sv_train_qsvc.npz'
    sv_test_cache_path  = config.FEATURES_DIR / 'sv_test_qsvc.npz'
    sv_cache_available  = sv_train_cache_path.exists() and sv_test_cache_path.exists()

    results = {}

    for n in SVD_DIMS:
        if SKIP_LARGE and n >= SKIP_THRESHOLD:
            print(f"\n── n={n:2d} qubits: SKIPPED (SKIP_LARGE=True, statevector dim={2**n:,})")
            results[n] = {'skipped': True, 'reason': f'statevector_dim={2**n}'}
            continue

        print(f"\n── n={n:2d} qubits (statevector dim={2**n}) ──────────────────")
        t0 = time.time()

        # ── SVD + encoding ────────────────────────────────────────────────────
        svd = TruncatedSVD(n_components=n, random_state=config.RANDOM_SEED)
        X_tr_svd = svd.fit_transform(X_train_raw)
        X_te_svd = svd.transform(X_test_raw)

        # minmax_0pi encoding (always use best encoding for this ablation)
        scaler = MinMaxScaler(feature_range=(0, 1))
        X_tr = scaler.fit_transform(X_tr_svd) * np.pi
        X_te = scaler.transform(X_te_svd)     * np.pi

        # ── Circuit: n qubits, reps=2, circular ───────────────────────────────
        feature_map = ZZFeatureMap(feature_dimension=n, reps=2, entanglement='circular')
        assert feature_map.num_qubits == n, f"Qubit mismatch: expected {n}, got {feature_map.num_qubits}"

        # ── Statevectors ──────────────────────────────────────────────────────
        if n == 9 and sv_cache_available:
            print("  n=9: Loading cached statevectors (sv_train_qsvc.npz / sv_test_qsvc.npz) …")
            sv_train = np.load(sv_train_cache_path, allow_pickle=True)['sv_train']
            sv_test  = np.load(sv_test_cache_path,  allow_pickle=True)['sv_test']
            # Sanity check: cached SVs must match minmax_0pi encoding shape
            if sv_train.shape[1] != 2**n:
                print(f"  WARNING: cached SV dim {sv_train.shape[1]} ≠ 2^{n}={2**n}. Recomputing.")
                sv_train = compute_statevectors(X_tr, feature_map)
                sv_test  = compute_statevectors(X_te, feature_map)
        else:
            print("  Computing train statevectors …")
            sv_train = compute_statevectors(X_tr, feature_map)
            print("  Computing test statevectors …")
            sv_test  = compute_statevectors(X_te, feature_map)

        K_train = build_kernel(sv_train, sv_train).astype(np.float32)
        K_test  = build_kernel(sv_test,  sv_train).astype(np.float32)

        # ── QSVC ──────────────────────────────────────────────────────────────
        print("  Training QSVC …")
        qsvc = SVC(kernel='precomputed', C=config.C_PARAM, random_state=config.RANDOM_SEED)
        qsvc.fit(K_train, y_train)
        y_pred = qsvc.predict(K_test)
        acc = accuracy_score(y_test, y_pred)
        f1  = f1_score(y_test, y_pred, average='macro', zero_division=0)
        elapsed = time.time() - t0

        print(f"  QSVC accuracy: {acc*100:.2f}%  F1-macro: {f1:.4f}  ({elapsed:.1f}s)")
        results[n] = {
            'accuracy':  round(float(acc), 6),
            'f1_macro':  round(float(f1),  6),
            'elapsed_s': round(elapsed,    1),
            'sv_dim':    int(2 ** n),
        }

    # ── Save JSON ─────────────────────────────────────────────────────────────
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_json = config.RESULTS_DIR / 'ablation_svd_dims.json'
    with open(out_json, 'w') as f:
        # Serialize with int keys
        json.dump({str(k): v for k, v in results.items()}, f, indent=2)
    print(f"\nResults saved → {out_json}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    _plot(results)
    return results


def _plot(results):
    computed = {n: v for n, v in results.items() if not v.get('skipped')}
    if not computed:
        print("No computed points to plot.")
        return

    ns   = sorted(computed.keys())
    accs = [computed[n]['accuracy'] * 100 for n in ns]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(ns, accs, marker='o', linewidth=2, color='#3b82f6', label='QSVC accuracy')

    # Annotate each point
    for n, acc in zip(ns, accs):
        ax.annotate(f'{acc:.1f}%', (n, acc), textcoords='offset points',
                    xytext=(0, 8), ha='center', fontsize=8)

    # Dashed vertical at n=9
    if 9 in computed:
        ax.axvline(x=9, color='#ef4444', linestyle='--', linewidth=1.2, label='Paper n=9')

    ax.set_xlabel('SVD Components (= Qubits)', fontsize=11)
    ax.set_ylabel('Test Accuracy (%)', fontsize=11)
    ax.set_title('ST-2A: SVD Dimensionality Ablation\n(ZZFeatureMap, reps=2, circular, minmax_0pi)',
                 fontsize=12)
    ax.set_xticks(ns)
    ax.set_ylim(50, 102)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    plt.tight_layout()

    out_png = config.RESULTS_DIR / 'ablation_svd_dims.png'
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved → {out_png}")


if __name__ == '__main__':
    run()
