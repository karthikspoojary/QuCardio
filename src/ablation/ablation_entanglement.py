"""
ST-2C — Entanglement Topology Ablation
======================================
Compares 5 ZZFeatureMap entanglement topologies:
  linear   — paper's topology (each qubit connected to next)
  full     — all-to-all connections
  circular — linear + wrap-around  (our best, used in QSVC 94.62%)
  sca      — shifted circular alternating (Qiskit native)
  pairwise — only adjacent pair entanglement

Both QSVC and Pegasos evaluated. Uses cached minmax_0pi 9D features.
Linear statevectors re-used from sv_train_pegasos.npz if shape matches.

Outputs:
  results/ablation_entanglement.json
  results/ablation_entanglement.png  (grouped bar chart, 300 DPI)
"""

import sys
import json
import time
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
import config

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

from qiskit.circuit.library import ZZFeatureMap
from qiskit.quantum_info import Statevector

CLASS_PAIRS = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]

TOPOLOGIES = ['linear', 'full', 'circular', 'sca', 'pairwise']


# ── Shared helpers ────────────────────────────────────────────────────────────

def compute_statevectors(X, feature_map):
    svs = []
    for x in tqdm(X, desc=f"  SV ({feature_map.entanglement})", leave=False):
        bound = feature_map.assign_parameters(x)
        svs.append(Statevector(bound).data)
    return np.array(svs)


def build_kernel(sv1, sv2):
    return np.abs(np.dot(sv1, sv2.conj().T)) ** 2


# ── Pegasos helpers ───────────────────────────────────────────────────────────

class _PegasosBinary:
    def __init__(self, C=1.0, num_steps=3000, seed=42):
        self.C, self.num_steps, self.seed = C, num_steps, seed

    def fit(self, K, y):
        rng = np.random.default_rng(self.seed)
        N = len(y)
        lam = 1.0 / (N * self.C)
        alpha = np.zeros(N)
        for t in range(1, self.num_steps + 1):
            i = int(rng.integers(0, N))
            score = (1.0 / (lam * t)) * np.dot(alpha * y, K[i])
            if y[i] * score < 1.0:
                alpha[i] += 1.0
        self.alpha_ = alpha
        self.y_train_ = y.copy()
        return self

    def predict(self, K_rows):
        return np.where(np.dot(K_rows, self.alpha_ * self.y_train_) >= 0, 1, -1).astype(np.int64)


def _remap(y, c1, c2):
    return np.where(y == c1, -1, 1).astype(np.float64)


def run_pegasos(K_train, K_test, y_train, y_test, C=1.0, num_steps=3000):
    models = {}
    for c1, c2 in CLASS_PAIRS:
        idx = np.where((y_train == c1) | (y_train == c2))[0]
        K_bin = K_train[np.ix_(idx, idx)]
        y_bin = _remap(y_train[idx], c1, c2)
        m = _PegasosBinary(C=C, num_steps=num_steps).fit(K_bin, y_bin)
        m.train_idx_ = idx
        models[(c1, c2)] = m

    y_pred = []
    for i in range(len(y_test)):
        def node(c1, c2):
            m = models[(c1, c2)]
            k = K_test[i, m.train_idx_].reshape(1, -1)
            p = m.predict(k)[0]
            return c1 if p == -1 else c2

        pred_01 = node(0, 1)
        pred_23 = node(2, 3)
        if pred_01 == 0:
            final = node(0, 2) if pred_23 == 2 else node(0, 3)
        else:
            final = node(1, 2) if pred_23 == 2 else node(1, 3)
        y_pred.append(final)

    y_pred = np.array(y_pred)
    return (accuracy_score(y_test, y_pred),
            f1_score(y_test, y_pred, average='macro', zero_division=0))


# ── Main ──────────────────────────────────────────────────────────────────────

def run():
    print("=" * 65)
    print("ST-2C — Entanglement Topology Ablation")
    print("=" * 65)

    # ── Load minmax_01 9D features, apply ×π → minmax_0pi ────────────────────
    feat_path = config.FEATURES_DIR / 'features_9d.npz'
    if not feat_path.exists():
        raise FileNotFoundError(f"{feat_path} not found. Run reduce_dimensions.py first.")

    print("Loading 9D features (minmax_01) …")
    d = np.load(feat_path)
    X_train = d['train_features'] * np.pi   # → minmax_0pi
    X_test  = d['test_features']  * np.pi
    y_train, y_test = d['y_train'], d['y_test']
    print(f"  Train: {X_train.shape}  Test: {X_test.shape}")

    # Cached linear statevectors (from Pegasos training) — shape (742, 512) expected
    sv_linear_path = config.FEATURES_DIR / 'sv_train_pegasos.npz'
    sv_linear_test_path = config.FEATURES_DIR / 'sv_test_pegasos.npz'

    results = {}

    for topology in TOPOLOGIES:
        print(f"\n── Entanglement: {topology} ──────────────────────────────")
        t0 = time.time()

        feature_map = ZZFeatureMap(feature_dimension=9, reps=2, entanglement=topology)

        # Reuse cached linear statevectors if available
        if topology == 'linear' and sv_linear_path.exists() and sv_linear_test_path.exists():
            print("  Reusing cached linear statevectors (sv_train_pegasos.npz) …")
            sv_train = np.load(sv_linear_path,    allow_pickle=True)['sv_train']
            sv_test  = np.load(sv_linear_test_path, allow_pickle=True)['sv_test']
            # Verify shape consistency (should be (742, 512))
            if sv_train.shape != (len(X_train), 2**9):
                print(f"  WARNING: cached shape {sv_train.shape} unexpected. Recomputing.")
                sv_train = compute_statevectors(X_train, feature_map)
                sv_test  = compute_statevectors(X_test,  feature_map)
        else:
            print("  Computing train statevectors …")
            sv_train = compute_statevectors(X_train, feature_map)
            print("  Computing test statevectors …")
            sv_test  = compute_statevectors(X_test,  feature_map)

        K_train = build_kernel(sv_train, sv_train).astype(np.float32)
        K_test  = build_kernel(sv_test,  sv_train).astype(np.float32)

        # ── QSVC ──────────────────────────────────────────────────────────────
        print("  Training QSVC …")
        qsvc = SVC(kernel='precomputed', C=config.C_PARAM, random_state=config.RANDOM_SEED)
        qsvc.fit(K_train, y_train)
        y_pred_q = qsvc.predict(K_test)
        qsvc_acc = accuracy_score(y_test, y_pred_q)
        qsvc_f1  = f1_score(y_test, y_pred_q, average='macro', zero_division=0)
        print(f"  QSVC  accuracy: {qsvc_acc*100:.2f}%  F1-macro: {qsvc_f1:.4f}")

        # ── Pegasos ────────────────────────────────────────────────────────────
        print("  Training Pegasos …")
        peg_acc, peg_f1 = run_pegasos(
            K_train.astype(np.float64), K_test.astype(np.float64),
            y_train, y_test, C=config.PEGASOS_C, num_steps=3000
        )
        print(f"  Pegasos accuracy: {peg_acc*100:.2f}%  F1-macro: {peg_f1:.4f}")

        results[topology] = {
            'qsvc_acc':    round(float(qsvc_acc), 6),
            'qsvc_f1':     round(float(qsvc_f1),  6),
            'pegasos_acc': round(float(peg_acc),   6),
            'pegasos_f1':  round(float(peg_f1),    6),
            'elapsed_s':   round(time.time() - t0,  1),
        }

    # ── Save JSON ─────────────────────────────────────────────────────────────
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_json = config.RESULTS_DIR / 'ablation_entanglement.json'
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {out_json}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    _plot(results)
    return results


def _plot(results):
    topologies  = list(results.keys())
    qsvc_vals   = [results[t]['qsvc_acc'] * 100   for t in topologies]
    pegasos_vals = [results[t]['pegasos_acc'] * 100 for t in topologies]

    x   = np.arange(len(topologies))
    w   = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    bars_q = ax.bar(x - w/2, qsvc_vals,   w, label='QSVC',    color='#3b82f6')
    bars_p = ax.bar(x + w/2, pegasos_vals, w, label='Pegasos', color='#7c3aed')

    for bar in bars_q:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=8)
    for bar in bars_p:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(topologies, fontsize=10)
    ax.set_ylabel('Test Accuracy (%)', fontsize=11)
    ax.set_ylim(50, 102)
    ax.set_title('ST-2C: Entanglement Topology Ablation\n(ZZFeatureMap, 9 qubits, reps=2, minmax_0pi)',
                 fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    out_png = config.RESULTS_DIR / 'ablation_entanglement.png'
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved → {out_png}")


if __name__ == '__main__':
    run()
