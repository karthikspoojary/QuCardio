"""
ST-2B — Feature Encoding Ablation (Primary Novel Contribution)
==============================================================
Formally quantifies the +4.84 pp improvement from minmax_0pi over the paper's
minmax_01 encoding. Tests 5 encoding variants on both QSVC and Pegasos.

Encodings tested:
  raw        — SVD output directly (no scaling)
  l2_norm    — unit-sphere normalisation
  minmax_01  — [0, 1]  (paper's approach, known QSVC=89.78%)
  minmax_0pi — [0, π]  (our best, known QSVC=94.62%, Pegasos=91.94%)
  minmax_02pi— [0, 2π]

Outputs:
  results/ablation_encoding.json
  results/ablation_encoding.png  (horizontal bar chart, 300 DPI)
"""

import sys
import json
import time
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
import config

from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MinMaxScaler, normalize
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

from qiskit.circuit.library import ZZFeatureMap
from qiskit.quantum_info import Statevector

# ── Re-use identical helpers from train_qsvc / train_pegasos ─────────────────

CLASS_PAIRS = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]


def compute_statevectors(X, feature_map):
    svs = []
    for x in tqdm(X, desc="  Statevectors", leave=False):
        bound = feature_map.assign_parameters(x)
        svs.append(Statevector(bound).data)
    return np.array(svs)


def build_kernel(sv1, sv2):
    return np.abs(np.dot(sv1, sv2.conj().T)) ** 2


# ── Pegasos SGD (copy of core logic from train_pegasos.py) ───────────────────

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
        decision = np.dot(K_rows, self.alpha_ * self.y_train_)
        return np.where(decision >= 0, 1, -1).astype(np.int64)


def _remap(y, c1, c2):
    return np.where(y == c1, -1, 1).astype(np.float64)


def pegasos_predict_multiclass(models, K_test_full, n_test):
    """Algorithm 1 from paper — binary decision tree."""
    y_pred = []
    for i in range(n_test):
        def node(c1, c2):
            m = models[(c1, c2)]
            idx = m.train_idx_
            k_row = K_test_full[i, idx].reshape(1, -1)
            p = m.predict(k_row)[0]
            return c1 if p == -1 else c2

        pred_01 = node(0, 1)
        pred_23 = node(2, 3)
        if pred_01 == 0:
            final = node(0, 2) if pred_23 == 2 else node(0, 3)
        else:
            final = node(1, 2) if pred_23 == 2 else node(1, 3)
        y_pred.append(final)
    return np.array(y_pred)


def train_pegasos_models(K_train, K_test, y_train, y_test, C=1.0, num_steps=3000):
    models = {}
    for c1, c2 in CLASS_PAIRS:
        idx = np.where((y_train == c1) | (y_train == c2))[0]
        K_bin = K_train[np.ix_(idx, idx)]
        y_bin = _remap(y_train[idx], c1, c2)
        m = _PegasosBinary(C=C, num_steps=num_steps).fit(K_bin, y_bin)
        m.train_idx_ = idx
        models[(c1, c2)] = m
    y_pred = pegasos_predict_multiclass(models, K_test, len(y_test))
    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average='macro', zero_division=0)
    return acc, f1


# ── Encoding functions ────────────────────────────────────────────────────────

def apply_encoding(X_train_svd, X_test_svd, name):
    """Return (X_tr_enc, X_te_enc) ready for ZZFeatureMap."""
    if name == 'raw':
        return X_train_svd.copy(), X_test_svd.copy()

    if name == 'l2_norm':
        return (normalize(X_train_svd, norm='l2'),
                normalize(X_test_svd,  norm='l2'))

    scaler = MinMaxScaler(feature_range=(0, 1))
    tr = scaler.fit_transform(X_train_svd)
    te = scaler.transform(X_test_svd)

    if name == 'minmax_01':
        return tr, te
    if name == 'minmax_0pi':
        return tr * np.pi, te * np.pi
    if name == 'minmax_02pi':
        return tr * 2 * np.pi, te * 2 * np.pi

    raise ValueError(f"Unknown encoding: {name}")


# ── Main ──────────────────────────────────────────────────────────────────────

ENCODINGS = ['raw', 'l2_norm', 'minmax_01', 'minmax_0pi', 'minmax_02pi']

# Known cached result — re-use to avoid recomputation
KNOWN = {
    'minmax_01':  {'qsvc_acc': 0.8978, 'qsvc_f1': None, 'pegasos_acc': None, 'pegasos_f1': None},
    'minmax_0pi': {'qsvc_acc': 0.9462, 'qsvc_f1': None, 'pegasos_acc': 0.9194, 'pegasos_f1': None},
}


def run():
    print("=" * 65)
    print("ST-2B — Feature Encoding Ablation")
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

    # ── Fit SVD once on training data (n=9) ───────────────────────────────────
    print("Fitting TruncatedSVD(9) on training features …")
    svd = TruncatedSVD(n_components=9, random_state=config.RANDOM_SEED)
    X_train_svd = svd.fit_transform(X_train_raw)
    X_test_svd  = svd.transform(X_test_raw)
    print(f"  Explained variance: {svd.explained_variance_ratio_.sum():.2%}")

    # Fixed circuit (9 qubits, reps=2, circular) — shared across all encodings
    feature_map = ZZFeatureMap(feature_dimension=9, reps=2, entanglement='circular')

    results = {}

    for enc in ENCODINGS:
        print(f"\n── Encoding: {enc} ──────────────────────────────")
        t0 = time.time()

        X_tr, X_te = apply_encoding(X_train_svd, X_test_svd, enc)

        # ── Compute statevectors & kernel ─────────────────────────────────────
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
        y_pred_q = qsvc.predict(K_test)
        qsvc_acc = accuracy_score(y_test, y_pred_q)
        qsvc_f1  = f1_score(y_test, y_pred_q, average='macro', zero_division=0)
        print(f"  QSVC  accuracy: {qsvc_acc*100:.2f}%  F1-macro: {qsvc_f1:.4f}")

        # ── Pegasos ────────────────────────────────────────────────────────────
        print("  Training Pegasos (6 binary, 3 000 steps each) …")
        peg_acc, peg_f1 = train_pegasos_models(
            K_train.astype(np.float64), K_test.astype(np.float64),
            y_train, y_test, C=config.PEGASOS_C, num_steps=3000
        )
        print(f"  Pegasos accuracy: {peg_acc*100:.2f}%  F1-macro: {peg_f1:.4f}")

        results[enc] = {
            'qsvc_acc':   round(float(qsvc_acc), 6),
            'qsvc_f1':    round(float(qsvc_f1),  6),
            'pegasos_acc': round(float(peg_acc),  6),
            'pegasos_f1':  round(float(peg_f1),   6),
            'elapsed_s':  round(time.time() - t0,  1),
        }

    # ── Save JSON ─────────────────────────────────────────────────────────────
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_json = config.RESULTS_DIR / 'ablation_encoding.json'
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {out_json}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    _plot(results)
    return results


def _plot(results):
    enc_labels = list(results.keys())
    qsvc_vals   = [results[e]['qsvc_acc'] * 100   for e in enc_labels]
    pegasos_vals = [results[e]['pegasos_acc'] * 100 for e in enc_labels]

    y_pos = np.arange(len(enc_labels))
    bar_h = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    bars_q = ax.barh(y_pos - bar_h / 2, qsvc_vals,   bar_h, label='QSVC',    color='#3b82f6')
    bars_p = ax.barh(y_pos + bar_h / 2, pegasos_vals, bar_h, label='Pegasos', color='#7c3aed')

    # Annotate
    for bar in bars_q:
        ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2,
                f'{bar.get_width():.1f}%', va='center', fontsize=8)
    for bar in bars_p:
        ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2,
                f'{bar.get_width():.1f}%', va='center', fontsize=8)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(enc_labels, fontsize=10)
    ax.set_xlabel('Test Accuracy (%)', fontsize=11)
    ax.set_title('ST-2B: Feature Encoding Ablation\n(ZZFeatureMap, 9 qubits, reps=2, circular)',
                 fontsize=12)
    ax.set_xlim(50, 102)
    ax.axvline(x=89.78, color='gray', linestyle='--', linewidth=0.8, label='Paper baseline (89.78%)')
    ax.legend(fontsize=9)
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()

    out_png = config.RESULTS_DIR / 'ablation_encoding.png'
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved → {out_png}")


if __name__ == '__main__':
    run()
