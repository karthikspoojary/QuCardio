"""
ST-5 — QSVC Scaling Study on Full Dataset (3023 training images)
=================================================================
Uses data/features_9d_full.npz (3023 train / 928 test, already SVD-reduced)
to sweep QSVC over increasing training set sizes.

Training sizes: 742 → 1500 → 2200 → 3023 (stratified subsamples)
  742  — matches the original baseline (direct comparison)
  3023 — full ECG_DATA training set (8.1× more data than baseline)

Encoding: minmax_0pi (confirmed best from ST-2B)
Circuit:  ZZFeatureMap(9, reps=2, circular)
Kernel:   precomputed statevector fidelity
Classifier: SVC(precomputed, C=5.0)

All statevectors for each subset are computed fresh (no cache reuse, since
the full-dataset features come from a different SVD fit than the 742-sample one).

Outputs:
  results/qsvc_scaling_study.json  — accuracy + F1 per training size
  results/qsvc_scaling_study.png   — line plot overlaid on existing SVM/Pegasos
                                     scaling curve (from full_dataset_metrics.json)
"""

import sys
import json
import time
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
import config

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

from qiskit.circuit.library import ZZFeatureMap
from qiskit.quantum_info import Statevector

# Training sizes to sweep
TRAIN_SIZES = [742, 1500, 2200, 3023]


def compute_statevectors(X, feature_map):
    svs = []
    for x in tqdm(X, desc=f"  SV (N={len(X)})", leave=False):
        bound = feature_map.assign_parameters(x)
        svs.append(Statevector(bound).data)
    return np.array(svs)


def build_kernel(sv1, sv2):
    return np.abs(np.dot(sv1, sv2.conj().T)) ** 2


def apply_minmax_0pi(X_train_raw, X_test_raw):
    """Fit MinMaxScaler on train, apply ×π to both — minmax_0pi encoding."""
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_tr = scaler.fit_transform(X_train_raw) * np.pi
    X_te = scaler.transform(X_test_raw)      * np.pi
    return X_tr, X_te


def run():
    print("=" * 65)
    print("ST-5 — QSVC Scaling Study (full dataset, 3023 train images)")
    print("=" * 65)

    # ── Load full-dataset 9D features ─────────────────────────────────────────
    feat_path = config.FEATURES_DIR / 'features_9d_full.npz'
    if not feat_path.exists():
        raise FileNotFoundError(
            f"{feat_path} not found.\n"
            "Expected data/features_9d_full.npz with keys: "
            "train_features (3023,9), test_features (928,9), y_train, y_test"
        )

    d = np.load(feat_path)
    X_train_full = d['train_features']   # (3023, 9)  minmax_01
    X_test_full  = d['test_features']    # (928, 9)
    y_train_full = d['y_train']
    y_test       = d['y_test']
    print(f"Full dataset: train={X_train_full.shape}  test={X_test_full.shape}")
    print(f"Class distribution (train): { {i: int((y_train_full==i).sum()) for i in range(4)} }")

    # Fixed circuit — 9 qubits, reps=2, circular
    feature_map = ZZFeatureMap(feature_dimension=9, reps=2, entanglement='circular')

    # Test statevectors are fixed for all runs (same test set)
    print("\nPre-computing test statevectors (928 samples, one-time) …")
    _, X_te_enc = apply_minmax_0pi(X_train_full, X_test_full)
    # Note: we encode test with the full-training scaler for the 3023 run.
    # For smaller subsets each gets its own scaler — test is re-encoded each time.

    results = {}
    feature_map = ZZFeatureMap(feature_dimension=9, reps=2, entanglement='circular')

    for n in TRAIN_SIZES:
        print(f"\n── N_train = {n} ──────────────────────────────────────────")
        t0 = time.time()

        # Stratified subsample from full training set
        if n >= len(X_train_full):
            X_sub = X_train_full
            y_sub = y_train_full
            print(f"  Using full training set ({len(X_sub)} samples)")
        else:
            sss = StratifiedShuffleSplit(n_splits=1, train_size=n,
                                         random_state=config.RANDOM_SEED)
            idx, _ = next(sss.split(X_train_full, y_train_full))
            X_sub  = X_train_full[idx]
            y_sub  = y_train_full[idx]
            print(f"  Stratified subsample: {len(X_sub)} samples  "
                  f"{ {i: int((y_sub==i).sum()) for i in range(4)} }")

        # Apply minmax_0pi — fit scaler on this subset only (fair: no test leakage)
        X_tr_enc, X_te_enc = apply_minmax_0pi(X_sub, X_test_full)

        # Compute statevectors
        print("  Computing train statevectors …")
        sv_train = compute_statevectors(X_tr_enc, feature_map)
        print("  Computing test statevectors …")
        sv_test  = compute_statevectors(X_te_enc, feature_map)

        # Build kernel matrices
        print("  Building kernel matrices …")
        K_train = build_kernel(sv_train, sv_train).astype(np.float32)
        K_test  = build_kernel(sv_test,  sv_train).astype(np.float32)
        print(f"  K_train: {K_train.shape}  ({K_train.nbytes / 1e6:.0f} MB)")

        # Train QSVC
        print("  Training QSVC …")
        qsvc = SVC(kernel='precomputed', C=config.C_PARAM,
                   random_state=config.RANDOM_SEED)
        qsvc.fit(K_train, y_sub)

        # Evaluate
        y_pred = qsvc.predict(K_test)
        acc = accuracy_score(y_test, y_pred)
        f1  = f1_score(y_test, y_pred, average='macro', zero_division=0)
        elapsed = time.time() - t0

        print(f"  ✅ Accuracy: {acc*100:.2f}%  F1-macro: {f1:.4f}  ({elapsed:.1f}s)")

        results[n] = {
            'accuracy':  round(float(acc), 6),
            'f1_macro':  round(float(f1),  6),
            'elapsed_s': round(elapsed,    1),
            'n_train':   int(n),
            'n_test':    int(len(y_test)),
        }

    # ── Save JSON ─────────────────────────────────────────────────────────────
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_json = config.RESULTS_DIR / 'qsvc_scaling_study.json'
    with open(out_json, 'w') as f:
        json.dump({str(k): v for k, v in results.items()}, f, indent=2)
    print(f"\nResults saved → {out_json}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    _plot(results)

    print("\n✅ ST-5 QSVC scaling study complete.")
    return results


def _plot(qsvc_results):
    """Overlay QSVC scaling on existing SVM+Pegasos curve."""
    # Load existing SVM + Pegasos scaling data if available
    existing_path = config.RESULTS_DIR / 'full_dataset_metrics.json'
    svm_pts, peg_pts = {}, {}
    if existing_path.exists():
        with open(existing_path) as f:
            full = json.load(f)
        for entry in full.get('scaling_results', []):
            n = entry.get('n_train') or entry.get('N') or entry.get('train_size')
            if n:
                if 'svm_accuracy' in entry:
                    svm_pts[n] = entry['svm_accuracy'] * 100
                if 'pegasos_accuracy' in entry:
                    peg_pts[n] = entry['pegasos_accuracy'] * 100

    ns_q   = sorted(qsvc_results.keys())
    accs_q = [qsvc_results[n]['accuracy'] * 100 for n in ns_q]

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(ns_q, accs_q, marker='D', linewidth=2, color='#7c3aed',
            label='QSVC (ZZFeatureMap, circular, minmax_0pi)')

    if svm_pts:
        ns_s = sorted(svm_pts.keys())
        ax.plot(ns_s, [svm_pts[n] for n in ns_s], marker='s', linewidth=1.5,
                color='#3b82f6', linestyle='--', label='Classical SVM')

    if peg_pts:
        ns_p = sorted(peg_pts.keys())
        ax.plot(ns_p, [peg_pts[n] for n in ns_p], marker='^', linewidth=1.5,
                color='#10b981', linestyle='--', label='Pegasos QSVC')

    # Annotate QSVC points
    for n, acc in zip(ns_q, accs_q):
        ax.annotate(f'{acc:.1f}%', (n, acc),
                    textcoords='offset points', xytext=(0, 8),
                    ha='center', fontsize=8, color='#7c3aed')

    ax.set_xlabel('Training Set Size', fontsize=11)
    ax.set_ylabel('Test Accuracy (%)', fontsize=11)
    ax.set_title('QSVC Scaling Study\n(Full ECG_DATA dataset, 928-sample test set)',
                 fontsize=12)
    ax.set_ylim(50, 105)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    plt.tight_layout()

    out_png = config.RESULTS_DIR / 'qsvc_scaling_study.png'
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved → {out_png}")


if __name__ == '__main__':
    run()
