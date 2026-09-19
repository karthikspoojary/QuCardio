"""
Pauli Feature Map Ablation
==========================
Compares four quantum feature map types on the same 9-D SVD-compressed ECG
features with [0, pi] encoding (our best configuration):

  ZFeatureMap       — single-qubit Rz only, no entanglement
  ZZFeatureMap      — Pauli-Z + ZZ interactions  (our baseline, known 94.62%)
  PauliFeatureMap_XX — Pauli-X + XX interactions
  PauliFeatureMap_YY — Pauli-Y + YY interactions

All use reps=2, circular entanglement (where applicable), C=5.0, [0,pi] scaling.
ZFeatureMap has no entanglement by design — forced to linear (no effect).

Outputs:
  results/paper/ablation/ablation_feature_maps.json
  results/paper/ablation/ablation_feature_maps.png
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

from qiskit.circuit.library import ZFeatureMap, ZZFeatureMap, PauliFeatureMap
from qiskit.quantum_info import Statevector


# ── Shared helpers ────────────────────────────────────────────────────────────

def compute_statevectors(X, feature_map):
    svs = []
    for x in tqdm(X, desc="  Statevectors", leave=False):
        bound = feature_map.assign_parameters(x)
        svs.append(Statevector(bound).data)
    return np.array(svs)


def build_kernel(sv1, sv2):
    return np.abs(np.dot(sv1, sv2.conj().T)) ** 2


# ── Feature map definitions ───────────────────────────────────────────────────

def get_feature_maps(n_features):
    """Return dict of name -> (feature_map, description)."""
    return {
        'ZFeatureMap': (
            ZFeatureMap(feature_dimension=n_features, reps=2),
            'Single-qubit Rz only, no entanglement'
        ),
        'ZZFeatureMap': (
            ZZFeatureMap(feature_dimension=n_features, reps=2, entanglement='circular'),
            'Pauli-Z + ZZ (our best config)'
        ),
        'PauliFeatureMap_XX': (
            PauliFeatureMap(feature_dimension=n_features, reps=2,
                            paulis=['X', 'XX'], entanglement='circular'),
            'Pauli-X + XX interactions, circular'
        ),
        'PauliFeatureMap_YY': (
            PauliFeatureMap(feature_dimension=n_features, reps=2,
                            paulis=['Y', 'YY'], entanglement='circular'),
            'Pauli-Y + YY interactions, circular'
        ),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def run():
    print("=" * 65)
    print("Pauli Feature Map Ablation")
    print("=" * 65)

    # ── Load pre-computed 9-D features ───────────────────────────────────────
    feat_path = config.FEATURES_DIR / 'features_9d.npz'
    if not feat_path.exists():
        raise FileNotFoundError(
            f"{feat_path} not found.\n"
            "Run src/features/reduce_dimensions.py first to produce features_9d.npz."
        )

    print(f"Loading 9-D features from {feat_path} ...")
    d = np.load(feat_path)
    X_train_01 = d['train_features'].astype(np.float64)
    X_test_01  = d['test_features'].astype(np.float64)
    y_train    = d['y_train']
    y_test     = d['y_test']
    print(f"  Train: {X_train_01.shape}  Test: {X_test_01.shape}")
    print(f"  Class distribution train: {np.bincount(y_train)}")

    # Apply [0, pi] encoding — same as our best QSVC config
    X_train = X_train_01 * np.pi
    X_test  = X_test_01  * np.pi
    print("  Applied [0, pi] encoding.")

    n_features = X_train.shape[1]
    feature_maps = get_feature_maps(n_features)

    results = {}
    OUT_DIR = Path('results/paper/ablation')
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for name, (fm, description) in feature_maps.items():
        print(f"\n── {name} ──")
        print(f"   {description}")
        t0 = time.time()

        # Statevectors
        print("  Computing train statevectors ...")
        sv_train = compute_statevectors(X_train, fm)
        print("  Computing test statevectors ...")
        sv_test  = compute_statevectors(X_test, fm)

        # Kernel matrices
        K_train = build_kernel(sv_train, sv_train).astype(np.float32)
        K_test  = build_kernel(sv_test,  sv_train).astype(np.float32)

        # QSVC (precomputed kernel SVM, C=5.0 — our tuned value)
        clf = SVC(kernel='precomputed', C=config.C_PARAM,
                  random_state=config.RANDOM_SEED)
        clf.fit(K_train, y_train)
        y_pred = clf.predict(K_test)

        acc = accuracy_score(y_test, y_pred)
        f1  = f1_score(y_test, y_pred, average='macro', zero_division=0)
        elapsed = round(time.time() - t0, 1)

        print(f"  Accuracy: {acc*100:.2f}%   F1-macro: {f1:.4f}   ({elapsed}s)")

        results[name] = {
            'accuracy':    round(float(acc), 6),
            'f1_macro':    round(float(f1),  6),
            'description': description,
            'elapsed_s':   elapsed,
        }

    # ── Save JSON ─────────────────────────────────────────────────────────────
    out_json = OUT_DIR / 'ablation_feature_maps.json'
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {out_json}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    _plot(results, OUT_DIR)

    return results


def _plot(results, out_dir):
    names  = list(results.keys())
    accs   = [results[n]['accuracy'] * 100 for n in names]
    f1s    = [results[n]['f1_macro']        for n in names]

    # Clean display labels
    labels = [
        'ZFeatureMap\n(Z only)',
        'ZZFeatureMap\n(Z+ZZ, ours)',
        'PauliFeatureMap\n(X+XX)',
        'PauliFeatureMap\n(Y+YY)',
    ]

    x     = np.arange(len(names))
    width = 0.35
    colors_acc = ['#94a3b8', '#3b82f6', '#f59e0b', '#10b981']
    colors_f1  = ['#cbd5e1', '#93c5fd', '#fcd34d', '#6ee7b7']

    fig, ax = plt.subplots(figsize=(10, 5))
    bars_a = ax.bar(x - width/2, accs, width, label='Accuracy (%)',
                    color=colors_acc, edgecolor='white', linewidth=0.8)
    bars_f = ax.bar(x + width/2, [v*100 for v in f1s], width,
                    label='Macro F1 (%)', color=colors_f1,
                    edgecolor='white', linewidth=0.8)

    # Value labels on bars
    for bar in bars_a:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{bar.get_height():.1f}', ha='center', va='bottom',
                fontsize=8.5, fontweight='bold')
    for bar in bars_f:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{bar.get_height():.1f}', ha='center', va='bottom',
                fontsize=8.5)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('Score (%)', fontsize=11)
    ax.set_title(
        'Pauli Feature Map Ablation\n'
        '(9 qubits, reps=2, circular entanglement, [0,π] encoding, C=5.0)',
        fontsize=11
    )
    ax.set_ylim(50, 105)
    ax.axhline(y=94.62, color='#3b82f6', linestyle='--', linewidth=1.2,
               label='ZZFeatureMap best (94.62%)')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    out_png = out_dir / 'ablation_feature_maps.png'
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved → {out_png}")


if __name__ == '__main__':
    run()
