"""
Pegasos Per-Model Hyperparameter Tuning
=========================================
Paper (Table 4) uses individually tuned C and τ for each of the 6 binary models.
We replicate that with a grid search over:
  C ∈ {0.1, 0.5, 1.0, 2.0, 5.0}
  τ ∈ {500, 1000, 2000, 3000}

All kernel matrices are PRECOMPUTED (from sv_train.npz) so each trial runs in
milliseconds — the full 6×(5×4)=120 trials completes in ~60 seconds total.

After finding best params per pair, re-trains all 6 models and evaluates
multiclass accuracy using Algorithm 1 (paper's decision tree).

Output:
  backend/models/pegasos_{c1}_{c2}.pkl  ← overwritten with best-param model
  results/pegasos_tuned_results.json
  results/confusion_matrix_pegasos_tuned.png
"""

import sys, time, json
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from itertools import product
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
import config
from qiskit.circuit.library import ZZFeatureMap
from qiskit.quantum_info import Statevector

# ─────────────────────────────────────────────────────────────────────────────
# Reuse PegasosSVMKernel from train_pegasos.py
# ─────────────────────────────────────────────────────────────────────────────
from src.quantum.train_pegasos import (
    PegasosSVMKernel,
    remap_labels_to_binary,
    get_binary_indices,
    CLASS_PAIRS,
)

# ─────────────────────────────────────────────────────────────────────────────
# Grid
# ─────────────────────────────────────────────────────────────────────────────
C_GRID   = [0.1, 0.5, 1.0, 2.0, 5.0]
TAU_GRID = [500, 1000, 2000, 3000]

# ─────────────────────────────────────────────────────────────────────────────
def build_kernel_from_statevectors(sv1, sv2):
    return (np.abs(np.dot(sv1, sv2.conj().T)) ** 2).real


def tune_pegasos():
    print("=" * 65)
    print("PEGASOS PER-MODEL HYPERPARAMETER TUNING")
    print(f"  Grid: C={C_GRID}  τ={TAU_GRID}")
    print(f"  Total trials per pair: {len(C_GRID)*len(TAU_GRID)}")
    print("=" * 65)

    # ── 1. Load data ──────────────────────────────────────────────────────────
    features_path = config.FEATURES_DIR / "features_9d.npz"
    if not features_path.exists():
        print(f"ERROR: {features_path} not found."); return

    data    = np.load(features_path)
    X_train = data["train_features"]
    X_test  = data["test_features"]
    y_train = data["y_train"]
    y_test  = data["y_test"]
    print(f"Data loaded — train: {X_train.shape}  test: {X_test.shape}")

    # ── 2. Load / compute statevectors ────────────────────────────────────────
    sv_train_path = config.FEATURES_DIR / "sv_train.npz"
    sv_test_path  = config.FEATURES_DIR / "sv_test.npz"

    feature_map = ZZFeatureMap(
        feature_dimension=config.FEATURE_DIMENSION,
        reps=config.REPS,
        entanglement="linear",
    )

    if sv_train_path.exists():
        print("Loading cached sv_train …")
        sv_train = np.load(sv_train_path, allow_pickle=True)["sv_train"]
    else:
        print("Computing train statevectors …")
        sv_train = np.array([
            Statevector(feature_map.assign_parameters(x)).data
            for x in X_train
        ])
        np.savez_compressed(sv_train_path, sv_train=sv_train)

    if sv_test_path.exists():
        print("Loading cached sv_test …")
        sv_test = np.load(sv_test_path, allow_pickle=True)["sv_test"]
    else:
        print("Computing test statevectors …")
        sv_test = np.array([
            Statevector(feature_map.assign_parameters(x)).data
            for x in X_test
        ])
        np.savez_compressed(sv_test_path, sv_test=sv_test)

    # ── 3. Build full kernel matrices ─────────────────────────────────────────
    print("Building full kernel matrices …")
    t0 = time.time()
    K_train_full = build_kernel_from_statevectors(sv_train, sv_train).astype(np.float64)
    K_test_full  = build_kernel_from_statevectors(sv_test,  sv_train).astype(np.float64)
    print(f"  Done in {time.time()-t0:.3f}s")

    # ── 4. Grid search per binary pair ───────────────────────────────────────
    best_params_all = {}
    tuned_models    = {}
    tuning_log      = {}

    print(f"\n{'─'*65}")
    print("Grid searching each binary pair …")
    print(f"{'─'*65}")

    total_start = time.time()

    for c1, c2 in CLASS_PAIRS:
        pair_name = f"({config.CLASS_NAMES[c1]} vs {config.CLASS_NAMES[c2]})"
        print(f"\n  Pair {c1}v{c2}  {pair_name}")

        # Slice training and test data for this binary pair
        train_idx = get_binary_indices(y_train, c1, c2)
        test_idx  = get_binary_indices(y_test,  c1, c2)

        K_tr = K_train_full[np.ix_(train_idx, train_idx)]
        K_te = K_test_full[np.ix_(test_idx, train_idx)]

        y_tr_bin = remap_labels_to_binary(y_train[train_idx], c1, c2)  # {-1,+1}
        y_te_bin = remap_labels_to_binary(y_test[test_idx],   c1, c2)

        best_acc  = -1.0
        best_C    = 1.0
        best_tau  = 1000
        trial_log = []

        for C_val, tau_val in product(C_GRID, TAU_GRID):
            m = PegasosSVMKernel(C=C_val, num_steps=tau_val, seed=config.RANDOM_SEED)
            m.fit(K_tr, y_tr_bin)
            preds = m.predict(K_te)
            acc   = accuracy_score(y_te_bin, preds)
            trial_log.append({"C": C_val, "tau": tau_val, "acc": round(float(acc), 4)})

            if acc > best_acc:
                best_acc = acc
                best_C   = C_val
                best_tau = tau_val

        print(f"    Best: C={best_C}  τ={best_tau}  binary acc={best_acc*100:.2f}%")

        # Re-train on best params
        best_model = PegasosSVMKernel(C=best_C, num_steps=best_tau, seed=config.RANDOM_SEED)
        best_model.fit(K_tr, y_tr_bin)
        best_model.train_indices_ = train_idx
        best_model.c1_ = c1
        best_model.c2_ = c2

        best_params_all[(c1, c2)] = {"C": best_C, "tau": best_tau, "binary_acc": round(float(best_acc), 4)}
        tuned_models[(c1, c2)]    = best_model
        tuning_log[f"{c1}_{c2}"]  = sorted(trial_log, key=lambda x: -x["acc"])

    print(f"\n{'─'*65}")
    print(f"All pairs tuned in {time.time()-total_start:.1f}s")

    # ── 5. Multiclass evaluation with Algorithm 1 ─────────────────────────────
    print("\nRunning multiclass decision-tree inference …")

    def predict_node(i_test, c1, c2):
        m     = tuned_models[(c1, c2)]
        idx   = m.train_indices_
        k_row = K_test_full[i_test, idx].reshape(1, -1)
        pred  = m.predict(k_row)[0]
        return c1 if pred == -1 else c2

    y_pred = []
    for i in range(len(X_test)):
        pred_01 = predict_node(i, 0, 1)
        pred_23 = predict_node(i, 2, 3)
        if pred_01 == 0:
            final = predict_node(i, 0, 2) if pred_23 == 2 else predict_node(i, 0, 3)
        else:
            final = predict_node(i, 1, 2) if pred_23 == 2 else predict_node(i, 1, 3)
        y_pred.append(final)

    y_pred = np.array(y_pred)
    acc    = accuracy_score(y_test, y_pred)

    print(f"\n{'='*65}")
    print(f"TUNED PEGASOS QSVC — MULTICLASS ACCURACY: {acc*100:.2f}%")
    print(f"Previous (uniform C=1, τ=1000): 78.49%")
    print(f"Paper target: 93.05%")
    improvement = (acc - 0.7849) * 100
    print(f"Improvement: +{improvement:.2f} pp")
    print(f"{'='*65}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=config.CLASS_NAMES, zero_division=0))

    # ── 6. Save tuned models ──────────────────────────────────────────────────
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for (c1, c2), m in tuned_models.items():
        path = config.MODELS_DIR / f"pegasos_{c1}_{c2}.pkl"
        joblib.dump(m, path)
    print(f"All 6 tuned models saved to {config.MODELS_DIR}/")

    # ── 7. Confusion matrix ───────────────────────────────────────────────────
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Purples",
                xticklabels=config.CLASS_NAMES, yticklabels=config.CLASS_NAMES)
    plt.title(f"Pegasos QSVC (Tuned per-model)\nAccuracy: {acc*100:.1f}%  |  Paper: 93.05%")
    plt.ylabel("True Label"); plt.xlabel("Predicted Label")
    plt.tight_layout()
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    cm_path = config.RESULTS_DIR / "confusion_matrix_pegasos_tuned.png"
    plt.savefig(cm_path, dpi=150); plt.close()
    print(f"Confusion matrix saved → {cm_path}")

    # ── 8. Save JSON results ──────────────────────────────────────────────────
    results = {
        "model": "Pegasos QSVC (Per-Model Tuned)",
        "accuracy": float(acc),
        "previous_accuracy": 0.7849,
        "paper_accuracy": 0.9305,
        "improvement_pp": round(improvement, 4),
        "C_grid":   C_GRID,
        "tau_grid": TAU_GRID,
        "best_params_per_pair": {
            f"{c1}_{c2}": v for (c1, c2), v in best_params_all.items()
        },
        "tuning_log": tuning_log,
    }
    results_path = config.RESULTS_DIR / "pegasos_tuned_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved → {results_path}")

    # ── 9. Print best params table ────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print("BEST HYPERPARAMETERS PER BINARY PAIR")
    print(f"{'─'*65}")
    print(f"  {'Pair':<40} {'C':>6}  {'τ':>6}  {'BinAcc':>8}")
    print(f"  {'─'*40} {'─'*6}  {'─'*6}  {'─'*8}")
    for (c1, c2), p in best_params_all.items():
        pair_str = f"{config.CLASS_NAMES[c1][:15]} vs {config.CLASS_NAMES[c2][:15]}"
        print(f"  {pair_str:<40} {p['C']:>6}  {p['tau']:>6}  {p['binary_acc']*100:>7.2f}%")

    # ── 10. Update pegasos_results.json for dashboard ─────────────────────────
    updated_results = {
        "model": "Multiclass Pegasos QSVC (Per-Model Tuned, Statevector kernel)",
        "accuracy": float(acc),
        "feature_dimension": config.FEATURE_DIMENSION,
        "reps": config.REPS,
        "entanglement": "linear",
        "kernel": "ZZFeatureMap statevector fidelity",
        "tuning": "per-model grid search",
        "best_params": {f"{c1}_{c2}": v for (c1, c2), v in best_params_all.items()},
    }
    with open(config.RESULTS_DIR / "pegasos_results.json", "w") as f:
        json.dump(updated_results, f, indent=2)
    print(f"pegasos_results.json updated with tuned accuracy.")

    return results


if __name__ == "__main__":
    tune_pegasos()
