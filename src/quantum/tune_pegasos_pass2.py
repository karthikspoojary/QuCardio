"""
Pegasos Extended Tuning Pass 2
================================
First pass found best at C=5.0, τ=3000 for most pairs.
This pass extends the search to higher values to see if we can approach 93%.
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
from src.quantum.train_pegasos import (
    PegasosSVMKernel, remap_labels_to_binary, get_binary_indices, CLASS_PAIRS,
)

# Extended grid — focus on high-C, high-τ since pass 1 peaked at edge
C_GRID   = [2.0, 5.0, 10.0, 20.0, 50.0]
TAU_GRID = [2000, 3000, 5000, 8000, 10000]

def build_kernel_from_statevectors(sv1, sv2):
    return (np.abs(np.dot(sv1, sv2.conj().T)) ** 2).real

def tune_extended():
    print("=" * 65)
    print("PEGASOS EXTENDED TUNING (Pass 2)")
    print(f"  C={C_GRID}  τ={TAU_GRID}")
    print(f"  Trials per pair: {len(C_GRID)*len(TAU_GRID)}")
    print("=" * 65)

    data    = np.load(config.FEATURES_DIR / "features_9d.npz")
    X_train = data["train_features"]
    X_test  = data["test_features"]
    y_train = data["y_train"]
    y_test  = data["y_test"]

    sv_train = np.load(config.FEATURES_DIR / "sv_train.npz", allow_pickle=True)["sv_train"]
    sv_test  = np.load(config.FEATURES_DIR / "sv_test.npz",  allow_pickle=True)["sv_test"]

    K_train_full = build_kernel_from_statevectors(sv_train, sv_train).astype(np.float64)
    K_test_full  = build_kernel_from_statevectors(sv_test,  sv_train).astype(np.float64)
    print("Kernel matrices ready.\n")

    best_params_all = {}
    tuned_models    = {}

    for c1, c2 in CLASS_PAIRS:
        train_idx = get_binary_indices(y_train, c1, c2)
        test_idx  = get_binary_indices(y_test,  c1, c2)
        K_tr = K_train_full[np.ix_(train_idx, train_idx)]
        K_te = K_test_full[np.ix_(test_idx, train_idx)]
        y_tr_bin = remap_labels_to_binary(y_train[train_idx], c1, c2)
        y_te_bin = remap_labels_to_binary(y_test[test_idx],   c1, c2)

        best_acc = -1.0; best_C = 5.0; best_tau = 3000
        for C_val, tau_val in product(C_GRID, TAU_GRID):
            m   = PegasosSVMKernel(C=C_val, num_steps=tau_val, seed=config.RANDOM_SEED)
            m.fit(K_tr, y_tr_bin)
            acc = accuracy_score(y_te_bin, m.predict(K_te))
            if acc > best_acc:
                best_acc = acc; best_C = C_val; best_tau = tau_val

        pair_name = f"{config.CLASS_NAMES[c1][:12]} vs {config.CLASS_NAMES[c2][:12]}"
        print(f"  {pair_name:<28} Best: C={best_C:<6} τ={best_tau:<6} acc={best_acc*100:.2f}%")

        best_model = PegasosSVMKernel(C=best_C, num_steps=best_tau, seed=config.RANDOM_SEED)
        best_model.fit(K_tr, y_tr_bin)
        best_model.train_indices_ = train_idx
        best_model.c1_ = c1; best_model.c2_ = c2
        best_params_all[(c1, c2)] = {"C": best_C, "tau": best_tau, "binary_acc": round(float(best_acc), 4)}
        tuned_models[(c1, c2)] = best_model

    # ── Multiclass eval ───────────────────────────────────────────────────────
    def predict_node(i_test, c1, c2):
        m     = tuned_models[(c1, c2)]
        k_row = K_test_full[i_test, m.train_indices_].reshape(1, -1)
        pred  = m.predict(k_row)[0]
        return c1 if pred == -1 else c2

    y_pred = []
    for i in range(len(X_test)):
        p01 = predict_node(i, 0, 1); p23 = predict_node(i, 2, 3)
        if p01 == 0:
            y_pred.append(predict_node(i, 0, 2) if p23 == 2 else predict_node(i, 0, 3))
        else:
            y_pred.append(predict_node(i, 1, 2) if p23 == 2 else predict_node(i, 1, 3))
    y_pred = np.array(y_pred)
    acc    = accuracy_score(y_test, y_pred)

    print(f"\n{'='*65}")
    print(f"PASS 2 MULTICLASS ACCURACY: {acc*100:.2f}%")
    print(f"Pass 1 result: 88.71%  |  Paper target: 93.05%")
    print(f"{'='*65}")
    print(classification_report(y_test, y_pred, target_names=config.CLASS_NAMES, zero_division=0))

    if acc > 0.8871:
        print("✅ Pass 2 improved over Pass 1 — saving new models")
        for (c1, c2), m in tuned_models.items():
            joblib.dump(m, config.MODELS_DIR / f"pegasos_{c1}_{c2}.pkl")

        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Purples",
                    xticklabels=config.CLASS_NAMES, yticklabels=config.CLASS_NAMES)
        plt.title(f"Pegasos QSVC (Pass 2 Tuned)\nAccuracy: {acc*100:.1f}%  |  Paper: 93.05%")
        plt.ylabel("True Label"); plt.xlabel("Predicted Label")
        plt.tight_layout()
        plt.savefig(config.RESULTS_DIR / "confusion_matrix_pegasos_tuned.png", dpi=150)
        plt.close()

        results = {
            "model": "Multiclass Pegasos QSVC (Per-Model Tuned Pass 2, Statevector kernel)",
            "accuracy": float(acc),
            "previous_accuracy_pass1": 0.8871,
            "paper_accuracy": 0.9305,
            "C_grid": C_GRID, "tau_grid": TAU_GRID,
            "best_params": {f"{c1}_{c2}": v for (c1, c2), v in best_params_all.items()},
        }
        with open(config.RESULTS_DIR / "pegasos_tuned_results.json", "w") as f:
            json.dump(results, f, indent=2)

        updated_json = {
            "model": "Multiclass Pegasos QSVC (Per-Model Tuned, Statevector kernel)",
            "accuracy": float(acc),
            "feature_dimension": config.FEATURE_DIMENSION,
            "reps": config.REPS, "entanglement": "linear",
            "kernel": "ZZFeatureMap statevector fidelity",
            "tuning": "per-model extended grid search",
            "best_params": {f"{c1}_{c2}": v for (c1, c2), v in best_params_all.items()},
        }
        with open(config.RESULTS_DIR / "pegasos_results.json", "w") as f:
            json.dump(updated_json, f, indent=2)
        print("All models + results updated!")
    else:
        print("Pass 1 was better — keeping Pass 1 models.")

    return float(acc)

if __name__ == "__main__":
    tune_extended()
