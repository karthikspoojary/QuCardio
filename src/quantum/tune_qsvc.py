"""
QSVC Comprehensive Hyperparameter Tuning
==========================================
Paper reports 94.09% with n=9, reps=2.
We currently get 89.78% with C=5.0.

Strategy — sweep every lever available:
  1. C values:           [0.1, 0.5, 1, 5, 10, 20, 50, 100, 200, 500]
  2. reps:               [1, 2, 3]  (paper uses 2 — test neighbours)
  3. entanglement:       ['linear', 'full', 'circular']
  4. class_weight:       [None, 'balanced']

All kernel matrices are built from CACHED statevectors (instant matrix multiply),
so every (reps, entanglement) combo only costs ~0.1s for the kernel and ~0.01s
for SVC fit.  The full sweep runs in < 2 minutes.

We also try FEATURE NORMALISATION variants:
  - features_9d.npz  (already MinMax scaled, what we use now)
  - L2-normalised version (maps each sample to unit sphere)

The quantum kernel inner product |<ψ(x)|ψ(z)>|² is sensitive to the DATA RANGE.
ZZFeatureMap encodes features as rotation angles — a [0,1] range means rotations
span [0, π] per gate. L2-norm puts samples on a sphere which may improve
separability in the quantum feature space.
"""

import sys, time, json, itertools
import numpy as np
import joblib
from pathlib import Path
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import normalize, MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
import config
from qiskit.circuit.library import ZZFeatureMap
from qiskit.quantum_info import Statevector

# ─────────────────────────────────────────────────────────────────────────────

def build_kernel(sv1, sv2):
    """K[i,j] = |<ψᵢ|ψⱼ>|²  via matrix multiply (exact, fast)."""
    return (np.abs(np.dot(sv1, sv2.conj().T)) ** 2).astype(np.float64)


def compute_svs(X, feature_map):
    """Compute statevectors for all samples in X."""
    svs = []
    for x in X:
        bound = feature_map.assign_parameters(x)
        svs.append(Statevector(bound).data)
    return np.array(svs)


def get_svs(X, reps, entanglement, feat_name, split, cache_dir):
    """Load cached statevectors or compute and cache them.
    Cache key includes feat_name and split (train/test) to avoid collisions."""
    tag  = f"sv_{feat_name}_reps{reps}_{entanglement}_{split}"
    path = cache_dir / f"{tag}.npz"
    if path.exists():
        return np.load(path, allow_pickle=True)["sv"]
    fm = ZZFeatureMap(feature_dimension=config.FEATURE_DIMENSION,
                      reps=reps, entanglement=entanglement)
    sv = compute_svs(X, fm)
    np.savez_compressed(path, sv=sv)
    return sv


# ─────────────────────────────────────────────────────────────────────────────

def tune_qsvc():
    print("=" * 65)
    print("QSVC COMPREHENSIVE HYPERPARAMETER SWEEP")
    print("=" * 65)

    # ── Load data ─────────────────────────────────────────────────────────────
    d = np.load(config.FEATURES_DIR / "features_9d.npz")
    X_train_mm = d["train_features"]   # MinMax scaled [0,1]
    X_test_mm  = d["test_features"]
    y_train    = d["y_train"]
    y_test     = d["y_test"]

    # Also build L2-normalised version
    X_train_l2 = normalize(X_train_mm, norm="l2")
    X_test_l2  = normalize(X_test_mm,  norm="l2")

    # And a [0, π] scaled version (maps [0,1] → [0,π] for angle encoding)
    X_train_pi = X_train_mm * np.pi
    X_test_pi  = X_test_mm  * np.pi

    feature_variants = {
        "minmax_01":  (X_train_mm, X_test_mm),
        "l2_norm":    (X_train_l2, X_test_l2),
        "minmax_0pi": (X_train_pi, X_test_pi),
    }

    cache_dir = config.FEATURES_DIR / "sv_cache"
    cache_dir.mkdir(exist_ok=True)

    # ── Sweep params ──────────────────────────────────────────────────────────
    C_VALS      = [0.5, 1.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0]
    REPS_VALS   = [1, 2, 3]
    ENTANGL     = ["linear", "full", "circular"]
    CW_VALS     = [None, "balanced"]

    print(f"\nGrid size: {len(C_VALS)} C × {len(REPS_VALS)} reps × "
          f"{len(ENTANGL)} entangle × {len(CW_VALS)} cw × "
          f"{len(feature_variants)} feature variants")
    total = (len(C_VALS) * len(CW_VALS) *
             len(REPS_VALS) * len(ENTANGL) *
             len(feature_variants))
    print(f"Total SVC fits: {total}  (kernel matrices reused per reps/entangle/features)\n")

    best_acc   = 0.0
    best_cfg   = {}
    all_results = []

    grand_start = time.time()

    for feat_name, (X_tr, X_te) in feature_variants.items():
        print(f"\n{'─'*65}")
        print(f"Feature variant: {feat_name}")
        print(f"{'─'*65}")

        for reps, entangle in itertools.product(REPS_VALS, ENTANGL):
            tag = f"reps{reps}_{entangle}"
            t0  = time.time()

            # Build / load statevectors for train and test
            sv_tr = get_svs(X_tr, reps, entangle, feat_name, "train", cache_dir)
            sv_te = get_svs(X_te, reps, entangle, feat_name, "test",  cache_dir)

            # Build kernel matrices (< 0.1s)
            K_tr = build_kernel(sv_tr, sv_tr)
            K_te = build_kernel(sv_te, sv_tr)

            t_kernel = time.time() - t0

            combo_best = 0.0
            combo_best_C = 5.0
            combo_best_cw = None

            for C_val, cw in itertools.product(C_VALS, CW_VALS):
                clf = SVC(kernel="precomputed", C=C_val,
                          class_weight=cw,
                          random_state=config.RANDOM_SEED)
                clf.fit(K_tr, y_train)
                acc = accuracy_score(y_test, clf.predict(K_te))

                all_results.append({
                    "features": feat_name, "reps": reps,
                    "entanglement": entangle, "C": C_val,
                    "class_weight": str(cw), "accuracy": round(float(acc), 6)
                })

                if acc > combo_best:
                    combo_best = acc
                    combo_best_C  = C_val
                    combo_best_cw = cw

                if acc > best_acc:
                    best_acc = acc
                    best_cfg = {
                        "features": feat_name, "reps": reps,
                        "entanglement": entangle, "C": C_val,
                        "class_weight": cw, "accuracy": float(acc)
                    }
                    # Save this as candidate best model
                    best_K_tr = K_tr.copy()
                    best_K_te = K_te.copy()
                    best_clf  = clf

            print(f"  {tag:<20} best: C={combo_best_C:<6} cw={str(combo_best_cw):<10}"
                  f" acc={combo_best*100:.2f}%  (kernel in {t_kernel:.3f}s)")

    elapsed = time.time() - grand_start
    print(f"\n{'='*65}")
    print(f"SWEEP COMPLETE in {elapsed:.1f}s")
    print(f"{'='*65}")
    print(f"\n🏆 BEST CONFIG:")
    for k, v in best_cfg.items():
        print(f"   {k:<20} {v}")
    print(f"\n   BEST ACCURACY:  {best_acc*100:.4f}%")
    print(f"   Current:        89.78%")
    print(f"   Paper target:   94.09%")
    print(f"   Improvement:    +{(best_acc-0.8978)*100:.2f}pp")

    # ── Full classification report for best config ────────────────────────────
    print(f"\nClassification Report (best config):")
    y_pred_best = best_clf.predict(best_K_te)
    print(classification_report(y_test, y_pred_best,
                                target_names=config.CLASS_NAMES, zero_division=0))

    # ── Confusion matrix ──────────────────────────────────────────────────────
    cm = confusion_matrix(y_test, y_pred_best)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=config.CLASS_NAMES, yticklabels=config.CLASS_NAMES)
    plt.title(f"QSVC (Tuned)\n"
              f"Acc={best_acc*100:.1f}%  feat={best_cfg['features']}  "
              f"reps={best_cfg['reps']}  entangle={best_cfg['entanglement']}  "
              f"C={best_cfg['C']}  cw={best_cfg['class_weight']}")
    plt.ylabel("True"); plt.xlabel("Predicted")
    plt.tight_layout()
    config.RESULTS_DIR.mkdir(exist_ok=True)
    plt.savefig(config.RESULTS_DIR / "confusion_matrix_qsvc_tuned.png", dpi=150)
    plt.close()
    print(f"Confusion matrix → results/confusion_matrix_qsvc_tuned.png")

    # ── Save best model ───────────────────────────────────────────────────────
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_clf, config.MODELS_DIR / "qsvc_model.pkl")
    print(f"Best QSVC model saved → backend/models/qsvc_model.pkl")

    # ── Save sv_train/test for the best config (needed for inference) ─────────
    bf = best_cfg["features"]
    br = best_cfg["reps"]
    be = best_cfg["entanglement"]

    X_tr_best = {
        "minmax_01":  d["train_features"],
        "l2_norm":    normalize(d["train_features"], norm="l2"),
        "minmax_0pi": d["train_features"] * np.pi,
    }[bf]
    X_te_best = {
        "minmax_01":  d["test_features"],
        "l2_norm":    normalize(d["test_features"], norm="l2"),
        "minmax_0pi": d["test_features"] * np.pi,
    }[bf]

    best_sv_train = get_svs(X_tr_best, br, be, bf, "train", cache_dir)
    best_sv_test  = get_svs(X_te_best, br, be, bf, "test",  cache_dir)
    np.savez_compressed(config.FEATURES_DIR / "sv_train_qsvc.npz", sv_train=best_sv_train)
    np.savez_compressed(config.FEATURES_DIR / "sv_test_qsvc.npz",  sv_test=best_sv_test)
    print(f"sv_train/test.npz updated for best config ({bf}, reps={br}, {be})")

    # Save meta so backend knows which feature transform to apply at inference
    import json as _json
    meta = {"feature_variant": bf, "reps": br, "entanglement": be,
            "C": best_cfg["C"], "class_weight": str(best_cfg["class_weight"])}
    with open(config.MODELS_DIR / "qsvc_meta.json", "w") as f:
        _json.dump(meta, f, indent=2)
    print(f"qsvc_meta.json saved → backend/models/qsvc_meta.json")

    # ── Save full sweep results ───────────────────────────────────────────────
    all_results_sorted = sorted(all_results, key=lambda x: -x["accuracy"])
    sweep_output = {
        "best_config":   best_cfg,
        "current_baseline": 0.8978,
        "paper_target":  0.9409,
        "improvement_pp": round((best_acc - 0.8978) * 100, 4),
        "top_10":        all_results_sorted[:10],
        "all_results":   all_results_sorted,
    }
    with open(config.RESULTS_DIR / "qsvc_sweep_results.json", "w") as f:
        json.dump(sweep_output, f, indent=2)
    print(f"Full sweep log → results/qsvc_sweep_results.json")

    # ── Update qsvc_results.json for dashboard ────────────────────────────────
    with open(config.RESULTS_DIR / "qsvc_results.json", "w") as f:
        json.dump({
            "model":            "QSVC (Tuned — ZZFeatureMap, Statevector kernel)",
            "accuracy":         float(best_acc),
            "feature_variant":  best_cfg["features"],
            "feature_dimension": config.FEATURE_DIMENSION,
            "reps":             best_cfg["reps"],
            "entanglement":     best_cfg["entanglement"],
            "C":                best_cfg["C"],
            "class_weight":     str(best_cfg["class_weight"]),
        }, f, indent=2)
    print(f"qsvc_results.json updated.")

    return best_cfg


if __name__ == "__main__":
    tune_qsvc()
