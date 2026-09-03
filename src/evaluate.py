"""
evaluate.py — QuCardio Side-by-Side Model Comparison
======================================================
Loads all trained models, runs inference on the test split,
prints a Table-6-style comparison (matching the paper's format),
generates a 3-panel confusion-matrix figure, and saves:
  results/metrics_report.json
  results/metrics_report.txt
  results/comparison_all_models.png

Usage:
  python src/evaluate.py
"""

import sys, os, json, time, warnings
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.preprocessing import normalize

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def specificity_score(y_true, y_pred, labels):
    """Macro-averaged specificity: TN/(TN+FP) per class."""
    specs = []
    for c in labels:
        tp = ((y_true == c) & (y_pred == c)).sum()
        fn = ((y_true == c) & (y_pred != c)).sum()
        fp = ((y_true != c) & (y_pred == c)).sum()
        tn = ((y_true != c) & (y_pred != c)).sum()
        specs.append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)
    return float(np.mean(specs))


def build_kernel(sv1, sv2):
    return (np.abs(np.dot(sv1, sv2.conj().T)) ** 2).astype(np.float64)


def load_data():
    d       = np.load(config.FEATURES_DIR / "features_9d.npz")
    return (d["train_features"], d["test_features"],
            d["y_train"],        d["y_test"])


# ─────────────────────────────────────────────────────────────────────────────
# Prediction functions
# ─────────────────────────────────────────────────────────────────────────────

def predict_classical(X_test, y_test):
    print("  [Classical SVM] loading model …")
    svm = joblib.load(config.MODELS_DIR / "svm_model.pkl")
    t0  = time.time()
    y_pred = svm.predict(X_test)
    elapsed = time.time() - t0
    return y_pred, elapsed


def predict_qsvc(X_train, X_test, y_train):
    print("  [QSVC] loading model + statevectors …")
    qsvc = joblib.load(config.MODELS_DIR / "qsvc_model.pkl")

    # Load meta (best feature variant, reps, entanglement)
    meta_path = config.MODELS_DIR / "qsvc_meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        feat_variant = meta.get("feature_variant", "minmax_0pi")
        reps         = int(meta.get("reps", 2))
        entanglement = meta.get("entanglement", "circular")
    else:
        feat_variant, reps, entanglement = "minmax_0pi", 2, "circular"

    print(f"         feat={feat_variant}  reps={reps}  entangle={entanglement}")

    # Apply feature transform
    def transform(X):
        if feat_variant == "minmax_0pi":
            return X * np.pi
        elif feat_variant == "l2_norm":
            return normalize(X, norm="l2")
        return X

    X_tr_t = transform(X_train)
    X_te_t = transform(X_test)

    # Load or recompute statevectors
    sv_tr_path = config.FEATURES_DIR / "sv_train_qsvc.npz"
    sv_te_path = config.FEATURES_DIR / "sv_test_qsvc.npz"

    if sv_tr_path.exists() and sv_te_path.exists():
        sv_tr = np.load(sv_tr_path, allow_pickle=True)["sv_train"]
        sv_te = np.load(sv_te_path, allow_pickle=True)["sv_test"]
        print(f"         sv_train loaded: {sv_tr.shape}")
    else:
        print("         Computing statevectors (this will take ~30s) …")
        from qiskit.circuit.library import ZZFeatureMap
        from qiskit.quantum_info import Statevector
        fm = ZZFeatureMap(feature_dimension=config.FEATURE_DIMENSION,
                          reps=reps, entanglement=entanglement)
        sv_tr = np.array([Statevector(fm.assign_parameters(x)).data for x in X_tr_t])
        sv_te = np.array([Statevector(fm.assign_parameters(x)).data for x in X_te_t])
        np.savez_compressed(sv_tr_path, sv_train=sv_tr)
        np.savez_compressed(sv_te_path, sv_test=sv_te)

    # Build kernel matrices
    K_tr = build_kernel(sv_tr, sv_tr).astype(np.float32)
    K_te = build_kernel(sv_te, sv_tr).astype(np.float32)

    t0 = time.time()
    y_pred = qsvc.predict(K_te)
    elapsed = time.time() - t0
    return y_pred, elapsed


def predict_pegasos(X_test, y_test):
    print("  [Pegasos QSVC] loading 6 binary models …")

    from src.quantum.train_pegasos import PegasosSVMKernel, CLASS_PAIRS

    sv_tr_path = config.FEATURES_DIR / "sv_train_pegasos.npz"
    sv_te_path = config.FEATURES_DIR / "sv_test_pegasos.npz"

    if not sv_tr_path.exists() or not sv_te_path.exists():
        print("         ERROR: sv_train_pegasos.npz / sv_test_pegasos.npz not found.")
        print("         Run scripts/fix_statevectors.py to generate Pegasos statevectors.")
        return None, 0

    sv_tr = np.load(sv_tr_path, allow_pickle=True)["sv_train"]
    sv_te = np.load(sv_te_path, allow_pickle=True)["sv_test"]

    K_tr_full = build_kernel(sv_tr, sv_tr)
    K_te_full = build_kernel(sv_te, sv_tr)

    models = {}
    for c1, c2 in CLASS_PAIRS:
        path = config.MODELS_DIR / f"pegasos_{c1}_{c2}.pkl"
        if path.exists():
            models[(c1, c2)] = joblib.load(path)
        else:
            print(f"         WARNING: {path.name} not found — Pegasos will be incomplete")

    if len(models) < 6:
        print(f"         Only {len(models)}/6 models found — skipping Pegasos")
        return None, 0

    def predict_node(i, c1, c2):
        m   = models[(c1, c2)]
        idx = m.train_indices_
        k   = K_te_full[i, idx].reshape(1, -1)
        p   = m.predict(k)[0]
        return c1 if p == -1 else c2

    t0 = time.time()
    y_pred = []
    for i in range(len(y_test)):
        p01 = predict_node(i, 0, 1)
        p23 = predict_node(i, 2, 3)
        if p01 == 0:
            y_pred.append(predict_node(i, 0, 2) if p23 == 2 else predict_node(i, 0, 3))
        else:
            y_pred.append(predict_node(i, 1, 2) if p23 == 2 else predict_node(i, 1, 3))
    elapsed = time.time() - t0
    return np.array(y_pred), elapsed


# ─────────────────────────────────────────────────────────────────────────────
# Main comparison
# ─────────────────────────────────────────────────────────────────────────────

def compare_all_models():
    print("\n" + "=" * 70)
    print("QuCardio — Full Model Comparison  (Prabhu et al. 2023 replication)")
    print("=" * 70)

    X_train, X_test, y_train, y_test = load_data()
    labels = list(range(len(config.CLASS_NAMES)))

    results  = {}
    y_preds  = {}

    # ── 1. Classical SVM ──────────────────────────────────────────────────────
    y_pred_svm, t_svm = predict_classical(X_test, y_test)
    y_preds["Classical SVM"] = y_pred_svm
    results["Classical SVM"] = {
        "accuracy":    accuracy_score(y_test,  y_pred_svm),
        "precision":   precision_score(y_test, y_pred_svm, average="macro", zero_division=0),
        "recall":      recall_score(y_test,    y_pred_svm, average="macro", zero_division=0),
        "f1":          f1_score(y_test,        y_pred_svm, average="macro", zero_division=0),
        "specificity": specificity_score(y_test, y_pred_svm, labels),
        "inf_time_ms": round(t_svm * 1000, 2),
        "paper_acc":   0.8333,
        "config": "RBF kernel, C=10, class_weight=balanced, CalibratedClassifierCV",
    }

    # ── 2. QSVC ───────────────────────────────────────────────────────────────
    y_pred_qsvc, t_qsvc = predict_qsvc(X_train, X_test, y_train)
    y_preds["QSVC"] = y_pred_qsvc
    results["QSVC"] = {
        "accuracy":    accuracy_score(y_test,  y_pred_qsvc),
        "precision":   precision_score(y_test, y_pred_qsvc, average="macro", zero_division=0),
        "recall":      recall_score(y_test,    y_pred_qsvc, average="macro", zero_division=0),
        "f1":          f1_score(y_test,        y_pred_qsvc, average="macro", zero_division=0),
        "specificity": specificity_score(y_test, y_pred_qsvc, labels),
        "inf_time_ms": round(t_qsvc * 1000, 2),
        "paper_acc":   0.9409,
        "config": "ZZFeatureMap reps=2 circular, feats×π, SVC precomputed C=5.0",
    }

    # ── 3. Pegasos QSVC ───────────────────────────────────────────────────────
    y_pred_peg, t_peg = predict_pegasos(X_test, y_test)
    if y_pred_peg is not None:
        y_preds["Pegasos QSVC"] = y_pred_peg
        results["Pegasos QSVC"] = {
            "accuracy":    accuracy_score(y_test,  y_pred_peg),
            "precision":   precision_score(y_test, y_pred_peg, average="macro", zero_division=0),
            "recall":      recall_score(y_test,    y_pred_peg, average="macro", zero_division=0),
            "f1":          f1_score(y_test,        y_pred_peg, average="macro", zero_division=0),
            "specificity": specificity_score(y_test, y_pred_peg, labels),
            "inf_time_ms": round(t_peg * 1000, 2),
            "paper_acc":   0.9305,
            "config": "6 binary models, per-model C & τ tuning (3-pass grid), Algorithm 1",
        }

    # ── Paper reference row ───────────────────────────────────────────────────
    paper = {
        "Classical SVM": {"accuracy":0.8333,"precision":0.8244,"recall":0.8333,"f1":0.8221,"specificity":0.9428},
        "QSVC":          {"accuracy":0.9409,"precision":0.9417,"recall":0.9409,"f1":0.9400,"specificity":0.9802},
        "Pegasos QSVC":  {"accuracy":0.9305,"precision":0.9350,"recall":0.9305,"f1":0.9271,"specificity":0.9766},
    }

    # ── Print Table ───────────────────────────────────────────────────────────
    cols = ["Accuracy", "Precision", "Recall", "F1", "Specificity"]
    w    = 14
    divider = "─" * (22 + w * 5)

    print(f"\n{'Model':<22}" + "".join(f"{c:>{w}}" for c in cols))
    print(divider)

    for name, r in results.items():
        vals = [r["accuracy"], r["precision"], r["recall"], r["f1"], r["specificity"]]
        print(f"{name:<22}" + "".join(f"{v*100:>{w}.2f}%" for v in vals))
        prow = paper.get(name, {})
        pvals = [prow.get("accuracy",0), prow.get("precision",0),
                 prow.get("recall",0),   prow.get("f1",0),
                 prow.get("specificity",0)]
        print(f"  {'(paper)':18}" + "".join(f"{v*100:>{w}.2f}%" for v in pvals))
        gains = [o - p for o, p in zip(vals, pvals)]
        gain_str = "".join(f"{('+' if g>=0 else '')}{g*100:>{w-1}.2f}%"
                           for g in gains)
        print(f"  {'(Δ vs paper)':18}" + gain_str)
        print()

    print(divider)

    # ── Classification reports ────────────────────────────────────────────────
    for name, y_pred in y_preds.items():
        print(f"\n{'─'*70}")
        print(f"Classification Report — {name}")
        print(f"{'─'*70}")
        print(classification_report(y_test, y_pred,
                                    target_names=config.CLASS_NAMES, zero_division=0))

    # ── 3-panel confusion matrix figure ───────────────────────────────────────
    model_names = list(y_preds.keys())
    n = len(model_names)
    colors = {"Classical SVM": "Blues", "QSVC": "Purples", "Pegasos QSVC": "Greens"}

    fig = plt.figure(figsize=(6 * n, 5.5))
    gs  = gridspec.GridSpec(1, n, figure=fig, wspace=0.35)

    for i, name in enumerate(model_names):
        cm  = confusion_matrix(y_test, y_preds[name])
        acc = results[name]["accuracy"]
        ax  = fig.add_subplot(gs[0, i])
        short_names = ["Normal", "Arrhy.", "MI", "Hist.MI"]
        sns.heatmap(cm, annot=True, fmt="d", cmap=colors.get(name, "Blues"),
                    xticklabels=short_names, yticklabels=short_names,
                    ax=ax, linewidths=0.5, annot_kws={"size": 11})
        ax.set_title(f"{name}\nAcc: {acc*100:.1f}%  |  Paper: {results[name]['paper_acc']*100:.1f}%",
                     fontsize=11, fontweight="bold", pad=10)
        ax.set_ylabel("True Label", fontsize=9)
        ax.set_xlabel("Predicted Label", fontsize=9)
        ax.tick_params(axis="both", labelsize=8)

    fig.suptitle("QuCardio — Model Comparison\n"
                 "ECG Cardiovascular Disease Classification (Prabhu et al. 2023 replication)",
                 fontsize=12, fontweight="bold", y=1.02)

    config.RESULTS_DIR.mkdir(exist_ok=True)
    out_path = config.RESULTS_DIR / "comparison_all_models.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nConfusion matrix panel saved → {out_path}")

    # ── Save JSON + TXT reports ───────────────────────────────────────────────
    report_json = {
        "dataset": "ECG Images — Ch. Pervaiz Elahi Institute of Cardiology",
        "total_samples": 929, "train": 742, "test": 186,
        "classes": config.CLASS_NAMES,
        "feature_pipeline": "ResNet50 pool1_pool (462,400-D) → TruncatedSVD(9) → MinMax[0,1]",
        "models": {}
    }
    for name, r in results.items():
        report_json["models"][name] = {
            "accuracy":    round(r["accuracy"],    4),
            "precision":   round(r["precision"],   4),
            "recall":      round(r["recall"],      4),
            "f1":          round(r["f1"],          4),
            "specificity": round(r["specificity"], 4),
            "paper_accuracy": r["paper_acc"],
            "delta_vs_paper": round(r["accuracy"] - r["paper_acc"], 4),
            "inference_time_ms": r["inf_time_ms"],
            "config": r["config"],
        }

    with open(config.RESULTS_DIR / "metrics_report.json", "w") as f:
        json.dump(report_json, f, indent=2)
    print(f"metrics_report.json saved → {config.RESULTS_DIR / 'metrics_report.json'}")

    # Plain text report
    lines = [
        "QuCardio — Model Comparison Report",
        "=" * 60,
        f"Dataset : ECG Images (Ch. Pervaiz Elahi Institute, Multan)",
        f"Samples : 929 total | Train 742 | Test 186",
        f"Classes : {', '.join(config.CLASS_NAMES)}",
        f"Pipeline: ResNet50 pool1_pool (462,400-D) → TruncatedSVD(9) → MinMax",
        "",
        f"{'Model':<22}{'Acc':>8}{'Prec':>8}{'Rec':>8}{'F1':>8}{'Spec':>8}{'Paper':>8}{'Δ':>8}",
        "-" * 78,
    ]
    for name, r in results.items():
        lines.append(
            f"{name:<22}"
            f"{r['accuracy']*100:>7.2f}%"
            f"{r['precision']*100:>7.2f}%"
            f"{r['recall']*100:>7.2f}%"
            f"{r['f1']*100:>7.2f}%"
            f"{r['specificity']*100:>7.2f}%"
            f"{r['paper_acc']*100:>7.2f}%"
            f"{(r['accuracy']-r['paper_acc'])*100:>+7.2f}%"
        )
    lines += ["", "Generated by src/evaluate.py", f"Date: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M')}"]

    txt_path = config.RESULTS_DIR / "metrics_report.txt"
    with open(txt_path, "w") as f:
        f.write("\n".join(lines))
    print(f"metrics_report.txt  saved → {txt_path}")

    print("\n" + "=" * 70)
    print("✅ Evaluation complete!")
    print("=" * 70)
    return report_json


if __name__ == "__main__":
    compare_all_models()
