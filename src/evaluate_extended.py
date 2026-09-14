"""
ST-3 — Extended Evaluation: ROC-AUC Curves, McNemar's Test, Circuit Visualisation
==================================================================================
Produces all publication-quality evaluation artifacts for the paper:

  ST-3A: Multi-class ROC curves (per-class + macro-AUC, all 3 models)
         → results/roc_curves_all_models.png
         → results/roc_auc_table.json

  ST-3B: McNemar's statistical significance test (QSVC vs SVM, QSVC vs Pegasos)
         → results/mcnemar_test.json

  ST-3C: Quantum circuit visualisation (9-qubit ZZFeatureMap, reps=2, circular)
         → results/zzfeaturemap_circuit.png

  ST-3D: Confidence calibration & reliability diagrams
         → results/calibration_all_models.png
         → results/ece_scores.json

Usage:
  python src/evaluate_extended.py [--roc] [--mcnemar] [--circuit] [--calibration] [--all]

  Default (no flags) = --all
"""

import sys
import argparse
import json
import warnings
import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
warnings.filterwarnings('ignore')

from sklearn.metrics import (
    roc_curve, auc,
    accuracy_score, f1_score,
)
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import label_binarize, normalize


# ─────────────────────────────────────────────────────────────────────────────
# Data & model loaders
# ─────────────────────────────────────────────────────────────────────────────

def load_data():
    d = np.load(config.FEATURES_DIR / 'features_9d.npz')
    return (d['train_features'], d['test_features'],
            d['y_train'],        d['y_test'])


def build_kernel(sv1, sv2):
    return (np.abs(np.dot(sv1, sv2.conj().T)) ** 2).astype(np.float64)


def _get_svm_proba(X_test):
    """CalibratedClassifierCV returns predict_proba (N, 4)."""
    svm = joblib.load(config.MODELS_DIR / 'svm_model.pkl')
    return svm.predict_proba(X_test), svm.predict(X_test)


def _get_qsvc_proba(X_train, X_test):
    """
    QSVC: SVC(precomputed) → decision_function → softmax → pseudo-probabilities.
    Returns (N, 4) probability array and (N,) hard predictions.
    """
    qsvc = joblib.load(config.MODELS_DIR / 'qsvc_model.pkl')

    # Apply minmax_0pi encoding (confirmed best)
    X_tr_enc = X_train * np.pi
    X_te_enc = X_test  * np.pi

    sv_tr_path = config.FEATURES_DIR / 'sv_train_qsvc.npz'
    sv_te_path = config.FEATURES_DIR / 'sv_test_qsvc.npz'

    if sv_tr_path.exists() and sv_te_path.exists():
        sv_tr = np.load(sv_tr_path, allow_pickle=True)['sv_train']
        sv_te = np.load(sv_te_path, allow_pickle=True)['sv_test']
    else:
        print("  QSVC: computing statevectors (one-time) …")
        from qiskit.circuit.library import ZZFeatureMap
        from qiskit.quantum_info import Statevector
        fm = ZZFeatureMap(feature_dimension=9, reps=2, entanglement='circular')
        sv_tr = np.array([Statevector(fm.assign_parameters(x)).data for x in X_tr_enc])
        sv_te = np.array([Statevector(fm.assign_parameters(x)).data for x in X_te_enc])
        np.savez_compressed(sv_tr_path, sv_train=sv_tr)
        np.savez_compressed(sv_te_path, sv_test=sv_te)

    K_tr = build_kernel(sv_tr, sv_tr).astype(np.float32)
    K_te = build_kernel(sv_te, sv_tr).astype(np.float32)

    # decision_function: (N, 4) for multiclass OVR SVC
    scores = qsvc.decision_function(K_te)          # (N, 4)
    # softmax to convert to pseudo-probabilities
    e_x = np.exp(scores - scores.max(axis=1, keepdims=True))
    proba = e_x / e_x.sum(axis=1, keepdims=True)   # (N, 4)
    y_pred = qsvc.predict(K_te)
    return proba, y_pred


def _get_pegasos_scores(X_train, X_test, y_train):
    """
    Pegasos: 6 binary decision_functions → aggregate into (N, 4) score matrix.
    Uses OVR: for class c, score = sum of binary scores where c is the positive class.
    Then softmax → pseudo-probabilities.
    """
    from src.quantum.train_pegasos import CLASS_PAIRS

    sv_tr_path = config.FEATURES_DIR / 'sv_train_pegasos.npz'
    sv_te_path = config.FEATURES_DIR / 'sv_test_pegasos.npz'

    if not sv_tr_path.exists() or not sv_te_path.exists():
        print("  Pegasos: sv_train_pegasos.npz not found — skipping")
        return None, None

    sv_tr = np.load(sv_tr_path, allow_pickle=True)['sv_train']
    sv_te = np.load(sv_te_path, allow_pickle=True)['sv_test']
    K_tr_full = build_kernel(sv_tr, sv_tr)
    K_te_full = build_kernel(sv_te, sv_tr)

    models = {}
    for c1, c2 in CLASS_PAIRS:
        p = config.MODELS_DIR / f'pegasos_{c1}_{c2}.pkl'
        if p.exists():
            models[(c1, c2)] = joblib.load(p)
        else:
            print(f"  WARNING: {p.name} not found — Pegasos will be incomplete")

    if len(models) < 6:
        print(f"  Only {len(models)}/6 Pegasos models found — skipping")
        return None, None

    N = K_te_full.shape[0]
    n_classes = 4
    # Aggregate class scores: each binary model votes for its positive class
    class_scores = np.zeros((N, n_classes))

    for (c1, c2), m in models.items():
        idx = m.train_indices_
        K_rows = K_te_full[:, idx]                # (N, |idx|)
        df = np.dot(K_rows, m.alpha_ * m.y_train_) # (N,)
        # Positive class (c2) gets +score, negative class (c1) gets -score
        class_scores[:, c2] += df
        class_scores[:, c1] -= df

    # Softmax
    e_x = np.exp(class_scores - class_scores.max(axis=1, keepdims=True))
    proba = e_x / e_x.sum(axis=1, keepdims=True)

    # Hard predictions via decision tree (Algorithm 1)
    y_pred = []
    for i in range(N):
        def node(c1, c2):
            m   = models[(c1, c2)]
            idx = m.train_indices_
            k   = K_te_full[i, idx].reshape(1, -1)
            p   = m.predict(k)[0]
            return c1 if p == -1 else c2
        pred_01 = node(0, 1)
        pred_23 = node(2, 3)
        if pred_01 == 0:
            final = node(0, 2) if pred_23 == 2 else node(0, 3)
        else:
            final = node(1, 2) if pred_23 == 2 else node(1, 3)
        y_pred.append(final)

    return proba, np.array(y_pred)


# ─────────────────────────────────────────────────────────────────────────────
# ST-3A — Multi-class ROC-AUC Curves
# ─────────────────────────────────────────────────────────────────────────────

def run_roc(X_train, X_test, y_train, y_test):
    print("\n── ST-3A: Multi-class ROC-AUC Curves ──────────────────────────")

    n_classes = 4
    y_bin = label_binarize(y_test, classes=list(range(n_classes)))  # (N, 4)

    models_data = {}

    # Classical SVM
    print("  SVM …")
    try:
        svm_proba, svm_pred = _get_svm_proba(X_test)
        models_data['SVM'] = (svm_proba, svm_pred)
    except Exception as e:
        print(f"  SVM failed: {e}")

    # QSVC
    print("  QSVC …")
    try:
        qsvc_proba, qsvc_pred = _get_qsvc_proba(X_train, X_test)
        models_data['QSVC'] = (qsvc_proba, qsvc_pred)
    except Exception as e:
        print(f"  QSVC failed: {e}")

    # Pegasos
    print("  Pegasos …")
    try:
        peg_proba, peg_pred = _get_pegasos_scores(X_train, X_test, y_train)
        if peg_proba is not None:
            models_data['Pegasos'] = (peg_proba, peg_pred)
    except Exception as e:
        print(f"  Pegasos failed: {e}")

    n_models = len(models_data)
    if n_models == 0:
        print("  No models available — skipping ROC plot")
        return {}

    auc_table = {}
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5), sharey=True)
    if n_models == 1:
        axes = [axes]

    colors = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444']

    for ax, (model_name, (proba, pred)) in zip(axes, models_data.items()):
        auc_table[model_name] = {}
        fprs, tprs, aucs_per_class = [], [], []

        for c in range(n_classes):
            fpr, tpr, _ = roc_curve(y_bin[:, c], proba[:, c])
            roc_auc = auc(fpr, tpr)
            auc_table[model_name][config.CLASS_NAMES[c]] = round(roc_auc, 4)
            aucs_per_class.append(roc_auc)
            fprs.append(fpr)
            tprs.append(tpr)
            ax.plot(fpr, tpr, color=colors[c], lw=1.5,
                    label=f'{config.CLASS_NAMES[c]} (AUC={roc_auc:.3f})')

        # Macro-average ROC
        all_fpr = np.unique(np.concatenate(fprs))
        mean_tpr = np.zeros_like(all_fpr)
        for fpr, tpr in zip(fprs, tprs):
            mean_tpr += np.interp(all_fpr, fpr, tpr)
        mean_tpr /= n_classes
        macro_auc = auc(all_fpr, mean_tpr)
        auc_table[model_name]['macro_AUC'] = round(macro_auc, 4)

        ax.plot(all_fpr, mean_tpr, 'k--', lw=2,
                label=f'Macro-avg (AUC={macro_auc:.3f})')
        ax.plot([0, 1], [0, 1], 'gray', linestyle=':', lw=1)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.02])
        ax.set_xlabel('False Positive Rate', fontsize=10)
        ax.set_ylabel('True Positive Rate', fontsize=10)
        acc_pct = accuracy_score(y_test, pred) * 100
        ax.set_title(f'{model_name}\nAcc={acc_pct:.1f}%  Macro-AUC={macro_auc:.3f}', fontsize=11)
        ax.legend(fontsize=7, loc='lower right')
        ax.grid(alpha=0.3)

    fig.suptitle('Multi-class ROC Curves (One-vs-Rest)', fontsize=13, y=1.01)
    plt.tight_layout()
    out_png = config.RESULTS_DIR / 'roc_curves_all_models.png'
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ROC plot saved → {out_png}")

    out_json = config.RESULTS_DIR / 'roc_auc_table.json'
    with open(out_json, 'w') as f:
        json.dump(auc_table, f, indent=2)
    print(f"  AUC table saved → {out_json}")

    # Print summary
    for model, aucs in auc_table.items():
        print(f"  {model}: macro-AUC={aucs.get('macro_AUC', '?'):.4f}")

    return auc_table


# ─────────────────────────────────────────────────────────────────────────────
# ST-3B — McNemar's Statistical Significance Test
# ─────────────────────────────────────────────────────────────────────────────

def run_mcnemar(X_train, X_test, y_train, y_test):
    print("\n── ST-3B: McNemar's Statistical Significance Test ──────────────")

    try:
        from statsmodels.stats.contingency_tables import mcnemar as mcnemar_test
    except ImportError:
        print("  statsmodels not installed. Run: pip install statsmodels")
        return {}

    preds = {}
    # SVM
    try:
        _, svm_pred = _get_svm_proba(X_test)
        preds['SVM'] = svm_pred
    except Exception as e:
        print(f"  SVM failed: {e}")

    # QSVC
    try:
        _, qsvc_pred = _get_qsvc_proba(X_train, X_test)
        preds['QSVC'] = qsvc_pred
    except Exception as e:
        print(f"  QSVC failed: {e}")

    # Pegasos
    try:
        _, peg_pred = _get_pegasos_scores(X_train, X_test, y_train)
        if peg_pred is not None:
            preds['Pegasos'] = peg_pred
    except Exception as e:
        print(f"  Pegasos failed: {e}")

    results = {}
    pairs = [('QSVC', 'SVM'), ('QSVC', 'Pegasos'), ('Pegasos', 'SVM')]

    for model_a, model_b in pairs:
        if model_a not in preds or model_b not in preds:
            print(f"  {model_a} vs {model_b}: skipped (model missing)")
            continue

        pred_a = preds[model_a]
        pred_b = preds[model_b]

        correct_a = (pred_a == y_test).astype(int)
        correct_b = (pred_b == y_test).astype(int)

        # 2×2 contingency table
        n00 = int(((correct_a == 0) & (correct_b == 0)).sum())  # both wrong
        n01 = int(((correct_a == 0) & (correct_b == 1)).sum())  # A wrong, B correct
        n10 = int(((correct_a == 1) & (correct_b == 0)).sum())  # A correct, B wrong
        n11 = int(((correct_a == 1) & (correct_b == 1)).sum())  # both correct

        table = np.array([[n11, n10], [n01, n00]])

        try:
            result = mcnemar_test(table, exact=False, correction=True)
            chi2 = float(result.statistic)
            pval = float(result.pvalue)
        except Exception as ex:
            chi2, pval = float('nan'), float('nan')
            print(f"  McNemar error ({model_a} vs {model_b}): {ex}")

        acc_a = accuracy_score(y_test, pred_a) * 100
        acc_b = accuracy_score(y_test, pred_b) * 100
        sig = 'p<0.001' if pval < 0.001 else ('p<0.05' if pval < 0.05 else f'p={pval:.3f}')
        conclusion = (f"The accuracy gain of {model_a} ({acc_a:.2f}%) over "
                      f"{model_b} ({acc_b:.2f}%) is "
                      + ("statistically significant" if pval < 0.05 else "NOT statistically significant")
                      + f" (McNemar χ²={chi2:.3f}, {sig}).")

        print(f"  {model_a} vs {model_b}: χ²={chi2:.3f}  {sig}")
        print(f"    n11={n11}, n10={n10}, n01={n01}, n00={n00}")
        print(f"    {conclusion}")

        results[f'{model_a}_vs_{model_b}'] = {
            'accuracy_a': round(acc_a, 4),
            'accuracy_b': round(acc_b, 4),
            'model_a': model_a,
            'model_b': model_b,
            'contingency_table': {'n11': n11, 'n10': n10, 'n01': n01, 'n00': n00},
            'chi2': round(chi2, 6) if not np.isnan(chi2) else None,
            'p_value': round(pval, 8) if not np.isnan(pval) else None,
            'significant': bool(pval < 0.05) if not np.isnan(pval) else None,
            'conclusion': conclusion,
        }

    out_json = config.RESULTS_DIR / 'mcnemar_test.json'
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  McNemar results saved → {out_json}")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# ST-3C — Quantum Circuit Visualisation
# ─────────────────────────────────────────────────────────────────────────────

def run_circuit_viz():
    print("\n── ST-3C: Quantum Circuit Visualisation ────────────────────────")
    try:
        from qiskit.circuit.library import ZZFeatureMap
        from qiskit.visualization import circuit_drawer

        fm = ZZFeatureMap(feature_dimension=9, reps=2, entanglement='circular')
        # decompose() expands to elementary gates for a more detailed diagram
        fig = circuit_drawer(
            fm.decompose(),
            output='mpl',
            fold=35,
            style={'backgroundcolor': '#FFFFFF'},
        )
        out_png = config.RESULTS_DIR / 'zzfeaturemap_circuit.png'
        fig.savefig(str(out_png), dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  Circuit diagram saved → {out_png}")
    except Exception as e:
        print(f"  Circuit viz failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# ST-3D — Confidence Calibration & Reliability Diagrams
# ─────────────────────────────────────────────────────────────────────────────

def _compute_ece(y_true, y_prob, n_bins=10):
    """Expected Calibration Error (ECE)."""
    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy='uniform')
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_lowers = bin_edges[:-1]
    bin_uppers = bin_edges[1:]
    n = len(y_true)
    ece = 0.0
    for lower, upper, acc, conf in zip(bin_lowers, bin_uppers, frac_pos, mean_pred):
        mask = (y_prob >= lower) & (y_prob < upper)
        n_b = mask.sum()
        if n_b > 0:
            ece += (n_b / n) * abs(acc - conf)
    return float(ece)


def run_calibration(X_train, X_test, y_train, y_test):
    print("\n── ST-3D: Confidence Calibration Diagrams ──────────────────────")

    n_classes = 4
    y_bin = label_binarize(y_test, classes=list(range(n_classes)))

    probas = {}
    try:
        svm_proba, _ = _get_svm_proba(X_test)
        probas['SVM'] = svm_proba
    except Exception as e:
        print(f"  SVM failed: {e}")

    try:
        qsvc_proba, _ = _get_qsvc_proba(X_train, X_test)
        probas['QSVC'] = qsvc_proba
    except Exception as e:
        print(f"  QSVC failed: {e}")

    try:
        peg_proba, _ = _get_pegasos_scores(X_train, X_test, y_train)
        if peg_proba is not None:
            probas['Pegasos'] = peg_proba
    except Exception as e:
        print(f"  Pegasos failed: {e}")

    n_models = len(probas)
    if n_models == 0:
        print("  No models available — skipping calibration")
        return {}

    ece_scores = {}
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 5), sharey=True)
    if n_models == 1:
        axes = [axes]

    colors = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444']

    for ax, (model_name, proba) in zip(axes, probas.items()):
        ece_scores[model_name] = {}
        macro_ece = 0.0
        for c in range(n_classes):
            frac_pos, mean_pred = calibration_curve(
                y_bin[:, c], proba[:, c], n_bins=10, strategy='uniform'
            )
            ece_c = _compute_ece(y_bin[:, c], proba[:, c])
            ece_scores[model_name][config.CLASS_NAMES[c]] = round(ece_c, 4)
            macro_ece += ece_c
            ax.plot(mean_pred, frac_pos, marker='s', color=colors[c], lw=1.5,
                    label=f'{config.CLASS_NAMES[c]} (ECE={ece_c:.3f})')

        macro_ece /= n_classes
        ece_scores[model_name]['macro_ECE'] = round(macro_ece, 4)
        ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Perfect calibration')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_xlabel('Mean Predicted Probability', fontsize=10)
        ax.set_ylabel('Fraction of Positives', fontsize=10)
        ax.set_title(f'{model_name}\nMacro-ECE={macro_ece:.3f}', fontsize=11)
        ax.legend(fontsize=7, loc='upper left')
        ax.grid(alpha=0.3)

    fig.suptitle('Reliability Diagrams (Confidence Calibration)', fontsize=13, y=1.01)
    plt.tight_layout()
    out_png = config.RESULTS_DIR / 'calibration_all_models.png'
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Calibration plot saved → {out_png}")

    out_json = config.RESULTS_DIR / 'ece_scores.json'
    with open(out_json, 'w') as f:
        json.dump(ece_scores, f, indent=2)
    print(f"  ECE scores saved → {out_json}")

    for model, scores in ece_scores.items():
        print(f"  {model}: macro-ECE={scores.get('macro_ECE', '?'):.4f}")

    return ece_scores


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Extended evaluation: ROC, McNemar, Circuit, Calibration')
    parser.add_argument('--roc',         action='store_true')
    parser.add_argument('--mcnemar',     action='store_true')
    parser.add_argument('--circuit',     action='store_true')
    parser.add_argument('--calibration', action='store_true')
    parser.add_argument('--all',         action='store_true')
    args = parser.parse_args()

    run_all = args.all or not any([args.roc, args.mcnemar, args.circuit, args.calibration])

    print("=" * 65)
    print("ST-3 — Extended Evaluation")
    print("=" * 65)

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    X_train, X_test, y_train, y_test = load_data()
    print(f"Test set: {X_test.shape[0]} samples  ({dict(zip(*np.unique(y_test, return_counts=True)))})")

    if run_all or args.roc:
        run_roc(X_train, X_test, y_train, y_test)

    if run_all or args.mcnemar:
        run_mcnemar(X_train, X_test, y_train, y_test)

    if run_all or args.circuit:
        run_circuit_viz()

    if run_all or args.calibration:
        run_calibration(X_train, X_test, y_train, y_test)

    print("\n✅ ST-3 complete.")


if __name__ == '__main__':
    main()
