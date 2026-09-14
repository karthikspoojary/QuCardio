"""
ST-4B — Classical Baselines: RF, KNN, Logistic Regression, MLP
===============================================================
Evaluates four classical model families on the identical 9-D SVD features
used by QSVC (minmax_01 scaled, same train/test split) to formally establish
Quantum Advantage.

All models operate on data/features_9d.npz — no image loading, no SVD refit.

Outputs:
  results/classical_baselines.json   — accuracy, F1, per-class metrics, timing
  results/classical_baselines.png    — grouped bar chart (7 models)
  results/comparison_all_models_extended.png — 7-panel confusion matrices
  results/metrics_report.json        — updated with all 7 models
"""

import sys
import json
import time
import warnings
import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report,
)

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
import config
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────────────────────────────────────
# Model definitions  (each is a (name, estimator) pair)
# ─────────────────────────────────────────────────────────────────────────────

def get_models():
    return [
        ('Random Forest', RandomForestClassifier(
            n_estimators=500,
            max_depth=None,
            class_weight='balanced',
            random_state=config.RANDOM_SEED,
            n_jobs=-1,
        )),
        ('KNN', KNeighborsClassifier(
            n_neighbors=5,
            weights='distance',
            metric='euclidean',
            n_jobs=-1,
        )),
        ('Logistic Regression', LogisticRegression(
            C=1.0,
            max_iter=1000,
            class_weight='balanced',
            solver='lbfgs',
            random_state=config.RANDOM_SEED,
        )),
        ('MLP', MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation='relu',
            solver='adam',
            alpha=1e-4,
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=config.RANDOM_SEED,
        )),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run():
    print("=" * 65)
    print("ST-4B — Classical Baselines (RF, KNN, LR, MLP)")
    print("=" * 65)

    # ── Load identical 9D features (minmax_01 — same as training) ─────────────
    feat_path = config.FEATURES_DIR / 'features_9d.npz'
    if not feat_path.exists():
        raise FileNotFoundError(f"{feat_path} not found. Run reduce_dimensions.py first.")

    d = np.load(feat_path)
    X_train = d['train_features']   # (742, 9)  minmax_01
    X_test  = d['test_features']    # (186, 9)
    y_train = d['y_train']
    y_test  = d['y_test']
    print(f"Features: train={X_train.shape}  test={X_test.shape}")

    # ── Load existing quantum/SVM results for comparison ──────────────────────
    existing = {}
    metrics_path = config.RESULTS_DIR / 'metrics_report.json'
    if metrics_path.exists():
        with open(metrics_path) as f:
            report = json.load(f)
        existing = report.get('models', {})

    # ── Evaluate each classical model ─────────────────────────────────────────
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_SEED)
    new_results = {}
    all_preds   = {}   # for confusion matrices

    for name, model in get_models():
        print(f"\n── {name} ──────────────────────────────────────────────────")

        # 5-fold cross-val on training set
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv,
                                    scoring='accuracy', n_jobs=-1)
        print(f"  5-fold CV accuracy: {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")

        # Fit on full training set
        t_fit = time.time()
        model.fit(X_train, y_train)
        fit_time = time.time() - t_fit

        # Inference timing
        t_inf = time.time()
        y_pred = model.predict(X_test)
        inf_ms = (time.time() - t_inf) * 1000

        acc  = accuracy_score(y_test, y_pred)
        f1   = f1_score(y_test, y_pred, average='macro', zero_division=0)
        prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
        rec  = recall_score(y_test, y_pred, average='macro', zero_division=0)

        # Per-class metrics
        per_class = {}
        for c, cname in enumerate(config.CLASS_NAMES):
            mask = y_test == c
            if mask.sum() > 0:
                per_class[cname] = {
                    'precision': round(float(precision_score(y_test == c, y_pred == c, zero_division=0)), 4),
                    'recall':    round(float(recall_score(y_test == c, y_pred == c, zero_division=0)), 4),
                    'f1':        round(float(f1_score(y_test == c, y_pred == c, zero_division=0)), 4),
                }

        print(f"  Test accuracy: {acc*100:.2f}%  F1-macro: {f1:.4f}  inf: {inf_ms:.1f}ms")
        print(classification_report(y_test, y_pred,
                                    target_names=config.CLASS_NAMES, zero_division=0))

        new_results[name] = {
            'accuracy':        round(float(acc),  6),
            'f1_macro':        round(float(f1),   6),
            'precision_macro': round(float(prec), 6),
            'recall_macro':    round(float(rec),  6),
            'cv_mean':         round(float(cv_scores.mean()), 6),
            'cv_std':          round(float(cv_scores.std()),  6),
            'fit_time_s':      round(fit_time, 3),
            'inference_ms':    round(inf_ms, 2),
            'per_class':       per_class,
        }
        all_preds[name] = y_pred

    # ── Save classical baselines JSON ──────────────────────────────────────────
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_json = config.RESULTS_DIR / 'classical_baselines.json'
    with open(out_json, 'w') as f:
        json.dump(new_results, f, indent=2)
    print(f"\nClassical baselines saved → {out_json}")

    # ── Comparison bar chart (7 models: 4 new + 3 existing quantum/SVM) ───────
    _plot_comparison(new_results, existing)

    # ── 7-panel confusion matrices ─────────────────────────────────────────────
    # Load existing quantum predictions for the CM grid
    existing_preds = _load_existing_preds(X_train, X_test, y_train, y_test)
    _plot_confusion_matrices(all_preds, existing_preds, y_test)

    # ── Update metrics_report.json with all 7 models ──────────────────────────
    _update_metrics_report(new_results, existing, metrics_path)

    print("\n✅ ST-4B complete.")
    return new_results


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _plot_comparison(new_results, existing):
    """Grouped bar chart: all 7 models, accuracy + F1."""
    # Build ordered model list: classical SVM first, then new classicals, then quantum
    model_order = ['Classical SVM', 'Random Forest', 'KNN', 'Logistic Regression', 'MLP',
                   'QSVC', 'Pegasos QSVC']

    accs, f1s, labels = [], [], []
    for m in model_order:
        if m in new_results:
            accs.append(new_results[m]['accuracy'] * 100)
            f1s.append(new_results[m]['f1_macro'] * 100)
            labels.append(m)
        elif m in existing:
            accs.append(existing[m]['accuracy'] * 100)
            f1s.append(existing[m]['f1'] * 100)
            labels.append(m)

    x   = np.arange(len(labels))
    w   = 0.38
    colors_acc = ['#3b82f6' if 'QSVC' in l or 'Pegasos' in l else '#6b7280' for l in labels]
    colors_f1  = ['#7c3aed' if 'QSVC' in l or 'Pegasos' in l else '#9ca3af' for l in labels]

    fig, ax = plt.subplots(figsize=(12, 5))
    b1 = ax.bar(x - w/2, accs, w, label='Accuracy', color=colors_acc, edgecolor='white')
    b2 = ax.bar(x + w/2, f1s,  w, label='F1-macro', color=colors_f1,  edgecolor='white')

    for bar in list(b1) + list(b2):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha='right', fontsize=9)
    ax.set_ylabel('Score (%)', fontsize=11)
    ax.set_ylim(40, 105)
    ax.set_title('All Models Comparison: Classical vs Quantum (9-D SVD Features)',
                 fontsize=12)
    ax.axvline(x=4.5, color='#ef4444', linestyle='--', lw=1, alpha=0.6)
    ax.text(4.55, 102, 'Quantum →', fontsize=8, color='#ef4444')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    out_png = config.RESULTS_DIR / 'classical_baselines.png'
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Comparison chart saved → {out_png}")


def _load_existing_preds(X_train, X_test, y_train, y_test):
    """Load SVM + QSVC + Pegasos predictions from saved models for CM grid."""
    preds = {}

    # Classical SVM
    svm_path = config.MODELS_DIR / 'svm_model.pkl'
    if svm_path.exists():
        try:
            svm = joblib.load(svm_path)
            preds['Classical SVM'] = svm.predict(X_test)
        except Exception as e:
            print(f"  SVM load failed: {e}")

    # QSVC
    qsvc_path = config.MODELS_DIR / 'qsvc_model.pkl'
    sv_tr_path = config.FEATURES_DIR / 'sv_train_qsvc.npz'
    sv_te_path = config.FEATURES_DIR / 'sv_test_qsvc.npz'
    if qsvc_path.exists() and sv_tr_path.exists() and sv_te_path.exists():
        try:
            qsvc = joblib.load(qsvc_path)
            sv_tr = np.load(sv_tr_path, allow_pickle=True)['sv_train']
            sv_te = np.load(sv_te_path, allow_pickle=True)['sv_test']
            K_te  = (np.abs(np.dot(sv_te, sv_tr.conj().T)) ** 2).astype(np.float32)
            K_tr  = (np.abs(np.dot(sv_tr, sv_tr.conj().T)) ** 2).astype(np.float32)
            preds['QSVC'] = qsvc.predict(K_te)
        except Exception as e:
            print(f"  QSVC load failed: {e}")

    # Pegasos
    try:
        from src.quantum.train_pegasos import CLASS_PAIRS
        sv_peg_tr = config.FEATURES_DIR / 'sv_train_pegasos.npz'
        sv_peg_te = config.FEATURES_DIR / 'sv_test_pegasos.npz'
        if sv_peg_tr.exists() and sv_peg_te.exists():
            sv_tr_p = np.load(sv_peg_tr, allow_pickle=True)['sv_train']
            sv_te_p = np.load(sv_peg_te, allow_pickle=True)['sv_test']
            K_te_p  = (np.abs(np.dot(sv_te_p, sv_tr_p.conj().T)) ** 2).astype(np.float64)
            models_p = {}
            for c1, c2 in CLASS_PAIRS:
                p = config.MODELS_DIR / f'pegasos_{c1}_{c2}.pkl'
                if p.exists():
                    models_p[(c1, c2)] = joblib.load(p)
            if len(models_p) == 6:
                y_pred_p = []
                for i in range(len(y_test)):
                    def node(c1, c2):
                        m   = models_p[(c1, c2)]
                        idx = m.train_indices_
                        k   = K_te_p[i, idx].reshape(1, -1)
                        pr  = m.predict(k)[0]
                        return c1 if pr == -1 else c2
                    p01 = node(0, 1); p23 = node(2, 3)
                    if p01 == 0:
                        y_pred_p.append(node(0, 2) if p23 == 2 else node(0, 3))
                    else:
                        y_pred_p.append(node(1, 2) if p23 == 2 else node(1, 3))
                preds['Pegasos'] = np.array(y_pred_p)
    except Exception as e:
        print(f"  Pegasos load failed: {e}")

    return preds


def _plot_confusion_matrices(new_preds, existing_preds, y_test):
    """7-panel confusion matrix grid."""
    all_models = {}
    # Order: SVM, RF, KNN, LR, MLP, QSVC, Pegasos
    for name in ['Classical SVM', 'Random Forest', 'KNN',
                 'Logistic Regression', 'MLP', 'QSVC', 'Pegasos']:
        if name in existing_preds:
            all_models[name] = existing_preds[name]
        elif name in new_preds:
            all_models[name] = new_preds[name]

    n = len(all_models)
    if n == 0:
        return

    cols = min(n, 4)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3.5))
    axes = np.array(axes).flatten()

    cmaps = ['Blues', 'Greens', 'Oranges', 'Purples', 'YlOrRd', 'PuBu', 'BuPu']

    for idx, (model_name, y_pred) in enumerate(all_models.items()):
        ax = axes[idx]
        cm = confusion_matrix(y_test, y_pred)
        acc = accuracy_score(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap=cmaps[idx % len(cmaps)],
                    xticklabels=['Nor', 'Arr', 'MI', 'H-MI'],
                    yticklabels=['Nor', 'Arr', 'MI', 'H-MI'],
                    ax=ax, cbar=False)
        ax.set_title(f'{model_name}\n{acc*100:.1f}%', fontsize=9)
        ax.set_xlabel('Predicted', fontsize=8)
        ax.set_ylabel('True', fontsize=8)
        ax.tick_params(labelsize=7)

    # Hide unused axes
    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle('Confusion Matrices — All 7 Models (9-D SVD Features)', fontsize=12)
    plt.tight_layout()
    out_png = config.RESULTS_DIR / 'comparison_all_models_extended.png'
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"7-panel CM saved → {out_png}")


def _update_metrics_report(new_results, existing, metrics_path):
    """Append classical baselines to metrics_report.json."""
    if not metrics_path.exists():
        print("  metrics_report.json not found — skipping update")
        return

    with open(metrics_path) as f:
        report = json.load(f)

    for name, res in new_results.items():
        report['models'][name] = {
            'accuracy':        res['accuracy'],
            'precision':       res['precision_macro'],
            'recall':          res['recall_macro'],
            'f1':              res['f1_macro'],
            'inference_time_ms': res['inference_ms'],
            'cv_5fold_mean':   res['cv_mean'],
            'cv_5fold_std':    res['cv_std'],
            'config':          f"ST-4B classical baseline, 9-D SVD features",
        }

    with open(metrics_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"metrics_report.json updated with {len(new_results)} new models")


if __name__ == '__main__':
    run()
