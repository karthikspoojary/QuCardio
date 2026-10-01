"""
Bootstrap 95% Confidence Intervals for All Models
==================================================
Uses the standard bootstrap (B=10,000 resamples of the 186-image test set)
to produce percentile CIs for accuracy.  More honest than the Wald interval
for small n because it does not assume Gaussian accuracy distributions.

Outputs:
  results/paper/statistics/confidence_intervals.json
"""

import json
import numpy as np
from pathlib import Path

RNG   = np.random.default_rng(42)
B     = 10_000          # bootstrap resamples
ALPHA = 0.05            # 95% CI

# ── Load true labels and all model predictions ─────────────────────────────
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
import config

from sklearn.metrics import accuracy_score


def _load_predictions():
    """Return dict of {model_name: y_pred array}."""
    import joblib

    feat  = np.load(config.FEATURES_DIR / 'features_9d.npz')
    y_test = feat['y_test']
    X_test = feat['test_features']

    preds = {}

    # ── Classical SVM ─────────────────────────────────────────────────────
    svm_path = config.MODELS_DIR / 'svm_model.pkl'
    if svm_path.exists():
        svm = joblib.load(svm_path)
        preds['Classical SVM'] = svm.predict(X_test)

    # ── QSVC ──────────────────────────────────────────────────────────────
    qsvc_path  = config.MODELS_DIR / 'qsvc_model.pkl'
    sv_tr_path = config.FEATURES_DIR / 'sv_train_qsvc.npz'
    sv_te_path = config.FEATURES_DIR / 'sv_test_qsvc.npz'
    if qsvc_path.exists() and sv_tr_path.exists() and sv_te_path.exists():
        qsvc  = joblib.load(qsvc_path)
        sv_tr = np.load(sv_tr_path, allow_pickle=True)['sv_train']
        sv_te = np.load(sv_te_path, allow_pickle=True)['sv_test']
        K_te  = (np.abs(np.dot(sv_te, sv_tr.conj().T)) ** 2).astype(np.float32)
        preds['QSVC'] = qsvc.predict(K_te)

    # ── Random Forest (train inline — fast, <1s on 9 features) ───────────
    from sklearn.ensemble import RandomForestClassifier
    X_train = feat['train_features']
    y_train = feat['y_train']
    rf = RandomForestClassifier(n_estimators=500, class_weight='balanced',
                                random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    preds['Random Forest'] = rf.predict(X_test)

    # ── Pegasos QSVC ──────────────────────────────────────────────────────
    peg_paths = [config.MODELS_DIR / f'pegasos_{c1}_{c2}.pkl'
                 for c1, c2 in [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]]
    sv_peg_tr = config.FEATURES_DIR / 'sv_train_pegasos.npz'
    sv_peg_te = config.FEATURES_DIR / 'sv_test_pegasos.npz'
    if all(p.exists() for p in peg_paths) and sv_peg_tr.exists() and sv_peg_te.exists():
        sv_tr_p = np.load(sv_peg_tr, allow_pickle=True)['sv_train']
        sv_te_p = np.load(sv_peg_te, allow_pickle=True)['sv_test']
        K_te_p  = (np.abs(np.dot(sv_te_p, sv_tr_p.conj().T)) ** 2).astype(np.float64)
        models_p = {(c1,c2): joblib.load(config.MODELS_DIR / f'pegasos_{c1}_{c2}.pkl')
                    for c1, c2 in [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]}
        y_pred_p = []
        for i in range(len(y_test)):
            def _node(c1, c2, _i=i):
                m   = models_p[(c1, c2)]
                idx = m.train_indices_
                k   = K_te_p[_i, idx].reshape(1, -1)
                return c1 if m.predict(k)[0] == -1 else c2
            p01 = _node(0,1); p23 = _node(2,3)
            y_pred_p.append(_node(0,2) if p01==0 and p23==2 else
                            _node(0,3) if p01==0 else
                            _node(1,2) if p23==2 else _node(1,3))
        preds['Pegasos QSVC'] = np.array(y_pred_p)

    # ── XGBoost & LightGBM (re-train if no saved model, takes <2s) ────────
    try:
        import xgboost as xgb
        clf_xgb = xgb.XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric='mlogloss', random_state=42, verbosity=0)
        clf_xgb.fit(X_train, y_train)
        preds['XGBoost'] = clf_xgb.predict(X_test)
    except ImportError:
        pass

    try:
        import lightgbm as lgb
        clf_lgb = lgb.LGBMClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            class_weight='balanced', random_state=42, verbose=-1)
        clf_lgb.fit(X_train, y_train)
        preds['LightGBM'] = clf_lgb.predict(X_test)
    except ImportError:
        pass

    return y_test, preds


def bootstrap_ci(y_true, y_pred, B=B, alpha=ALPHA, rng=RNG):
    """Percentile bootstrap CI for accuracy."""
    n        = len(y_true)
    correct  = (y_true == y_pred).astype(float)
    boot_acc = np.array([
        correct[rng.integers(0, n, n)].mean() for _ in range(B)
    ])
    lo = float(np.percentile(boot_acc, 100 * alpha / 2))
    hi = float(np.percentile(boot_acc, 100 * (1 - alpha / 2)))
    point = float(correct.mean())
    return {
        "accuracy":  round(point, 6),
        "n":         n,
        "ci_lower":  round(lo, 6),
        "ci_upper":  round(hi, 6),
        "margin":    round((hi - lo) / 2, 6),
        "formatted": (
            f"{point*100:.2f}% "
            f"[95% CI {lo*100:.2f}%\u2013{hi*100:.2f}%]"
        ),
        "method": f"bootstrap percentile, B={B}",
    }


def main():
    print(f"Loading predictions …")
    y_test, preds = _load_predictions()
    print(f"  y_test: {len(y_test)} samples")
    print(f"  Models found: {list(preds.keys())}")

    results = {}
    for name, y_pred in preds.items():
        r = bootstrap_ci(y_test, y_pred)
        results[name] = r
        print(f"  {name:<20} {r['formatted']}")

    out = Path('results/paper/statistics/confidence_intervals.json')
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {out}")


if __name__ == '__main__':
    main()
