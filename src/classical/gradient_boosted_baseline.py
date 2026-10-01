"""
Gradient-Boosted Classical Baselines: XGBoost and LightGBM
===========================================================
Evaluates XGBoost and LightGBM on the identical 9-D SVD features used by
the QSVC (minmax_01 scaled, same 80:20 stratified split, random_state=42).

This is the closest reviewer objection to the quantum advantage claim:
"did you compare against gradient-boosted trees?"  This script answers it.

Outputs:
  results/paper/main/gradient_boosted_results.json
  results/paper/main/gradient_boosted_baselines.png
"""

import sys
import json
import time
import warnings
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix
)

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
import config

warnings.filterwarnings('ignore')

OUT_DIR = Path('results/paper/main')
OUT_DIR.mkdir(parents=True, exist_ok=True)


def run():
    print("=" * 65)
    print("Gradient-Boosted Classical Baselines: XGBoost + LightGBM")
    print("=" * 65)

    feat_path = config.FEATURES_DIR / 'features_9d.npz'
    if not feat_path.exists():
        raise FileNotFoundError(f"{feat_path} not found. Run reduce_dimensions.py first.")

    d = np.load(feat_path)
    X_train = d['train_features']   # (742, 9)  minmax_01
    X_test  = d['test_features']    # (186, 9)
    y_train = d['y_train']
    y_test  = d['y_test']
    print(f"Features: train={X_train.shape}  test={X_test.shape}")

    results = {}

    # ── XGBoost ───────────────────────────────────────────────────────────────
    try:
        import xgboost as xgb
        print("\n── XGBoost ──────────────────────────────────────────────────")
        t0 = time.time()
        # scale_pos_weight not used for multiclass; use sample_weight instead
        clf_xgb = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric='mlogloss',
            random_state=config.RANDOM_SEED,
            verbosity=0,
        )
        clf_xgb.fit(X_train, y_train)
        y_pred_xgb = clf_xgb.predict(X_test)
        elapsed = time.time() - t0

        acc  = accuracy_score(y_test, y_pred_xgb)
        f1   = f1_score(y_test, y_pred_xgb, average='macro', zero_division=0)
        prec = precision_score(y_test, y_pred_xgb, average='macro', zero_division=0)
        rec  = recall_score(y_test, y_pred_xgb, average='macro', zero_division=0)

        print(f"  Test accuracy: {acc*100:.2f}%  F1-macro: {f1:.4f}  ({elapsed:.2f}s)")
        print(classification_report(y_test, y_pred_xgb,
                                    target_names=config.CLASS_NAMES, zero_division=0))

        results['XGBoost'] = {
            'accuracy':        round(float(acc),  6),
            'f1_macro':        round(float(f1),   6),
            'precision_macro': round(float(prec), 6),
            'recall_macro':    round(float(rec),  6),
            'fit_time_s':      round(elapsed, 3),
            'hyperparams': 'n_estimators=300, max_depth=6, lr=0.1',
        }
    except ImportError:
        print("  XGBoost not installed — skipping. pip install xgboost")

    # ── LightGBM ──────────────────────────────────────────────────────────────
    try:
        import lightgbm as lgb
        print("\n── LightGBM ─────────────────────────────────────────────────")
        t0 = time.time()
        clf_lgb = lgb.LGBMClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            class_weight='balanced',
            random_state=config.RANDOM_SEED,
            verbose=-1,
        )
        clf_lgb.fit(X_train, y_train)
        y_pred_lgb = clf_lgb.predict(X_test)
        elapsed = time.time() - t0

        acc  = accuracy_score(y_test, y_pred_lgb)
        f1   = f1_score(y_test, y_pred_lgb, average='macro', zero_division=0)
        prec = precision_score(y_test, y_pred_lgb, average='macro', zero_division=0)
        rec  = recall_score(y_test, y_pred_lgb, average='macro', zero_division=0)

        print(f"  Test accuracy: {acc*100:.2f}%  F1-macro: {f1:.4f}  ({elapsed:.2f}s)")
        print(classification_report(y_test, y_pred_lgb,
                                    target_names=config.CLASS_NAMES, zero_division=0))

        results['LightGBM'] = {
            'accuracy':        round(float(acc),  6),
            'f1_macro':        round(float(f1),   6),
            'precision_macro': round(float(prec), 6),
            'recall_macro':    round(float(rec),  6),
            'fit_time_s':      round(elapsed, 3),
            'hyperparams': 'n_estimators=300, max_depth=6, lr=0.1, balanced',
        }
    except ImportError:
        print("  LightGBM not installed — skipping. pip install lightgbm")

    # ── Save ──────────────────────────────────────────────────────────────────
    out_json = OUT_DIR / 'gradient_boosted_results.json'
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {out_json}")

    # Print summary vs QSVC
    print("\n── Summary vs QSVC (94.62%) ─────────────────────────────────")
    for name, r in results.items():
        gap = 0.9462 - r['accuracy']
        print(f"  {name:<12} {r['accuracy']*100:.2f}%  F1={r['f1_macro']:.4f}"
              f"  (QSVC margin: {gap*100:+.2f} pp)")

    return results


if __name__ == '__main__':
    run()
