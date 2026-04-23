import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import joblib

def train_classical_svm():
    """
    Classical SVM — Generalization-Focused Tuning

    QuCardio Paper (Table 6):
    - Classical SVM: 83.33% accuracy
    - Our goal: Be robust on outside images, not just the test split
    """
    if not Path('data/features_9d.npz').exists():
        print("data/features_9d.npz not found! Run dimensionality reduction first.")
        return

    print("Loading 9D features...")
    data = np.load('data/features_9d.npz')

    X_train = data['train_features']
    X_test  = data['test_features']
    y_train = data['y_train']
    y_test  = data['y_test']
    class_names = data['class_names']

    print(f"Train: {X_train.shape}")
    print(f"Test:  {X_test.shape}")

    # ── Grid Search — Generalization-Focused ─────────────────────────────────
    # KEY CHANGES vs the old 91% model:
    #
    # 1. C values capped at 10 (was 500):
    #    High C → model memorizes training quirks → fails on outside images.
    #    Low  C → allows some errors → smoother decision boundary → generalizes.
    #
    # 2. class_weight='balanced':
    #    Dataset is imbalanced (284 Normal vs 172 History_of_MI).
    #    Without this, SVM is biased toward predicting the majority class.
    #    Balanced weights give rare classes equal importance.
    #
    # 3. scoring='f1_macro' (was default accuracy):
    #    Accuracy rewards predicting Normal (majority class) more.
    #    F1-macro weighs ALL 4 classes equally — better for medical classification.
    # ─────────────────────────────────────────────────────────────────────────
    print("\n🔍 Running Grid Search (Generalization-Focused)...")
    print("Lower C values + balanced class weights + F1-macro scoring\n")

    param_grid = {
        'C':            [0.01, 0.1, 1, 5, 10],
        'gamma':        ['scale', 'auto', 0.1, 0.01, 0.001],
        'kernel':       ['rbf'],
        'class_weight': ['balanced'],
    }

    from sklearn.model_selection import StratifiedKFold
    from sklearn.calibration import CalibratedClassifierCV

    # StratifiedKFold: every CV fold contains all 4 classes (important for small datasets)
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    base_svm = SVC(probability=False, random_state=42)

    grid_search = GridSearchCV(
        base_svm,
        param_grid,
        cv=cv_strategy,
        scoring='f1_macro',
        verbose=1,
        n_jobs=-1
    )

    import time
    start = time.time()
    grid_search.fit(X_train, y_train)
    train_time = time.time() - start

    best_params = grid_search.best_params_
    print(f"\n✅ Grid Search Complete in {train_time:.1f} seconds")
    print(f"⭐ BEST PARAMETERS FOUND: {best_params}")

    # ── Probability Calibration ───────────────────────────────────────────────
    # SVC with probability=True uses Platt scaling which is often over-confident.
    # CalibratedClassifierCV with isotonic regression gives honest probability
    # scores — crucial for the dashboard's confidence bar to be meaningful.
    print("\n📊 Fitting CalibratedClassifierCV for honest confidence scores...")
    best_base = SVC(
        **{k: v for k, v in best_params.items()},
        random_state=42,
        probability=False
    )
    svm = CalibratedClassifierCV(best_base, cv=3, method='isotonic')
    svm.fit(X_train, y_train)

    # ── Metrics ──────────────────────────────────────────────────────────────
    y_pred_train = svm.predict(X_train)
    y_pred_test  = svm.predict(X_test)

    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc  = accuracy_score(y_test, y_pred_test)
    precision = precision_score(y_test, y_pred_test, average='macro', zero_division=0)
    recall    = recall_score(y_test, y_pred_test, average='macro', zero_division=0)
    f1        = f1_score(y_test, y_pred_test, average='macro', zero_division=0)
    gap       = train_acc - test_acc

    print(f"\n{'='*55}")
    print(f"CLASSICAL SVM RESULTS (Generalization-Tuned)")
    print(f"{'='*55}")
    print(f"Train Accuracy:         {train_acc:.4f} ({train_acc*100:.2f}%)")
    print(f"Test  Accuracy:         {test_acc:.4f}  ({test_acc*100:.2f}%)")
    print(f"Precision (macro):      {precision:.4f} ({precision*100:.2f}%)")
    print(f"Recall    (macro):      {recall:.4f}    ({recall*100:.2f}%)")
    print(f"F1-Score  (macro):      {f1:.4f}        ({f1*100:.2f}%)")
    print(f"Generalization gap:     {gap*100:.1f}%  (train - test)")
    print(f"{'='*55}")
    print(f"\nQuCardio Paper baseline: 83.33%")
    print(f"Your Result:             {test_acc*100:.2f}%")

    if gap < 0.10:
        print("✅ Low gap → model generalizes well to unseen/outside images.")
    elif gap < 0.20:
        print("⚠️  Moderate gap → acceptable generalization.")
    else:
        print("❌ Large gap → consider reducing C further.")

    if test_acc >= 0.83:
        print("✅ Paper baseline matched or exceeded!")
    else:
        print("⚠️  Below paper baseline — but more robust on outside images.")

    # ── Save ─────────────────────────────────────────────────────────────────
    Path('results').mkdir(exist_ok=True)
    Path('backend/models').mkdir(parents=True, exist_ok=True)

    print("\nSaving SVM model for backend usage...")
    joblib.dump(svm, 'backend/models/svm_model.pkl')

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred_test)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title(
        f'Classical SVM — Generalization-Tuned\n'
        f'Test Acc: {test_acc*100:.1f}%  |  F1-macro: {f1*100:.1f}%  |  Gap: {gap*100:.1f}%'
    )
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('results/classical_svm_cm.png', dpi=150, bbox_inches='tight')
    print("Confusion matrix saved: results/classical_svm_cm.png")

    results_dict = {
        'model': 'Classical SVM (Generalization-Tuned)',
        'best_params': best_params,
        'accuracy': float(test_acc),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'generalization_gap': float(gap),
        'train_time_seconds': float(train_time)
    }

    with open('results/classical_svm_results.json', 'w') as f:
        json.dump(results_dict, f, indent=2)

    print("Results saved: results/classical_svm_results.json")

    return svm, results_dict

if __name__ == "__main__":
    import time
    start = time.time()
    train_classical_svm()
    print(f"\n⏱️ Total time: {(time.time() - start)/60:.1f} minutes")