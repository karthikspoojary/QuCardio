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
    Classical SVM baseline with Grid Search Tuning
    
    QuCardio Paper (Table 6):
    - Classical SVM: 83.33% accuracy
    - Our goal: Match or exceed this by finding the optimal C and gamma
    """
    if not Path('data/features_9d.npz').exists():
        print("data/features_9d.npz not found! Run dimensionality reduction first.")
        return
        
    print("Loading 9D features...")
    data = np.load('data/features_9d.npz')
    
    X_train = data['train_features']
    X_test = data['test_features']
    y_train = data['y_train']
    y_test = data['y_test']
    class_names = data['class_names']

    print(f"Train: {X_train.shape}")
    print(f"Test:  {X_test.shape}")
    
    # Train classical SVM using Grid Search
    print("\n🔍 Running Grid Search to find the best SVM parameters...")
    print("This will test dozens of combinations to maximize accuracy...")
    
    param_grid = {
        'C': [0.1, 1, 10, 50, 100, 500],
        'gamma': [1, 0.1, 0.01, 0.001, 'scale', 'auto'],
        'kernel': ['rbf']
    }
    
    base_svm = SVC(probability=True, random_state=42)
    grid_search = GridSearchCV(base_svm, param_grid, cv=5, verbose=1, n_jobs=-1)
    
    import time
    start = time.time()
    
    # This automatically finds the best settings AND trains the final model
    grid_search.fit(X_train, y_train)
    train_time = time.time() - start
    
    svm = grid_search.best_estimator_
    
    print(f"\n✅ Grid Search Complete in {train_time:.1f} seconds")
    print(f"⭐ BEST PARAMETERS FOUND: {grid_search.best_params_}")
    
    # Predict using the best model
    y_pred_train = svm.predict(X_train)
    y_pred_test = svm.predict(X_test)
    
    # Metrics
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)
    precision = precision_score(y_test, y_pred_test, average='macro', zero_division=0)
    recall = recall_score(y_test, y_pred_test, average='macro', zero_division=0)
    f1 = f1_score(y_test, y_pred_test, average='macro', zero_division=0)
    
    print(f"\n{'='*50}")
    print(f"CLASSICAL SVM RESULTS (TUNED)")
    print(f"{'='*50}")
    print(f"Train Accuracy:  {train_acc:.4f} ({train_acc*100:.2f}%)")
    print(f"Test Accuracy:   {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"Precision:       {precision:.4f} ({precision*100:.2f}%)")
    print(f"Recall:          {recall:.4f} ({recall*100:.2f}%)")
    print(f"F1-Score:        {f1:.4f} ({f1*100:.2f}%)")
    print(f"{'='*50}")
    print(f"\nQuCardio Paper: 83.33%")
    print(f"Your Result:    {test_acc*100:.2f}%")
    
    if test_acc >= 0.83:
        print(f"✅ Baseline matched or exceeded!")
    else:
        print(f"⚠️ Still below baseline, but likely highly improved over defaults.")
    
    # Setup directories
    Path('results').mkdir(exist_ok=True)
    Path('backend/models').mkdir(parents=True, exist_ok=True)
    
    # Save the model for the backend
    print("\nSaving SVM model for backend usage...")
    joblib.dump(svm, 'backend/models/svm_model.pkl')
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred_test)
    
    plt.figure(figsize=(10, 8)) 
    
    # Added xticklabels and yticklabels 
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, 
                yticklabels=class_names)
    
    plt.title(f'Tuned Classical SVM Confusion Matrix\nAccuracy: {test_acc*100:.1f}%')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45) 
    plt.tight_layout()      
    
    plt.savefig('results/classical_svm_cm.png', dpi=150, bbox_inches='tight')
    print(f"\nConfusion matrix saved: results/classical_svm_cm.png")
    
    # Save results
    results_dict = {
        'model': 'Classical SVM (Tuned)',
        'best_params': grid_search.best_params_,
        'accuracy': float(test_acc),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'train_time_seconds': float(train_time)
    }
    
    with open('results/classical_svm_results.json', 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    print(f"Results saved: results/classical_svm_results.json")
    
    return svm, results_dict

if __name__ == "__main__":
    import time
    start = time.time()
    train_classical_svm()
    print(f"\n⏱️ Total time: {(time.time() - start)/60:.1f} minutes")