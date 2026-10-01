import sys
import json
import time
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
from qiskit.circuit.library import ZZFeatureMap
from qiskit.quantum_info import Statevector

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

def compute_statevectors(X, feature_map):
    svs = []
    for i, x in enumerate(X):
        bound = feature_map.assign_parameters(x)
        svs.append(Statevector(bound).data)
    return np.array(svs)

def build_kernel(sv1, sv2):
    return np.abs(np.dot(sv1, sv2.conj().T)) ** 2

def main():
    print("="*60)
    print("UNBIASED EVALUATION: NESTED VALIDATION (HOLD-OUT TEST SET)")
    print("="*60)

    print("Loading data...")
    d_feat = np.load('data/features_9d.npz')
    X_train_svd = d_feat['train_features']
    X_test_svd = d_feat['test_features']
    y_train = d_feat['y_train']
    y_test = d_feat['y_test']
    
    print(f"Train samples: {len(X_train_svd)}")
    print(f"Test samples (Untouched): {len(X_test_svd)}")
    
    scalings = [1.0, np.pi]
    reps_list = [1, 2, 3]
    entanglements = ['linear', 'circular', 'full']
    C_values = [0.1, 1.0, 5.0, 10.0]
    
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    best_val_acc = 0
    best_params = None
    
    print("\nStarting 5-Fold Cross-Validation on Train Set for Model Selection...")
    
    for scaling in scalings:
        scaling_name = "0-1" if scaling == 1.0 else "0-pi"
        X_train_scaled = X_train_svd * scaling
        
        for rep in reps_list:
            for ent in entanglements:
                print(f"  Computing SVs for: range={scaling_name}, reps={rep}, ent={ent}...")
                start_time = time.time()
                fm = ZZFeatureMap(9, reps=rep, entanglement=ent)
                sv_train = compute_statevectors(X_train_scaled, fm)
                K_train_full = build_kernel(sv_train, sv_train)
                elapsed = time.time() - start_time
                print(f"    (Statevectors and Kernel computed in {elapsed:.1f}s)")
                
                for C in C_values:
                    fold_accs = []
                    for train_idx, val_idx in kf.split(X_train_svd, y_train):
                        # Slice precomputed kernel
                        K_fold_train = K_train_full[train_idx][:, train_idx]
                        K_fold_val = K_train_full[val_idx][:, train_idx]
                        
                        svc = SVC(kernel='precomputed', C=C, class_weight='balanced')
                        svc.fit(K_fold_train, y_train[train_idx])
                        preds = svc.predict(K_fold_val)
                        fold_accs.append(accuracy_score(y_train[val_idx], preds))
                    
                    avg_acc = np.mean(fold_accs)
                    if avg_acc > best_val_acc:
                        best_val_acc = avg_acc
                        best_params = (scaling, scaling_name, rep, ent, C)
                        print(f"    -> NEW BEST: Acc = {best_val_acc:.4f} (C={C})")

    best_scaling, best_scaling_name, best_rep, best_ent, best_C = best_params
    print("\n" + "="*60)
    print("HYPERPARAMETER SEARCH COMPLETE")
    print(f"Best Validation Accuracy: {best_val_acc:.4f}")
    print(f"Best Configuration:")
    print(f"  Encoding Range: {best_scaling_name}")
    print(f"  Repetitions:    {best_rep}")
    print(f"  Entanglement:   {best_ent}")
    print(f"  C value:        {best_C}")
    print("="*60)
    
    print("\nRetraining on full Train Set with best configuration...")
    X_train_best = X_train_svd * best_scaling
    X_test_best = X_test_svd * best_scaling
    
    fm_best = ZZFeatureMap(9, reps=best_rep, entanglement=best_ent)
    sv_train_best = compute_statevectors(X_train_best, fm_best)
    sv_test_best = compute_statevectors(X_test_best, fm_best)
    
    K_train_final = build_kernel(sv_train_best, sv_train_best)
    K_test_final = build_kernel(sv_test_best, sv_train_best)
    
    final_svc = SVC(kernel='precomputed', C=best_C, class_weight='balanced')
    final_svc.fit(K_train_final, y_train)
    
    final_preds = final_svc.predict(K_test_final)
    final_test_acc = accuracy_score(y_test, final_preds)
    final_test_f1 = f1_score(y_test, final_preds, average='macro')
    
    print("\n" + "="*60)
    print(f"UNBIASED FINAL TEST RESULTS (on {len(y_test)} untouched images)")
    print(f"Final Test Accuracy: {final_test_acc:.4f} ({final_test_acc*100:.2f}%)")
    print(f"Final Test Macro F1: {final_test_f1:.4f}")
    print("="*60)

if __name__ == "__main__":
    main()
