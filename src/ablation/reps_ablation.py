import sys
import json
import time
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
from qiskit.circuit.library import ZZFeatureMap
from qiskit.quantum_info import Statevector
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def compute_statevectors(X, feature_map):
    svs = []
    # Print progress manually to avoid tqdm dependency in headless script
    for i, x in enumerate(X):
        if i % 100 == 0:
            print(f"    Computed {i}/{len(X)} statevectors...")
        bound = feature_map.assign_parameters(x)
        svs.append(Statevector(bound).data)
    return np.array(svs)

def build_kernel(sv1, sv2):
    return np.abs(np.dot(sv1, sv2.conj().T)) ** 2

def main():
    print("Loading data...")
    d_feat = np.load('data/features_9d.npz')
    X_train_svd = d_feat['train_features']
    X_test_svd = d_feat['test_features']
    y_train = d_feat['y_train']
    y_test = d_feat['y_test']
    
    # Apply minmax_0pi scaling (best encoding)
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_train_scaled = scaler.fit_transform(X_train_svd) * np.pi
    X_test_scaled = scaler.transform(X_test_svd) * np.pi
    
    reps_to_test = [1, 2, 3, 4]
    results = {}
    
    for rep in reps_to_test:
        print(f"\n--- Testing reps={rep} ---")
        start_time = time.time()
        
        feature_map = ZZFeatureMap(feature_dimension=9, reps=rep, entanglement='circular')
        
        print("  Computing Train Statevectors...")
        sv_train = compute_statevectors(X_train_scaled, feature_map)
        
        print("  Computing Test Statevectors...")
        sv_test = compute_statevectors(X_test_scaled, feature_map)
        
        print("  Building Kernel Matrices...")
        K_train = build_kernel(sv_train, sv_train)
        K_test = build_kernel(sv_test, sv_train)
        
        print("  Training QSVC...")
        svc = SVC(kernel='precomputed', C=1.0)
        svc.fit(K_train, y_train)
        
        y_pred = svc.predict(K_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')
        
        elapsed = time.time() - start_time
        print(f"  Result: Acc={acc:.4f}, F1={f1:.4f} (Took {elapsed:.1f}s)")
        
        results[f"reps_{rep}"] = {
            "accuracy": float(acc),
            "f1_macro": float(f1),
            "time_s": float(elapsed)
        }
        
    Path("results").mkdir(exist_ok=True)
    with open("results/ablation_reps.json", "w") as f:
        json.dump(results, f, indent=2)
        
    # Plotting
    accs = [results[f"reps_{r}"]["accuracy"] * 100 for r in reps_to_test]
    plt.figure(figsize=(8, 5))
    plt.plot(reps_to_test, accs, marker='o', linewidth=2, color='#2c3e50')
    plt.title('QSVC Accuracy vs ZZFeatureMap Repetitions', fontsize=14)
    plt.xlabel('Number of Repetitions (Depth)', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.xticks(reps_to_test)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Highlight highest point
    best_rep = reps_to_test[np.argmax(accs)]
    plt.axvline(best_rep, color='#e74c3c', linestyle='--', alpha=0.5, label=f'Optimal: reps={best_rep}')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("results/ablation_reps.png", dpi=300)
    print("\nSaved results to results/ablation_reps.json and ablation_reps.png")

if __name__ == '__main__':
    main()
