import sys
import json
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

def compute_kta():
    print("Loading data...")
    # Load features and labels
    d_feat = np.load('data/features_9d.npz')
    y_train = d_feat['y_train']
    
    # Load precomputed training kernel matrix (QSVC minmax_0pi circular reps=2)
    d_K = np.load('data/K_train.npz')
    K = d_K['K_train']
    
    print(f"K shape: {K.shape}, y_train shape: {y_train.shape}")
    
    # For multiclass KTA, we define the target kernel matrix Y:
    # Y_ij = 1 if y_i == y_j else -1 / (num_classes - 1)
    # This centers the matrix appropriately.
    # Alternatively, Y_ij = 1 if y_i == y_j else 0 is also standard.
    # We will use the centered version Y_ij = 1 if match else -1/(C-1)
    
    num_classes = len(np.unique(y_train))
    N = len(y_train)
    Y = np.zeros((N, N))
    
    for i in range(N):
        for j in range(N):
            if y_train[i] == y_train[j]:
                Y[i, j] = 1.0
            else:
                Y[i, j] = -1.0 / (num_classes - 1)
                
    # Center both matrices in feature space (Kernel centering)
    # K_c = K - 1_N K - K 1_N + 1_N K 1_N
    # 1_N is matrix with all entries 1/N
    H = np.eye(N) - np.ones((N, N)) / N
    K_c = H @ K @ H
    Y_c = H @ Y @ H
    
    # Compute KTA
    numerator = np.trace(K_c.T @ Y_c)
    denominator = np.sqrt(np.trace(K_c.T @ K_c) * np.trace(Y_c.T @ Y_c))
    
    kta_score = numerator / denominator
    print(f"Kernel Target Alignment (KTA): {kta_score:.4f}")
    
    # Save result
    results = {
        "kta_score": float(kta_score),
        "kernel": "ZZFeatureMap",
        "scaling": "minmax_0pi",
        "reps": 2,
        "entanglement": "circular"
    }
    
    Path("results").mkdir(exist_ok=True)
    with open("results/kta_score.json", "w") as f:
        json.dump(results, f, indent=2)
        
    print("Saved to results/kta_score.json")

if __name__ == '__main__':
    compute_kta()
