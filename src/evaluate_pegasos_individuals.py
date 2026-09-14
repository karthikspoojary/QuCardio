import sys
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config

def build_kernel(sv1, sv2):
    return (np.abs(np.dot(sv1, sv2.conj().T)) ** 2).astype(np.float64)

def main():
    print("Loading test and train data...")
    data = np.load(config.FEATURES_DIR / "features_9d.npz")
    y_test = data["y_test"]
    
    sv_tr_path = config.FEATURES_DIR / "sv_train_pegasos.npz"
    sv_te_path = config.FEATURES_DIR / "sv_test_pegasos.npz"
    
    sv_tr = np.load(sv_tr_path, allow_pickle=True)["sv_train"]
    sv_te = np.load(sv_te_path, allow_pickle=True)["sv_test"]
    
    print("Building full kernel matrices...")
    K_te_full = build_kernel(sv_te, sv_tr)
    
    pairs = [
        (1, 2, "AH", "MI"),
        (0, 3, "NP", "H. MI"),
        (2, 3, "MI", "H. MI"),
        (0, 2, "NP", "MI"),
        (1, 3, "AH", "H. MI"),
        (0, 1, "NP", "AH")
    ]
    
    fig, axes = plt.subplots(3, 2, figsize=(12, 16))
    axes = axes.flatten()
    
    for idx, (c1, c2, name1, name2) in enumerate(pairs):
        # We enforce c1 < c2 for the filename because that's how we saved them
        mc1, mc2 = min(c1, c2), max(c1, c2)
        model_path = config.MODELS_DIR / f"pegasos_{mc1}_{mc2}.pkl"
        
        if not model_path.exists():
            print(f"Model {model_path} not found!")
            continue
            
        model = joblib.load(model_path)
        
        # Filter test data for ONLY the two classes this model was trained on
        mask = (y_test == c1) | (y_test == c2)
        y_test_subset = y_test[mask]
        K_test_subset = K_te_full[mask][:, model.train_indices_]
        
        y_pred = model.predict(K_test_subset)
        
        # The Pegasos predict method returns -1 for mc1 and +1 for mc2.
        # Let's map -1 -> mc1, +1 -> mc2
        y_pred_mapped = np.where(y_pred == -1, mc1, mc2)
        
        cm = confusion_matrix(y_test_subset, y_pred_mapped, labels=[c1, c2])
        
        # Plot
        ax = axes[idx]
        sns.heatmap(cm, annot=True, fmt="d", cmap="rocket", 
                    xticklabels=[name1, name2], yticklabels=[name1, name2],
                    ax=ax, cbar=True)
        ax.set_title(f"Pegasos model ({name1} vs {name2})", fontweight="bold")
        ax.set_ylabel("True Label")
        ax.set_xlabel("Predicted Label")
        
    plt.tight_layout()
    out_path = config.RESULTS_DIR / "pegasos_individual_cms.png"
    plt.savefig(out_path, dpi=150)
    print(f"Saved figure to {out_path}")

if __name__ == "__main__":
    main()
