import json
import numpy as np
from pathlib import Path

def compute_95ci(p, n):
    # p = proportion (accuracy), n = sample size
    # 95% CI using Normal Approximation (Wald interval)
    z = 1.96
    se = np.sqrt(p * (1 - p) / n)
    margin = z * se
    return {
        "accuracy": float(p),
        "n": int(n),
        "ci_lower": float(p - margin),
        "ci_upper": float(p + margin),
        "margin": float(margin),
        "formatted": f"{p*100:.2f}% ± {margin*100:.2f}% [{((p-margin)*100):.2f}% - {((p+margin)*100):.2f}%]"
    }

def main():
    results = {}
    
    # Baseline Test set size
    n_test = 186
    
    # Key Models
    models = {
        "QSVC (minmax_0pi, circular, reps=2)": 0.9462,
        "SVM (RBF)": 0.8978,
        "Random Forest (500 trees)": 0.9194,
        "KNN": 0.8763
    }
    
    for name, acc in models.items():
        results[name] = compute_95ci(acc, n_test)
        
    Path("results").mkdir(exist_ok=True)
    with open("results/confidence_intervals.json", "w") as f:
        json.dump(results, f, indent=2)
        
    for name, r in results.items():
        print(f"{name}: {r['formatted']}")
        
if __name__ == '__main__':
    main()
