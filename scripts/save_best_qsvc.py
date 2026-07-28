"""
Save best QSVC config: minmax_0pi, reps=2, circular, C=5.0
Updates sv_train.npz, sv_test.npz, qsvc_meta.json, qsvc_results.json
"""
import sys, numpy as np, joblib, json
from pathlib import Path
from sklearn.preprocessing import normalize
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from qiskit.circuit.library import ZZFeatureMap
from qiskit.quantum_info import Statevector

BEST = {"feature_variant": "minmax_0pi", "reps": 2,
        "entanglement": "circular",  "C": 5.0, "accuracy": 0.9462365591397849}

d      = np.load(config.FEATURES_DIR / "features_9d.npz")
X_tr   = d["train_features"] * np.pi    # minmax_0pi
X_te   = d["test_features"]  * np.pi

cache  = config.FEATURES_DIR / "sv_cache"
cache.mkdir(exist_ok=True)

fm = ZZFeatureMap(feature_dimension=config.FEATURE_DIMENSION,
                  reps=BEST["reps"], entanglement=BEST["entanglement"])

for split, X, key, outfile in [
    ("train", X_tr, "sv_train", config.FEATURES_DIR / "sv_train.npz"),
    ("test",  X_te, "sv_test",  config.FEATURES_DIR / "sv_test.npz"),
]:
    tag  = f"sv_{BEST['feature_variant']}_reps{BEST['reps']}_{BEST['entanglement']}_{split}"
    path = cache / f"{tag}.npz"
    if path.exists():
        print(f"Loading {split} statevectors from cache …")
        sv = np.load(path, allow_pickle=True)["sv"]
    else:
        print(f"Computing {split} statevectors …")
        sv = np.array([Statevector(fm.assign_parameters(x)).data for x in X])
        np.savez_compressed(path, sv=sv)

    np.savez_compressed(outfile, **{key: sv})
    print(f"  {outfile.name} updated  shape={sv.shape}")

# Save meta
with open(config.MODELS_DIR / "qsvc_meta.json", "w") as f:
    json.dump(BEST, f, indent=2)
print("qsvc_meta.json saved")

# Update results JSON
with open(config.RESULTS_DIR / "qsvc_results.json", "w") as f:
    json.dump({
        "model":             "QSVC (Tuned — ZZFeatureMap, Statevector kernel)",
        "accuracy":          BEST["accuracy"],
        "feature_variant":   BEST["feature_variant"],
        "feature_dimension": config.FEATURE_DIMENSION,
        "reps":              BEST["reps"],
        "entanglement":      BEST["entanglement"],
        "C":                 BEST["C"],
        "class_weight":      "None",
        "paper_accuracy":    0.9409,
        "improvement_vs_paper": round((BEST["accuracy"] - 0.9409) * 100, 3),
    }, f, indent=2)
print("qsvc_results.json updated")
print(f"\n✅ All done — QSVC accuracy: {BEST['accuracy']*100:.2f}%  (paper: 94.09%)")
