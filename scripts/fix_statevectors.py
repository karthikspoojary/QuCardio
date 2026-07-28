import os
import sys
import numpy as np
from pathlib import Path
from qiskit.circuit.library import ZZFeatureMap
from qiskit.quantum_info import Statevector
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
import json

def compute_svs(X, feature_map, variant="minmax_01"):
    svs = []
    for x in tqdm(X):
        if variant == "minmax_0pi": x_scaled = x * np.pi
        elif variant == "l2_norm":
            from sklearn.preprocessing import normalize
            x_scaled = normalize(x.reshape(1, -1), norm="l2")[0]
        else: x_scaled = x
        bound = feature_map.assign_parameters(x_scaled)
        svs.append(Statevector(bound).data)
    return np.array(svs)

def main():
    print("Loading 9D features...")
    data = np.load(config.FEATURES_DIR / 'features_9d.npz')
    X_train = data['train_features']
    
    print("Computing Pegasos statevectors (linear)...")
    fm_pegasos = ZZFeatureMap(feature_dimension=9, reps=2, entanglement='linear')
    sv_train_pegasos = compute_svs(X_train, fm_pegasos, variant="minmax_01")
    np.savez_compressed(config.FEATURES_DIR / 'sv_train_pegasos.npz', sv_train=sv_train_pegasos)
    
    print("Computing QSVC statevectors...")
    meta_path = config.MODELS_DIR / 'qsvc_meta.json'
    if meta_path.exists():
        with open(meta_path) as f: meta = json.load(f)
        qsvc_variant = meta.get("feature_variant", "minmax_0pi")
        qsvc_reps = int(meta.get("reps", 2))
        qsvc_entangle = meta.get("entanglement", "circular")
    else:
        qsvc_variant = "minmax_0pi"; qsvc_reps = 2; qsvc_entangle = "circular"
        
    fm_qsvc = ZZFeatureMap(feature_dimension=9, reps=qsvc_reps, entanglement=qsvc_entangle)
    sv_train_qsvc = compute_svs(X_train, fm_qsvc, variant=qsvc_variant)
    np.savez_compressed(config.FEATURES_DIR / 'sv_train_qsvc.npz', sv_train=sv_train_qsvc)
    print("Done generating separate statevectors.")

if __name__ == "__main__":
    main()
