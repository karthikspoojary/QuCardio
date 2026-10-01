"""
Verify whether sv_train_pegasos.npz was computed with [0,pi] or [0,1] features
by re-computing a small batch and comparing.
"""
import numpy as np
from qiskit.circuit.library import ZZFeatureMap
from qiskit.quantum_info import Statevector

d = np.load('data/features_9d.npz')
X_train = d['train_features']  # [0,1]

fm_linear = ZZFeatureMap(feature_dimension=9, reps=2, entanglement='linear')

# Compute first 3 statevectors with [0,1] encoding
svs_01 = []
for x in X_train[:3]:
    bound = fm_linear.assign_parameters(x)
    svs_01.append(Statevector(bound).data)

# Compute first 3 statevectors with [0,pi] encoding
svs_0pi = []
for x in X_train[:3] * np.pi:
    bound = fm_linear.assign_parameters(x)
    svs_0pi.append(Statevector(bound).data)

# Load cached
cached = np.load('data/sv_train_pegasos.npz', allow_pickle=True)['sv_train']

for i in range(3):
    diff_01  = np.max(np.abs(svs_01[i]  - cached[i]))
    diff_0pi = np.max(np.abs(svs_0pi[i] - cached[i]))
    print(f"Sample {i}: diff_[0,1]={diff_01:.6f}  diff_[0,pi]={diff_0pi:.6f}  "
          f"→ cached matches {'[0,1]' if diff_01 < diff_0pi else '[0,pi]'}")
