import sys
import numpy as np
import joblib
from pathlib import Path
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from qiskit.circuit.library import ZZFeatureMap
from qiskit.quantum_info import Statevector

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

def compute_sv(X, feature_map):
    svs = []
    for x in X:
        bound = feature_map.assign_parameters(x)
        svs.append(Statevector(bound).data)
    return np.array(svs)

d = np.load('data/features_9d.npz')
X_test = d['test_features']
y_test = d['y_test']

# Classical SVM
svm = joblib.load('backend/models/svm_model.pkl')
preds_svm = svm.predict(X_test)

# Quantum SVM Best
X_train_best = d['train_features'] * np.pi
X_test_best = X_test * np.pi
fm = ZZFeatureMap(9, reps=1, entanglement='full')
sv_train = compute_sv(X_train_best, fm)
sv_test = compute_sv(X_test_best, fm)
K_train = np.abs(np.dot(sv_train, sv_train.conj().T))**2
K_test = np.abs(np.dot(sv_test, sv_train.conj().T))**2

svc = SVC(kernel='precomputed', C=5.0, class_weight='balanced')
svc.fit(K_train, d['y_train'])
preds_qsvc = svc.predict(K_test)

# McNemar
b = np.sum((preds_qsvc == y_test) & (preds_svm != y_test))
c = np.sum((preds_qsvc != y_test) & (preds_svm == y_test))
chi2 = ((np.abs(b - c) - 1)**2) / (b + c)
from scipy.stats import chi2 as chi2_dist
p = chi2_dist.sf(chi2, 1)

print(f"QSVC Acc: {accuracy_score(y_test, preds_qsvc)*100:.2f}%")
print(f"SVM Acc: {accuracy_score(y_test, preds_svm)*100:.2f}%")
print(f"McNemar: chi2={chi2:.2f}, p={p:.4f}")
