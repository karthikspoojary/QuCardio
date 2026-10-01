"""Check Qiskit ZZFeatureMap default entanglement"""
from qiskit.circuit.library import ZZFeatureMap
fm_default = ZZFeatureMap(feature_dimension=4)
print("ZZFeatureMap default entanglement:", fm_default.entanglement)
print("ZZFeatureMap(reps=2) entanglement:", ZZFeatureMap(4, reps=2).entanglement)
print("ZZFeatureMap(4, reps=2, entanglement='full') entanglement:", ZZFeatureMap(4, reps=2, entanglement='full').entanglement)
# Also confirm full == default
fm9 = ZZFeatureMap(feature_dimension=9)
print("\n9-qubit ZZFeatureMap default entanglement:", fm9.entanglement)
