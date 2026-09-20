import re
import sys

filepath = 'Doc/Review/QuCardio_Report_v9.md'
try:
    with open(filepath, 'r') as f:
        content = f.read()
except FileNotFoundError:
    print("File not found")
    sys.exit(1)

# Chapter 2 headings
content = content.replace("## 2.2 QuCardio: Application of quantum machine learning for detection of cardiovascular diseases [1]", "## 2.2 Baseline Quantum ECG Classification [1]")
content = content.replace("## 2.3 Supervised learning with quantum-enhanced feature spaces [2]", "## 2.3 Quantum-Enhanced Feature Spaces and Theoretical Limits [2]")
content = content.replace("## 2.4 Quantum machine learning [3]", "## 2.4 Foundations of Quantum Machine Learning [3]")
content = content.replace("## 2.5 Pegasos: Primal estimated sub-gradient solver for SVM [4]", "## 2.5 The Multiclass Pegasos Optimization Algorithm [4]")
content = content.replace("## 2.6 Deep residual learning for image recognition [5]", "## 2.6 Deep Residual Networks for Spatial Feature Extraction [5]")
content = content.replace("## 2.7 A hybrid quantum-classical deep learning algorithm for efficient arrhythmia classification [6]", "## 2.7 Variational Quantum Circuits for Binary ECG Classification [6]")
content = content.replace("## 2.8 Quantum-assisted cardiac diseases diagnosis and prediction using ECG images [7]", "## 2.8 Quantum Kernels with PCA Dimensionality Reduction [7]")
content = content.replace("## 2.9 Performance evaluation of quantum-based machine learning algorithms for cardiac arrhythmia classification [8]", "## 2.9 Scaling Behaviour of Quantum SVMs [8]")
content = content.replace("## 2.10 Quantum-based convolutional neural network model for efficient cardiovascular disease prediction [9]", "## 2.10 Quantum Convolutional Neural Networks for Tabular Risk Prediction [9]")
content = content.replace("## 2.11 Arrhythmia detection using deep convolutional neural network with long duration ECG signals [10]", "## 2.11 1D Convolutional Neural Networks for ECG Signals [10]")
content = content.replace("## 2.12 Automatic classification of cardiac arrhythmias using deep learning techniques: A systematic review [11]", "## 2.12 Classical Deep Learning for Arrhythmia Detection [11]")
content = content.replace("## 2.13 Early cardiovascular disease detection using hierarchical quantum ensemble model [12]", "## 2.13 Hierarchical Quantum Ensembles for Cardiovascular Risk [12]")
content = content.replace("## 2.14 A systematic literature review of quantum machine learning for medical applications: Trends, datasets, topics and methods [13]", "## 2.14 Trends and Datasets in Medical Quantum Machine Learning [13]")
content = content.replace("## 2.15 Comparison of SVM, QSVM and Pegasos-QSVM algorithms on different medical datasets [14]", "## 2.15 Benchmarking Quantum SVMs on Tabular Medical Data [14]")
content = content.replace("## 2.16 New cardiovascular disease prediction approach using support vector machine and quantum-behaved particle swarm optimisation [15]", "## 2.16 Quantum-Behaved Particle Swarm Optimisation [15]")
content = content.replace("## 2.17 Quantum-enhanced machine learning algorithms for heart disease prediction [16]", "## 2.17 Feature Selection with Hybrid Quantum Algorithms [16]")

# IT-05 HTTP code
content = content.replace("HTTP 400 rejection", "HTTP 422 rejection")

# Chapter 8 and Random Forest conclusion
orig_chap8 = "QuCardio successfully demonstrates an end-to-end, hybrid classical-quantum machine learning system for classifying multi-class paper ECG images. The system introduces a robust three-stage Gating Layer to validate clinical inputs before applying a Classification Engine composed of a frozen ResNet50 feature extractor, SVD dimensionality reduction, and a 9-qubit ZZFeatureMap Quantum Support Vector Classifier. Evaluated on a 928-image dataset, the QSVC achieves 94.62% accuracy, significantly outperforming the classical SVM (p=0.0013) while matching state-of-the-art classical models like Random Forest. A factorial ablation confirms that $[0, \\pi]$ feature encoding combined with circular entanglement drives the quantum advantage. The entire pipeline is deployed as a React-FastAPI web application, providing automated, near-real-time diagnostic screening with calibrated confidence scores."
new_chap8 = "The proposed system successfully demonstrates an end-to-end, hybrid classical-quantum machine learning system for classifying multi-class paper ECG images. The system introduces a robust three-stage Gating Layer to validate clinical inputs before applying a Classification Engine composed of a frozen ResNet50 feature extractor, SVD dimensionality reduction, and a 9-qubit ZZFeatureMap Quantum Support Vector Classifier. Evaluated on a 928-image dataset, the QSVC achieves 94.62% accuracy, significantly outperforming the classical SVM (p=0.0013) while remaining numerically ahead of the Random Forest baseline by 2.68 points, a margin that is not statistically significant. A factorial ablation confirms that [0, π] feature encoding combined with circular entanglement drives the quantum advantage. The entire pipeline is deployed as a React-FastAPI web application, providing automated, near-real-time diagnostic screening with calibrated confidence scores."
content = content.replace(orig_chap8, new_chap8)

# Other QuCardio -> proposed system
content = content.replace("For QuCardio this is of particular importance", "For the proposed system this is of particular importance")

# Numerical error 37.7 -> 40.0
content = content.replace("The 37.7-point separation", "The 40.0-point separation")

# Missing table rows
orig_table = """| Ozpolat & Karabatak [8] | Arrhythmia (signal) | PCA + QSVM (3-9 qubits) | 80.90% vs 78.84% SVM | Signal-domain features only |
| Yildirim et al. [10] | 17-class arrhythmia | 1D CNN on raw signal | 91.33% | Requires digital signal file |
| Soon et al. [12] | CVD risk (tabular) | QNN + XGBoost + LightGBM | 97% accuracy, 98% AUC | Tabular only; high complexity |
| Setiawan et al. [13] | QML healthcare review | Review of 94 studies | Hybrid CNN-QSVM dominant | Review; no new experiments |
| Aksoy & Ozpolat [14] | Cancer and CVD (tabular) | SVM / QSVM / Pegasos-QSVM | QSVM most stable; [0, pi] | Tabular data; no ablation |
| **Proposed System**"""
new_table = """| Ozpolat & Karabatak [8] | Arrhythmia (signal) | PCA + QSVM (3-9 qubits) | 80.90% vs 78.84% SVM | Signal-domain features only |
| Gummadi et al. [9] | CVD risk (tabular) | CNN + Quantum Layers | Improved over classical CNN | Tabular data; no spatial processing |
| Yildirim et al. [10] | 17-class arrhythmia | 1D CNN on raw signal | 91.33% | Requires digital signal file |
| Vasquez-Iturralde et al. [11] | Arrhythmia classification | Systematic Deep Learning Review | Identified missing calibration | Review paper |
| Soon et al. [12] | CVD risk (tabular) | QNN + XGBoost + LightGBM | 97% accuracy, 98% AUC | Tabular only; high complexity |
| Setiawan et al. [13] | QML healthcare review | Review of 94 studies | Hybrid CNN-QSVM dominant | Review; no new experiments |
| Aksoy & Ozpolat [14] | Cancer and CVD (tabular) | SVM / QSVM / Pegasos-QSVM | QSVM most stable; [0, pi] | Tabular data; no ablation |
| Elsedimy et al. [15] | CVD prediction (tabular) | QPSO + SVM | 96.31% | Classical optimiser; not quantum |
| Alotaibi et al. [16] | Heart disease (tabular) | QPSO + QML | 96.7% | Tabular data; no spatial pipeline |
| **Proposed System**"""
content = content.replace(orig_table, new_table)
content = content.replace("| **Ideal simulation; single-source dataset** |", "| **Ideal simulation; single-source dataset** |\n\nNote: References [4] (Pegasos optimization algorithm) and [5] (ResNet50 architecture) are foundational methods incorporated into the proposed system, rather than competing pipelines, and are therefore excluded from this comparative summary.")

# Section 5.8 11 -> 12
content = content.replace("loads eleven serialised objects", "loads twelve serialised objects")
content = content.replace("and the MobileNetV2 gatekeeper", "the MobileNetV2 gatekeeper, and the Mahalanobis reference mean/covariance matrix")

# Kernel equation
orig_eq = "K[i, j] = |sv[i] · conj(sv[j])|²"
new_eq = "K[i, j] = |⟨ψ_i|ψ_j⟩|² = |Σ_k conj(sv_i[k]) sv_j[k]|²"
content = content.replace(orig_eq, new_eq)

# Table renaming and references
table_map = {
    "Table 7.7": "Table 7.9",
    "Table 7.6": "Table 7.8",
    "Table 7.5": "Table 7.7",
    "Table 7.4": "Table 7.6",
    "Table 7.3B": "Table 7.5",
    "Table 7.3": "Table 7.4",
    "Table 7.2": "Table 7.3",
    "Table 7.1-B": "Table 7.2",
}

for old, new in table_map.items():
    content = content.replace(old, new)

# Math and symbol standardization
content = content.replace("2x2", "2 × 2")
content = content.replace("2 \\times 2", "2 × 2")
content = content.replace("[0, pi]", "[0, π]")
content = content.replace("2pi", "2π")
content = content.replace("1/2^9", "1/2⁹")
content = content.replace("2^9", "2⁹")
content = content.replace("chi-squared", "χ²")
content = content.replace("World Health Organisation", "World Health Organization")
content = content.replace("OTSU", "Otsu")

with open(filepath, 'w') as f:
    f.write(content)

print("done")
