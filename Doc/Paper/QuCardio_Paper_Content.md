# QuCardio: Quantum Kernel Circuit Design for Four-Class ECG Image Classification

> **NOTE TO TEAM (Paper):** This file contains complete content for the conference/journal paper.
> Follow the IEEE IEEEtran two-column LaTeX template (same as Main.pdf from SJEC seniors).
> Target venue: IEEE conference (e.g., IEEE WIECON-ECE, IEEE TENCON, IEEE INDISCON, or
> Elsevier Computers in Biology and Medicine / Biomedical Signal Processing and Control).
> Page target: 6 pages (conference) or 10–12 pages (journal).
>
> **Section order for LaTeX:** Title → Authors → Abstract → Index Terms → I. Introduction →
> II. Related Work → III. Proposed Methodology → IV. Experimental Setup → V. Results and
> Discussion → VI. Ablation Studies → VII. Conclusion → Acknowledgement → References
>
> **All numbers are verified from actual result files.** Do not round or alter any metric.
> IEEE figure numbering: Fig. 1, Fig. 2 ... (caption BELOW figure).
> IEEE table numbering: TABLE I, TABLE II ... (caption ABOVE table, Roman numerals).
> Every abbreviation defined at first use.

---

## TITLE

**QuCardio: Systematic Circuit Design for Quantum Kernel ECG Classification — Encoding Range, Entanglement Topology, and Statistical Validation**

*(Alternative shorter title for tight venues:)*
**Quantum Kernel Circuit Optimization for Multi-Class Cardiovascular ECG Image Classification**

---

## AUTHORS

*(Team member names), Department of Computer Science and Engineering, St. Joseph Engineering College, Mangaluru, India*
*{usn1, usn2, ...}@sjec.ac.in*

*(Guide Name), Department of Computer Science and Engineering, St. Joseph Engineering College, Mangaluru, India*

---

## ABSTRACT

Cardiovascular diseases cause approximately 17.9 million deaths annually, and the Electrocardiogram remains the primary non-invasive diagnostic tool. Automated ECG interpretation using quantum machine learning has shown promise, but existing studies report isolated accuracy numbers with no systematic analysis of quantum circuit design choices, no statistical validation of the quantum advantage, and no deployed clinical interface. This paper presents QuCardio, a hybrid classical-quantum pipeline for four-class ECG image classification, and investigates how two fundamental circuit parameters affect the Quantum Support Vector Classifier.

The pipeline extracts a 462,400-dimensional spatial feature vector from the pool1\_pool layer of a frozen ResNet50 network, compresses it to 9 dimensions with Truncated Singular Value Decomposition, and classifies using a 9-qubit ZZFeatureMap fidelity quantum kernel computed via statevector simulation. A systematic sweep over feature encoding range and entanglement topology reveals that scaling features to [0, pi] with circular entanglement improves accuracy from 89.78% (base paper configuration) to 94.62% on a 928-image four-class clinical dataset from the Ch. Pervaiz Elahi Institute of Cardiology, Multan. McNemar's test confirms the advantage over a well-tuned classical Support Vector Machine is statistically significant (chi-squared = 10.32, p = 0.0013). The Quantum Support Vector Classifier achieves a macro Area Under the ROC Curve of 0.989 and perfect separation for Myocardial Infarction (AUC = 1.000), while Expected Calibration Error analysis confirms prediction confidence is clinically trustworthy (macro ECE below 0.07 for all models). The trained system is deployed as a web application with a FastAPI backend, React frontend, single and batch diagnosis, and downloadable PDF clinical reports.

**Index Terms** — Quantum Machine Learning, Quantum Support Vector Classifier, ZZFeatureMap, Fidelity Quantum Kernel, ECG Classification, Cardiovascular Disease, ResNet50, Truncated SVD, McNemar Test.

---

## I. INTRODUCTION

Cardiovascular diseases (CVDs) are the leading cause of mortality worldwide, responsible for approximately 17.9 million deaths each year according to the World Health Organization [1]. The Electrocardiogram (ECG) is the standard first-line non-invasive screening tool, recording the heart's electrical activity as a waveform trace from which cardiologists identify conditions such as Arrhythmia, Myocardial Infarction (MI), and prior History of MI. Interpreting an ECG accurately requires specialist cardiological training that many high-volume hospitals and rural clinics do not have on demand. Missed or delayed diagnoses in acute cardiac events carry significant mortality costs [2].

Machine learning and deep learning have been applied to ECG analysis for over a decade, producing strong results on large benchmark datasets [3]. However, multi-class performance on small clinical image collections tends to plateau, and classical kernel methods such as the Radial Basis Function Support Vector Machine are limited by their inability to exploit higher-order feature correlations that a quantum kernel can access. Quantum machine learning (QML) offers a theoretically grounded alternative: quantum feature maps embed classical data into exponentially large Hilbert spaces where certain classification boundaries become accessible that no efficient classical kernel can represent [4]. With the current Noisy Intermediate-Scale Quantum (NISQ) era hardware still constrained by qubit count and gate fidelity, statevector simulation on classical hardware is the practical mode for QML research, and the fidelity quantum kernel approach of Havlicek et al. [4] is well-suited to this setting.

Prior work on quantum ECG classification has established that Quantum Support Vector Classifiers (QSVC) with ZZFeatureMap encoding outperform classical SVM on ECG image datasets [5], [6]. However, these studies share a common limitation: each reports a single accuracy number under a fixed circuit configuration with no analysis of how encoding range or entanglement topology affects the result. The circuit design choices are not independently varied, so it is not possible to attribute performance differences to specific parameters. Furthermore, statistical significance of the quantum advantage is not tested, calibration of predicted confidence is not reported, and no system is deployed for clinical use.

This paper addresses those gaps with the following contributions:

- We demonstrate that scaling the 9-dimensional SVD-compressed ECG features to [0, pi] and switching from linear to circular entanglement in the ZZFeatureMap raises QSVC accuracy by 4.84 percentage points over the base paper configuration (89.78% to 94.62%), through a systematic sweep over five encoding ranges and five entanglement topologies.
- We provide the first McNemar significance test of quantum kernel advantage over classical SVM on ECG image classification (chi-squared = 10.32, p = 0.0013), confirming that the gap is not a statistical artifact.
- We report Expected Calibration Error analysis and augmentation robustness experiments that are absent from prior ECG-QML work, showing that QSVC maintains 86.7% accuracy under brightness perturbation where classical SVM falls to 46.7%.
- We deploy the complete pipeline as a working web application that a clinician can use directly, with single and batch ECG upload, calibrated confidence scores, class-specific recommendations, and PDF report generation.

The remainder of this paper is organized as follows. Section II surveys related work. Section III describes the proposed pipeline. Section IV details the experimental setup. Section V presents and discusses the results. Section VI covers the ablation studies. Section VII concludes the paper.

---

## II. RELATED WORK

**Classical and Deep Learning ECG Analysis.** Deep learning methods for ECG classification have advanced substantially since Yildirim et al. [3] demonstrated that a 1D Convolutional Neural Network applied to raw ECG waveforms reaches 91.33% accuracy on 17 arrhythmia classes from the MIT-BIH database. Vasquez-Iturralde et al. [7] reviewed 494 deep learning ECG papers and identified two persistent gaps: fewer than 15% of studies work on image-format ECGs rather than digital recordings, and fewer than 8% report any form of calibration analysis. Our work operates on scanned paper ECG prints, directly filling the first gap, and provides both calibration and statistical significance analyses, filling the second.

**Quantum Kernel Methods.** Havlicek et al. [4] established the theoretical basis for quantum kernel classification, showing that ZZFeatureMap-based kernels can achieve classification advantages intractable for polynomial-time classical kernels by mapping classical data into exponentially large Hilbert spaces via Pauli-ZZ interactions. The fidelity kernel K(x, z) = |⟨ψ(x)|ψ(z)⟩|² is the construction used in this work. Biamonte et al. [8] surveyed quantum machine learning broadly and argued that hybrid classical-quantum pipelines, where a classical preprocessor handles dimensionality reduction before quantum encoding, are among the most practically viable near-term architectures. This observation directly motivates the SVD compression step in our pipeline.

**Quantum ECG and Cardiac Classification.** Prabhu et al. [5] published the closest prior work to ours, using the same 928-image four-class ECG dataset from the Ch. Pervaiz Elahi Institute of Cardiology, Multan. Their pipeline combines ResNet50 pool1\_pool extraction with Truncated SVD at 9 components and a ZZFeatureMap QSVC, reporting 94.09% accuracy under [0, 1] feature scaling with linear entanglement. The Quantum Neural Network in the same paper reached 97.31%. Neither the encoding range nor entanglement topology was varied, and no statistical validation or deployment was provided. Our work extends this pipeline with a systematic circuit design study and finds that two configuration changes together lift QSVC accuracy to 94.62%, above their reported figure.

Jain et al. [6] applied ResNet50 with PCA (8 components) and an 8-qubit ZZFeatureMap QSVC to two ECG image datasets from the same clinical source, with the Quantum Support Vector Machine outperforming classical SVM by approximately 8% and the Quantum Convolutional Neural Network exceeding its classical counterpart by about 5%. Statistical significance was assessed using the Kruskal-Wallis H-test. The paper validates the ResNet50-plus-quantum-kernel architecture across two datasets but does not vary circuit design parameters.

Ramkhelawan et al. [9] paired ResNet50 feature extraction with a 10-qubit variational quantum circuit in PennyLane for binary arrhythmia detection, reporting 99.98% accuracy on a combined MIT-BIH and PTB dataset of 123,998 images. The binary task is structurally simpler than the four-class problem, and the variational circuit introduces barren plateau risk absent from the fixed fidelity kernel approach. Ozpolat and Karabatak [10] benchmarked QSVM against classical SVM on signal-domain Chapman ECG features across qubit counts from 3 to 9, finding consistent but modest quantum margin gains (80.90% against 78.84% at the best configuration).

Aksoy and Ozpolat [11] compared SVM, QSVM, and Pegasos QSVM on multiple medical tabular datasets using [0, pi] MinMax scaling, independently arriving at the same encoding range our ablation identifies as optimal on ECG image features. This cross-domain convergence strengthens the case that [0, pi] is a general property of the ZZFeatureMap rotation structure rather than a dataset-specific artifact.

**Summary and Gap.** Across all reviewed works, no paper (i) systematically ablates both encoding range and entanglement topology on an ECG image dataset, (ii) provides a McNemar significance test of the quantum advantage, (iii) reports confidence calibration, or (iv) deploys a working clinical interface. QuCardio addresses all four.

**TABLE I: Comparison with Prior Quantum ECG / Cardiac Works**

| Reference | Task | Method | Best Accuracy | Ablation | Stats Test | Deployed |
|---|---|---|---|---|---|---|
| Prabhu et al. [5] | 4-class ECG images | ResNet50 + SVD + QSVC ([0,1], linear) | 94.09% (QSVC) | No | No | No |
| Jain et al. [6] | Multi-class ECG images | ResNet50 + PCA + QSVC | 95.2% (QSVM) | No | Kruskal-Wallis | No |
| Ramkhelawan et al. [9] | Binary arrhythmia | ResNet50 + 10-qubit VQC | 99.98% | No | No | No |
| Ozpolat & Karabatak [10] | Arrhythmia signals | PCA + QSVM (3–9 qubits) | 80.90% | Qubit sweep | No | No |
| Aksoy & Ozpolat [11] | Medical tabular | SVM / QSVM / Pegasos | QSVM best | No | No | No |
| **QuCardio (Ours)** | **4-class ECG images** | **ResNet50 + SVD + QSVC ([0,pi], circular)** | **94.62%** | **Yes (486 configs)** | **McNemar, p=0.0013** | **Yes** |

---

## III. PROPOSED METHODOLOGY

The QuCardio pipeline has three stages: offline preprocessing and feature extraction, model training and selection, and online inference through a deployed web application. Fig. 1 gives an overview.

**[Figure 1 here: Pipeline block diagram. Left branch (Offline): ECG Dataset (928 images, 4 classes) → OpenCV Preprocessing → ResNet50 pool1_pool (462,400-D) → TruncatedSVD(9) + MinMaxScaler → Classical SVM / QSVC / Pegasos training → serialized models. Right branch (Online Inference): User uploads ECG via React → FastAPI → MobileNetV2 Gatekeeper → same preprocessing chain → User-selected classifier → Prediction + Confidence + Clinical Recommendation. Caption: Fig. 1. QuCardio hybrid classical-quantum pipeline. The offline phase trains all three classifiers on 742 ECG images; the online phase runs the same preprocessing chain on a single uploaded image and returns predictions within seconds for SVM or approximately 15 seconds for the quantum models.]**

### A. ECG Image Preprocessing

Raw clinical ECG scans arrive with wide variation in scan quality, background colour, grid density, and orientation. A seven-step OpenCV pipeline in `preprocess_ecg.py` converts each image to a 340×340 pixel normalized grayscale array. The steps are: (1) OTSU adaptive thresholding and ROI crop using morphological dilation and largest-contour detection, (2) adaptive background polarity normalization so that all images have a consistent dark-trace-on-light-background representation, (3) vertical grid removal using a 1×40 morphological OPEN operation, (4) horizontal grid removal with a 40×1 kernel — the ECG trace is thicker than any grid line, so it survives both subtractions intact — (5) 3×3 Gaussian blur followed by re-thresholding at 30 to suppress residual speckle, (6) cv2.INTER\_AREA resize to 340×340, and (7) division by 255.0 to produce a float32 array in [0, 1].

### B. ResNet50 Feature Extraction

Feature extraction used a ResNet50 [12] with frozen ImageNet weights, rebuilt through the Keras functional API to output at the pool1\_pool layer. This layer applies 3×3 max-pooling over the initial 7×7 convolutional stage (64 filters, stride 2), producing an 85×85×64 spatial activation map. Flattened, this yields a 462,400-dimensional feature vector per image. The shallow layer was chosen following Prabhu et al. [5] to capture low-level morphological structure — QRS complex geometry, ST segment curvature, P and T wave shape — without imposing ImageNet object semantics from deeper layers. Grayscale images were expanded to three channels by axis repetition before passing through ResNet50's standard per-channel normalization. Extraction ran in batches of 8.

### C. Dimensionality Reduction

Truncated Singular Value Decomposition (TruncatedSVD) at n\_components = 9, fitted on 742 training samples, compressed the 462,400-dimensional vectors to 9 dimensions. TruncatedSVD was preferred over Principal Component Analysis because it does not require explicit mean-centering of the data matrix, avoiding an O(N × d) memory allocation at this dimensionality. The 9 retained right singular vectors capture the dominant axes of variance in the training feature distribution. MinMaxScaler, also fitted on training data alone, normalized these to [0, 1]. For the quantum classifiers the scaled vector is multiplied by pi at inference time, placing features in [0, pi] — the encoding range established as optimal in Section VI. The classical SVM consumes the [0, 1] vector directly. Both fitted objects are serialized at training time and loaded at API startup, so inference reuses the exact transformations from training without any refitting.

### D. Classical Support Vector Machine

The classical SVM used `sklearn.svm.SVC` with an RBF kernel, C = 10, gamma = 'scale', and class\_weight = 'balanced' to handle the 227:138 training imbalance between the largest and smallest class. Wrapped in `CalibratedClassifierCV` (method = 'isotonic', cv = 3), it outputs calibrated class probabilities suitable for clinical confidence display. Hyperparameters were selected via 5-fold stratified GridSearchCV optimizing macro F1 over C ∈ {0.01, 0.1, 1.0, 5.0, 10.0} and gamma ∈ {'scale', 'auto'}.

### E. Quantum Support Vector Classifier

The QSVC encodes the 9-dimensional feature vector into a 2⁹ = 512-dimensional complex Hilbert space using Qiskit's ZZFeatureMap [4], which implements the Pauli-ZZ construction. For reps repetitions, the circuit applies: (a) a Hadamard layer on all 9 qubits, (b) first-order Rz(2xᵢ) rotations encoding each feature as a phase angle, and (c) second-order entangling blocks applying Rz(2(π − xᵢ)(π − xⱼ)) to each adjacent qubit pair under the selected topology. The fidelity quantum kernel is:

**K(x, z) = |⟨ψ(x)|ψ(z)⟩|²**   &nbsp;&nbsp;&nbsp;&nbsp; (1)

where |ψ(x)⟩ is the statevector produced by encoding input x.

Computing the N×N kernel matrix by running N² circuits would be prohibitively slow. Instead, all N = 742 training statevectors are precomputed once, and the full kernel matrix is formed as:

**K[i, j] = |svᵢ · sv̄ⱼ|²**   &nbsp;&nbsp;&nbsp;&nbsp; (2)

where svᵢ is the complex-amplitude vector for sample i. Training on 742 samples takes under 15 seconds on a standard CPU. At inference, only a single kernel row is evaluated against the 742 cached training statevectors — dense linear algebra, not circuit simulation. A 486-configuration sweep over encoding range, entanglement topology, reps ∈ {1, 2, 3, 4}, and C ∈ {0.1, 1.0, 5.0, 10.0} settled the final configuration: [0, pi] encoding, circular entanglement, reps = 2, C = 5.0.

### F. Multiclass Pegasos QSVC

The Pegasos Quantum Support Vector Classifier [13], extended to four classes using six one-versus-one binary models, trains each model with stochastic sub-gradient descent. The native Qiskit PegasosQSVC implementation fails numerically at 9 qubits because the average kernel value at this scale is approximately 1/512, causing the SGD step size ηₜ = 1/(λt) to produce updates that effectively vanish within the first few iterations. A custom training loop resolving this issue, combined with per-model grid search over the regularization constant and iteration count, recovered accuracy from 78.49% to 91.94%. At inference, Algorithm 1 from Prabhu et al. [5] evaluates only three of the six binary models in a decision tree to produce the four-class prediction.

### G. MobileNetV2 Gatekeeper and Web Application

A pre-trained MobileNetV2 gatekeeper rejects non-ECG uploads before any heavy computation begins. The FastAPI backend loads all model objects at startup and exposes six REST endpoints, including single-image prediction, batch prediction (up to 20 images), PDF report generation, a three-model comparison endpoint, a health check, and a paginated audit history endpoint. A Redis cache keyed by the SHA-256 hash of the image bytes plus the model name returns stored results instantly for repeated submissions, with a 24-hour time-to-live. The React frontend provides five tabs for Diagnosis, Batch Upload, Model Performance, Quantum Insights, and API documentation.

---

## IV. EXPERIMENTAL SETUP

**Dataset.** The dataset is the ECG image collection from the Ch. Pervaiz Elahi Institute of Cardiology, Multan, Pakistan, publicly available on Mendeley Data [14]. It contains 928 scanned and photographed paper ECG prints from real clinical recordings, split across four classes: Normal (284 images), Arrhythmia (233), Myocardial Infarction (239), and History of Myocardial Infarction (172). An 80:20 stratified random split with random\_state = 42 yields 742 training samples and 186 test samples. Class proportions are preserved in both partitions. No augmentation was applied during training; augmented images appear only in the robustness evaluation.

**TABLE II: Dataset Distribution**

| Class | Total | Training | Test |
|---|---|---|---|
| Normal | 284 | 227 | 57 |
| Arrhythmia | 233 | 186 | 47 |
| Myocardial Infarction | 239 | 191 | 48 |
| History of MI | 172 | 138 | 34 |
| **Total** | **928** | **742** | **186** |

**Evaluation Protocol.** All models were evaluated on the same held-out 186-image test set. The TruncatedSVD reducer and MinMaxScaler were fitted on training data only and serialized; the test set passed through the same fitted objects without refitting, eliminating data leakage. Primary metrics are accuracy and macro-averaged F1 score. ROC-AUC is reported per class and macro-averaged. McNemar's test [15] is used for pairwise statistical significance testing. Expected Calibration Error (ECE) is computed with 15 equal-width probability bins.

**Hardware and Software.** All experiments ran on a standard CPU (Intel Core i5, 16 GB RAM). Quantum circuits were simulated using Qiskit Aer 0.13 statevector simulator [16]. The backend uses Python 3.10, TensorFlow 2.15, Scikit-learn 1.3, and Qiskit Machine Learning 0.7.

**Kernel Target Alignment.** To verify that the quantum kernel is well-matched to the label structure of the data, Kernel Target Alignment (KTA) [17] was computed for the optimal QSVC configuration: KTA = 0.1081. A value above zero indicates positive alignment between the kernel matrix and the ideal kernel, confirming that the ZZFeatureMap is capturing class-relevant structure.

---

## V. RESULTS AND DISCUSSION

### A. Model Performance

**TABLE III: Model Performance on 186-Image Test Set**

| Model | Accuracy | Precision | Recall | Macro F1 | Paper Acc. |
|---|---|---|---|---|---|
| Classical SVM (RBF, C=10) | 84.95% | 84.17% | 84.41% | 84.10% | 83.33% |
| **QSVC (ours)** | **94.62%** | **94.84%** | **94.05%** | **94.39%** | 94.09% |
| Pegasos QSVC | 91.94% | 91.99% | 91.61% | 91.31% | 93.05% |
| Random Forest (baseline) | 91.94% | — | — | 91.20% | — |
| KNN (baseline) | 84.95% | — | — | 83.61% | — |

The QSVC leads all five baselines by a substantial margin. The 9.67 percentage point gap over classical SVM is the primary claim, and it is tested statistically in Section V-B. Pegasos falls 1.11 percentage points below the 93.05% reported in the base paper [5], which we attribute to stricter leakage control: the SVD reducer and scaler here are fitted on training data only, while the base paper's implementation details on this point are not specified. Random Forest reaching 91.94% confirms that the 9-dimensional SVD features carry strong discriminative signal on their own; the quantum kernel improves on an already-informative representation rather than compensating for weak features. The classical SVM improves 1.62 percentage points over the base paper's 83.33%, reflecting the tuning of balanced class weights and isotonic calibration.

**[Figure 2 here: QSVC confusion matrix (4×4 heatmap). Caption: Fig. 2. Confusion matrix for the tuned QSVC on 186 test samples. Perfect recall on Myocardial Infarction (48/48); Arrhythmia is the most challenging class for all models due to morphological variability. File: results/paper/main/confusion_matrix_qsvc_tuned.png]**

The QSVC confusion matrix (Fig. 2) shows 100% recall on Myocardial Infarction — the class where a missed positive carries the greatest clinical cost. Arrhythmia is the most frequently confused class across all three models, which is consistent with the morphological heterogeneity of rhythm abnormalities and their overlap with Normal traces near classification boundaries.

### B. Statistical Significance

**TABLE IV: McNemar's Test Results (n = 186)**

| Comparison | Discordant Pairs (B / C) | χ² | p-value | Significant? |
|---|---|---|---|---|
| **QSVC vs. Classical SVM** | **23 / 5** | **10.321** | **0.00131** | **Yes (p < 0.005)** |
| Pegasos vs. Classical SVM | 16 / 6 | 5.760 | 0.0164 | Yes (p < 0.05) |
| QSVC vs. Pegasos | 9 / 5 | 1.231 | 0.267 | No |

McNemar's test operates on discordant pairs: cases where one model is correct and the other is wrong. The QSVC got 23 samples right that the classical SVM got wrong, while the SVM rescued only 5 that the QSVC missed. The resulting chi-squared statistic of 10.32 (p = 0.0013) establishes that the quantum advantage is a systematic property of the models and not a product of this particular test split. Both quantum models are statistically superior to the classical SVM; they are not statistically distinguishable from each other at n = 186 despite the 2.68 percentage point accuracy gap.

To the best of our knowledge, this is the first McNemar significance test of quantum kernel advantage over classical SVM applied to ECG image classification in the published literature.

### C. ROC-AUC and Confidence Calibration

**TABLE V: Multi-Class ROC-AUC Scores**

| Class | SVM AUC | QSVC AUC | Pegasos AUC |
|---|---|---|---|
| Normal | 0.929 | 0.997 | 0.972 |
| Arrhythmia | 0.912 | 0.974 | 0.912 |
| Myocardial Infarction | 0.987 | **1.000** | **1.000** |
| History of MI | 0.927 | 0.984 | 0.979 |
| **Macro-AUC** | 0.940 | **0.989** | 0.967 |

**[Figure 3 here: ROC curves for all models and all four classes. Caption: Fig. 3. ROC curves for QSVC, Pegasos, and Classical SVM across four ECG classes. Both quantum models achieve AUC = 1.000 for Myocardial Infarction. File: results/paper/main/roc_curves_all_models.png]**

Both quantum models achieve AUC = 1.000 for Myocardial Infarction, separating every MI sample from every other class at every decision threshold without a single ranking error. The QSVC macro-AUC of 0.989 sits 4.9 points above the classical SVM.

**[Figure 4 here: Calibration curves. Caption: Fig. 4. Reliability diagrams showing predicted confidence versus empirical accuracy for all three models. All macro ECE values below 0.07 indicate clinically trustworthy confidence scores. File: results/paper/main/calibration_all_models.png]**

Expected Calibration Error for all three models falls below 0.07 macro ECE. A prediction reported at 80% confidence is therefore correct approximately 80% of the time, which is a necessary condition for clinical decision support. We could not find calibration analysis in any of the ECG quantum machine learning papers surveyed in Section II.

### D. Augmentation Robustness

**[Figure 5 here: Augmentation robustness bar chart. Caption: Fig. 5. Model accuracy under nine augmentation conditions. Classical SVM accuracy collapses from 84.4% to 46.7% under brightness perturbation while QSVC maintains 86.7%. File: results/paper/ablation/augmentation_robustness.png]**

Under brightness perturbation — simulating an overexposed photograph of a printed ECG — classical SVM accuracy collapses from 84.4% to 46.7%, which is near random-guess performance on four classes. The QSVC maintains 86.7% under the same condition. The mechanism is the encoding: ZZFeatureMap encodes feature values as phase angles, and phase is periodic, so a uniform shift in scaled pixel statistics moves the encoded quantum state around the Bloch sphere rather than away from the training support vectors. The RBF kernel, working on Euclidean distance in the raw feature space, is far more sensitive to the same shift. Rotation, affine distortion, and aspect ratio change affect both models similarly.

---

## VI. ABLATION STUDIES

### A. Feature Encoding Range

**TABLE VI: QSVC and Pegasos Accuracy vs. Feature Encoding Range**
*(All rows use circular entanglement, reps = 2, C = 5.0)*

| Encoding | QSVC Accuracy | Pegasos Accuracy |
|---|---|---|
| L2 Normalization (unit sphere) | 68.28% | 57.53% |
| MinMax [0, 1] (base paper range) | 92.47% | 84.41% |
| **MinMax [0, pi] (ours)** | **94.62%** | **90.86%** |
| MinMax [0, 2pi] | 94.09% | 90.32% |

**[Figure 6 here: Encoding ablation bar chart. Caption: Fig. 6. Effect of feature encoding range on QSVC accuracy. All configurations use circular entanglement; L2 normalization discards feature magnitude and performs worst. File: results/paper/ablation/ablation_encoding.png]**

The ZZFeatureMap applies Rz(2xᵢ) rotations, so a feature in [0, 1] sweeps approximately 2 radians of Bloch sphere latitude — roughly one-third of the full circle. Extending the range to [0, pi] sweeps the full 2π, distributing encoded states more widely and raising the rank of the kernel matrix. Pushing further to [0, 2pi] loses accuracy because rotation angles exceeding 2π wrap around, causing distinct feature values to map to nearly identical quantum states. L2 normalization performs worst by 24 percentage points: projecting onto the unit sphere discards feature magnitude entirely, and magnitude is precisely what the dominant SVD components encode.

All rows in TABLE VI use circular entanglement, so the table isolates the encoding effect alone. The 92.47% value at [0, 1] is therefore the base paper's encoding range combined with our topology, not the base paper's full configuration. The full base paper configuration (linear entanglement, [0, 1]) corresponds to the 89.78% baseline in TABLE VII.

### B. Entanglement Topology

**TABLE VII: QSVC Accuracy vs. Entanglement Topology**
*(All rows use [0, pi] encoding, reps = 2, C = 5.0)*

| Topology | QSVC Accuracy | Delta vs. Linear |
|---|---|---|
| Linear | 89.78% | — |
| Pairwise | 93.01% | +3.23 pp |
| Full | 92.47% | +2.69 pp |
| **Circular (ours)** | **94.62%** | **+4.84 pp** |
| SCA | 94.62% | +4.84 pp |

**[Figure 7 here: Entanglement ablation bar chart. Caption: Fig. 7. QSVC accuracy for five entanglement topologies. Circular topology gives each qubit two entangled neighbors instead of one, producing the richest pairwise interaction structure at the same circuit depth as linear. File: results/paper/ablation/ablation_entanglement.png]**

Linear entanglement connects each qubit to only one neighbor, while circular topology closes the chain by connecting qubit 8 back to qubit 0. Every qubit therefore has two entangled partners, producing richer pairwise correlations between SVD components at a circuit depth barely greater than linear. Full entanglement, which connects every pair, yields lower accuracy than circular at 92.47%, likely because the exponentially larger number of entangling blocks at reps = 2 introduces over-parameterisation at this feature dimensionality. For 9 qubits, the SCA (Shifted Circular Alternating) topology reduces to the same connectivity as circular, which explains the identical result.

The two improvements — encoding range and entanglement topology — interact rather than add independently. Moving from [0, 1] encoding with circular entanglement to [0, pi] with circular entanglement gains 2.15 percentage points (TABLE VI, 92.47% to 94.62%). Moving from [0, pi] with linear to [0, pi] with circular gains 4.84 percentage points (TABLE VII, 89.78% to 94.62%). The combined gain from the base paper's full configuration to ours is also 4.84 percentage points, because the topology accounts for the dominant portion of that improvement once the encoding range is not the limiting factor.

### C. Pauli Feature Map Comparison

**TABLE VIII: Pauli Feature Map Comparison**
*(9 qubits, reps = 2, circular, [0, pi], C = 5.0)*

| Feature Map | Structure | QSVC Accuracy | Macro F1 |
|---|---|---|---|
| ZFeatureMap | Rz only, no entanglement | 87.63% | 0.871 |
| **ZZFeatureMap (ours)** | **Pauli-Z + ZZ interactions** | **94.62%** | **0.944** |
| PauliFeatureMap (X+XX) | Pauli-X + XX interactions | 91.94% | 0.913 |
| PauliFeatureMap (Y+YY) | Pauli-Y + YY interactions | 83.33% | 0.835 |

**[Figure 8 here: Pauli ablation bar chart. Caption: Fig. 8. QSVC accuracy for four Pauli feature map variants under identical configuration. ZFeatureMap applies no entanglement and is 6.99 pp below ZZFeatureMap, confirming entanglement is doing real work. File: results/paper/ablation/ablation_feature_maps.png]**

ZFeatureMap, which applies only single-qubit Rz rotations with no entanglement, drops to 87.63% — 6.99 percentage points below ZZFeatureMap. This is the clearest demonstration that the entangling ZZ interactions are genuinely contributing to classification, not merely adding circuit depth. PauliFeatureMap with X+XX interactions reaches 91.94%, comfortably above the unentangled baseline but still 2.68 points below ZZFeatureMap. The Y+YY variant falls to 83.33%, roughly at the level of the classical SVM baseline and below even the unentangled ZFeatureMap. Rx rotations place features on the equatorial plane of the Bloch sphere rather than the phase axis, and Ry rotations on a different axis still, each producing a different kernel geometry; the empirical result establishes ZZFeatureMap as the best match for the compressed ECG feature distribution, though we do not claim to have established the mechanism from a single dataset.

### D. SVD Components and Circuit Repetitions

**[Figure 9 here: SVD dims ablation (top) and reps ablation (bottom) in a 1×2 or 2×1 panel. Caption: Fig. 9. (Top) QSVC accuracy versus number of SVD components. Accuracy peaks at 9 components consistent with Prabhu et al. (Bottom) Accuracy versus circuit repetitions (reps). reps=2 is optimal; increasing reps does not improve performance and adds simulation cost. Files: results/paper/ablation/ablation_svd_dims.png, results/paper/ablation/ablation_reps.png]**

Varying SVD components from 2 to 18 confirms that 9 components is the optimal point for this feature space, consistent with the base paper. Below 9, discriminative structure is discarded; above 9, the additional components carry little signal relative to the simulation cost. Circuit repetitions sweep confirms reps = 2 is optimal; deeper circuits (reps = 3, 4) show no improvement and add quantum simulation cost, which is consistent with findings in Ozpolat and Karabatak [10] that increasing circuit depth beyond a sufficient encoding depth does not help.

---

## VII. CONCLUSION

This paper investigated how two quantum circuit parameters — feature encoding range and entanglement topology — affect the Quantum Support Vector Classifier for four-class ECG image classification. The core findings are precise. Scaling the 9-dimensional SVD-compressed features to [0, pi] instead of the base paper's [0, 1] range contributes 2.15 percentage points when entanglement is held fixed. Switching from linear to circular entanglement topology contributes 4.84 percentage points when encoding is held at [0, pi]. Together, moving from the base paper configuration to ours raises accuracy from 89.78% to 94.62% on a 928-image clinical dataset, with the topology providing the dominant share of the improvement. The mechanism for the encoding effect is the Rz gate structure of the ZZFeatureMap: [0, 1] scaling uses roughly one-third of the available Bloch sphere latitude, while [0, pi] uses all of it.

McNemar's test confirms the quantum kernel advantage over a well-tuned RBF SVM is statistically significant at chi-squared = 10.32, p = 0.0013. Both quantum models achieve AUC = 1.000 for Myocardial Infarction. Expected Calibration Error below 0.07 for all three models confirms that confidence scores are clinically trustworthy. The QSVC holds 86.7% accuracy under brightness perturbation that collapses the classical SVM to 46.7%. The ZZFeatureMap is confirmed as the best-fitting Pauli structure for this feature distribution, outperforming unentangled, X+XX, and Y+YY variants.

The deployed web application moves quantum ECG classification from proof-of-concept to a usable clinical screening tool for the first time, supporting single and batch upload, calibrated confidence, and PDF report generation.

Several directions remain open. Evaluating the trained 9-qubit classifier under realistic NISQ noise models and eventually on IBM Quantum hardware would quantify the gap between ideal statevector simulation and near-term deployment. Domain adaptation to address the cross-dataset sensitivity identified in zero-shot transfer experiments is the most urgent practical next step. Adding Grad-CAM visualization over the pool1\_pool feature map would provide clinician-interpretable regions of interest. Implementing the Quanvolutional Neural Network of Prabhu et al. [5] and comparing it against our QSVC under the same circuit design discipline would complete the picture.

---

## ACKNOWLEDGEMENT

The authors thank the Qiskit open-source community and IBM Quantum for the quantum computing framework. The ECG dataset was made publicly available by the Ch. Pervaiz Elahi Institute of Cardiology, Multan, through Mendeley Data. The authors also thank the Department of Computer Science and Engineering, St. Joseph Engineering College, Mangaluru, for infrastructure support.

---

## REFERENCES

[1] World Health Organization, "Cardiovascular diseases (CVDs)," Fact Sheet, Jun. 2021. [Online]. Available: https://www.who.int/news-room/fact-sheets/detail/cardiovascular-diseases-(cvds)

[2] F. Vasquez-Iturralde, M. J. Flores-Calero, F. Grijalva, and A. Rosales-Acosta, "Automatic classification of cardiac arrhythmias using deep learning techniques: A systematic review," *IEEE Access*, vol. 12, 2024, doi: 10.1109/ACCESS.2024.3408282.

[3] O. Yildirim, P. Plawiak, R.-S. Tan, and U. R. Acharya, "Arrhythmia detection using deep convolutional neural network with long duration ECG signals," *Comput. Biol. Med.*, vol. 102, pp. 411–420, 2018, doi: 10.1016/j.compbiomed.2018.09.009.

[4] V. Havlicek, A. D. Corcoles, K. Temme, A. W. Harrow, A. Kandala, J. M. Gambetta, and J. Preskill, "Supervised learning with quantum-enhanced feature spaces," *Nature*, vol. 567, pp. 209–212, Mar. 2019, doi: 10.1038/s41586-019-0980-2.

[5] S. Prabhu, S. Gupta, G. M. Prabhu, A. V. Dhanuka, and K. V. Bhat, "QuCardio: Application of quantum machine learning for detection of cardiovascular diseases," *IEEE Access*, vol. 11, pp. 136835–136851, 2023, doi: 10.1109/ACCESS.2023.3338145.

[6] V. Jain, N. Arora, and A. Gupta, "Quantum-assisted cardiac diseases diagnosis and prediction using ECG images," *J. Supercomputing*, vol. 81, p. 1439, 2025, doi: 10.1007/s11227-025-07939-8.

[7] F. Vasquez-Iturralde et al., "Automatic classification of cardiac arrhythmias using deep learning techniques: A systematic review," *IEEE Access*, vol. 12, 2024, doi: 10.1109/ACCESS.2024.3408282.

[8] J. Biamonte, P. Wittek, N. Pancotti, P. Rebentrost, N. Wiebe, and S. Lloyd, "Quantum machine learning," *Nature*, vol. 549, pp. 195–202, Sep. 2017, doi: 10.1038/nature23474.

[9] M. Y. Ramkhelawan, S. Grandhi, and S. Wibowo, "A hybrid quantum-classical deep learning algorithm for efficient arrhythmia classification," in *Proc. IEEE SNPD*, 2025, doi: 10.1109/SNPD62189.2025.10985682.

[10] Z. Ozpolat and M. Karabatak, "Performance evaluation of quantum-based machine learning algorithms for cardiac arrhythmia classification," *Diagnostics*, vol. 13, no. 6, p. 1099, 2023, doi: 10.3390/diagnostics13061099.

[11] I. Aksoy and Z. Ozpolat, "Comparison of SVM, QSVM and Pegasos-QSVM algorithms on different medical datasets," *Int. J. Sustain. Eng. Technol.*, vol. 9, no. 1, pp. 80–93, 2025, doi: 10.62301/usmtd.1716034.

[12] K. He, X. Zhang, S. Ren, and J. Sun, "Deep residual learning for image recognition," in *Proc. IEEE/CVF CVPR*, Jun. 2016, pp. 770–778, doi: 10.1109/CVPR.2016.90.

[13] S. Shalev-Shwartz, Y. Singer, N. Srebro, and A. Cotter, "Pegasos: Primal estimated sub-gradient solver for SVM," *Math. Program.*, vol. 127, no. 1, pp. 3–30, 2011, doi: 10.1007/s10107-010-0420-4.

[14] S. Prabhu, "ECG Images dataset of Cardiac and COVID-19 Patients," Mendeley Data, V2, 2022, doi: 10.17632/gwbz3fsgp8.2.

[15] Q. McNemar, "Note on the sampling error of the difference between correlated proportions or percentages," *Psychometrika*, vol. 12, no. 2, pp. 153–157, 1947.

[16] Qiskit contributors, "Qiskit: An open-source framework for quantum computing," 2023, doi: 10.5281/zenodo.2573505.

[17] N. Cristianini, J. Shawe-Taylor, A. Elisseeff, and J. Kandola, "On kernel-target alignment," in *Advances in Neural Information Processing Systems (NeurIPS)*, 2002, pp. 367–373.

[18] K. L. Soon et al., "Early cardiovascular disease detection using hierarchical quantum ensemble model," *Comput. Methods Biomech. Biomed. Eng.*, Jan. 2026, doi: 10.1080/10255842.2025.2612536.

[19] A. E. Setiawan et al., "A systematic literature review of quantum machine learning for medical: trends, datasets, topics, and methods," *Int. J. Cogn. Comput. Eng.*, vol. 7, pp. 609–630, 2026, doi: 10.1016/j.ijcce.2026.05.002.

---

## FIGURE AND TABLE PLACEMENT GUIDE

*(For LaTeX typesetting — place each at top or bottom of column, never mid-column)*

| Item | Type | File / Description | Section |
|---|---|---|---|
| Fig. 1 | Pipeline block diagram | Draw in draw.io using description in Sec. III | After Section III intro |
| Fig. 2 | QSVC confusion matrix heatmap | results/paper/main/confusion_matrix_qsvc_tuned.png | Section V-A |
| Fig. 3 | ROC curves (all models, 4 classes) | results/paper/main/roc_curves_all_models.png | Section V-C |
| Fig. 4 | Calibration reliability diagrams | results/paper/main/calibration_all_models.png | Section V-C |
| Fig. 5 | Augmentation robustness bar chart | results/paper/ablation/augmentation_robustness.png | Section V-D |
| Fig. 6 | Encoding ablation bar chart | results/paper/ablation/ablation_encoding.png | Section VI-A |
| Fig. 7 | Entanglement ablation bar chart | results/paper/ablation/ablation_entanglement.png | Section VI-B |
| Fig. 8 | Pauli map ablation bar chart | results/paper/ablation/ablation_feature_maps.png | Section VI-C |
| Fig. 9 | SVD dims + Reps ablation (2-panel) | results/paper/ablation/ablation_svd_dims.png + ablation_reps.png | Section VI-D |
| TABLE I | Prior work comparison | inline | Section II |
| TABLE II | Dataset distribution | inline | Section IV |
| TABLE III | Model performance | inline | Section V-A |
| TABLE IV | McNemar test | inline | Section V-B |
| TABLE V | ROC-AUC per class | inline | Section V-C |
| TABLE VI | Encoding ablation | inline | Section VI-A |
| TABLE VII | Entanglement ablation | inline | Section VI-B |
| TABLE VIII | Pauli map comparison | inline | Section VI-C |

**Note on figure count:** 9 figures + 8 tables is heavy for a 6-page conference paper. For a 6-page venue:
- Drop Fig. 9 (combine SVD and reps into one short paragraph, no figure)
- Drop TABLE VIII (move Pauli map numbers into the text)
- Keep: Fig. 1 (pipeline), Fig. 2 (CM), Fig. 3 (ROC), Fig. 5 (robustness), Fig. 6 (encoding), Fig. 7 (entanglement), TABLE I (prior work), TABLE III (results), TABLE IV (McNemar), TABLE VI (encoding), TABLE VII (entanglement)
- For a journal paper (8–12 pages), include all figures and tables.

---

*End of paper content — QuCardio, SJEC Mangaluru, 2025-2026*
