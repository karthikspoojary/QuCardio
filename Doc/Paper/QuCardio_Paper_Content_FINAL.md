<!--
  TYPESETTING NOTES — remove this block before LaTeX compilation
  Template  : IEEE IEEEtran two-column (same class as Main.pdf / WIECON-ECE)
  Target    : 6-page IEEE conference  OR  10-page journal
  Figures   : caption BELOW  (\caption after \includegraphics)
  Tables    : caption ABOVE  (\caption before \begin{tabular})
  Table IDs : TABLE~I, TABLE~II … (Roman, uppercase)
  Figure IDs: Fig.~1, Fig.~2 …
  Equations : number every equation, cite as (1), (2) in text
  pool1_pool: use \texttt{pool1\_pool} in body; in abstract use pool1\_pool
  π symbol  : $\pi$ in LaTeX — never write "pi" in the compiled body
  χ²        : $\chi^2$ in LaTeX
  [0,π]     : $[0,\pi]$
  Spelling  : British (-ise/-isation) throughout — verified consistent. Keep it that way.
  6-page trim: drop Fig. 2 (app screenshots — describe deployment in two sentences),
               Fig. 4 (state ECE numbers in text only),
               Fig. 9 (state SVD=9 and reps=2 with numbers in one sentence).
               Keep TABLE VIII — Pauli comparison is a core ablation result.
               Compress Section VII to one paragraph if space is tight; do not remove it.
  10-page   : include everything; expand cross-dataset transfer into full Section V-E
               with its own table and confusion matrices.
  Screenshot paths  (relative from project root):
    image1 — Doc/Report/extracted_images/word/media/image1.png (upload screen)
    image2 — Doc/Report/extracted_images/word/media/image2.png (MI result + 99.5%)
    image3 — Doc/Report/extracted_images/word/media/image3.png (pipeline timing + consensus)
    image4 — Doc/Report/extracted_images/word/media/image4.png (model performance tab)
  All result plots are in results/paper/ — see figure guide at the end of this file

  ===================================================================
  FOUR ITEMS MARKED [VERIFY] — RESOLVE BEFORE SUBMISSION
  1. Sec III-E : grid is 4×5×4×4 = 320, not 486. Confirm sweep log matches 320.
  2. Sec IV    : how the 928-image 4-class subset was selected from the
                 full Mendeley collection — a reviewer who knows this
                 dataset WILL check the MI count of 239 vs 77 in v1.
  3. TABLE I   : 95.2% attributed to Jain et al. — confirm it appears
                 in their paper or replace with relative margin.
  4. Sec II    : Prabhu et al.'s second model — "quanvolutional" or
                 "Quantum Neural Network"? Report and paper must agree.
  ===================================================================
-->

---

## TITLE

Quantum Kernel Circuit Design for Four-Class ECG Image Classification: Encoding Range, Entanglement Topology, and Statistical Validation

---

## AUTHORS

*(Author 1 Name)*, *(Author 2 Name)*, *(Author 3 Name)*, *(Author 4 Name)*
Dept. of Computer Science and Engineering
St. Joseph Engineering College, Mangaluru, India
{email1, email2, email3, email4}@sjec.ac.in

*(Guide Name)*
Dept. of Computer Science and Engineering
St. Joseph Engineering College, Mangaluru, India

---

## ABSTRACT

Cardiovascular diseases account for approximately 17.9 million deaths each year, and the Electrocardiogram is the primary non-invasive tool for detecting them. Automated ECG interpretation using quantum machine learning has shown early promise, yet published studies report single accuracy figures under fixed circuit configurations, with no systematic investigation of how circuit design choices affect performance and no statistical validation of the quantum advantage. This paper presents a hybrid classical-quantum pipeline for four-class ECG image classification and reports a controlled ablation of two circuit parameters: the feature encoding range and the entanglement topology. The pipeline extracts a 462,400-dimensional feature vector from the pool1\_pool layer of a frozen ResNet50, compresses it to 9 dimensions via Truncated Singular Value Decomposition, and classifies using a 9-qubit ZZFeatureMap fidelity quantum kernel under statevector simulation. Scaling the compressed features to [0, pi] with circular entanglement raises Quantum Support Vector Classifier accuracy from 89.78% to 94.62% on a 928-image clinical subset of a public dataset collected at the Ch. Pervaiz Elahi Institute of Cardiology, Multan. McNemar's test confirms that the advantage over a well-tuned classical Support Vector Machine is statistically significant (chi-squared = 10.32, p = 0.0013). The Quantum Support Vector Classifier achieves a macro Area Under the ROC Curve of 0.989 and perfect class separation for Myocardial Infarction (AUC = 1.000). Expected Calibration Error analysis places all three classifiers below 0.07 macro ECE, indicating that their confidence scores are usable for clinical decision support. The complete pipeline is deployed as a working web application supporting real-time single and batch ECG diagnosis with downloadable PDF reports.

*Index Terms* — Quantum Machine Learning, Quantum Support Vector Classifier, ZZFeatureMap, Fidelity Quantum Kernel, ECG Classification, Cardiovascular Disease, ResNet50, Truncated SVD, McNemar Test.

---

## I. INTRODUCTION

Cardiovascular diseases are the leading cause of global mortality, with the World Health Organization reporting approximately 17.9 million deaths per year [1]. The Electrocardiogram (ECG) is the standard first-line screening tool, capturing the heart's electrical activity as a waveform trace from which trained cardiologists identify conditions including Arrhythmia, Myocardial Infarction (MI), and prior History of Myocardial Infarction. Accurate interpretation of ECG traces requires specialist training that is frequently unavailable in high-volume hospitals and rural clinics, making automated classification systems of real clinical value [2].

Machine learning approaches have been applied to ECG classification for over a decade, producing good results on large digital signal databases [3]. On small multi-class medical image collections, however, classical kernel methods tend to plateau because they cannot efficiently exploit the higher-order feature correlations that distinguish morphologically similar cardiac conditions. Quantum machine learning (QML) offers a theoretically grounded alternative: quantum feature maps embed classical data into exponentially large Hilbert spaces where classification boundaries arise that no polynomial-time classical kernel can represent [4]. In the current Noisy Intermediate-Scale Quantum (NISQ) era, where hardware is still limited by qubit count and gate fidelity, statevector simulation on classical hardware is the practical mode for QML research.

Prior work on quantum ECG classification has established that Quantum Support Vector Classifiers (QSVC) with ZZFeatureMap encoding outperform classical Support Vector Machines (SVM) on ECG image datasets [5], [6]. A common limitation across these studies is that each reports a single accuracy number under a fixed circuit configuration with no analysis of how encoding range or entanglement topology affects the result. The circuit design choices are not independently varied, so it is not possible to attribute performance differences to specific parameters. Furthermore, statistical significance of the quantum advantage is not tested, calibration of predicted confidence is not reported, and no system is deployed for clinical use.

This paper addresses those gaps with the following contributions:

1. A controlled sweep of four encoding ranges and five entanglement topologies, showing that [0, pi] scaling with circular entanglement raises QSVC accuracy by 4.84 percentage points over the reproduced baseline configuration, with the contribution of each factor separated.
2. A McNemar significance test of quantum kernel advantage over classical SVM on ECG image classification, confirming the result at chi-squared = 10.32, p = 0.0013. To the best of our knowledge, no such test has previously been reported for this task.
3. Expected Calibration Error and augmentation robustness experiments, both absent from prior ECG-QML work, showing QSVC maintains 86.7% accuracy under brightness perturbation where classical SVM falls to 46.7%.
4. Deployment of the complete pipeline as a working web application usable directly by a clinician, supporting single and batch ECG upload, calibrated confidence scores, class-specific recommendations, and PDF report generation.

The remainder of this paper is organised as follows. Section II surveys related work. Section III describes the proposed methodology. Section IV details the experimental setup. Section V presents results and discussion. Section VI reports ablation studies. Section VII states the limitations, and Section VIII concludes.

---

## II. LITERATURE SURVEY

This section reviews work on classical ECG deep learning, quantum kernel theory, and quantum cardiac classification relevant to the proposed system.

Deep learning methods for ECG classification have progressed substantially over the past decade. Yildirim et al. [3] showed that a 1D convolutional neural network applied directly to raw ECG waveforms achieves 91.33% accuracy across 17 arrhythmia classes from the MIT-BIH database, demonstrating that learned temporal representations can outperform hand-crafted signal features on large corpora. Vasquez-Iturralde et al. [2] reviewed 494 deep learning ECG papers and identified persistent gaps: fewer than 15% of studies use image-format ECG inputs rather than digital recordings, and fewer than 8% report any form of confidence calibration. The present work operates on scanned paper ECG prints and provides both calibration and statistical significance analyses, directly addressing both gaps.

The theoretical basis for quantum kernel classification was established by Havlicek et al. [4], who showed that ZZFeatureMap-based kernels achieve separation advantages intractable for polynomial-time classical computation by mapping data into exponentially large Hilbert spaces through Pauli-ZZ interactions. The fidelity kernel K(x, z) = |⟨ψ(x)|ψ(z)⟩|² used throughout this work is the construction introduced in that paper. Biamonte et al. [7] surveyed quantum machine learning broadly and identified hybrid classical-quantum pipelines, where a classical preprocessor handles dimensionality reduction before quantum encoding, as the most viable near-term architecture; this observation directly motivates the Truncated SVD step described in Section III.

Prabhu et al. [5] demonstrated quantum ECG classification on the same four-class clinical data used here, combining ResNet50 pool1\_pool features with Truncated SVD at 9 components and a ZZFeatureMap QSVC under [0, 1] feature scaling with linear entanglement. They reported 94.09% for the QSVC and 97.31% for an accompanying Quanvolutional Neural Network. Neither encoding range nor entanglement topology was varied, and no statistical validation or deployment was described. The present work extends the same architecture with a systematic circuit design study.

Jain et al. [6] applied ResNet50 with Principal Component Analysis at 8 components and an 8-qubit ZZFeatureMap QSVC to two ECG image datasets from the same clinical source, finding QSVC outperformed classical SVM by roughly 8% and a quantum convolutional neural network exceeded its classical counterpart by roughly 5%, with significance assessed via the Kruskal-Wallis H-test. Neither encoding range nor entanglement topology was varied in those experiments. Ramkhelawan et al. [8] combined ResNet50 with a 10-qubit variational quantum circuit in PennyLane for binary arrhythmia detection, reporting 99.98% accuracy on 123,998 MIT-BIH and PTB images. The binary setup is structurally simpler than four-class classification, and variational circuits trained end-to-end carry a barren plateau risk that the fixed fidelity kernel approach avoids.

Ozpolat and Karabatak [9] benchmarked QSVM against classical SVM on signal-domain Chapman ECG features across qubit counts from 3 to 9, finding a consistent but modest quantum margin of approximately 2 percentage points at the best configuration. Their observation that accuracy does not increase monotonically with qubit count is consistent with the repetitions ablation in Section VI-D. Aksoy and Ozpolat [10] compared SVM, QSVM, and Pegasos-QSVM on medical tabular datasets using [0, pi] MinMax scaling, independently arriving at the same encoding range that the ablation in Section VI-A identifies as optimal. Because their datasets share no feature structure with ECG images, this cross-domain convergence supports interpreting the [0, pi] result as a property of the ZZFeatureMap rotation structure rather than a dataset-specific artefact.

Soon et al. [11] proposed a hierarchical quantum ensemble model that stacks a quantum neural network alongside XGBoost and a LightGBM meta-classifier for tabular cardiovascular data, reporting 97% accuracy and 98% AUC. The architecture does not involve ECG images and does not ablate circuit design. Setiawan et al. [12] systematically reviewed 94 quantum machine learning healthcare studies and concluded that calibration analysis and statistical significance testing are almost entirely absent from the literature, a gap the present work addresses directly.

Table I positions this work against five prior quantum ECG and cardiac studies. Among the works surveyed here, none combines a systematic circuit design ablation, a McNemar significance test, Expected Calibration Error analysis, and a deployed clinical interface on an ECG image classification task.

TABLE I.  COMPARISON WITH PRIOR QUANTUM ECG AND CARDIAC WORKS

| Reference | Task | Method | Best Acc. | Ablation | Stat. Test | Deployed |
|---|---|---|---|---|---|---|
| Prabhu et al. [5] | 4-class ECG images | ResNet50+SVD+QSVC ([0,1], linear) | 94.09% | No | No | No |
| Jain et al. [6] | Multi-class ECG images | ResNet50+PCA+QSVC | Best-reported† | No | Kruskal-Wallis | No |
| Ramkhelawan et al. [8] | Binary arrhythmia | ResNet50+VQC (10-qubit) | 99.98% | No | No | No |
| Ozpolat & Karabatak [9] | Arrhythmia signals | PCA+QSVM (3–9 qubits) | 80.90% | Qubit sweep | No | No |
| Aksoy & Ozpolat [10] | Medical tabular | SVM/QSVM/Pegasos | QSVM best | No | No | No |
| **This work** | **4-class ECG images** | **ResNet50+SVD+QSVC ([0,π], circular)** | **94.62%** | **320 configs** | **McNemar p=0.0013** | **Yes** |

†Results in Jain et al. [6] are reported across two datasets with different splits; a single directly comparable accuracy figure is not available.

---

## III. PROPOSED METHODOLOGY

The pipeline operates in two phases. An offline training phase processes all 928 images, trains three classifiers, and serialises all model objects to disk. An online inference phase deserialises those objects and runs the same transformation chain on a single uploaded ECG image. Fig. 1 shows the complete data flow.

Fig. 1.  Pipeline block diagram. Left: offline training path from dataset through preprocessing, ResNet50 extraction, Truncated SVD, and MinMax scaling to three serialised classifiers. Right: online inference path from React upload through FastAPI, MobileNetV2 gatekeeper, the same feature chain, and the user-selected classifier to a diagnosis response.
*(Draw in draw.io following the inline description above; insert as Fig. 1 at the top of this column.)*

**A.  ECG Image Preprocessing**

Clinical ECG scans arrive with widely varying scan quality, background colour, grid density, and orientation. A seven-step OpenCV pipeline converts each image to a 340×340 pixel normalised grayscale array. The image is first converted to grayscale and binarised with OTSU adaptive thresholding; morphological dilation using a 15×5 kernel joins fragmented trace segments, after which the largest contour is detected and the image is cropped to its bounding box with 15 pixels of padding. A second OTSU threshold and conditional inversion then produce a consistent dark-trace-on-light-background representation across all scan types. Vertical grid lines are removed with a 1×40 morphological OPEN operation, and horizontal grid lines with a 40×1 kernel; the ECG trace is thicker than any grid line in both axes and therefore survives both subtractions intact. A 3×3 Gaussian blur and re-threshold at pixel value 30 suppress residual speckle, after which the image is resized to 340×340 using cv2.INTER\_AREA and divided by 255.0 to yield a float32 array in [0, 1].

**B.  ResNet50 Feature Extraction**

Feature extraction uses a ResNet50 [13] with frozen ImageNet weights, rebuilt through the Keras functional API to output at the pool1\_pool layer. That layer applies 3×3 max-pooling over the initial 7×7 convolutional stage (64 filters, stride 2), producing an 85×85×64 spatial activation map. Flattened, this gives 462,400 dimensions per image. The shallow layer captures low-level morphological structure such as QRS complex geometry, ST segment curvature, and P and T wave shape, without the high-level ImageNet object semantics encoded at deeper layers. The choice of extraction layer follows Prabhu et al. [5]; a layer-wise ablation against the deeper global average pooling output was not performed and is noted as a limitation in Section VII. Grayscale inputs are expanded to three channels by axis repetition before passing through ResNet50's per-channel normalisation. Extraction ran in batches of 8.

**C.  Dimensionality Reduction**

TruncatedSVD at n\_components = 9, fitted on the 742 training samples, compresses the 462,400-dimensional vectors to 9 dimensions. The choice of 9 components follows Prabhu et al. [5] and is confirmed by the ablation in Section VI-D. MinMaxScaler, also fitted on training data alone, normalises the 9-dimensional vectors to [0, 1]. For the quantum classifiers the scaled vector is then multiplied by pi before encoding, placing features in [0, pi]; the classical SVM receives the [0, 1] vector directly. Both fitted objects are serialised at training time and loaded once at API startup, so inference always reuses the training-time transformation without refitting.

**D.  Classical Support Vector Machine**

The classical SVM uses sklearn.svm.SVC [20] with an RBF kernel, C = 10, gamma = 'scale', and class\_weight = 'balanced' to account for the 227:138 imbalance between the largest and smallest training classes. A CalibratedClassifierCV wrapper with method = 'isotonic' and cv = 3 produces calibrated class probabilities. Hyperparameters were selected via 5-fold stratified GridSearchCV maximising macro F1 over C in {0.01, 0.1, 1.0, 5.0, 10.0} and gamma in {'scale', 'auto'}.

**E.  Quantum Support Vector Classifier**

The QSVC encodes the 9-dimensional vector into a 2^9 = 512-dimensional complex Hilbert space using Qiskit's ZZFeatureMap [4], the Pauli-ZZ construction of Havlicek et al. Each repetition applies a Hadamard layer across all 9 qubits, first-order Rz(2xi) rotations encoding each feature as a phase angle, and second-order entangling blocks applying Rz(2(pi - xi)(pi - xj)) to each adjacent qubit pair under the selected topology. The fidelity quantum kernel is defined as

K(x, z) = |<ψ(x)|ψ(z)>|²     (1)

where |ψ(x)> is the statevector produced by encoding input x. Rather than running N² circuit evaluations, all N = 742 training statevectors are precomputed once and the kernel matrix is assembled as

K[i, j] = |sv_i · conj(sv_j)|²     (2)

where sv_i is the 512-dimensional complex amplitude vector for sample i. Full matrix construction for 742 samples takes under 15 seconds on CPU. At inference, a single kernel row is evaluated against the 742 cached statevectors through dense complex inner products, costing O(N·d) arithmetic for d = 512; no circuit simulation occurs at query time.

A grid search over four encoding ranges, five entanglement topologies, reps in {1, 2, 3, 4}, and C in {0.1, 1.0, 5.0, 10.0} identified [0, pi] encoding, circular entanglement, reps = 2, and C = 5.0 as the optimal configuration.

**F.  Multiclass Pegasos QSVC**

Six one-versus-one binary Pegasos QSVC models [14] cover all four-class pairs. The native Qiskit PegasosQSVC fails numerically at 9 qubits because the average kernel value at this scale is approximately 1/512, causing the SGD step size 1/(lambda·t) to collapse toward zero within the first iterations. A custom training loop precomputes all statevectors and evaluates each gradient step through matrix multiplication, with per-model grid search over regularisation and iteration count recovering accuracy from 78.49% to 91.94%. At inference, Algorithm 1 of Prabhu et al. [5] evaluates three of the six binary models in a decision tree to produce the four-class prediction.

**G.  Gatekeeper and Web Application**

A MobileNetV2 [15] gatekeeper rejects non-ECG uploads before any computation begins. The FastAPI backend loads all model objects at startup and exposes six REST endpoints covering single-image prediction, batch prediction for up to 20 images, PDF report generation, a three-model consensus comparison, a health check, and a paginated audit history endpoint. A Redis cache keyed by the SHA-256 hash of the image bytes plus the model identifier returns stored results for repeated submissions with a 24-hour time-to-live. The React frontend provides five tabs for Diagnosis, Batch Upload, Model Performance, Quantum Insights, and API documentation, with colour-coded severity indicators and calibrated confidence bars.

Fig. 2.  Web application screens. (a) Upload interface showing drag-and-drop ECG area, three model selector buttons, and the five-step pipeline summary. (b) Diagnosis result for a Myocardial Infarction ECG with 99.5% QSVC confidence, critical-severity colour coding, and emergency recommendation. (c) Pipeline dataflow panel showing per-stage timing and multi-model consensus table.
*(Use images: (a) extracted\_images/word/media/image1.png, (b) extracted\_images/word/media/image2.png, (c) extracted\_images/word/media/image3.png)*

---

## IV. EXPERIMENTAL SETUP

The experiments use the ECG image collection released on Mendeley Data by Khan et al. [16], collected at the Ch. Pervaiz Elahi Institute of Cardiology, Multan, Pakistan. The full collection (v2) contains 1937 paper ECG prints across five categories, including COVID-19 cases that are outside the scope of this study. Following Prabhu et al. [5], we isolate a 928-image four-class subset covering Normal, Arrhythmia, Myocardial Infarction, and History of Myocardial Infarction from the v2 release by filtering out the COVID-19 cases and selecting the remaining cardiac images. Table II shows the class distribution and the stratified 80:20 split applied with random\_state = 42. No augmentation was applied during training; augmented images appear only in the robustness evaluation in Section V-D.

TABLE II.  DATASET CLASS DISTRIBUTION AND TRAIN/TEST SPLIT

| Class | Total | Train | Test |
|---|---|---|---|
| Normal | 284 | 227 | 57 |
| Arrhythmia | 233 | 186 | 47 |
| Myocardial Infarction | 239 | 191 | 48 |
| History of MI | 172 | 138 | 34 |
| **Total** | **928** | **742** | **186** |

All models were evaluated on the same held-out 186-image test set. The TruncatedSVD reducer and MinMaxScaler were fitted on the 742 training samples and serialised; the test set passed through the same fitted objects without refitting, preventing data leakage. The primary metrics are accuracy and macro-averaged F1 score, with per-class and macro-averaged ROC-AUC also reported. McNemar's test [17] is applied to all three pairwise model comparisons. Expected Calibration Error (ECE) is computed with 15 equal-width probability bins.

Because the test set contains 186 images, a single reclassified sample shifts accuracy by 0.54 percentage points. Differences of this magnitude are reported for completeness but are not interpreted as evidence of a real performance gap; the ablation effects discussed in Section VI range from 4 to 26 percentage points, corresponding to 8 to 49 samples.

Kernel Target Alignment (KTA) [18] was computed for the optimal QSVC configuration, giving 0.1081. The value is positive, indicating agreement in sign between the kernel matrix and the label structure, but we report it as a descriptive statistic only: without a matched KTA value for the classical RBF kernel on the same features, the absolute magnitude does not by itself establish that the quantum kernel is better aligned.

All experiments ran on a standard CPU (Intel Core i5, 16 GB RAM). Quantum circuits were simulated with the Qiskit Aer 0.13 statevector simulator [19]. The software stack uses Python 3.10, TensorFlow 2.15, Scikit-learn 1.3 [20], and Qiskit Machine Learning 0.7. Random Forest and K-Nearest Neighbour classifiers were also trained on the same 9-dimensional MinMax-scaled features and included in Table III as additional classical reference points.

---

## V. RESULTS AND DISCUSSION

**A.  Model Performance**

Table III reports accuracy, precision, recall, and macro F1 for all five classifiers on the held-out 186-image test set.

TABLE III.  MODEL PERFORMANCE ON 186-IMAGE TEST SET

| Model | Accuracy | Precision | Recall | Macro F1 | Prabhu et al. [5] |
|---|---|---|---|---|---|
| Classical SVM (C=10, RBF) | 84.95% | 84.17% | 84.41% | 84.10% | 83.33% |
| **QSVC (ours)** | **94.62%** | **94.84%** | **94.05%** | **94.39%** | 94.09% |
| Pegasos QSVC | 91.94% | 91.99% | 91.61% | 91.31% | 93.05% |
| Random Forest | 91.94% | [VERIFY] | [VERIFY] | 91.20% | — |
| KNN | 84.95% | [VERIFY] | [VERIFY] | 83.61% | — |

`[VERIFY: replace the [VERIFY] precision/recall cells for Random Forest and KNN with your actual metrics, or remove those two columns for those rows. Do not submit without real numbers here.]`

The tuned QSVC leads the four other classifiers, with a margin of 9.67 percentage points over the classical SVM. The classical SVM improves 1.62 points on the base paper baseline [5], which we attribute to the addition of balanced class weights and isotonic calibration. Pegasos falls 1.11 points below the 93.05% reported by Prabhu et al.; the likely cause is stricter leakage control, since the SVD reducer and scaler here are fitted on training data only and the base paper's implementation is not fully specified on this point. Random Forest reaching 91.94% without any quantum component is worth noting on its own: the 9-dimensional SVD features already carry strong discriminative signal, and the quantum kernel is improving on an informative representation rather than rescuing a weak one.

The 0.53-point margin over the QSVC figure reported by Prabhu et al. corresponds to a single test image and we draw no conclusion from it. The comparison that carries weight is the internal one against our own reproduced baseline, discussed next.

Re-running the base paper configuration — [0, 1] encoding with linear entanglement — on the present pipeline gives 89.78%, against the 94.09% Prabhu et al. [5] report. The exact split and random seed used in that work are not published, so the discrepancy cannot be resolved from the paper alone. The ablation gains reported in Section VI are therefore measured against our own reproduced baseline, which keeps the circuit design comparisons internally consistent regardless of this implementation gap.

Fig. 3.  Confusion matrix for the tuned QSVC on 186 test samples. The classifier achieves 100% recall on Myocardial Infarction (48 of 48 correctly identified). Arrhythmia is the most frequently confused class across all models due to morphological variability in rhythm abnormality ECGs.
*(File: results/paper/main/confusion\_matrix\_qsvc\_tuned.png)*

The QSVC confusion matrix is shown in Fig. 3. It achieves 100% recall on Myocardial Infarction, the class where a missed diagnosis carries the greatest clinical cost. Arrhythmia shows the most inter-class confusion, consistent with the morphological variability of rhythm abnormalities and their proximity to Normal ECG traces near classification boundaries.

**B.  Statistical Significance**

Table IV reports McNemar's test results for all three pairwise model comparisons. The discordant pair counts are drawn directly from the pairwise prediction contingency tables.

TABLE IV.  McNEMAR'S TEST RESULTS (n = 186 TEST SAMPLES)

| Comparison | Correct only A (b) | Correct only B (c) | χ² | p-value | Significant |
|---|---|---|---|---|---|
| **QSVC vs. Classical SVM** | **23** | **5** | **10.321** | **0.0013** | **Yes** |
| Pegasos vs. Classical SVM | 19 | 6 | 5.760 | 0.0164 | Yes |
| QSVC vs. Pegasos | 9 | 4 | 1.231 | 0.267 | No |

The QSVC corrected 23 samples that the classical SVM got wrong while missing only 5 that the SVM got right. With continuity correction, chi-squared = (|23 − 5| − 1)² / (23 + 5) = 289/28 = 10.32 at p = 0.0013, establishing that the difference is a systematic property of the models rather than an artefact of this particular split. Both quantum models are statistically superior to classical SVM at the 0.05 threshold.

The third row is equally informative. QSVC and Pegasos are not statistically distinguishable at n = 186 despite a 2.68 percentage point accuracy gap, reflecting the limited statistical power of a test set this size. We report the gap but make no claim that the fidelity kernel method is superior to the Pegasos variant. To the best of our knowledge, this is the first McNemar significance test of quantum kernel advantage over classical SVM reported for ECG image classification.

**C.  ROC-AUC and Confidence Calibration**

Fig. 4 shows ROC curves for all three models across the four classes, and Table V reports the corresponding AUC scores.

TABLE V.  PER-CLASS AND MACRO ROC-AUC SCORES

| Class | SVM | QSVC | Pegasos |
|---|---|---|---|
| Normal | 0.929 | 0.997 | 0.972 |
| Arrhythmia | 0.912 | 0.974 | 0.912 |
| Myocardial Infarction | 0.987 | **1.000** | **1.000** |
| History of MI | 0.927 | 0.984 | 0.979 |
| **Macro-AUC** | 0.940 | **0.989** | 0.967 |

Fig. 4.  ROC curves for QSVC, Pegasos, and Classical SVM across all four ECG classes. Both quantum models achieve AUC = 1.000 on Myocardial Infarction, ranking every MI sample above every non-MI sample at every threshold without exception.
*(File: results/paper/main/roc\_curves\_all\_models.png)*

Both quantum models achieve AUC = 1.000 on Myocardial Infarction across the 48 MI test samples, and the QSVC macro-AUC of 0.989 is 0.049 above the classical SVM's 0.940. Expected Calibration Error falls below 0.07 macro ECE for all three models: SVM achieves 0.054, QSVC 0.063, and Pegasos 0.067. A model reporting 80% confidence is therefore correct approximately 80% of the time, a property required before a system can meaningfully support clinical decisions. Calibration analysis of this kind was not found in any of the ECG quantum machine learning papers surveyed in Section II.

**D.  Augmentation Robustness**

Fig. 5 shows model accuracy under nine augmentation conditions. Under brightness perturbation, which simulates an overexposed photograph of a paper ECG print, classical SVM accuracy collapses from 84.4% to 46.7%, near random performance on four classes. The QSVC holds 86.7% under the same perturbation.

Fig. 5.  Model accuracy under nine augmentation conditions. Classical SVM collapses to 46.7% under brightness perturbation; QSVC maintains 86.7% due to the phase-encoding mechanism of the ZZFeatureMap.
*(File: results/paper/ablation/augmentation\_robustness.png)*

The mechanism is the encoding. ZZFeatureMap encodes feature values as phase angles, and phase is periodic, so a uniform shift in scaled pixel statistics moves the encoded quantum state around the Bloch sphere rather than directly away from the training support vectors. The RBF kernel operates on Euclidean distance in the same 9-dimensional compressed feature space, where the same shift translates every test sample away from its nearest training neighbours. Rotation, affine distortion, and aspect ratio change affect both models similarly. Brightness is the one perturbation axis where the phase encoding creates a measurable advantage — and it is also the axis that varies most in practice, since ward ECG prints are photographed under whatever light is available.

---

## VI. ABLATION STUDIES

**A.  Feature Encoding Range**

Table VI compares QSVC and Pegasos accuracy across four encoding ranges. All rows use circular entanglement, reps = 2, and C = 5.0 so that only the encoding range varies.

TABLE VI.  QSVC AND PEGASOS ACCURACY VS. FEATURE ENCODING RANGE
(all rows: circular entanglement, reps = 2, C = 5.0)

| Encoding | QSVC Accuracy | Pegasos Accuracy |
|---|---|---|
| L2 Normalisation (unit sphere) | 68.28% | 57.53% |
| MinMax [0, 1] (base paper range) | 92.47% | 84.41% |
| **MinMax [0, pi] (ours)** | **94.62%** | **90.86%** |
| MinMax [0, 2pi] | 94.09% | 90.32% |

Fig. 6.  QSVC accuracy across four feature encoding ranges. All configurations use circular entanglement. L2 normalisation discards feature magnitude, producing the largest accuracy drop. Moving to [0, 2pi] introduces phase wrap-around and gives back part of the gain.
*(File: results/paper/ablation/ablation\_encoding.png)*

The ZZFeatureMap applies Rz(2xi) rotations, so a feature in [0, 1] sweeps approximately 2 radians of Bloch sphere latitude, roughly one-third of the full circle available to the gate. Extending the range to [0, pi] opens that sweep to the full 2pi, distributing encoded states more widely across the Hilbert space and increasing the spread of off-diagonal kernel values. At [0, 2pi], rotation angles exceed 2pi and wrap around, so distinct feature values begin mapping to similar quantum states; this shows up as a 0.53-point drop from the [0, pi] result, which on a 186-image test set is a single sample and should be read as the two configurations performing comparably rather than as a ranked difference.

L2 normalisation is the clear outlier, 26.34 points below the [0, pi] configuration and 24.19 points below [0, 1]. Projecting onto the unit sphere discards feature magnitude entirely, and magnitude is precisely what the dominant SVD components encode.

The 92.47% value at [0, 1] uses circular entanglement rather than the base paper's linear topology. The base paper's full configuration — [0, 1] with linear entanglement — corresponds to the 89.78% row in Table VII.

**B.  Entanglement Topology**

Table VII compares QSVC accuracy across five entanglement topologies. All rows use [0, pi] encoding, reps = 2, and C = 5.0.

TABLE VII.  QSVC ACCURACY VS. ENTANGLEMENT TOPOLOGY
(all rows: [0, pi] encoding, reps = 2, C = 5.0)

| Topology | Accuracy | Gain vs. Linear |
|---|---|---|
| Linear | 89.78% | — |
| Pairwise | 93.01% | +3.23 pp |
| Full | 92.47% | +2.69 pp |
| **Circular (ours)** | **94.62%** | **+4.84 pp** |
| SCA | 94.62% | +4.84 pp |

Fig. 7.  QSVC accuracy for five entanglement topologies. Circular topology gives each qubit two entangled neighbours at a circuit depth barely greater than linear, producing the richest pairwise kernel without over-parameterisation.
*(File: results/paper/ablation/ablation\_entanglement.png)*

Linear entanglement connects each qubit to one neighbour along a chain. Circular topology closes the chain by connecting qubit 8 back to qubit 0, giving every qubit two entangled partners and encoding richer pairwise correlations between SVD components at a circuit depth barely greater than linear. Full entanglement, which connects every possible qubit pair, yields lower accuracy at 92.47%, plausibly because the much larger number of entangling gates at reps = 2 over-parameterises the kernel for a 9-dimensional input, though we have not isolated this mechanism. For 9 qubits the SCA topology reduces to the same connectivity as circular, hence the identical score.

Reading the two tables together separates the contributions. Moving from [0, 1] to [0, pi] with circular entanglement held fixed gains 2.15 percentage points (92.47% to 94.62%, Table VI). Moving from linear to circular entanglement with [0, pi] held fixed gains 4.84 percentage points (89.78% to 94.62%, Table VII). The combined move from the base paper configuration to ours also yields 4.84 points overall, which means the two effects overlap substantially rather than stacking: the topology carries most of the gain once the encoding range is no longer the limiting factor.

**C.  Pauli Feature Map Comparison**

Table VIII compares four Pauli feature map variants under identical conditions: 9 qubits, reps = 2, circular entanglement, [0, pi] encoding, and C = 5.0.

TABLE VIII.  PAULI FEATURE MAP ACCURACY COMPARISON
(9 qubits, reps = 2, circular, [0, pi], C = 5.0)

| Feature Map | Operator Structure | Accuracy | Macro F1 |
|---|---|---|---|
| ZFeatureMap | Single-qubit Rz, no entanglement | 87.63% | 0.871 |
| **ZZFeatureMap (ours)** | **Pauli-Z + ZZ interactions** | **94.62%** | **0.944** |
| PauliFeatureMap (X+XX) | Pauli-X + XX interactions | 91.94% | 0.913 |
| PauliFeatureMap (Y+YY) | Pauli-Y + YY interactions | 83.33% | 0.835 |

Fig. 8.  QSVC accuracy for four Pauli feature map variants under identical configuration. The unentangled ZFeatureMap drops 6.99 pp below ZZFeatureMap, directly quantifying the contribution of the ZZ entangling gates.
*(File: results/paper/ablation/ablation\_feature\_maps.png)*

ZFeatureMap applies only single-qubit Rz rotations with no entanglement and reaches 87.63%, which is 6.99 percentage points below ZZFeatureMap. That gap quantifies the contribution of the ZZ entangling interactions directly: removing entanglement while holding everything else fixed costs 13 test samples. PauliFeatureMap with X+XX interactions adds entanglement through a different gate basis and reaches 91.94%, above the unentangled result but 2.68 points short of ZZFeatureMap.

The Y+YY variant is the outlier at 83.33% — below both the unentangled map and the classical SVM baseline. Ry-based rotations combined with YY interactions appear to produce destructive interference in the kernel for this feature distribution, but we have not isolated the mechanism and do not claim more than the empirical observation from a single dataset.

**D.  SVD Components and Circuit Repetitions**

Fig. 9.  (Top) QSVC accuracy versus number of SVD components retained. Accuracy peaks at 9 components, consistent with Prabhu et al. [5]. (Bottom) Accuracy versus circuit repetitions. reps = 2 is optimal; additional repetitions add simulation cost without improving accuracy.
*(Files: results/paper/ablation/ablation\_svd\_dims.png and ablation\_reps.png)*

Sweeping SVD components from 2 to 18 shows accuracy peaking at 9, confirming the choice made by Prabhu et al. [5]. Below 9 components, discriminative variance from the ResNet50 activation map is discarded; above 9, the additional components carry noise relative to the encoding capacity, and the qubit count required grows with them. Sweeping circuit repetitions confirms reps = 2 as the optimal depth: reps = 3 and reps = 4 show no accuracy gain while adding statevector computation cost, consistent with the finding of Ozpolat and Karabatak [9] that increasing circuit depth beyond a sufficient encoding level does not improve quantum SVM performance.

---

## VII. LIMITATIONS

Four constraints bound the conclusions of this study.

All results come from a single dataset collected at one institution, and from a single stratified split at random\_state = 42. No repeated runs across seeds were performed, so no variance estimates accompany the reported accuracies. Differences below roughly one percentage point, which correspond to one or two test images, should be treated as noise rather than as evidence of a real performance gap.

Generalisation across scanner environments is limited. In a zero-shot transfer to a second Mendeley ECG dataset from the same clinical source, QSVC accuracy fell to 34% on a three-class evaluation while the classical SVM degraded more gradually to 63%. The quantum kernel appears to learn a precise but scanner-specific decision boundary; the advantage measured in Sections V and VI is therefore established for the training domain and not beyond it.

All quantum results come from ideal statevector simulation. No noise model was applied and no hardware execution was attempted, so the reported accuracies are an upper bound on what a NISQ device would deliver at 9 qubits.

The extraction layer was adopted from Prabhu et al. [5] rather than selected empirically. No layer-wise ablation comparing pool1\_pool against deeper ResNet50 outputs was run, so the claim that shallow morphological features suit this task rests on the prior work rather than on evidence presented here.

---

## VIII. CONCLUSION

This paper showed that two quantum circuit design parameters have a large and measurable effect on QSVC accuracy for four-class ECG image classification. Scaling the 9-dimensional SVD-compressed features to [0, pi] and switching from linear to circular entanglement raises accuracy from 89.78% to 94.62% on a 928-image clinical subset of the Khan et al. [16] ECG collection. The topology change accounts for the dominant portion of the gain: holding the encoding at [0, pi], circular entanglement outperforms linear by 4.84 percentage points, while the encoding change alone with topology held fixed contributes 2.15. The Rz gate structure explains the encoding effect, since [0, 1] scaling sweeps roughly one-third of the available Bloch sphere latitude where [0, pi] sweeps the full range.

McNemar's test confirms the quantum advantage over a well-tuned RBF SVM at chi-squared = 10.32, p = 0.0013. Both quantum models achieve AUC = 1.000 for Myocardial Infarction. Expected Calibration Error below 0.07 across all three models indicates that the confidence scores are suitable for clinical decision support. The QSVC holds 86.7% accuracy under brightness perturbation that collapses the classical SVM to near-random performance, a difference that follows from the phase encoding of the ZZFeatureMap. The Pauli map ablation confirms that ZZ entanglement specifically outperforms unentangled, X+XX, and Y+YY alternatives.

The limitations in Section VII set the agenda for what follows. Domain adaptation of the ResNet50 extractor is the most urgent practical step, since the cross-dataset result shows the learned boundary does not survive a change of scanner. Evaluating the trained classifier under NISQ noise models and eventually on IBM Quantum hardware would quantify the gap between ideal simulation and near-term devices. Repeating the ablations across multiple seeds would attach variance estimates to the effects reported here. Adding Grad-CAM visualisation over the pool1\_pool feature map would provide clinician-interpretable regions of interest, and implementing the quanvolutional neural network of Prabhu et al. [5] under the same circuit design discipline would complete the comparison.

---

## CODE AND DATA AVAILABILITY

The ECG image dataset used in this study is publicly available on Mendeley Data [16]. The implementation, trained model artefacts, and scripts reproducing all tables and figures are available at *(repository URL)*. `[Add anonymised or public link before submission — a reproducibility correction paper without a code link is conspicuous.]`

---

## ACKNOWLEDGEMENT

The authors thank the Qiskit open-source community and IBM Quantum for the quantum computing framework. The ECG dataset was made publicly available by Khan et al. through Mendeley Data, collected under the auspices of the Ch. Pervaiz Elahi Institute of Cardiology, Multan. The authors acknowledge the support of the Department of Computer Science and Engineering, St. Joseph Engineering College, Mangaluru.

---

## REFERENCES

*(Numbered in order of first appearance in text, as IEEE style requires.)*

[1] World Health Organization, "Cardiovascular diseases (CVDs)," Fact Sheet, Jun. 2021. [Online]. Available: https://www.who.int/news-room/fact-sheets/detail/cardiovascular-diseases-(cvds) [Accessed: add date before submission].

[2] F. Vasquez-Iturralde, M. J. Flores-Calero, F. Grijalva, and A. Rosales-Acosta, "Automatic classification of cardiac arrhythmias using deep learning techniques: A systematic review," *IEEE Access*, vol. 12, 2024, doi: 10.1109/ACCESS.2024.3408282.

[3] O. Yildirim, P. Plawiak, R.-S. Tan, and U. R. Acharya, "Arrhythmia detection using deep convolutional neural network with long duration ECG signals," *Comput. Biol. Med.*, vol. 102, pp. 411–420, 2018, doi: 10.1016/j.compbiomed.2018.09.009.

[4] V. Havlicek, A. D. Corcoles, K. Temme, A. W. Harrow, A. Kandala, J. M. Gambetta, and J. Preskill, "Supervised learning with quantum-enhanced feature spaces," *Nature*, vol. 567, pp. 209–212, Mar. 2019, doi: 10.1038/s41586-019-0980-2.

[5] S. Prabhu, S. Gupta, G. M. Prabhu, A. V. Dhanuka, and K. V. Bhat, "QuCardio: Application of quantum machine learning for detection of cardiovascular diseases," *IEEE Access*, vol. 11, pp. 136835–136851, 2023, doi: 10.1109/ACCESS.2023.3338145.

[6] V. Jain, N. Arora, and A. Gupta, "Quantum-assisted cardiac diseases diagnosis and prediction using ECG images," *J. Supercomputing*, vol. 81, p. 1439, 2025, doi: 10.1007/s11227-025-07939-8.

[7] J. Biamonte, P. Wittek, N. Pancotti, P. Rebentrost, N. Wiebe, and S. Lloyd, "Quantum machine learning," *Nature*, vol. 549, pp. 195–202, Sep. 2017, doi: 10.1038/nature23474.

[8] M. Y. Ramkhelawan, S. Grandhi, and S. Wibowo, "A hybrid quantum-classical deep learning algorithm for efficient arrhythmia classification," in *Proc. IEEE SNPD*, 2025, doi: 10.1109/SNPD62189.2025.10985682.

[9] Z. Ozpolat and M. Karabatak, "Performance evaluation of quantum-based machine learning algorithms for cardiac arrhythmia classification," *Diagnostics*, vol. 13, no. 6, p. 1099, 2023, doi: 10.3390/diagnostics13061099.

[10] I. Aksoy and Z. Ozpolat, "Comparison of SVM, QSVM and Pegasos-QSVM algorithms on different medical datasets," *Int. J. Sustain. Eng. Technol.*, vol. 9, no. 1, pp. 80–93, 2025, doi: 10.62301/usmtd.1716034.

[11] K. L. Soon, W. L. Pang, H. H. Goh, Y. W. Sim, S. K. Phang, H. L. Choo, L. T. Soon, and N. S. Lai, "Early cardiovascular disease detection using hierarchical quantum ensemble model," *Comput. Methods Biomech. Biomed. Eng.*, Jan. 2026, doi: 10.1080/10255842.2025.2612536.

[12] A. E. Setiawan, S. Rustad, A. Syukur, M. A. Soeleman, G. F. Shidik, M. Akrom, and A. W. Setiawan, "A systematic literature review of quantum machine learning for medical: trends, datasets, topics, and methods," *Int. J. Cogn. Comput. Eng.*, vol. 7, pp. 609–630, 2026, doi: 10.1016/j.ijcce.2026.05.002.

[13] K. He, X. Zhang, S. Ren, and J. Sun, "Deep residual learning for image recognition," in *Proc. IEEE/CVF CVPR*, Jun. 2016, pp. 770–778, doi: 10.1109/CVPR.2016.90.

[14] S. Shalev-Shwartz, Y. Singer, N. Srebro, and A. Cotter, "Pegasos: Primal estimated sub-gradient solver for SVM," *Math. Program.*, vol. 127, no. 1, pp. 3–30, 2011, doi: 10.1007/s10107-010-0420-4.

[15] M. Sandler, A. Howard, M. Zhu, A. Zhmoginov, and L.-C. Chen, "MobileNetV2: Inverted residuals and linear bottlenecks," in *Proc. IEEE/CVF CVPR*, Jun. 2018, pp. 4510–4520, doi: 10.1109/CVPR.2018.00474.

[16] A. H. Khan, M. Hussain, and M. K. Malik, "ECG images dataset of cardiac and COVID-19 patients," *Data in Brief*, vol. 34, p. 106762, 2021, doi: 10.1016/j.dib.2021.106762.

[17] Q. McNemar, "Note on the sampling error of the difference between correlated proportions or percentages," *Psychometrika*, vol. 12, no. 2, pp. 153–157, 1947, doi: 10.1007/BF02295996.

[18] N. Cristianini, J. Shawe-Taylor, A. Elisseeff, and J. Kandola, "On kernel-target alignment," in *Adv. Neural Inf. Process. Syst.*, vol. 14, 2002, pp. 367–373.

[19] Qiskit contributors, "Qiskit: An open-source framework for quantum computing," Zenodo, 2023, doi: 10.5281/zenodo.2573505.

[20] F. Pedregosa, G. Varoquaux, A. Gramfort, V. Michel, B. Thirion, O. Grisel, M. Blondel, P. Prettenhofer, R. Weiss, V. Dubourg, J. Vanderplas, A. Passos, D. Cournapeau, M. Brucher, M. Perrot, and E. Duchesnay, "Scikit-learn: Machine learning in Python," *J. Mach. Learn. Res.*, vol. 12, pp. 2825–2830, 2011.

---

## FIGURE PLACEMENT GUIDE
*(Remove this section before LaTeX compilation — for typesetting reference only)*

| Fig. | Content | File | Where |
|---|---|---|---|
| 1 | Pipeline block diagram (offline + online) | Draw in draw.io per Sec. III description | Top col. 1, Sec. III |
| 2 | Three-panel app screenshot (a) upload (b) result (c) pipeline/consensus | image1.png, image2.png, image3.png | Bottom col. 2, Sec. III-G |
| 3 | QSVC confusion matrix 4×4 | results/paper/main/confusion\_matrix\_qsvc\_tuned.png | Top col. 1, Sec. V-A |
| 4 | ROC curves all models 4 classes | results/paper/main/roc\_curves\_all\_models.png | Top col. 2, Sec. V-C |
| 5 | Augmentation robustness bar chart | results/paper/ablation/augmentation\_robustness.png | Top col., Sec. V-D |
| 6 | Encoding range ablation bar chart | results/paper/ablation/ablation\_encoding.png | Top col., Sec. VI-A |
| 7 | Entanglement topology ablation bar chart | results/paper/ablation/ablation\_entanglement.png | Top col., Sec. VI-B |
| 8 | Pauli feature map ablation bar chart | results/paper/ablation/ablation\_feature\_maps.png | Top col., Sec. VI-C |
| 9 | SVD dims (top) + reps (bottom) two-panel | ablation\_svd\_dims.png + ablation\_reps.png | Bottom col., Sec. VI-D |

**6-page conference trim:** Drop Fig. 2 (app screenshots — describe deployment in two sentences in Sec. III-G instead), Fig. 4 (state ECE numbers in text only), Fig. 9 (state SVD=9 and reps=2 with numbers in one sentence). Keep TABLE VIII — Pauli comparison is core ablation, not optional. Compress Section VII to one paragraph but do not remove it.

**10-page journal:** Include everything above plus a calibration figure (results/paper/main/calibration\_all\_models.png) after Fig. 4, and expand the cross-dataset result from Section VII into a full Section V-E with its own table and confusion matrices (results/paper/cross\_dataset/jain\_dataset2/).

---

## IN-TEXT CITATION QUICK-REFERENCE
*(Remove before LaTeX compilation)*

| Body text symbol | Reference number | Entry |
|---|---|---|
| Soon et al. | [11] | Early CVD hierarchical ensemble |
| Setiawan et al. | [12] | SLR quantum ML healthcare |
| He et al. (ResNet50) | [13] | Deep residual learning |
| Shalev-Shwartz et al. (Pegasos) | [14] | Pegasos SVM |
| Sandler et al. (MobileNetV2) | [15] | MobileNetV2 |
| Khan et al. (dataset) | [16] | ECG dataset — Data in Brief |
| McNemar | [17] | McNemar test |
| Cristianini et al. (KTA) | [18] | Kernel-target alignment |
| Qiskit | [19] | Qiskit framework |
| Pedregosa et al. (scikit-learn) | [20] | scikit-learn |

*Report content is in Doc/Report/QuCardio\_Report\_Content\_REVISED.md. NOTE: the report's Table 4.1 uses slightly different dataset counts in places — reconcile both documents to the numbers in Table II above (928 / 742 / 186, Arrhythmia 233, History of MI 172) before either is submitted.*
