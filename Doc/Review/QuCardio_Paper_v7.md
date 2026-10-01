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

Quantum machine learning has been applied to automated ECG analysis, but prior studies frequently report isolated accuracy figures without parametric ablation, statistical validation, or confidence calibration. This paper presents a hybrid classical-quantum pipeline for four-class ECG image classification and systematically ablates five quantum circuit variables. We extract features from a frozen ResNet50, compress them to nine dimensions via TruncatedSVD, and classify them with a nine-qubit ZZFeatureMap fidelity kernel. On a 928-image clinical dataset, the tuned Quantum Support Vector Classifier reaches 94.62% accuracy. A $2 \times 2$ factorial ablation shows that expanding the feature encoding range to $[0, \pi]$ and closing the qubit chain into a circular topology each improve accuracy, yielding a joint 4.84 percentage-point gain over the $[0, 1]$ linear baseline. The quantum kernel achieves a significant margin over a classical SVM ($\chi^2 = 10.32, p = 0.0013$), though its 2.68-point margin over Random Forest is not statistically significant. The classifier attains complete Myocardial Infarction separation (AUC = 1.000), Expected Calibration Error below 0.07, and retains 86.7% accuracy under brightness perturbation where the classical SVM falls to 46.7%. Zero-shot transfer to a second scanner domain reverses the ordering, indicating that generalisation is currently bounded by classical feature extraction. The pipeline is deployed as an end-to-end clinical screening web application.

*Index Terms:* Quantum machine learning, quantum kernel methods, ZZFeatureMap, ECG classification, circuit ablation, statistical validation.

---

## I. INTRODUCTION

Cardiovascular diseases are the leading cause of death worldwide, responsible for approximately 17.9 million deaths annually according to World Health Organization estimates [1]. The term covers a broad group of conditions affecting the heart and blood vessels, including myocardial infarction, arrhythmia, coronary artery disease and stroke, with risk driven by a combination of smoking, hypertension, elevated cholesterol, obesity and sedentary lifestyle. The Electrocardiogram (ECG) remains the standard first-line, non-invasive tool for detecting these conditions. It captures the heart's electrical activity as a waveform trace, and trained cardiologists read that trace to identify Arrhythmia, Myocardial Infarction (MI) and prior History of Myocardial Infarction. Specialist interpretation of this kind is frequently unavailable in high-volume hospitals and rural clinics, which is what gives an automated classification system real clinical value.

Machine learning applied to ECG classification has progressed considerably over the past decade, moving from hand-crafted signal features toward representations learned directly by deep networks, a shift that mirrors the broader rise of deep learning and large labelled datasets across medical imaging. Progress specifically on ECG *image* classification, as opposed to classification of the raw digitised signal, has been slower, because clinical image datasets of this kind remain small and multi-class, and because the non-linear correlations between different regions of a waveform are difficult for a classical feature space of modest dimension to separate cleanly. It is this specific difficulty, rather than a general shortage of machine learning methods for cardiology, that motivates looking beyond purely classical feature spaces for the task addressed here.

Quantum machine learning offers one route around that limitation. A quantum feature map encodes classical data into the state of a multi-qubit system, and because that system occupies a Hilbert space whose dimension grows exponentially with qubit count, superposition and entanglement give the resulting quantum kernel access to a far richer set of non-linear decision boundaries than a compact classical kernel operating on the same input dimension. Present-day quantum hardware remains in the Noisy Intermediate-Scale Quantum (NISQ) era, in which qubit count, coherence time and gate fidelity are all limited, so near-term quantum machine learning research of this kind is evaluated almost exclusively on classical simulators rather than physical devices, as is the case throughout this study.

Prior work applying quantum machine learning to ECG classification has established that the approach is workable and frequently competitive with classical baselines, but it has generally done so by reporting a single accuracy figure under one fixed circuit configuration, typically a software framework's default entanglement topology paired with an unexamined encoding range. What is missing from that literature is any systematic account of which circuit design choices are actually responsible for the reported accuracy, whether the resulting margin over a classical baseline is statistically distinguishable from noise, and whether the model's confidence scores are calibrated well enough to be useful in a clinical decision.

This paper addresses that gap directly. Rather than treating the quantum feature map as a fixed and unexamined component, the classical portion of the pipeline is held constant while five circuit design variables are systematically varied: encoding range, entanglement topology, Pauli operator structure, circuit repetition depth and the number of retained classical features. Every configuration is evaluated on the same held-out test set, the resulting margin over classical baselines is tested for statistical significance, and the model's confidence scores are checked for calibration before the complete system is deployed as a web application intended for clinical screening use.

The main contributions of this study are summarised as follows:

1. **Controlled $2 \times 2$ factorial circuit ablation.** We show that scaling features to $[0, \pi]$ and closing the entanglement chain into a ring each raise accuracy at both levels of the other factor, and that their joint effect is sub-additive by 4.84 against 5.92 percentage points. The result is consistent with full $2\pi$ phase coverage on the Bloch sphere equator and ring connectivity acting on an overlapping set of misclassified samples.
2. **Statistical validation against classical baselines.** We apply McNemar's test to every pairwise model comparison, including the strongest classical baseline. To the best of our knowledge this is the first significance test of quantum kernel advantage reported for ECG image classification.
3. **Calibration and perturbation robustness.** We report Expected Calibration Error for all three deployed models, all below 0.07, and measure accuracy under nine scan-quality perturbations, where phase encoding retains 86.7% under brightness distortion against 46.7% for the classical SVM.
4. **Domain boundary.** We report a zero-shot cross-scanner evaluation in which the ordering reverses, which identifies feature-extractor domain adaptation as the blocking step for deployment beyond the training scanner.
5. **Deployed system.** We release a web application with automated input gatekeeping, three-model consensus comparison and PDF report generation, together with the code and scripts that reproduce every table and figure.

The remainder of this paper is organised as follows. Section II reviews related work in classical ECG classification, the theoretical foundations of quantum kernel methods, and quantum machine learning for cardiac and ECG classification specifically. Section III describes the proposed methodology, covering image preprocessing, feature extraction, dimensionality reduction and the three classifiers evaluated. Section IV details the experimental setup, including the dataset, the evaluation protocol and the software and hardware environment. Section V presents the results, covering model performance, statistical significance, calibration, perturbation robustness and cross-dataset transfer. Section VI reports the ablation studies across all five circuit design variables. Section VII states the limitations of the study, and Section VIII concludes.

---

## II. RELATED WORK

This section reviews work in four areas relevant to the proposed system: classical deep learning for ECG classification, the theoretical foundations of quantum kernel methods, quantum machine learning applied specifically to cardiac and ECG classification, and a summary positioning the present study against that literature.

**A.  Classical ECG Classification**

Deep learning methods for ECG classification have progressed substantially over the past decade. Yildirim et al. [3] showed that a one-dimensional convolutional neural network applied directly to raw ECG waveforms achieves 91.33% accuracy across 17 arrhythmia classes from the MIT-BIH database, demonstrating that learned temporal representations can outperform hand-crafted signal features on large corpora. Vasquez-Iturralde et al. [2] reviewed 494 deep learning ECG publications using the PRISMA 2020 methodology and identified persistent gaps, among them the scarcity of studies operating on image-format ECG inputs rather than digital recordings, and the near-absence of any confidence calibration reporting.

Working directly with image-format ECGs is distinctly more challenging than signal processing. The lack of a digital waveform precludes standard frequency-domain techniques, requiring classifiers to extract morphological structures while ignoring binarisation artifacts, grid lines, and scan-quality variations. Consequently, the dominant classical approach applies transfer learning from large vision models; for example, Prabhu et al. [7] report a classical baseline using a frozen ResNet50 paired with an SVM, reaching 83.33% accuracy on a four-class clinical dataset. The present work operates on these more challenging scanned paper ECG prints and provides both calibration and statistical significance analyses, directly addressing the gaps identified in the broader literature.

**B.  Theoretical Foundations of Quantum Kernel Methods**

The theoretical basis for quantum kernel classification was established by Havlicek et al. [4], who constructed ZZFeatureMap-based kernels that map data into exponentially large Hilbert spaces through Pauli-ZZ interactions and argued that such kernels are not efficiently reproducible by a classical algorithm. Their demonstration used synthetic data constructed to favour the quantum kernel, and a rigorous learning separation was established only subsequently, for a specially constructed problem, so the advantage on natural datasets of the kind used here remains an empirical question rather than a proven one. The fidelity kernel used throughout this work is the construction that their paper introduced.

Huang et al. [5] sharpened that caution considerably. Their analysis shows that once training data is available, a classical model can often match a quantum model even on problems built to favour the quantum side, and they propose a geometric criterion for judging in advance whether a quantum advantage is plausible for a given dataset at all. That result is the reason the present work treats the accuracy margins it reports as evidence about one pipeline and one dataset, rather than as a claim of a general separation between quantum and classical learning. Biamonte et al. [6] surveyed quantum machine learning broadly and identified hybrid classical-quantum pipelines, in which a classical preprocessor handles dimensionality reduction before quantum encoding, as the most practical near-term architecture; that observation directly motivates the Truncated SVD stage described in Section III.

**C.  Quantum Machine Learning for Cardiac and ECG Classification**

Prabhu et al. [7] demonstrated quantum ECG classification on the same four-class clinical data used here, combining ResNet50 pool1\_pool features with Truncated SVD at nine components and a ZZFeatureMap QSVC under [0, 1] feature scaling. The paper does not state the entanglement topology; Qiskit's ZZFeatureMap default is full entanglement, so the configuration used was almost certainly [0, 1] + full. Re-running that configuration on the present pipeline gives 93.01%, reducing the reproduction gap to 1.08 percentage points; we attribute most of the remaining difference to the split and seed. They reported 94.09% for the QSVC, 93.05% for a Multiclass Pegasos QSVC and 97.31% for an accompanying Quanvolutional Neural Network. Neither the encoding range nor the entanglement topology was varied, and no statistical validation, calibration analysis or deployment was described. The present work retains that architecture and adds the circuit design study.

Jain et al. [8] applied ResNet50 with Principal Component Analysis at eight components and an eight-qubit ZZFeatureMap QSVC to two ECG image datasets from the same clinical source. One of those collections, covering three classes (Normal, Arrhythmia, and History of Myocardial Infarction), is exactly the second dataset we use for the cross-dataset transfer study in Section V-E. They reported that the QSVC outperformed a classical SVM by roughly eight percentage points, with significance assessed by a non-parametric rank test. While their PCA-based eight-component reduction differs slightly from our nine-component SVD, their result establishes a strong baseline on that specific dataset. However, neither the encoding range nor the entanglement topology was varied in their experiments, which leaves the circuit design space largely unexplored. Ramkhelawan et al. [9] combined ResNet50 with a ten-qubit variational quantum circuit in PennyLane for binary arrhythmia detection, reporting 99.98% accuracy on 123,998 MIT-BIH and PTB images. The binary setup is structurally simpler than four-class classification and the dataset is two orders of magnitude larger, and variational circuits trained end to end also carry a barren plateau risk that the fixed fidelity kernel approach used here avoids.

Ozpolat and Karabatak [10] benchmarked a QSVM against a classical SVM on signal-domain Chapman ECG features across qubit counts from three to nine, finding a consistent but modest margin of approximately two percentage points at the best configuration. Their observation that accuracy does not increase monotonically with qubit count is consistent with the circuit-repetition ablation reported in Section VI-E. Aksoy and Ozpolat [11] compared SVM, QSVM and Pegasos-QSVM on medical tabular datasets using MinMax scaling to [0, π], independently adopting the same encoding range that the ablation in Section VI-A identifies as optimal here. Because their datasets share no feature structure with ECG images, this cross-domain agreement supports reading the [0, π] result as a property of the ZZFeatureMap rotation structure rather than as a dataset-specific artefact.

Soon et al. [12] proposed a hierarchical quantum ensemble that stacks a quantum neural network alongside XGBoost with a LightGBM meta-classifier for tabular cardiovascular data, reporting 97% accuracy and 98% AUC; the architecture does not involve ECG images and does not ablate circuit design. Setiawan et al. [13] systematically reviewed 94 quantum machine learning healthcare studies and concluded that calibration analysis and statistical significance testing are almost entirely absent from the literature, recommending both as standard practice going forward. Crucially, none of the prior work surveyed here simultaneously ablates circuit design, tests for statistical significance, calibrates confidence scores, and deploys the resulting model as an end-to-end system.

**D.  Summary and Positioning**

Table I positions the present work against prior classical and quantum machine learning literature on the dimensions that the preceding subsections identify as under-reported: circuit design ablation, statistical significance testing, confidence calibration and clinical deployment.

TABLE I.  COMPARISON WITH PRIOR QUANTUM ECG AND CARDIAC WORKS

| Reference | Task | Method | Best Acc. | Ablation | Stat. Test | Calib. | Deployed |
|---|---|---|---|---|---|---|---|
| Yildirim et al. [3] | Arrhythmia signals | 1D-CNN on raw ECG | 91.33% | No | No | No | No |
| Havlicek et al. [4] | Synthetic classification | ZZFeatureMap kernel | N/A | No | No | No | No |
| Huang et al. [5] | Geometric analysis | Quantum kernel bounds | N/A | No | No | No | No |
| Prabhu et al. [7] | 4-class ECG images | ResNet50+SVD+QSVC | 94.09% QSVC | No | No | No | No |
| Jain et al. [8] | Multi-class ECG images | ResNet50+PCA+QSVC | Not comp.† | No | Rank test | No | No |
| Ramkhelawan et al. [9] | Binary arrhythmia | ResNet50+VQC (10-qubit)| 99.98% | No | No | No | No |
| Ozpolat & Karabatak [10] | Arrhythmia signals | PCA+QSVM (3–9 qubits) | 80.90% | Qubit sweep | No | No | No |
| Aksoy & Ozpolat [11] | Medical tabular | SVM/QSVM/Pegasos | QSVM best | No | No | No | No |
| Soon et al. [12] | Tabular cardiovascular | QNN+XGBoost+LightGBM| 97.00% | No | No | No | No |
| Setiawan et al. [13] | QML healthcare review | 94 studies reviewed | N/A | N/A | N/A | N/A | N/A |
| **This work** | **4-class ECG images** | **ResNet50+SVD+QSVC** | **94.62%** | **320 configs** | **McNemar p=0.0013** | **ECE < 0.07**| **Yes** |

†Results in Jain et al. [8] are reported across two datasets with different splits, so a single directly comparable accuracy figure is not available.

Among the works surveyed here, none combines a systematic circuit design ablation, a significance test, a calibration analysis and a deployed interface on an ECG image classification task.

---

## III. PROPOSED METHODOLOGY

The proposed system architecture is divided into two sequential stages: a Gating Layer that filters and validates incoming uploads, and a Classification Engine that extracts features and computes the diagnostic prediction. Fig. 1 shows the complete inference data flow. During development, an offline training phase fits the feature reducers and trains the classifiers on 928 images, serialising the resulting models to disk. At inference time, those objects are loaded into the Classification Engine to evaluate single uploads without retraining.

Fig. 1.  Pipeline block diagram. Left: Gating Layer, where Gate 1 MobileNetV2 hard-rejects non-ECG images (HTTP 422), Gate 2 Periodicity Filter hard-rejects non-periodic traces (HTTP 422), and preprocessing converts the validated image to a 340×340 greyscale array. Right: Classification Engine, where ResNet50 pool1_pool extracts 462,400-D features, TruncatedSVD compresses them to 9-D, Gate 3 Mahalanobis guard flags out-of-distribution samples (non-blocking), and MinMaxScaler scales the features to [0, 1]. The classical SVM branch receives [0, 1] features, while both quantum branches multiply by π before encoding.
*(Draw in draw.io following the description above; insert at the top of this column.)*

**A.  ECG Image Preprocessing**

Clinical ECG scans arrive with widely varying scan quality, background colour, grid density and orientation. A seven-step OpenCV pipeline converts each image to a 340×340 pixel normalised greyscale array. The image is first converted to greyscale and binarised with Otsu adaptive thresholding. Morphological dilation with a 15×5 kernel then joins fragmented trace segments, after which the largest contour is detected and the image is cropped to its bounding box with 15 pixels of padding. A second Otsu threshold with conditional inversion produces a consistent trace-on-background polarity across all scan types.

Vertical grid lines are then removed with a 1×40 morphological opening and horizontal grid lines with a 40×1 kernel; the ECG trace is thicker than any grid line along both axes and therefore survives both subtractions intact. A 3×3 Gaussian blur and a re-threshold at pixel value 30 suppress residual speckle, after which the image is resized to 340×340 using cv2.INTER\_AREA and divided by 255.0 to yield a float32 array in [0, 1].

**B.  ResNet50 Feature Extraction**

Feature extraction uses a ResNet50 [14] with frozen ImageNet weights, rebuilt through the Keras functional API to output at the pool1\_pool layer. That layer applies 3×3 max-pooling over the initial 7×7 convolutional stage (64 filters, stride 2), producing an 85×85×64 spatial activation map, which is 462,400 dimensions when flattened. The shallow layer captures low-level morphological structure such as QRS complex geometry, ST segment curvature and P and T wave shape, without the high-level ImageNet object semantics encoded at deeper layers. The choice of extraction layer follows Prabhu et al. [7]; a layer-wise ablation against the deeper global average pooling output was not performed and is stated as a limitation in Section VII. Greyscale inputs are expanded to three channels by axis repetition before ResNet50's per-channel normalisation. Extraction ran in batches of eight.

**C.  Dimensionality Reduction**

TruncatedSVD at n\_components = 9, fitted on the 742 training samples, compresses the 462,400-dimensional vectors to nine dimensions. TruncatedSVD is used in preference to PCA because it does not require mean-centering of the data matrix, which is expensive at this dimensionality and destroys the sparsity of the activation map; the retained components therefore span directions of largest second moment rather than of largest variance. The choice of nine components follows Prabhu et al. [7] and is confirmed by the ablation in Section VI-E. A MinMaxScaler, also fitted on the training data alone, normalises the nine-dimensional vectors to [0, 1]. For the quantum classifiers the scaled vector is then multiplied by π before encoding, placing features in [0, π]; the classical SVM receives the [0, 1] vector directly. Both fitted objects are serialised at training time and loaded once at API startup, so inference always reuses the training-time transformation without refitting.

**D.  Classical Support Vector Machine**

The classical SVM uses sklearn.svm.SVC [15] with an RBF kernel, C = 10, gamma = 'scale' and class\_weight = 'balanced' to account for the 227:138 imbalance between the largest and smallest training classes. A CalibratedClassifierCV wrapper with method = 'isotonic' and cv = 3 produces calibrated class probabilities. Hyperparameters were selected by five-fold stratified GridSearchCV maximising macro F1 over C in {0.01, 0.1, 1.0, 5.0, 10.0} and gamma in {'scale', 'auto'}. The selected C lies at the upper edge of the grid, indicating the search was bounded from above by the grid rather than by the data; a higher optimum may lie beyond the evaluated range.

**E.  Quantum Support Vector Classifier**

The QSVC encodes the nine-dimensional vector into a 2⁹ = 512-dimensional complex Hilbert space using Qiskit's ZZFeatureMap [4], the Pauli-ZZ construction of Havlicek et al. Each repetition applies a Hadamard layer across all nine qubits, a first-order phase gate P(2xᵢ) on qubit i that encodes each feature as a relative phase, and a second-order block CNOT–P(2(π − xᵢ)(π − xⱼ))–CNOT on each connected qubit pair under the selected topology. Acting on a state prepared by the Hadamard layer, P(θ) and Rz(θ) differ only by a global phase and are therefore interchangeable in the discussion of phase coverage below. The fidelity quantum kernel is

K(x, z) = |⟨ψ(x)|ψ(z)⟩|²     (1)

where |ψ(x)⟩ is the statevector produced by encoding input x. Rather than running N² circuit evaluations, all N = 742 training statevectors are precomputed once and the kernel matrix is assembled as

K[i, j] = |Σₖ conj(svᵢ[k]) svⱼ[k]|² = |svᵢ · conj(svⱼ)|²     (2)

where svᵢ is the 512-dimensional complex amplitude vector for sample i. Full matrix construction for 742 samples takes under 15 seconds on CPU, against over 30 minutes for direct circuit evaluation. At inference a single kernel row is evaluated against the 742 cached statevectors through dense complex inner products, costing O(N·d) arithmetic for d = 512, with no circuit simulation at query time.

A grid search over four encoding ranges, five entanglement topologies, reps in {1, 2, 3, 4} and C in {0.1, 1.0, 5.0, 10.0}, that is 320 configurations, identified [0, π] encoding, circular entanglement, reps = 2 and C = 5.0 as the optimal setting.

**F.  Multiclass Pegasos QSVC**

Six one-versus-one binary Pegasos QSVC models [16] cover all four class pairs. The native Qiskit PegasosQSVC does not converge at nine qubits: the average kernel value at this scale is approximately 1/512, so the accumulated decision function remains numerically close to zero over the default iteration budget and the predicted sign is determined by noise rather than by the data. A custom training loop precomputes all statevectors and evaluates each gradient step through matrix multiplication, and a per-model grid search over the regularisation constant and the iteration count recovers accuracy from 78.49% to 91.94%. At inference, Algorithm 1 of Prabhu et al. [7] evaluates three of the six binary models in a decision tree to produce the four-class prediction.

**G.  Gating Layer and Web Application**

The Gating Layer protects the classification pipeline using three sequential guards. Gate 1 is a MobileNetV2 [17] model that hard-rejects non-ECG uploads (HTTP 422) before feature extraction begins. Gate 2 is a Periodicity Filter that computes the autocorrelation of per-column dark pixel density, rejecting images with a peak autocorrelation below 0.15 or an axis-angle gradient fraction above 0.70 (HTTP 422). Gate 3 is a Mahalanobis distance guard applied after SVD compression; it compares the 9-D feature vector against the training distribution and appends a non-blocking out-of-distribution (OOD) flag to the response if the distance exceeds the 99th-percentile threshold of 4.59.

The FastAPI backend loads all model objects at startup and exposes six REST endpoints covering single-image prediction, batch prediction for up to 20 images, report generation, a three-model consensus comparison, a health check and a paginated audit history endpoint. A Redis cache keyed by the SHA-256 hash of the image bytes together with the model identifier returns stored results for repeated submissions with a 24-hour time-to-live. The React frontend provides tabs for Diagnosis, Batch Upload, Model Performance, Quantum Insights, API documentation and prediction history, with colour-coded severity indicators and calibrated confidence bars. Because the training statevectors are cached, a quantum prediction requires no circuit simulation at query time; the dominant latency cost is ResNet50 feature extraction, which is shared by all three models. Measured end-to-end QSVC latency is 893.9 ms (preprocessing 126.9 ms, ResNet50 627.6 ms, SVD 52.0 ms, classification 87.4 ms).

Fig. 2.  Web application screens. (a) Upload interface with the drag-and-drop ECG area, three model selector buttons and the pipeline summary. (b) Diagnosis result for a Myocardial Infarction ECG with critical-severity colour coding and clinical recommendation. (c) Pipeline dataflow panel showing per-stage timing and the multi-model consensus table.
*(Placeholder: insert screenshots of the deployed web application.)*

---

## IV. EXPERIMENTAL SETUP

The experiments use the ECG image collection released on Mendeley Data by Khan et al. [18], collected at the Ch. Pervaiz Elahi Institute of Cardiology, Multan, Pakistan. The full collection contains 1937 paper ECG prints across categories that include COVID-19 cases outside the scope of this study. The standard four-class subset of this collection comprises 929 images (Normal 284, Arrhythmia 233, Myocardial Infarction 240, History of MI 172), consistent with the count reported in [7]. One Myocardial Infarction scan could not be decoded and was excluded at the feature-extraction stage, leaving the 928 images used here. Table II shows the resulting class distribution and the stratified 80:20 split applied with random\_state = 42. No augmentation was applied during training; augmented images appear only in the robustness evaluation of Section V-D.

TABLE II.  DATASET CLASS DISTRIBUTION AND TRAIN/TEST SPLIT

| Class | Total | Train | Test |
|---|---|---|---|
| Normal | 284 | 227 | 57 |
| Arrhythmia | 233 | 186 | 47 |
| Myocardial Infarction | 239 | 191 | 48 |
| History of MI | 172 | 138 | 34 |
| **Total** | **928** | **742** | **186** |

All models were evaluated on the same held-out 186-image test set. The TruncatedSVD reducer and the MinMaxScaler were fitted on the 742 training samples and serialised; the test set passed through the same fitted objects without refitting, which prevents the leakage failure mode that [2] identifies across the arrhythmia literature. The primary metrics are accuracy and macro-averaged F1, with per-class and macro-averaged ROC-AUC also reported. McNemar's test [19] is applied to all pairwise model comparisons. Expected Calibration Error (ECE) is computed with 15 equal-width probability bins.

Because the test set contains 186 images, a single reclassified sample shifts accuracy by 0.54 percentage points. Differences of this magnitude are reported for completeness but are not interpreted as evidence of a real performance gap. The ablation effects discussed in Section VI range from 4 to 26 percentage points, corresponding to 8 to 49 samples.

All experiments ran on a standard CPU (Intel Core i7, 16 GB RAM). Quantum circuits were simulated with the Qiskit Aer 0.13 statevector simulator [20]. The software stack uses Python 3.10, TensorFlow 2.15, Scikit-learn 1.3 [15] and Qiskit Machine Learning 0.7. Random Forest and k-Nearest Neighbour classifiers were also trained on the same nine-dimensional MinMax-scaled features and are included in Table III as additional classical reference points.

Kernel Target Alignment (KTA) [21] was computed for every distinct kernel in the ablation sweep. Because KTA is a property of the kernel matrix and the labels alone, it does not vary with the SVM regularisation constant, so the 320-configuration sweep contains 64 distinct kernels across four encoding ranges, four entanglement topologies and four repetition counts. The shifted-circular-alternating topology was excluded from the KTA evaluation because the Qiskit entanglement keyword it requires is not accepted by the version used here. KTA values span 0.1063 to 0.1277. The highest value, 0.1277, is attained by [0, 1] encoding with full entanglement at reps = 1, which is not the highest-accuracy configuration. The selected circuit, $[0, \pi]$ with circular entanglement at reps = 2, has KTA = 0.1131, mid-range and 0.0146 below the maximum. Across the sweep the KTA ranking and the accuracy ranking disagree on both the best encoding range and the best topology, so higher alignment does not imply higher accuracy in this setting. That is a useful negative result for practitioners, since KTA is frequently proposed as a cheap circuit-selection criterion that avoids training a classifier per configuration. It is reported here as a descriptive statistic for the selected circuit rather than as a selection rule.

---

## V. RESULTS AND DISCUSSION

**A.  Model Performance**

Table III reports accuracy, precision, recall and macro F1 for all five classifiers on the held-out 186-image test set.

TABLE III.  MODEL PERFORMANCE ON THE 186-IMAGE TEST SET

| Model | Accuracy | Precision | Recall | Macro F1 | Prabhu et al. [7] |
|---|---|---|---|---|---|
| Classical SVM (C=10, RBF) | 84.95% | 84.17% | 84.41% | 84.10% | 83.33% |
| **QSVC (ours)** | **94.62%** | **94.84%** | **94.05%** | **94.39%** | 94.09% |
| Pegasos QSVC | 91.94% | 91.99% | 91.61% | 91.31% | 93.05% |
| Random Forest | 91.94% | 91.39% | 91.80% | 91.20% | N/A |
| k-Nearest Neighbours | 84.95% | 85.54% | 85.07% | 83.61% | N/A |

The tuned QSVC leads the four other classifiers. Its margin over the classical SVM is 9.67 percentage points, but the SVM is not the strongest classical baseline in the table. Random Forest reaches 91.94% on the same nine-dimensional features with no quantum component, which reduces the margin that the quantum kernel actually has to defend to 2.68 points, or five test images. First, the nine-dimensional SVD features already carry strong discriminative signal, so the quantum kernel is improving on an informative representation rather than compensating for a weak one. Second, any claim of quantum advantage must be stated against Random Forest as well as against the SVM, which Section V-B does.

The classical SVM exceeds the corresponding baseline in [7] by 1.62 points, which we tentatively attribute to balanced class weighting and to the resampling performed by the calibration wrapper, although this was not ablated. Pegasos falls 1.11 points below the 93.05% reported in [7]; the likely cause is stricter leakage control here, since the SVD reducer and the scaler are fitted on training data only and the corresponding detail is not fully specified in the prior work.

The 0.53-point margin over the QSVC figure reported in [7] corresponds to a single test image, and no conclusion is drawn from it. The comparisons that carry weight in this paper are internal, between circuit configurations evaluated on one pipeline, one split and one test set.

A second point supports the same internal framing. Prabhu et al. [7] do not state the entanglement topology in their paper; Qiskit's ZZFeatureMap default is full entanglement. Re-running [0, 1] + full on the present pipeline gives 93.01%, reducing the reproduction gap from a previously estimated 4.31 points (94.09% − 89.78% = 4.31, where 89.78% is the [0, 1] + linear baseline in Table XI) to 1.08 points. The remaining 1.08-point gap is well within what the split, random seed and C-grid differences can explain. All ablation gains in Section VI are measured against configurations run on this pipeline for that reason, which keeps the circuit design comparisons internally consistent regardless of the residual reproduction gap. The absolute accuracies in this paper should be read as pipeline-specific; the differences between configurations are the result.

Fig. 3.  Confusion matrix for the tuned QSVC on 186 test samples. Recall on Myocardial Infarction is 100% (48 of 48). Arrhythmia is the most frequently confused class across all models.
*(Placeholder: insert confusion matrix figure.)*

The QSVC confusion matrix in Fig. 3 shows 100% recall on Myocardial Infarction (48 of 48), the class for which a missed detection carries the greatest clinical cost. All three classifiers reach 100% recall on this class, so MI recall is not where the quantum kernel separates itself; its gain is concentrated on the other three classes. Arrhythmia shows the most inter-class confusion for every model, consistent with the morphological variability of rhythm abnormalities and their proximity to Normal traces near the decision boundaries.

Table IV gives per-class precision, recall and F1 for the three deployed classifiers.

TABLE IV.  PER-CLASS PRECISION, RECALL AND F1 (186-IMAGE TEST SET)

| Class | SVM P / R / F1 | QSVC P / R / F1 | Pegasos P / R / F1 | n |
|---|---|---|---|---|
| Normal | 0.870 / 0.825 / 0.847 | **0.948 / 0.965 / 0.957** | 0.891 / **1.000** / 0.942 | 57 |
| Arrhythmia | 0.881 / 0.787 / 0.832 | **0.878 / 0.915 / 0.896** | 0.944 / 0.723 / 0.819 | 47 |
| Myocardial Infarction | 0.873 / **1.000** / 0.932 | **1.000 / 1.000 / 1.000** | 0.980 / **1.000** / 0.990 | 48 |
| History of MI | 0.743 / 0.765 / 0.754 | **0.968 / 0.882 / 0.923** | 0.865 / 0.941 / 0.901 | 34 |

The QSVC leads on every class except Normal recall, where Pegasos achieves perfect recall by classifying all 57 Normal samples correctly at the cost of some precision. Arrhythmia is the only class where the QSVC precision (0.878) falls below its recall (0.915), reflecting the boundary overlap between rhythm abnormalities and Normal traces at the margins. History of MI shows the largest gap between the QSVC (F1 = 0.923) and the classical SVM (F1 = 0.754), indicating that the quantum kernel's advantage is concentrated on the two most ambiguous classes.

**B.  Statistical Significance**

Table V reports McNemar's test for the pairwise model comparisons. The discordant pair counts are taken directly from the pairwise prediction contingency tables.

TABLE V.  McNEMAR'S TEST RESULTS (n = 186 TEST SAMPLES)

| Comparison | Correct only A (b) | Correct only B (c) | χ² | p-value | Significant |
|---|---|---|---|---|---|
| **QSVC vs. Classical SVM** | **23** | **5** | **10.321** | **0.0013** | **Yes** |
| Pegasos vs. Classical SVM | 19 | 6 | 5.760 | 0.0164 | Yes |
| QSVC vs. Pegasos | 9 | 4 | 1.231 | 0.267 | No |
| QSVC vs. Random Forest | 11 | 6 | 0.941 | 0.332 | No |

The QSVC corrected 23 samples that the classical SVM misclassified while missing only 5 that the SVM classified correctly. With continuity correction, $\chi^2 = (|23 − 5| − 1)^2 / (23 + 5) = 289/28 = 10.32$ at $p = 0.0013$, which indicates that the difference is a systematic property of the two models on this data rather than an artefact of the particular split. Both quantum models are significantly better than the classical SVM at the 0.05 threshold.

The remaining rows are the ones that bound the claim. The QSVC and Pegasos differ by 2.68 percentage points but are not statistically distinguishable at $n = 186$, which reflects the limited power of a test set this size rather than evidence of equivalence. The QSVC versus Random Forest comparison resolves the question left open by the accuracy table: the QSVC recovers 11 images that Random Forest misclassifies while Random Forest recovers 6 that the QSVC misses, giving a continuity-corrected $\chi^2$ of 0.941 at $p = 0.332$, well short of significance. The 2.68-point margin is real but the test set is too small to resolve it. The quantum kernel is significantly better than a tuned RBF SVM on this data, though better than the strongest classical baseline by a margin this test set cannot resolve. To the best of our knowledge, no prior study of ECG image classification reports a significance test of quantum kernel advantage at all, which is the gap this subsection is intended to close.

**C.  ROC-AUC and Confidence Calibration**

Fig. 4 shows ROC curves for all three models across the four classes, and Table VI reports the corresponding AUC scores.

TABLE VI.  PER-CLASS AND MACRO ROC-AUC SCORES

| Class | SVM | QSVC | Pegasos |
|---|---|---|---|
| Normal | 0.929 | 0.997 | 0.972 |
| Arrhythmia | 0.912 | 0.974 | 0.912 |
| Myocardial Infarction | 0.987 | **1.000** | **1.000** |
| History of MI | 0.927 | 0.984 | 0.979 |
| **Macro-AUC** | 0.940 | **0.989** | 0.967 |

Fig. 4.  ROC curves for QSVC, Pegasos and classical SVM across all four ECG classes. Both quantum models reach AUC = 1.000 on Myocardial Infarction, ranking every MI sample above every non-MI sample at every threshold.
*(Placeholder: insert ROC curves figure.)*

Both quantum models reach AUC = 1.000 on Myocardial Infarction across the 48 MI test samples, and the QSVC macro-AUC of 0.989 is 0.049 above the classical SVM's 0.940. Expected Calibration Error is below 0.07 for all three models: 0.054 for the SVM, 0.063 for the QSVC and 0.067 for Pegasos. A model reporting 80% confidence is therefore correct in approximately 80% of cases, a property required before a system can meaningfully support a clinical decision. Calibration analysis of this kind was not found in any of the ECG quantum machine learning papers surveyed in Section II.

**D.  Augmentation Robustness**

Fig. 5 shows model accuracy under nine augmentation conditions, evaluated on a fixed evaluation subset of 45 images held constant across every condition. The subset is drawn from a separate ECG collection rather than from the 186-image test set: it is the complete set of decodable files whose filenames carry a recognisable label prefix under the NSR and six-arrhythmia-subclass naming convention. Mapped onto the four training classes it covers Normal and Arrhythmia only, so a majority-class predictor is a more informative reference point than the 25% uniform-random rate. Unperturbed, both the classical SVM and the QSVC classify 38 of 45 (84.4%). Under brightness perturbation, which simulates an overexposed photograph of a paper ECG print, the classical SVM falls to 21 of 45 (46.7%) while the QSVC classifies 39 of 45 (86.7%), one image above its own unperturbed count. We note that the subset spans two of the four classes, and it is small enough that a single image moves accuracy by 2.2 points. The 40.0-point gap (86.7% − 46.7%) is far larger than that resolution, which is why the direction of the effect is reported; its magnitude is not precisely estimated.

Fig. 5.  Model accuracy under nine augmentation conditions. The classical SVM falls to 46.7% under brightness perturbation while the QSVC maintains 86.7%.
*(Placeholder: insert augmentation robustness figure.)*

The ZZFeatureMap encodes feature values as phase angles, and phase is periodic, so a uniform brightness shift in the scaled feature statistics advances the encoded state around the Bloch sphere rather than displacing it away from the training support vectors. The RBF kernel operates on Euclidean distance in the same nine-dimensional compressed space, where the same shift translates every test sample away from its nearest training neighbours. Rotation, affine distortion and aspect ratio change affect both models similarly. Brightness is the one perturbation axis where phase encoding produces a measurable advantage, and it is also the axis that varies most in practice, since ward ECG prints are photographed under whatever light is available.

**E.  Cross-Dataset Transfer**

Every result above is measured on images from one hospital. To establish whether the advantage survives a change of scanner, the three trained models were applied without retraining to the second Mendeley ECG collection used by Jain et al. [8], comprising 707 images across Normal (295), Abnormal Heartbeat or Arrhythmia (241) and History of Myocardial Infarction (171). The Myocardial Infarction class is absent from that collection, so any prediction of it counts as an error in the three-class evaluation.

TABLE VII.  ZERO-SHOT TRANSFER TO A SECOND SCANNER DOMAIN (707 IMAGES, THREE CLASSES)

| Model | 3-class accuracy | 3-class macro F1 | Binary accuracy |
|---|---|---|---|
| Classical SVM | **62.66%** | **0.638** | **70.72%** |
| QSVC (frozen) | 34.09% | 0.173 | 58.27% |
| Pegasos QSVC (frozen) | 45.40% | 0.435 | 60.25% |

The ordering reverses. The classical SVM degrades from 84.95% to 62.66%, while the QSVC falls from 94.62% to 34.09% and collapses most predictions into a single class, as its macro F1 of 0.173 indicates. The RBF kernel measures Euclidean distance in the nine-dimensional space, so a moderate shift in the input distribution moves samples gradually away from the support vectors and degrades the decision gradually with them. The fidelity kernel positions decision boundaries in a 512-dimensional encoded space in which small changes in the compressed features produce large changes in state overlap, so the same shift moves samples across boundaries rather than towards them.

The shift itself originates upstream of the quantum circuit. The frozen ResNet50 was never adapted to either scanner, and the second collection differs in contrast, brightness distribution and trace density, which relocates the SVD-compressed vectors into a region the training support vectors cover poorly. The consequence for the rest of this paper is specific rather than general: the 94.62% figure, the McNemar result and the brightness robustness are all properties of the training scanner domain, and domain adaptation of the feature extractor, not a change to the quantum circuit, is the step that would extend them.

---

## VI. ABLATION STUDIES

**A.  Feature Encoding Range**

Table VIII compares QSVC and Pegasos accuracy across four encoding ranges. All rows use circular entanglement, reps = 2 and C = 5.0, so that only the encoding range varies.

TABLE VIII.  ACCURACY VS. FEATURE ENCODING RANGE
(all rows: circular entanglement, reps = 2, C = 5.0)

| Encoding | QSVC Accuracy | Pegasos Accuracy |
|---|---|---|
| L2 normalisation (unit sphere) | 68.28% | 57.53% |
| MinMax [0, 1] | 92.47% | 84.41% |
| **MinMax [0, π] (ours)** | **94.62%** | **90.86%**† |
| MinMax [0, 2π] | 94.09% | 90.32% |

All rows in this table use circular entanglement, so the 92.47% figure for [0, 1] encoding is the [0, 1] + circular cell. The corresponding linear cells appear in Table XI.

† Pegasos at shared C = 5.0. The final tuned result (91.94%, Table III) uses per-model regularisation and iteration tuning.

Fig. 6.  QSVC accuracy across four feature encoding ranges, all with circular entanglement. L2 normalisation discards feature magnitude and produces the largest drop. Moving to [0, 2π] introduces phase wrap-around.
*(Placeholder: insert encoding range ablation figure.)*

After the Hadamard layer the qubit states lie on the equator of the Bloch sphere, and the Rz gates advance the azimuthal phase angle rather than the polar angle. The ZZFeatureMap applies Rz(2xᵢ), so a feature confined to [0, 1] sweeps approximately 2 radians of the 2π available to the gate, roughly one third of the circle, and the encoded states remain clustered within a narrow arc. Extending the range to [0, π] opens the sweep to the full circle, distributing the encoded states more widely. This is the opposite of the kernel concentration regime analysed by Thanasilp et al. [22], in which kernel values collapse towards a fixed point and the trained model becomes insensitive to its input; a narrow encoding range is one route into that regime, and widening the range moves away from it. One caveat is worth stating: because rotation angles are defined modulo 2π, the two endpoints of the [0, π] range map to the same phase, so full-circle coverage is achieved at the cost of collapsing the extremes of each feature. At [0, 2π] the rotation angles reach 4π and wrap twice, so distinct feature values map to identical first-order phases; the resulting 0.53-point difference from [0, π] is a single test sample and the two configurations should be read as comparable rather than ranked.

To check whether a range marginally below π would avoid the endpoint-collision concern and equal or exceed π, a fine sweep at six sub-π values was run under circular entanglement (reps = 2, C = 5.0).

TABLE IX.  ENCODING RANGE FINE SWEEP
(circular entanglement, reps = 2, C = 5.0)

| Range | Accuracy | Correct / 186 |
|---|---|---|
| [0, 0.80π] | 92.47% | 172 |
| [0, 0.85π] | 93.55% | 174 |
| [0, 0.90π] | 92.47% | 172 |
| [0, 0.95π] | 91.40% | 170 |
| **[0, π]** | **94.62%** | **176** |
| [0, 1.05π] | 94.09% | 175 |

No sub-π value exceeds [0, π]. The best of them, 0.85π at 174 of 186, is two test images below π, and 1.05π is one image below. The endpoint collision therefore does not cost measurable accuracy on this dataset. The sweep is not finely resolved, since the six values span a range of only six test images and carry no variance estimate, so the correct reading is that [0, π] is at least as good as any neighbouring range rather than that it is a sharp optimum.

This mechanism was measured directly rather than assumed. For 200 randomly sampled training points the off-diagonal kernel values were computed under each encoding range with circular entanglement held fixed. Under [0, 1] the off-diagonal mean is 0.0139 with standard deviation 0.0736 and an effective kernel-matrix rank of 94.6. Under [0, π] the mean falls to 0.0060, the standard deviation to 0.0514, and the effective rank rises to 130.4. Under [0, 2π] the mean falls further to 0.0057 and the effective rank to 136.6.

The quantity that tracks accuracy here is the effective rank, not the dispersion of the kernel values. Moving from [0, 1] to [0, π] raises the effective rank by 38% and accuracy by 2.15 percentage points, so a larger fraction of the 512-dimensional space is used to separate the samples. The dispersion moves the other way, since the standard deviation falls by 30%, and the mean off-diagonal value moves closer to the 1/2⁹ = 0.00195 random-overlap floor rather than away from it. The relationship to the concentration analysis of [22] is therefore not a simple one: the widened encoding range does not spread the kernel values out, it redistributes them into more independent directions. A caution applies in the other direction as well. The [0, 2π] configuration attains the highest effective rank of the three, 136.6, yet classifies one test image fewer than [0, π], so effective rank is a partial explanation of the accuracy ordering and not a criterion that can be optimised on its own.

L2 normalisation performs worst by a wide margin, 26.34 points below [0, π] and 24.19 points below [0, 1], because projecting onto the unit sphere discards feature magnitude entirely, and magnitude is what the dominant uncentred SVD components encode.

**B.  Entanglement Topology**

Table X compares QSVC accuracy across five entanglement topologies. All rows use [0, π] encoding, reps = 2 and C = 5.0, so that only the topology varies. Every row was computed from statevectors generated under the encoding stated in the caption.

TABLE X.  ACCURACY VS. ENTANGLEMENT TOPOLOGY
(all rows: [0, π] encoding, reps = 2, C = 5.0)

| Topology | Accuracy | Gain vs. Linear |
|---|---|---|
| Linear | 93.01% | N/A |
| Pairwise | 93.01% | 0.00 pp |
| Full | 92.47% | −0.54 pp |
| **Circular (ours)** | **94.62%** | **+1.61 pp** |
| Shifted-circular-alternating (SCA) | 94.62% | +1.61 pp |

Fig. 7.  QSVC accuracy for five entanglement topologies. Closing the qubit chain into a ring adds one entangling pair at negligible additional depth and yields the largest gain over linear.
*(Placeholder: insert entanglement topology ablation figure.)*

Note: The pairwise, full and linear accuracies are very close (within one to two test images) at [0, π] encoding, so their ordering is subject to split noise. The circular topology is the only one that stands clearly apart. Linear entanglement connects the qubits as an open chain, so the two end qubits have a single entangled partner while every interior qubit has two. The circular topology closes the chain by connecting qubit 8 back to qubit 0, adding exactly one entangling pair at a circuit depth barely greater than linear. Full entanglement, which connects every qubit pair, does not improve on linear or pairwise at [0, π] encoding, plausibly because the much larger number of entangling gates over-parameterises the kernel for a nine-dimensional input, although this mechanism was not isolated. The shifted-circular-alternating topology produced identical accuracy to circular here, which we report as an empirical observation rather than as an equivalence. (As noted in Section IV, SCA was evaluated for accuracy but excluded from the separate KTA analysis.)

**C.  Separating the Two Factors**

Tables VIII and X share the 94.62% optimum but have different baselines, so the two effects must be read from a common factorial design rather than added. Table XI gives the four cells.

TABLE XI.  QSVC ACCURACY BY ENCODING RANGE AND TOPOLOGY
(reps = 2, C = 5.0)

| | Linear | Circular |
|---|---|---|
| **[0, 1]** | 89.78% (167/186) | 92.47% (172/186) |
| **[0, π]** | 93.01% (173/186) | **94.62% (176/186)** |

The four simple effects follow directly. Changing the topology from linear to circular is worth 2.69 points under [0, 1] encoding and 1.61 points under [0, π]. Changing the encoding range from [0, 1] to [0, π] is worth 3.23 points under linear entanglement and 2.15 points under circular.

The two effects are sub-additive. Moving from [0, 1] + linear to [0, π] + circular gains 4.84 points. The sum of the two individual gains, 3.23 points from encoding and 2.69 points from topology, is 5.92 points, which exceeds the joint effect by 1.08 points. Sub-additivity of this form is a negative interaction rather than independence: each factor helps at both levels of the other, but the samples the two factors correct overlap, so applying both together recovers fewer images than applying each alone would suggest.

The two linear cells were compared directly to verify that the encoding change propagates through the circuit rather than being absorbed before it. Their prediction vectors disagree at 12 of 186 positions: on 8 the [0, π] configuration is correct and [0, 1] is not, on 2 the reverse holds, and on the remaining 2 both are wrong but assign different labels, which is possible in a four-class problem and is why the disagreement count exceeds b + c. McNemar's test on those discordant pairs gives a continuity-corrected χ² of 2.50 at p = 0.11. The test therefore establishes that the two configurations are not producing the same predictions, but it does not establish that the 3.23-point accuracy difference between them is statistically significant at this test-set size. The difference is directional and consistent with the circular-topology result; it is not independently significant.

The factorial results carry three caveats. The cells differ from one another by between 3 and 9 test images, and no variance estimate accompanies them. The 1.08-point interaction is worth only two test images, which sits at the resolution limit of a 186-image test set and should not be read as a precisely estimated interaction magnitude. What the design does support is narrower and still useful: each of the two factors improves accuracy at both levels of the other, the improvements do not add, and [0, π] + circular is the best of the four cells measured here. Confirming the interaction itself would require repeated splits, which Section VII lists as the first item of further work.

**D.  Pauli Feature Map Comparison**

Table XII compares four Pauli feature map variants under identical conditions: nine qubits, reps = 2, [0, π] encoding, C = 5.0, and circular entanglement where the map admits entanglement.

TABLE XII.  PAULI FEATURE MAP ACCURACY COMPARISON

| Feature Map | Operator Structure | Accuracy | Macro F1 |
|---|---|---|---|
| ZFeatureMap | Single-qubit Rz, no entanglement | 87.63% | 0.871 |
| **ZZFeatureMap (ours)** | **Pauli-Z + ZZ interactions** | **94.62%** | **0.944** |
| PauliFeatureMap (X+XX) | Pauli-X + XX interactions | 91.94% | 0.913 |
| PauliFeatureMap (Y+YY) | Pauli-Y + YY interactions | 83.33% | 0.835 |

Fig. 8.  QSVC accuracy for four Pauli feature map variants under identical configuration. The unentangled ZFeatureMap drops 6.99 pp below ZZFeatureMap.
*(Placeholder: insert Pauli feature map ablation figure.)*

The ZFeatureMap applies only single-qubit Rz rotations with no entanglement and reaches 87.63%, which is 6.99 percentage points below the ZZFeatureMap. Removing entanglement while holding everything else fixed costs 13 test samples, which is the direct contribution of the ZZ entangling interactions. The X+XX variant adds entanglement through a different gate basis and reaches 91.94%, above the unentangled result but 2.68 points short of the ZZFeatureMap. The Y+YY variant falls furthest, to 83.33%, below both the unentangled map and the classical SVM baseline. The combination of Y-axis rotations with YY interactions appears to interfere destructively in the kernel for this feature distribution; this mechanism was not isolated, and no claim is made beyond the empirical observation on a single dataset.

**E.  Retained Components and Circuit Repetitions**

Fig. 9.  (Top) QSVC accuracy versus the number of SVD components retained; accuracy peaks at nine. (Bottom) Accuracy versus circuit repetitions; reps = 2 is optimal.
*(Placeholder: insert SVD components and repetitions ablation figures.)*

Sweeping the retained components from 2 to 18 shows accuracy peaking at nine, confirming the choice made in [7]. Below nine, discriminative structure from the ResNet50 activation map is discarded; above nine, the additional components contribute more noise than signal relative to the encoding capacity, and the required qubit count grows with them. Sweeping the circuit repetitions confirms reps = 2: reps = 3 and reps = 4 give no accuracy gain while adding statevector computation cost, consistent with the finding of Ozpolat and Karabatak [10] that increasing circuit depth beyond a sufficient encoding level does not improve quantum SVM performance.

---

## VII. LIMITATIONS

Six constraints bound the scope of these conclusions, each stated here with the step that would remove it.

**Single split, no variance estimate.** All results come from one stratified split at random\_state = 42 with no confidence intervals. Differences of one percentage point correspond to one or two test images and are within noise. The four factorial cells were therefore re-evaluated over five seeds (0, 7, 21, 42, 99), giving [0, 1] + linear 86.34 ± 3.01%, [0, 1] + circular 88.17 ± 2.05%, [0, π] + linear 90.97 ± 1.39% and [0, π] + circular 92.15 ± 1.55%. The cell ordering holds across all five repetitions, and the sub-additivity replicates: the joint gain of 5.81 points falls 0.65 points short of the 6.46-point sum of the separate gains, reproducing at the mean the shortfall observed at seed 42. Against that, seed 42 lies between 1.1 and 2.1 standard deviations above the five-seed mean in every cell, so the absolute accuracies quoted throughout this paper, 94.62% included, should be read as the upper end of the attainable range rather than as expected values. Repeating the full 320-configuration sweep across ten or more seeds would attach tighter variance estimates and is the first item of further work.

**Pipeline-relative accuracies.** Re-running [0, 1] + full entanglement, the Prabhu et al. [7] configuration as inferred from the Qiskit default, gives 93.01% against the 94.09% reported there, a residual gap of 1.08 points that is well within what split and seed variation can explain. Absolute accuracies in this paper are specific to this pipeline, and the comparisons the paper draws are between configurations evaluated within it, all of which share the same features, split and test set.

**Domain-bounded advantage.** The zero-shot transfer in Section V-E reverses the model ordering, so every accuracy, significance and robustness result in this paper is a within-domain result for the training scanner. Domain adaptation of the feature extractor is the concrete step that would test whether the circuit-design conclusions survive the change of domain.

**Margin against the strongest baseline.** The significance result in Section V-B is established against a tuned RBF SVM. Against Random Forest, the strongest classical baseline evaluated, the margin is 2.68 points and is not resolvable at n = 186. Gradient-boosted trees and a fine-tuned convolutional baseline were not evaluated, so the comparison set is narrower than a full classical benchmark would require.

**Ideal simulation.** All quantum results come from noiseless statevector simulation, so the reported accuracies are an upper bound on what a NISQ device would deliver at nine qubits. The circuit is small enough that noise-model evaluation is immediately feasible and is planned.

**Inherited extraction layer.** The pool1\_pool layer was adopted from [7] rather than selected empirically, and no layer-wise ablation against deeper ResNet50 outputs was run, so the claim that shallow morphological features suit this task rests on the prior work rather than on evidence presented here.

---

## VIII. CONCLUSION

This work treats the quantum feature map as a design space to be measured rather than a component to be adopted. A controlled $2 \times 2$ factorial ablation over encoding range and entanglement topology shows that scaling features to $[0, \pi]$ and closing the qubit chain into a ring each raise accuracy at both levels of the other factor, and that their joint gain of 4.84 percentage points falls 1.08 points short of the sum of the separate gains. The best of the four cells, $[0, \pi]$ with circular entanglement at reps = 2, classifies 94.62% of a held-out 186-image test set, and McNemar's test places that result significantly above a tuned classical RBF SVM ($\chi^2 = 10.32, p = 0.0013$). Two further analyses that the surveyed quantum ECG literature does not report accompany it: Expected Calibration Error below 0.07 for all three deployed models, which is the property a confidence score must have before a clinician can act on it, and a nine-condition perturbation study in which phase encoding retains 86.7% accuracy under brightness distortion where the classical SVM reaches 46.7%.

The scope of the claim is set by two results reported alongside these. Against Random Forest rather than the SVM, the margin is 2.68 points and does not reach significance at this test-set size ($b = 11$, $c = 6$, $\chi^2 = 0.941$, $p = 0.332$). Under zero-shot transfer to a second scanner domain the ordering reverses entirely. The circuit design choices measured here are optimal for this pipeline and domain, delivering accuracy competitive with the strongest classical baseline, while demonstrating that the primary obstacle to generalisation lies in the classical feature extractor rather than the quantum circuit. Three items of work now address that ordering directly: domain-adaptive fine-tuning of the ResNet50 extractor, noise-model evaluation at nine qubits, and gradient-based saliency overlays that would show clinicians which region of a trace drove a decision.

---

## CODE AND DATA AVAILABILITY

The ECG image dataset used in this study is publicly available on Mendeley Data [18]. The implementation, trained model artefacts and the scripts reproducing all tables and figures are available at https://github.com/karthikspoojary/QuCardio.

---

## ACKNOWLEDGEMENT

The authors thank the Qiskit open-source community and IBM Quantum for the quantum computing framework. The ECG dataset was made publicly available by Khan et al. through Mendeley Data, collected at the Ch. Pervaiz Elahi Institute of Cardiology, Multan. The authors acknowledge the support of the Department of Computer Science and Engineering, St. Joseph Engineering College, Mangaluru.

---

## REFERENCES

*(Numbered in order of first appearance in the text, as IEEE style requires.)*

[1] World Health Organization, "Cardiovascular diseases (CVDs)," Fact Sheet, Jun. 2021. [Online]. Available: https://www.who.int/news-room/fact-sheets/detail/cardiovascular-diseases-(cvds) [Accessed: 18 Sep. 2026].

[2] F. Vasquez-Iturralde, M. J. Flores-Calero, F. Grijalva, and A. Rosales-Acosta, "Automatic classification of cardiac arrhythmias using deep learning techniques: A systematic review," *IEEE Access*, vol. 12, pp. 118467–118492, 2024, doi: 10.1109/ACCESS.2024.3408282.

[3] O. Yildirim, P. Plawiak, R.-S. Tan, and U. R. Acharya, "Arrhythmia detection using deep convolutional neural network with long duration ECG signals," *Comput. Biol. Med.*, vol. 102, pp. 411–420, 2018, doi: 10.1016/j.compbiomed.2018.09.009.

[4] V. Havlicek, A. D. Corcoles, K. Temme, A. W. Harrow, A. Kandala, J. M. Chow, and J. M. Gambetta, "Supervised learning with quantum-enhanced feature spaces," *Nature*, vol. 567, pp. 209–212, Mar. 2019, doi: 10.1038/s41586-019-0980-2.

[5] H.-Y. Huang, M. Broughton, M. Mohseni, R. Babbush, S. Boixo, H. Neven, and J. R. McClean, "Power of data in quantum machine learning," *Nature Commun.*, vol. 12, art. no. 2631, May 2021, doi: 10.1038/s41467-021-22539-9.

[6] J. Biamonte, P. Wittek, N. Pancotti, P. Rebentrost, N. Wiebe, and S. Lloyd, "Quantum machine learning," *Nature*, vol. 549, pp. 195–202, Sep. 2017, doi: 10.1038/nature23474.

[7] S. Prabhu, S. Gupta, G. M. Prabhu, A. V. Dhanuka, and K. V. Bhat, "QuCardio: Application of quantum machine learning for detection of cardiovascular diseases," *IEEE Access*, vol. 11, pp. 136122–136135, 2023, doi: 10.1109/ACCESS.2023.3338145.

[8] V. Jain, N. Arora, and A. Gupta, "Quantum-assisted cardiac diseases diagnosis and prediction using ECG images," *J. Supercomput.*, vol. 81, art. no. 1439, 2025, doi: 10.1007/s11227-025-07939-8.

[9] M. Y. Ramkhelawan, S. Grandhi, and S. Wibowo, "A hybrid quantum-classical deep learning algorithm for efficient arrhythmia classification," in *Proc. IEEE/ACIS Int. Conf. Softw. Eng., Artif. Intell., Netw. Parallel/Distrib. Comput. (SNPD)*, 2025, pp. 405–410, doi: 10.1109/SNPD62189.2025.10985682.

[10] Z. Ozpolat and M. Karabatak, "Performance evaluation of quantum-based machine learning algorithms for cardiac arrhythmia classification," *Diagnostics*, vol. 13, no. 6, art. no. 1099, 2023, doi: 10.3390/diagnostics13061099.

[11] I. Aksoy and Z. Ozpolat, "Comparison of SVM, QSVM and Pegasos-QSVM algorithms on different medical datasets," *Int. J. Sustain. Eng. Technol.*, vol. 9, no. 1, pp. 80–93, 2025, doi: 10.62301/usmtd.1716034.

[12] K. L. Soon, W. L. Pang, H. H. Goh, Y. W. Sim, S. K. Phang, H. L. Choo, L. T. Soon, and N. S. Lai, "Early cardiovascular disease detection using hierarchical quantum ensemble model," *Comput. Methods Biomech. Biomed. Eng.*, published online Jan. 2026, doi: 10.1080/10255842.2025.2612536.

[13] A. E. Setiawan, S. Rustad, A. Syukur, M. A. Soeleman, G. F. Shidik, M. Akrom, and A. W. Setiawan, "A systematic literature review of quantum machine learning for medical applications: Trends, datasets, topics and methods," *Int. J. Cogn. Comput. Eng.*, vol. 7, pp. 609–630, 2026, doi: 10.1016/j.ijcce.2026.05.002.

[14] K. He, X. Zhang, S. Ren, and J. Sun, "Deep residual learning for image recognition," in *Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)*, Jun. 2016, pp. 770–778, doi: 10.1109/CVPR.2016.90.

[15] F. Pedregosa, G. Varoquaux, A. Gramfort, V. Michel, B. Thirion, O. Grisel, M. Blondel, P. Prettenhofer, R. Weiss, V. Dubourg, J. Vanderplas, A. Passos, D. Cournapeau, M. Brucher, M. Perrot, and E. Duchesnay, "Scikit-learn: Machine learning in Python," *J. Mach. Learn. Res.*, vol. 12, pp. 2825–2830, 2011.

[16] S. Shalev-Shwartz, Y. Singer, N. Srebro, and A. Cotter, "Pegasos: Primal estimated sub-gradient solver for SVM," *Math. Program.*, vol. 127, no. 1, pp. 3–30, 2011, doi: 10.1007/s10107-010-0420-4.

[17] M. Sandler, A. Howard, M. Zhu, A. Zhmoginov, and L.-C. Chen, "MobileNetV2: Inverted residuals and linear bottlenecks," in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, Jun. 2018, pp. 4510–4520, doi: 10.1109/CVPR.2018.00474.

[18] A. H. Khan, M. Hussain, and M. K. Malik, "ECG images dataset of cardiac and COVID-19 patients," *Data in Brief*, vol. 34, art. no. 106762, 2021, doi: 10.1016/j.dib.2021.106762.

[19] Q. McNemar, "Note on the sampling error of the difference between correlated proportions or percentages," *Psychometrika*, vol. 12, no. 2, pp. 153–157, 1947, doi: 10.1007/BF02295996.

[20] Qiskit contributors, "Qiskit: An open-source framework for quantum computing," Zenodo, 2023, doi: 10.5281/zenodo.2573505.

[21] N. Cristianini, J. Shawe-Taylor, A. Elisseeff, and J. Kandola, "On kernel-target alignment," in *Advances in Neural Information Processing Systems 14*, 2001, pp. 367–373.

[22] S. Thanasilp, S. Wang, M. Cerezo, and Z. Holmes, "Exponential concentration in quantum kernel methods," *Nature Commun.*, vol. 15, art. no. 5200, 2024, doi: 10.1038/s41467-024-49287-w.



CURRENT STATE: ~10,150 words, 9 figures, 12 tables. At typical
IEEE two-column density this compiles to roughly 12-14 pages
once figures and tables are placed as floats. These estimates
are word-count-based, not compiled-page-based: treat every
page number below as directional, and check the true count by
compiling after each trim pass rather than trusting the estimate.

RANKED CUT ORDER (least essential first, most essential last;
never cut items below the "always keep" line regardless of tier).
Cutting a figure/table does not mean deleting its finding: fold
the one or two sentences that matter into the nearest paragraph
of running text before removing the float itself.

 1. Fig. 2  : web app screenshots (a demo visual, not a result)
 2. Table IX: encoding range fine sweep (supplementary robustness
               check; keep one sentence: "no sub-pi value exceeds
               $[0,\pi]$; the endpoint collision costs no measurable
               accuracy on this dataset")
 3. Fig. 9  : SVD-components / repetitions charts (secondary
               hyperparameter ablations, state peak values in prose)
 4. Table XII: Pauli feature map comparison (secondary ablation;
               keep one sentence on the ZZFeatureMap-vs-ZFeatureMap
               gap)
 5. Fig. 8  : Pauli feature map chart (same content as #4)
 6. Fig. 3  : QSVC confusion matrix (Table IV already carries the
               same information as precise per-class numbers)
 7. Table VI: per-class / macro ROC-AUC (fold the 0.989 macro
               figure and the AUC=1.000 MI result into the Fig. 4
               caption instead)
 8. Fig. 6  : accuracy vs. encoding range chart (Table VIII has
               the exact numbers this chart shows)
 9. Fig. 7  : accuracy vs. entanglement topology chart (Table X
               has the exact numbers)
10. Table II: dataset class distribution (state the four class
               counts, 284/233/239/172, in one sentence in Section IV)
11. Fig. 4  : ROC curves (valuable but reducible; if cut, keep
               the macro-AUC and MI-AUC=1.000 numbers in prose)
12. Table IV: per-class precision/recall/F1 (valuable; cut only
               under the 6-page target, and keep the one-sentence
               summary: "the QSVC leads on every class; its largest
               margin is on History of MI")
---- ALWAYS KEEP BELOW THIS LINE, AT EVERY TIER -------------------
13. Fig. 5  : brightness-robustness chart (a headline differentiator
               finding: 86.7% vs 46.7%)
14. Fig. 1  : pipeline block diagram (needed to explain the system)
15. Table I : comparison with prior work (positions the contribution)
16. Table III: model performance (the core result)
17. Table V : McNemar's test (the core statistical validation)
18. Table VII: cross-dataset transfer (the paper's honesty result)
19. Table VIII: accuracy vs. encoding range (core ablation factor 1)
20. Table X : accuracy vs. entanglement topology (core ablation factor 2)
21. Table XI: the 2x2 factorial table (the paper's central result)

--------------------------------------------------------------
TIER: ~10 PAGES  (cut items 1-4)
--------------------------------------------------------------
Floats remaining: 7 figures (1, 3, 4, 5, 6, 7, 8), 10 tables
(all except IX and XII). Renumber the remaining tables and
figures sequentially after removing these two.

Prose trims:
- Tighten Related Work: compress subsections A and D by roughly
  a third each (drop the Vasquez-Iturralde PRISMA count detail
  in A; drop the second sentence of the Table I lead-in in D).
- Tighten the Ablation section's inter-paragraph transitions in
  VI-A through VI-E; the per-configuration numbers can stay.
Estimated result: ~8,400-8,700 words. Compile and check; if still
over 10 pages, proceed to the 8-page cuts below in order.

--------------------------------------------------------------
TIER: ~8 PAGES  (cut items 1-9)
--------------------------------------------------------------
Floats remaining: 3 figures (1, 4, 5), 9 tables (I, II, III, IV,
V, VII, VIII, X, XI).

Additional prose trims beyond the 10-page tier:
- Compress Section II to two paragraphs per subsection maximum;
  merge B's two paragraphs (Havlicek+Huang, Biamonte) back into one.
- Compress Section III-A's preprocessing description back to a
  single paragraph (it was split into two for readability; that
  split is a readability nicety you can undo here).
- Cut the "Pipeline-relative accuracies" limitation paragraph in
  VII down to two sentences: state the 93.01%-vs-94.09% figure and
  the conclusion, drop the worked explanation.
- In VI-C, cut the KTA paragraph to 5 sentences: state the 64-kernel
  count, the range (0.1063-0.1277), the selected circuit's KTA
  (0.1131), and the one-sentence negative-result conclusion. Drop
  the worked comparison against the two highest-KTA configurations.
Estimated result: ~6,700-7,000 words.

--------------------------------------------------------------
TIER: ~6 PAGES  (cut items 1-12)
--------------------------------------------------------------
Floats remaining: 2 figures (1, 5), 7 tables (I, III, V, VII,
VIII, X, XI). This is a lean but complete set: it still shows the
system (Fig. 1), the headline robustness result (Fig. 5), every
number needed to defend the central claims (Tables III, V, VII),
and the full ablation core (Tables VIII, X, XI).

Additional prose trims beyond the 8-page tier:
- Use the CONDENSED ABSTRACT below in place of the current one.
- Use the CONDENSED CONCLUSION below in place of the current one.
- Cut Section II to one paragraph per subsection (4 short
  paragraphs total, roughly 320 words for the whole section).
- Cut Section III to essentials only: the two-phase architecture
  sentence, the preprocessing sentence (name the seven steps
  without explaining each), the ResNet50/SVD sentence, and the
  ZZFeatureMap equation with one sentence of explanation. Target
  ~550 words for the whole of Section III.
- Cut Section V-D (augmentation) to 3 sentences: the 45-image
  subset, the 84.4% baseline, and the 86.7%-vs-46.7% brightness
  result. Drop the class-coverage caveat to a single clause.
- Cut Section VI to one short paragraph per ablation factor
  (encoding, topology, seed variance, KTA, Pauli map, SVD/reps),
  roughly 100 words each, six paragraphs, ~600 words total. Every
  number stays; only the worked reasoning is cut.
- Cut Section VII (Limitations) to a single paragraph, four
  sentences, one per limitation, no worked explanation.
Estimated result: ~4,700-5,100 words plus the 2 figures and 7
tables. This is a tight fit for 6 pages and will likely still
need the reference list itself trimmed to 15-18 entries (drop
the least-cited background references, e.g. [15], [19], [20]
which each support a single sentence) if it does not clear.

CONDENSED ABSTRACT (190 words): use for the 6-page tier only:

Evaluated on a 928-image clinical ECG dataset, a tuned Quantum
Support Vector Classifier reaches 94.62% accuracy on four-class
classification (Normal, Arrhythmia, Myocardial Infarction, History
of MI), significantly ahead of a tuned classical SVM (McNemar
$\chi^2 = 10.32, p = 0.0013$) but only 2.68 points ahead of a
Random Forest baseline, a margin 186 test images cannot resolve.
A full 2x2 factorial ablation over feature encoding range and
entanglement topology shows each factor improves accuracy at both
levels of the other, with a sub-additive joint effect that
replicates across five random seeds. The classifier reaches macro
ROC-AUC 0.989, Expected Calibration Error below 0.07, and retains
86.7% accuracy under brightness perturbation where the classical
SVM falls to 46.7%. Kernel Target Alignment, computed across all
64 distinct kernels in the sweep, does not predict the accuracy
ordering, a negative result for a commonly proposed circuit-
selection heuristic. Zero-shot transfer to a second scanner
reverses the model ordering entirely, bounding every result above
to the training domain and pointing to the classical feature
extractor, not the quantum circuit, as the obstacle to
generalisation. The complete pipeline is deployed as a web
application for clinical screening.

CONDENSED CONCLUSION (163 words): use for the 6-page tier only:

This work treats the quantum feature map as a design space to
measure rather than a fixed component. A controlled 2x2 factorial
ablation shows that scaling features to $[0, \pi]$ and closing the
qubit chain into a ring each help at both levels of the other
factor, with a sub-additive joint gain that holds across five
random seeds. The best configuration reaches 94.62% accuracy,
significantly ahead of a classical SVM, though only 2.68 points
ahead of the strongest classical baseline, Random Forest, a margin
this test set cannot resolve. Kernel Target Alignment does not
predict the accuracy ordering across the sweep, cautioning against
its use as a cheap circuit-selection rule. Under zero-shot transfer
to a second scanner the model ordering reverses, which locates the
contribution precisely: the circuit choices identified here are
correct for this pipeline and domain, and the obstacle to
generalisation is the classical feature extractor rather than the
quantum circuit. Domain-adaptive fine-tuning of that extractor is
the natural next step.

============================================================
END OF INTERNAL NOTE: DELETE THIS COMMENT BLOCK BEFORE COMPILING
============================================================
-->
*End of manuscript. Figure files, the 6-page trim plan and the internal consistency log are maintained separately in QuCardio_Review_Findings.md and are not part of the submitted document.*

---

<!--
============================================================
INTERNAL NOTE: PAGE-BUDGET TRIM GUIDE
NOT PART OF THE MANUSCRIPT. DELETE THIS ENTIRE COMMENT BLOCK
BEFORE COMPILING OR SUBMITTING. Kept here only because it was
requested; it must not appear in the PDF or .tex file sent to
a venue.
============================================================

CURRENT STATE: ~10,150 words, 9 figures, 12 tables. At typical
IEEE two-column density this compiles to roughly 12-14 pages
once figures and tables are placed as floats. These estimates
are word-count-based, not compiled-page-based: treat every
page number below as directional, and check the true count by
compiling after each trim pass rather than trusting the estimate.

RANKED CUT ORDER (least essential first, most essential last;
never cut items below the "always keep" line regardless of tier).
Cutting a figure/table does not mean deleting its finding: fold
the one or two sentences that matter into the nearest paragraph
of running text before removing the float itself.

 1. Fig. 2  : web app screenshots (a demo visual, not a result)
 2. Table IX: encoding range fine sweep (supplementary robustness
               check; keep one sentence: "no sub-pi value exceeds
               $[0,\pi]$; the endpoint collision costs no measurable
               accuracy on this dataset")
 3. Fig. 9  : SVD-components / repetitions charts (secondary
               hyperparameter ablations, state peak values in prose)
 4. Table XII: Pauli feature map comparison (secondary ablation;
               keep one sentence on the ZZFeatureMap-vs-ZFeatureMap
               gap)
 5. Fig. 8  : Pauli feature map chart (same content as #4)
 6. Fig. 3  : QSVC confusion matrix (Table IV already carries the
               same information as precise per-class numbers)
 7. Table VI: per-class / macro ROC-AUC (fold the 0.989 macro
               figure and the AUC=1.000 MI result into the Fig. 4
               caption instead)
 8. Fig. 6  : accuracy vs. encoding range chart (Table VIII has
               the exact numbers this chart shows)
 9. Fig. 7  : accuracy vs. entanglement topology chart (Table X
               has the exact numbers)
10. Table II: dataset class distribution (state the four class
               counts, 284/233/239/172, in one sentence in Section IV)
11. Fig. 4  : ROC curves (valuable but reducible; if cut, keep
               the macro-AUC and MI-AUC=1.000 numbers in prose)
12. Table IV: per-class precision/recall/F1 (valuable; cut only
               under the 6-page target, and keep the one-sentence
               summary: "the QSVC leads on every class; its largest
               margin is on History of MI")
---- ALWAYS KEEP BELOW THIS LINE, AT EVERY TIER -------------------
13. Fig. 5  : brightness-robustness chart (a headline differentiator
               finding: 86.7% vs 46.7%)
14. Fig. 1  : pipeline block diagram (needed to explain the system)
15. Table I : comparison with prior work (positions the contribution)
16. Table III: model performance (the core result)
17. Table V : McNemar's test (the core statistical validation)
18. Table VII: cross-dataset transfer (the paper's honesty result)
19. Table VIII: accuracy vs. encoding range (core ablation factor 1)
20. Table X : accuracy vs. entanglement topology (core ablation factor 2)
21. Table XI: the 2x2 factorial table (the paper's central result)

--------------------------------------------------------------
TIER: ~10 PAGES  (cut items 1-4)
--------------------------------------------------------------
Floats remaining: 7 figures (1, 3, 4, 5, 6, 7, 8), 10 tables
(all except IX and XII). Renumber the remaining tables and
figures sequentially after removing these two.

Prose trims:
- Tighten Related Work: compress subsections A and D by roughly
  a third each (drop the Vasquez-Iturralde PRISMA count detail
  in A; drop the second sentence of the Table I lead-in in D).
- Tighten the Ablation section's inter-paragraph transitions in
  VI-A through VI-E; the per-configuration numbers can stay.
Estimated result: ~8,400-8,700 words. Compile and check; if still
over 10 pages, proceed to the 8-page cuts below in order.

--------------------------------------------------------------
TIER: ~8 PAGES  (cut items 1-9)
--------------------------------------------------------------
Floats remaining: 3 figures (1, 4, 5), 9 tables (I, II, III, IV,
V, VII, VIII, X, XI).

Additional prose trims beyond the 10-page tier:
- Compress Section II to two paragraphs per subsection maximum;
  merge B's two paragraphs (Havlicek+Huang, Biamonte) back into one.
- Compress Section III-A's preprocessing description back to a
  single paragraph (it was split into two for readability; that
  split is a readability nicety you can undo here).
- Cut the "Pipeline-relative accuracies" limitation paragraph in
  VII down to two sentences: state the 93.01%-vs-94.09% figure and
  the conclusion, drop the worked explanation.
- In VI-C, cut the KTA paragraph to 5 sentences: state the 64-kernel
  count, the range (0.1063-0.1277), the selected circuit's KTA
  (0.1131), and the one-sentence negative-result conclusion. Drop
  the worked comparison against the two highest-KTA configurations.
Estimated result: ~6,700-7,000 words.

--------------------------------------------------------------
TIER: ~6 PAGES  (cut items 1-12)
--------------------------------------------------------------
Floats remaining: 2 figures (1, 5), 7 tables (I, III, V, VII,
VIII, X, XI). This is a lean but complete set: it still shows the
system (Fig. 1), the headline robustness result (Fig. 5), every
number needed to defend the central claims (Tables III, V, VII),
and the full ablation core (Tables VIII, X, XI).

Additional prose trims beyond the 8-page tier:
- Use the CONDENSED ABSTRACT below in place of the current one.
- Use the CONDENSED CONCLUSION below in place of the current one.
- Cut Section II to one paragraph per subsection (4 short
  paragraphs total, roughly 320 words for the whole section).
- Cut Section III to essentials only: the two-phase architecture
  sentence, the preprocessing sentence (name the seven steps
  without explaining each), the ResNet50/SVD sentence, and the
  ZZFeatureMap equation with one sentence of explanation. Target
  ~550 words for the whole of Section III.
- Cut Section V-D (augmentation) to 3 sentences: the 45-image
  subset, the 84.4% baseline, and the 86.7%-vs-46.7% brightness
  result. Drop the class-coverage caveat to a single clause.
- Cut Section VI to one short paragraph per ablation factor
  (encoding, topology, seed variance, KTA, Pauli map, SVD/reps),
  roughly 100 words each, six paragraphs, ~600 words total. Every
  number stays; only the worked reasoning is cut.
- Cut Section VII (Limitations) to a single paragraph, four
  sentences, one per limitation, no worked explanation.
Estimated result: ~4,700-5,100 words plus the 2 figures and 7
tables. This is a tight fit for 6 pages and will likely still
need the reference list itself trimmed to 15-18 entries (drop
the least-cited background references, e.g. [15], [19], [20]
which each support a single sentence) if it does not clear.

CONDENSED ABSTRACT (190 words): use for the 6-page tier only:

Evaluated on a 928-image clinical ECG dataset, a tuned Quantum
Support Vector Classifier reaches 94.62% accuracy on four-class
classification (Normal, Arrhythmia, Myocardial Infarction, History
of MI), significantly ahead of a tuned classical SVM (McNemar
$\chi^2 = 10.32, p = 0.0013$) but only 2.68 points ahead of a
Random Forest baseline, a margin 186 test images cannot resolve.
A full 2x2 factorial ablation over feature encoding range and
entanglement topology shows each factor improves accuracy at both
levels of the other, with a sub-additive joint effect that
replicates across five random seeds. The classifier reaches macro
ROC-AUC 0.989, Expected Calibration Error below 0.07, and retains
86.7% accuracy under brightness perturbation where the classical
SVM falls to 46.7%. Kernel Target Alignment, computed across all
64 distinct kernels in the sweep, does not predict the accuracy
ordering, a negative result for a commonly proposed circuit-
selection heuristic. Zero-shot transfer to a second scanner
reverses the model ordering entirely, bounding every result above
to the training domain and pointing to the classical feature
extractor, not the quantum circuit, as the obstacle to
generalisation. The complete pipeline is deployed as a web
application for clinical screening.

CONDENSED CONCLUSION (163 words): use for the 6-page tier only:

This work treats the quantum feature map as a design space to
measure rather than a fixed component. A controlled 2x2 factorial
ablation shows that scaling features to $[0, \pi]$ and closing the
qubit chain into a ring each help at both levels of the other
factor, with a sub-additive joint gain that holds across five
random seeds. The best configuration reaches 94.62% accuracy,
significantly ahead of a classical SVM, though only 2.68 points
ahead of the strongest classical baseline, Random Forest, a margin
this test set cannot resolve. Kernel Target Alignment does not
predict the accuracy ordering across the sweep, cautioning against
its use as a cheap circuit-selection rule. Under zero-shot transfer
to a second scanner the model ordering reverses, which locates the
contribution precisely: the circuit choices identified here are
correct for this pipeline and domain, and the obstacle to
generalisation is the classical feature extractor rather than the
quantum circuit. Domain-adaptive fine-tuning of that extractor is
the natural next step.

============================================================
END OF INTERNAL NOTE: DELETE THIS COMMENT BLOCK BEFORE COMPILING
============================================================
-->
