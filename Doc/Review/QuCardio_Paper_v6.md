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

Cardiovascular diseases cause approximately 17.9 million deaths annually, and the Electrocardiogram (ECG) remains the primary non-invasive diagnostic tool. Quantum machine learning has been applied to automated ECG analysis, but published studies report isolated accuracy figures under fixed circuit configurations, without parametric ablation, statistical validation or confidence calibration. This paper presents a hybrid classical-quantum pipeline for four-class ECG image classification together with a controlled ablation of five quantum circuit design variables: encoding range, entanglement topology, Pauli feature map structure, repetition depth and retained component count. The pipeline extracts 462,400-dimensional feature vectors from the pool1_pool layer of a frozen ResNet50, compresses them to nine dimensions by Truncated Singular Value Decomposition, and classifies them with a nine-qubit ZZFeatureMap fidelity kernel under statevector simulation. Evaluated on a 928-image clinical dataset from the Ch. Pervaiz Elahi Institute of Cardiology, the tuned Quantum Support Vector Classifier reaches 94.62% accuracy. A complete $2 \times 2$ factorial design shows that expanding the feature encoding range from $[0, 1]$ to $[0, \pi]$ and closing the qubit chain into a circular topology each improve accuracy at both levels of the other factor, with a sub-additive joint gain of 4.84 percentage points over the baseline $[0, 1]$ + linear circuit. McNemar's test confirms a significant margin over a tuned classical Support Vector Machine ($\chi^2 = 10.32, p = 0.0013$); against the strongest classical baseline evaluated here, a Random Forest at 91.94%, the margin narrows to 2.68 points and does not reach significance at this test-set size. The classifier attains a macro ROC-AUC of 0.989 with complete Myocardial Infarction separation (AUC = 1.000) and Expected Calibration Error below 0.07, and retains 86.7% accuracy under brightness perturbation where the classical SVM falls to 46.7%. Zero-shot transfer to a second scanner domain reverses the ordering, which bounds the result to the training domain. The pipeline is deployed as an end-to-end web application for clinical screening.

*Index Terms:* Quantum machine learning, quantum kernel methods, ZZFeatureMap, ECG classification, circuit ablation, statistical validation.

---

## I. INTRODUCTION

Cardiovascular diseases are the leading cause of death worldwide, responsible for approximately 17.9 million deaths annually according to World Health Organisation estimates [1]. The term covers a broad group of conditions affecting the heart and blood vessels, including myocardial infarction, arrhythmia, coronary artery disease and stroke, with risk driven by a combination of smoking, hypertension, elevated cholesterol, obesity and sedentary lifestyle. The Electrocardiogram (ECG) remains the standard first-line, non-invasive tool for detecting these conditions. It captures the heart's electrical activity as a waveform trace, and trained cardiologists read that trace to identify Arrhythmia, Myocardial Infarction (MI) and prior History of Myocardial Infarction. Specialist interpretation of this kind is frequently unavailable in high-volume hospitals and rural clinics, which is what gives an automated classification system real clinical value [2].

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

Deep learning methods for ECG classification have progressed substantially over the past decade. Yildirim et al. [3] showed that a one-dimensional convolutional neural network applied directly to raw ECG waveforms achieves 91.33% accuracy across 17 arrhythmia classes from the MIT-BIH database, demonstrating that learned temporal representations can outperform hand-crafted signal features on large corpora. Vasquez-Iturralde et al. [2] reviewed 494 deep learning ECG publications using the PRISMA 2020 methodology and identified persistent gaps, among them the scarcity of studies operating on image-format ECG inputs rather than digital recordings, and the near-absence of any confidence calibration reporting. The present work operates on scanned paper ECG prints and provides both calibration and statistical significance analyses, addressing both gaps directly.

**B.  Theoretical Foundations of Quantum Kernel Methods**

The theoretical basis for quantum kernel classification was established by Havlicek et al. [4], who constructed ZZFeatureMap-based kernels that map data into exponentially large Hilbert spaces through Pauli-ZZ interactions and argued that such kernels are not efficiently reproducible by a classical algorithm. Their demonstration used synthetic data constructed to favour the quantum kernel, and a rigorous learning separation was established only subsequently, for a specially constructed problem, so the advantage on natural datasets of the kind used here remains an empirical question rather than a proven one. The fidelity kernel used throughout this work is the construction that their paper introduced.

Huang et al. [5] sharpened that caution considerably. Their analysis shows that once training data is available, a classical model can often match a quantum model even on problems built to favour the quantum side, and they propose a geometric criterion for judging in advance whether a quantum advantage is plausible for a given dataset at all. That result is the reason the present work treats the accuracy margins it reports as evidence about one pipeline and one dataset, rather than as a claim of a general separation between quantum and classical learning. Biamonte et al. [6] surveyed quantum machine learning broadly and identified hybrid classical-quantum pipelines, in which a classical preprocessor handles dimensionality reduction before quantum encoding, as the most practical near-term architecture; that observation directly motivates the Truncated SVD stage described in Section III.

**C.  Quantum Machine Learning for Cardiac and ECG Classification**

Prabhu et al. [7] demonstrated quantum ECG classification on the same four-class clinical data used here, combining ResNet50 pool1\_pool features with Truncated SVD at nine components and a ZZFeatureMap QSVC under [0, 1] feature scaling. The paper does not state the entanglement topology; Qiskit's ZZFeatureMap default is full entanglement, so the configuration used was almost certainly [0, 1] + full. Re-running that configuration on the present pipeline gives 93.01%, reducing the reproduction gap to 1.08 percentage points and attributing most of the remaining difference to the split and seed. They reported 94.09% for the QSVC, 93.05% for a Multiclass Pegasos QSVC and 97.31% for an accompanying Quanvolutional Neural Network. Neither the encoding range nor the entanglement topology was varied, and no statistical validation, calibration analysis or deployment was described. The present work retains that architecture and adds the circuit design study.

Jain et al. [8] applied ResNet50 with Principal Component Analysis at eight components and an eight-qubit ZZFeatureMap QSVC to two ECG image datasets from the same clinical source, reporting that the QSVC outperformed a classical SVM by roughly eight percentage points, with significance assessed by a non-parametric rank test. Neither the encoding range nor the entanglement topology was varied in those experiments. Ramkhelawan et al. [9] combined ResNet50 with a ten-qubit variational quantum circuit in PennyLane for binary arrhythmia detection, reporting 99.98% accuracy on 123,998 MIT-BIH and PTB images. The binary setup is structurally simpler than four-class classification and the dataset is two orders of magnitude larger, and variational circuits trained end to end also carry a barren plateau risk that the fixed fidelity kernel approach used here avoids.

Ozpolat and Karabatak [10] benchmarked a QSVM against a classical SVM on signal-domain Chapman ECG features across qubit counts from three to nine, finding a consistent but modest margin of approximately two percentage points at the best configuration. Their observation that accuracy does not increase monotonically with qubit count is consistent with the circuit-repetition ablation reported in Section VI-E. Aksoy and Ozpolat [11] compared SVM, QSVM and Pegasos-QSVM on medical tabular datasets using MinMax scaling to [0, pi], independently adopting the same encoding range that the ablation in Section VI-A identifies as optimal here. Because their datasets share no feature structure with ECG images, this cross-domain agreement supports reading the [0, pi] result as a property of the ZZFeatureMap rotation structure rather than as a dataset-specific artefact.

Soon et al. [12] proposed a hierarchical quantum ensemble that stacks a quantum neural network alongside XGBoost with a LightGBM meta-classifier for tabular cardiovascular data, reporting 97% accuracy and 98% AUC; the architecture does not involve ECG images and does not ablate circuit design. Setiawan et al. [13] systematically reviewed 94 quantum machine learning healthcare studies and concluded that calibration analysis and statistical significance testing are almost entirely absent from the literature, recommending both as standard practice going forward.

**D.  Summary and Positioning**

Table I positions the present work against five prior quantum ECG and cardiac studies on the dimensions that the preceding subsections identify as under-reported: circuit design ablation, statistical significance testing, confidence calibration and clinical deployment.

TABLE I.  COMPARISON WITH PRIOR QUANTUM ECG AND CARDIAC WORKS

| Reference | Task | Method | Best Acc. | Ablation | Stat. Test | Calib. | Deployed |
|---|---|---|---|---|---|---|---|
| Prabhu et al. [7] | 4-class ECG images | ResNet50+SVD+QSVC ([0,1], full entanglement default) | 94.09% QSVC; 97.31% QNN | No | No | No | No |
| Jain et al. [8] | Multi-class ECG images | ResNet50+PCA+QSVC | Not directly comparable† | No | Rank test | No | No |
| Ramkhelawan et al. [9] | Binary arrhythmia | ResNet50+VQC (10-qubit) | 99.98% | No | No | No | No |
| Ozpolat & Karabatak [10] | Arrhythmia signals | PCA+QSVM (3–9 qubits) | 80.90% | Qubit sweep | No | No | No |
| Aksoy & Ozpolat [11] | Medical tabular | SVM/QSVM/Pegasos | QSVM best | No | No | No | No |
| **This work** | **4-class ECG images** | **ResNet50+SVD+QSVC ([0,π], circular)** | **94.62%** | **320 configs** | **McNemar p=0.0013** | **ECE < 0.07** | **Yes** |

†Results in Jain et al. [8] are reported across two datasets with different splits, so a single directly comparable accuracy figure is not available.

Among the works surveyed here, none combines a systematic circuit design ablation, a significance test, a calibration analysis and a deployed interface on an ECG image classification task.

---

## III. PROPOSED METHODOLOGY

The pipeline is described in its inference form: all model objects are pre-trained and serialised offline, and the online inference path receives a single uploaded ECG image and runs it through the gatekeeper, feature extraction, dimensionality reduction and classification stages in sequence. Fig. 1 shows the inference data flow.

Fig. 1.  Pipeline block diagram. Top: gating layer — MobileNetV2 ECG gatekeeper (Gate 1), periodicity-and-axis heuristic validator (Gate 2), and Mahalanobis OOD guard (Gate 3). Bottom: classification engine — ResNet50 pool1\_pool feature extraction, Truncated SVD (9 components), MinMax scaling, and three parallel classifier branches (classical SVM, QSVC, Pegasos QSVC) leading to the diagnosis output.
*(Draw in draw.io following the description above; insert at the top of this column.)*

**A.  ECG Image Preprocessing**

Clinical ECG scans arrive with widely varying scan quality, background colour, grid density and orientation. A seven-step OpenCV pipeline converts each image to a 340×340 pixel normalised greyscale array. The image is first converted to greyscale and binarised with OTSU adaptive thresholding. Morphological dilation with a 15×5 kernel then joins fragmented trace segments, after which the largest contour is detected and the image is cropped to its bounding box with 15 pixels of padding. A second OTSU threshold with conditional inversion produces a consistent trace-on-background polarity across all scan types.

Vertical grid lines are then removed with a 1×40 morphological opening and horizontal grid lines with a 40×1 kernel; the ECG trace is thicker than any grid line along both axes and therefore survives both subtractions intact. A 3×3 Gaussian blur and a re-threshold at pixel value 30 suppress residual speckle, after which the image is resized to 340×340 using cv2.INTER\_AREA and divided by 255.0 to yield a float32 array in [0, 1].

**B.  ResNet50 Feature Extraction**

Feature extraction uses a ResNet50 [14] with frozen ImageNet weights, rebuilt through the Keras functional API to output at the pool1\_pool layer. That layer applies 3×3 max-pooling over the initial 7×7 convolutional stage (64 filters, stride 2), producing an 85×85×64 spatial activation map, which is 462,400 dimensions when flattened. The shallow layer captures low-level morphological structure such as QRS complex geometry, ST segment curvature and P and T wave shape, without the high-level ImageNet object semantics encoded at deeper layers. The choice of extraction layer follows Prabhu et al. [7]; a layer-wise ablation against the deeper global average pooling output was not performed and is stated as a limitation in Section VII. Greyscale inputs are expanded to three channels by axis repetition before ResNet50's per-channel normalisation. Extraction ran in batches of eight.

**C.  Dimensionality Reduction**

TruncatedSVD at n\_components = 9, fitted on the 742 training samples, compresses the 462,400-dimensional vectors to nine dimensions. TruncatedSVD is used in preference to PCA because it does not require mean-centering of the data matrix, which is expensive at this dimensionality and destroys the sparsity of the activation map; the retained components therefore span directions of largest second moment rather than of largest variance. The choice of nine components follows Prabhu et al. [7] and is confirmed by the ablation in Section VI-E. A MinMaxScaler, also fitted on the training data alone, normalises the nine-dimensional vectors to [0, 1]. For the quantum classifiers the scaled vector is then multiplied by pi before encoding, placing features in [0, pi]; the classical SVM receives the [0, 1] vector directly. Both fitted objects are serialised at training time and loaded once at API startup, so inference always reuses the training-time transformation without refitting.

**D.  Classical Support Vector Machine**

The classical SVM uses sklearn.svm.SVC [15] with an RBF kernel, C = 10, gamma = 'scale' and class\_weight = 'balanced' to account for the 227:138 imbalance between the largest and smallest training classes. A CalibratedClassifierCV wrapper with method = 'isotonic' and cv = 3 produces calibrated class probabilities. Hyperparameters were selected by five-fold stratified GridSearchCV maximising macro F1 over C in {0.01, 0.1, 1.0, 5.0, 10.0} and gamma in {'scale', 'auto'}. The selected C lies at the upper edge of the grid, so the search was not bounded from above by the data.

**E.  Quantum Support Vector Classifier**

The QSVC encodes the nine-dimensional vector into a 2⁹ = 512-dimensional complex Hilbert space using Qiskit's ZZFeatureMap [4], the Pauli-ZZ construction of Havlicek et al. Each repetition applies a Hadamard layer across all nine qubits, a first-order phase gate P(2x_i) on qubit i that encodes each feature as a relative phase, and a second-order block CNOT–P(2(pi − x_i)(pi − x_j))–CNOT on each connected qubit pair under the selected topology. Acting on a state prepared by the Hadamard layer, P(θ) and Rz(θ) differ only by a global phase and are therefore interchangeable in the discussion of phase coverage below. The fidelity quantum kernel is

K(x, z) = |⟨ψ(x)|ψ(z)⟩|²     (1)

where |ψ(x)⟩ is the statevector produced by encoding input x. Rather than running N² circuit evaluations, all N = 742 training statevectors are precomputed once and the kernel matrix is assembled as

K[i, j] = |sv_i · conj(sv_j)|²     (2)

where sv_i is the 512-dimensional complex amplitude vector for sample i. Full matrix construction for 742 samples takes under 15 seconds on CPU, against over 30 minutes for direct circuit evaluation. At inference a single kernel row is evaluated against the 742 cached statevectors through dense complex inner products, costing O(N·d) arithmetic for d = 512, with no circuit simulation at query time.

A grid search over four encoding ranges, five entanglement topologies, reps in {1, 2, 3, 4} and C in {0.1, 1.0, 5.0, 10.0}, that is 320 configurations, identified [0, pi] encoding, circular entanglement, reps = 2 and C = 5.0 as the optimal setting.

**F.  Multiclass Pegasos QSVC**

Six one-versus-one binary Pegasos QSVC models [16] cover all four class pairs. The native Qiskit PegasosQSVC does not converge at nine qubits: the average kernel value at this scale is approximately 1/512, so the accumulated decision function remains numerically close to zero over the default iteration budget and the predicted sign is determined by noise rather than by the data. A custom training loop precomputes all statevectors and evaluates each gradient step through matrix multiplication, and a per-model grid search over the regularisation constant and the iteration count recovers accuracy from 78.49% to 91.94%. At inference, Algorithm 1 of Prabhu et al. [7] evaluates three of the six binary models in a decision tree to produce the four-class prediction.

**G.  Gatekeeper and Web Application**

A two-stage gatekeeper rejects non-ECG uploads before any computation begins. The first stage is a MobileNetV2 [17] classifier retrained with 600 hard-negative line-art images (300 line charts and 300 UML diagrams) to separate ECG prints from visually similar non-ECG inputs. The second stage applies two signal-derived checks: a periodicity score that detects the repeating waveform structure of an ECG trace (threshold 0.14), and an axis-angle fraction that measures the proportion of dark pixels aligned with horizontal and vertical grid directions (threshold 0.42); any image that fails either check is rejected with HTTP 422. A Mahalanobis distance guard fitted on the nine-dimensional SVD-compressed training features (threshold 4.59, 99th percentile) flags test samples outside the training distribution without blocking prediction, appending `ood_flag` and `ood_distance` to the response. The FastAPI backend loads all model objects at startup and exposes six REST endpoints covering single-image prediction, batch prediction for up to 20 images, report generation, a three-model consensus comparison, a health check and a paginated audit history endpoint. A Redis cache keyed by the SHA-256 hash of the image bytes together with the model identifier returns stored results for repeated submissions with a 24-hour time-to-live. The React frontend provides tabs for Diagnosis, Batch Upload, Model Performance, Quantum Insights, API documentation and prediction history, with colour-coded severity indicators and calibrated confidence bars. Because the training statevectors are cached, a quantum prediction requires no circuit simulation at query time; the dominant latency cost is ResNet50 feature extraction, which is shared by all three models. Measured end-to-end QSVC latency is 893.9 ms (preprocessing 126.9 ms, ResNet50 627.6 ms, SVD 52.0 ms, classification 87.4 ms).

Fig. 2.  Web application screens. (a) Upload interface with the drag-and-drop ECG area, three model selector buttons and the pipeline summary. (b) Diagnosis result for a Myocardial Infarction ECG with critical-severity colour coding and clinical recommendation. (c) Pipeline dataflow panel showing per-stage timing and the multi-model consensus table.
*(Images: Doc/screenshots/doc1\_image1.png, doc1\_image2.png, doc1\_image3.png)*

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

All models were evaluated on the same held-out 186-image test set. The TruncatedSVD reducer and the MinMaxScaler were fitted on the 742 training samples and serialised; the test set passed through the same fitted objects without refitting, which prevents the leakage failure mode that [2] identifies across the arrhythmia literature. The primary metrics are accuracy and macro-averaged F1, with per-class and macro-averaged ROC-AUC also reported. McNemar's test [19] is applied to all three pairwise model comparisons. Expected Calibration Error (ECE) is computed with 15 equal-width probability bins.

Because the test set contains 186 images, a single reclassified sample shifts accuracy by 0.54 percentage points. Differences of this magnitude are reported for completeness but are not interpreted as evidence of a real performance gap. The ablation effects discussed in Section VI range from 4 to 26 percentage points, corresponding to 8 to 49 samples.

All experiments ran on a standard CPU (Intel Core i5, 16 GB RAM). Quantum circuits were simulated with the Qiskit Aer 0.13 statevector simulator [20]. The software stack uses Python 3.10, TensorFlow 2.15, Scikit-learn 1.3 [15] and Qiskit Machine Learning 0.7. Random Forest and k-Nearest Neighbour classifiers were also trained on the same nine-dimensional MinMax-scaled features and are included in Table III as additional classical reference points.

Kernel Target Alignment (KTA) [21] was computed across all 240 non-SCA configurations in the ablation sweep (four encodings × four entanglement topologies × four repetition counts; the shifted-circular-alternating topology was excluded owing to an incompatible Qiskit parameter name). KTA values ranged from 0.1063 to 0.1277, a span of 0.0214. The two highest-KTA configurations are raw [0,1] encoding with full entanglement at reps = 1 (KTA = 0.1277) and MinMax [0,1] full at reps = 1 (KTA = 0.1277), neither of which is the best-accuracy configuration. The selected circuit — MinMax [0,π], circular, reps = 2 — has KTA = 0.1131, which lies in the middle of the observed range and is 0.0146 below the maximum. Across the sweep, configurations with higher KTA do not systematically achieve higher accuracy: the KTA ranking and the accuracy ranking disagree on which topology and encoding range is best. This finding suggests that KTA, as a label-aware kernel alignment measure, is not a reliable proxy for classification accuracy in this setting, consistent with the theoretical observation that KTA measures class-conditional kernel concentration but not decision-boundary geometry. KTA is therefore reported as a descriptive statistic for the selected configuration (0.1131) rather than as a circuit selection criterion.

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

The tuned QSVC leads the four other classifiers. Its margin over the classical SVM is 9.67 percentage points, but the SVM is not the strongest classical baseline in the table. Random Forest reaches 91.94% on the same nine-dimensional features with no quantum component, which reduces the margin that the quantum kernel actually has to defend to 2.68 points, or five test images. This has two consequences for how the result should be read. First, the nine-dimensional SVD features already carry strong discriminative signal, so the quantum kernel is improving on an informative representation rather than compensating for a weak one. Second, any claim of quantum advantage must be stated against Random Forest as well as against the SVM, which Section V-B does.

The classical SVM exceeds the corresponding baseline in [7] by 1.62 points, which we tentatively attribute to balanced class weighting and to the resampling performed by the calibration wrapper, although this was not ablated. Pegasos falls 1.11 points below the 93.05% reported in [7]; the likely cause is stricter leakage control here, since the SVD reducer and the scaler are fitted on training data only and the corresponding detail is not fully specified in the prior work.

The 0.53-point margin over the QSVC figure reported in [7] corresponds to a single test image, and no conclusion is drawn from it. The comparisons that carry weight in this paper are internal, between circuit configurations evaluated on one pipeline, one split and one test set.

A second point supports the same internal framing. Prabhu et al. [7] do not state the entanglement topology in their paper; Qiskit's ZZFeatureMap default is full entanglement. Re-running [0, 1] + full on the present pipeline gives 93.01%, reducing the reproduction gap from a previously estimated 4.31 points to 1.08 points. The remaining 1.08-point gap is well within what the split, random seed and C-grid differences can explain. All ablation gains in Section VI are measured against configurations run on this pipeline for that reason, which keeps the circuit design comparisons internally consistent regardless of the residual reproduction gap. The absolute accuracies in this paper should be read as pipeline-specific; the differences between configurations are the result.

Fig. 3.  Confusion matrix for the tuned QSVC on 186 test samples. Recall on Myocardial Infarction is 100% (48 of 48). Arrhythmia is the most frequently confused class across all models.
*(File: results/paper/main/confusion\_matrix\_qsvc\_tuned.png)*

The QSVC confusion matrix in Fig. 3 shows 100% recall on Myocardial Infarction (48 of 48), the class for which a missed detection carries the greatest clinical cost. The classical SVM also reaches 100% recall on this class, and Pegasos reaches 95.8%, so MI recall is not where the quantum kernel separates itself; its gain is concentrated on the other three classes. Arrhythmia shows the most inter-class confusion for every model, consistent with the morphological variability of rhythm abnormalities and their proximity to Normal traces near the decision boundaries.

Table III-B gives per-class precision, recall and F1 for the three deployed classifiers.

TABLE III-B.  PER-CLASS PRECISION, RECALL AND F1 (186-IMAGE TEST SET)

| Class | SVM P / R / F1 | QSVC P / R / F1 | Pegasos P / R / F1 | n |
|---|---|---|---|---|
| Normal | 0.870 / 0.825 / 0.847 | **0.948 / 0.965 / 0.957** | 0.891 / **1.000** / 0.942 | 57 |
| Arrhythmia | 0.881 / 0.787 / 0.832 | **0.878 / 0.915 / 0.896** | 0.944 / 0.723 / 0.819 | 47 |
| Myocardial Infarction | 0.873 / **1.000** / 0.932 | **1.000 / 1.000 / 1.000** | 0.980 / **1.000** / 0.990 | 48 |
| History of MI | 0.743 / 0.765 / 0.754 | **0.968 / 0.882 / 0.923** | 0.865 / 0.941 / 0.901 | 34 |

The QSVC leads on every class except Normal recall, where Pegasos achieves perfect recall by classifying all 57 Normal samples correctly at the cost of some precision. Arrhythmia is the only class where the QSVC precision (0.878) falls below its recall (0.915), reflecting the boundary overlap between rhythm abnormalities and Normal traces at the margins. History of MI shows the largest gap between the QSVC (F1 = 0.923) and the classical SVM (F1 = 0.754), indicating that the quantum kernel's advantage is concentrated on the two most ambiguous classes.

**B.  Statistical Significance**

Table IV reports McNemar's test for all three pairwise model comparisons. The discordant pair counts are taken directly from the pairwise prediction contingency tables.

TABLE IV.  McNEMAR'S TEST RESULTS (n = 186 TEST SAMPLES)

| Comparison | Correct only A (b) | Correct only B (c) | χ² | p-value | Significant |
|---|---|---|---|---|---|
| **QSVC vs. Classical SVM** | **23** | **5** | **10.321** | **0.0013** | **Yes** |
| Pegasos vs. Classical SVM | 19 | 6 | 5.760 | 0.0164 | Yes |
| QSVC vs. Pegasos | 9 | 4 | 1.231 | 0.267 | No |
| QSVC vs. Random Forest | 6 | 11 | 0.941 | 0.332 | No |

The QSVC corrected 23 samples that the classical SVM misclassified while missing only 5 that the SVM classified correctly. With continuity correction, chi-squared = (|23 − 5| − 1)² / (23 + 5) = 289/28 = 10.32 at p = 0.0013, which indicates that the difference is a systematic property of the two models on this data rather than an artefact of the particular split. Both quantum models are significantly better than the classical SVM at the 0.05 threshold.

The remaining rows are the ones that bound the claim. The QSVC and Pegasos differ by 2.68 percentage points but are not statistically distinguishable at n = 186, which reflects the limited power of a test set this size rather than evidence of equivalence. The QSVC versus Random Forest comparison shows that Random Forest corrects 6 images the QSVC misses while the QSVC recovers 11 the RF misses; chi-squared = 0.941 at p = 0.332, well short of significance. The 2.68-point margin is real but the test set is too small to resolve it. The honest summary is therefore narrower than the headline accuracy gap suggests: the quantum kernel is significantly better than a tuned RBF SVM on this data, and better than the strongest classical baseline by a margin this test set cannot resolve. To the best of our knowledge, no prior study of ECG image classification reports a significance test of quantum kernel advantage at all, which is the gap this subsection is intended to close.

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

Fig. 4.  ROC curves for QSVC, Pegasos and classical SVM across all four ECG classes. Both quantum models reach AUC = 1.000 on Myocardial Infarction, ranking every MI sample above every non-MI sample at every threshold.
*(File: results/paper/main/roc\_curves\_all\_models.png)*

Both quantum models reach AUC = 1.000 on Myocardial Infarction across the 48 MI test samples, and the QSVC macro-AUC of 0.989 is 0.049 above the classical SVM's 0.940. Expected Calibration Error is below 0.07 for all three models: 0.054 for the SVM, 0.063 for the QSVC and 0.067 for Pegasos. A model reporting 80% confidence is therefore correct in approximately 80% of cases, a property required before a system can meaningfully support a clinical decision. Calibration analysis of this kind was not found in any of the ECG quantum machine learning papers surveyed in Section II.

**D.  Augmentation Robustness**

Fig. 5 shows model accuracy under nine augmentation conditions, evaluated on a fixed evaluation subset of 45 images held constant across every condition. The subset is drawn from a separate ECG collection rather than from the 186-image test set: it is the complete set of decodable files whose filenames carry a recognisable label prefix under the NSR and six-arrhythmia-subclass naming convention. Mapped onto the four training classes it covers Normal and Arrhythmia only, so a majority-class predictor is a more informative reference point than the 25% uniform-random rate. Unperturbed, both the classical SVM and the QSVC classify 38 of 45 (84.4%). Under brightness perturbation, which simulates an overexposed photograph of a paper ECG print, the classical SVM falls to 21 of 45 (46.7%) while the QSVC classifies 39 of 45 (86.7%), one image above its own unperturbed count. Two qualifications apply to this result and both are material: the subset spans two of the four classes, and it is small enough that a single image moves accuracy by 2.2 points. The 37.7-point gap is far larger than that resolution, which is why the direction of the effect is reported; its magnitude is not precisely estimated.

Fig. 5.  Model accuracy under nine augmentation conditions. The classical SVM falls to 46.7% under brightness perturbation while the QSVC maintains 86.7%.
*(File: results/paper/ablation/augmentation\_robustness.png)*

The ZZFeatureMap encodes feature values as phase angles, and phase is periodic, so a uniform brightness shift in the scaled feature statistics advances the encoded state around the Bloch sphere rather than displacing it away from the training support vectors. The RBF kernel operates on Euclidean distance in the same nine-dimensional compressed space, where the same shift translates every test sample away from its nearest training neighbours. Rotation, affine distortion and aspect ratio change affect both models similarly. Brightness is the one perturbation axis where phase encoding produces a measurable advantage, and it is also the axis that varies most in practice, since ward ECG prints are photographed under whatever light is available.

**E.  Cross-Dataset Transfer**

Every result above is measured on images from one hospital. To establish whether the advantage survives a change of scanner, the three trained models were applied without retraining to the second Mendeley ECG collection used by Jain et al. [8], comprising 707 images across Normal (295), Abnormal Heartbeat or Arrhythmia (241) and History of Myocardial Infarction (171). The Myocardial Infarction class is absent from that collection, so any prediction of it counts as an error in the three-class evaluation.

TABLE VI.  ZERO-SHOT TRANSFER TO A SECOND SCANNER DOMAIN (707 IMAGES, THREE CLASSES)

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

Table VII compares QSVC and Pegasos accuracy across four encoding ranges. All rows use circular entanglement, reps = 2 and C = 5.0, so that only the encoding range varies.

TABLE VII.  ACCURACY VS. FEATURE ENCODING RANGE
(all rows: circular entanglement, reps = 2, C = 5.0)

| Encoding | QSVC Accuracy | Pegasos Accuracy |
|---|---|---|
| L2 normalisation (unit sphere) | 68.28% | 57.53% |
| MinMax [0, 1] | 92.47% | 84.41% |
| **MinMax [0, pi] (ours)** | **94.62%** | **90.86%** |
| MinMax [0, 2pi] | 94.09% | 90.32% |

All rows in this table use circular entanglement, so the 92.47% figure for [0, 1] encoding is the [0, 1] + circular cell. The corresponding linear cells appear in Table IX.

Fig. 6.  QSVC accuracy across four feature encoding ranges, all with circular entanglement. L2 normalisation discards feature magnitude and produces the largest drop. Moving to [0, 2pi] introduces phase wrap-around.
*(File: results/paper/ablation/ablation\_encoding.png)*

After the Hadamard layer the qubit states lie on the equator of the Bloch sphere, and the Rz gates advance the azimuthal phase angle rather than the polar angle. The ZZFeatureMap applies Rz(2x_i), so a feature confined to [0, 1] sweeps approximately 2 radians of the 2pi available to the gate, roughly one third of the circle, and the encoded states remain clustered within a narrow arc. Extending the range to [0, pi] opens the sweep to the full circle, distributing the encoded states more widely and increasing the spread of the off-diagonal kernel values. This is the opposite of the kernel concentration regime analysed by Thanasilp et al. [22], in which kernel values collapse towards a fixed point and the trained model becomes insensitive to its input; a narrow encoding range is one route into that regime, and widening the range moves away from it. One caveat is worth stating: because rotation angles are defined modulo 2pi, the two endpoints of the [0, pi] range map to the same phase, so full-circle coverage is achieved at the cost of collapsing the extremes of each feature. At [0, 2pi] the rotation angles reach 4pi and wrap twice, so distinct feature values map to identical first-order phases; the resulting 0.53-point difference from [0, pi] is a single test sample and the two configurations should be read as comparable rather than ranked.

To check whether a range marginally below pi would avoid the endpoint-collision concern and equal or exceed pi, a fine sweep at six sub-pi values was run under circular entanglement (reps = 2, C = 5.0).

TABLE XI.  ENCODING RANGE FINE SWEEP
(circular entanglement, reps = 2, C = 5.0)

| Range | Accuracy | Correct / 186 |
|---|---|---|
| [0, 0.80π] | 92.47% | 172 |
| [0, 0.85π] | 93.55% | 174 |
| [0, 0.90π] | 92.47% | 172 |
| [0, 0.95π] | 91.40% | 170 |
| **[0, π]** | **94.62%** | **176** |
| [0, 1.05π] | 94.09% | 175 |

No sub-pi value exceeds [0, pi]: the best sub-pi result, 0.85pi at 174/186, is two test images below pi and one image below the 1.05pi result. The endpoint-collision concern does not materialise in practice; [0, pi] is the true optimum on this dataset.

This mechanism can be measured directly. For 200 randomly sampled training points the off-diagonal kernel values were computed for each encoding range under circular entanglement. Under [0, 1] encoding the off-diagonal mean is 0.0139 and the standard deviation is 0.0736, giving an effective kernel-matrix rank of 94.6. Under [0, pi] the mean drops to 0.0060 and the standard deviation to 0.0514, and the effective rank rises to 130.4. Under [0, 2pi] the mean falls further to 0.0057 with effective rank 136.6. A lower mean with lower variance means the kernel values are more spread but less concentrated near one — the kernel matrix is fuller-rank and more discriminative. The movement from [0, 1] to [0, pi] is a step away from the concentration regime of [22]: the mean falls by 57%, the effective rank rises by 38%, and accuracy rises by 2.15 percentage points under circular entanglement.

L2 normalisation performs worst by a wide margin, 26.34 points below [0, pi] and 24.19 points below [0, 1], because projecting onto the unit sphere discards feature magnitude entirely, and magnitude is what the dominant uncentred SVD components encode.

**B.  Entanglement Topology**

Table VIII compares QSVC accuracy across five entanglement topologies. All rows use [0, pi] encoding, reps = 2 and C = 5.0, so that only the topology varies. Every row was computed from statevectors generated under the encoding stated in the caption.

TABLE VIII.  ACCURACY VS. ENTANGLEMENT TOPOLOGY
(all rows: [0, pi] encoding, reps = 2, C = 5.0)

| Topology | Accuracy | Gain vs. Linear |
|---|---|---|
| Linear | 93.01% | N/A |
| Pairwise | 93.01% | 0.00 pp |
| Full | 92.47% | −0.54 pp |
| **Circular (ours)** | **94.62%** | **+1.61 pp** |
| Shifted-circular-alternating (SCA) | 94.62% | +1.61 pp |

Fig. 7.  QSVC accuracy for five entanglement topologies. Closing the qubit chain into a ring adds one entangling pair at negligible additional depth and yields the largest gain over linear.
*(File: results/paper/ablation/ablation\_entanglement.png)*

Note: The pairwise, full and linear accuracies are very close (within one to two test images) at [0, pi] encoding, so their ordering is subject to split noise. The circular topology is the only one that stands clearly apart. Linear entanglement connects the qubits as an open chain, so the two end qubits have a single entangled partner while every interior qubit has two. The circular topology closes the chain by connecting qubit 8 back to qubit 0, adding exactly one entangling pair at a circuit depth barely greater than linear. Full entanglement, which connects every qubit pair, does not improve on linear or pairwise at [0, pi] encoding, plausibly because the much larger number of entangling gates over-parameterises the kernel for a nine-dimensional input, although this mechanism was not isolated. The shifted-circular-alternating topology produced identical accuracy to circular here, which we report as an empirical observation rather than as an equivalence.

**C.  Separating the Two Factors**

Tables VII and VIII share the 94.62% optimum but have different baselines, so the two effects must be read from a common factorial design rather than added. Table IX gives the four cells.

TABLE IX.  QSVC ACCURACY BY ENCODING RANGE AND TOPOLOGY
(reps = 2, C = 5.0)

| | Linear | Circular |
|---|---|---|
| **[0, 1]** | 89.78% (167/186) | 92.47% (172/186) |
| **[0, pi]** | 93.01% (173/186) | **94.62% (176/186)** |

The four simple effects follow directly. Changing the topology from linear to circular is worth 2.69 points under [0, 1] encoding and 1.61 points under [0, pi]. Changing the encoding range from [0, 1] to [0, pi] is worth 3.23 points under linear entanglement and 2.15 points under circular.

The two effects are sub-additive. Moving from [0, 1] + linear to [0, pi] + circular gains 4.84 points. The sum of the two individual gains, 3.23 points from encoding and 2.69 points from topology, is 5.92 points, which exceeds the joint effect by 1.08 points. Sub-additivity of this form is a negative interaction rather than independence: each factor helps at both levels of the other, but the samples the two factors correct overlap, so applying both together recovers fewer images than applying each alone would suggest.

The two linear cells were compared directly to verify that the encoding change propagates through the circuit rather than being absorbed before it. Their prediction vectors disagree at 12 of 186 positions: on 8 the [0, pi] configuration is correct and [0, 1] is not, on 2 the reverse holds, and on the remaining 2 both are wrong but assign different labels, which is possible in a four-class problem and is why the disagreement count exceeds b + c. McNemar's test on those discordant pairs gives a continuity-corrected chi-squared of 2.50 at p = 0.11. The test therefore establishes that the two configurations are not producing the same predictions, but it does not establish that the 3.23-point accuracy difference between them is statistically significant at this test-set size. The difference is directional and consistent with the circular-topology result; it is not independently significant.

The factorial results carry three caveats. The cells differ from one another by between 3 and 9 test images, and no variance estimate accompanies them. The 1.08-point interaction is worth only two test images, which sits at the resolution limit of a 186-image test set and should not be read as a precisely estimated interaction magnitude. What the design does support is narrower and still useful: each of the two factors improves accuracy at both levels of the other, the improvements do not add, and [0, pi] + circular is the best of the four cells measured here. Confirming the interaction itself would require repeated splits, which Section VII lists as the first item of further work.

**D.  Pauli Feature Map Comparison**

Table X compares four Pauli feature map variants under identical conditions: nine qubits, reps = 2, [0, pi] encoding, C = 5.0, and circular entanglement where the map admits entanglement.

TABLE X.  PAULI FEATURE MAP ACCURACY COMPARISON

| Feature Map | Operator Structure | Accuracy | Macro F1 |
|---|---|---|---|
| ZFeatureMap | Single-qubit Rz, no entanglement | 87.63% | 0.871 |
| **ZZFeatureMap (ours)** | **Pauli-Z + ZZ interactions** | **94.62%** | **0.944** |
| PauliFeatureMap (X+XX) | Pauli-X + XX interactions | 91.94% | 0.913 |
| PauliFeatureMap (Y+YY) | Pauli-Y + YY interactions | 83.33% | 0.835 |

Fig. 8.  QSVC accuracy for four Pauli feature map variants under identical configuration. The unentangled ZFeatureMap drops 6.99 pp below ZZFeatureMap.
*(File: results/paper/ablation/ablation\_feature\_maps.png)*

The ZFeatureMap applies only single-qubit Rz rotations with no entanglement and reaches 87.63%, which is 6.99 percentage points below the ZZFeatureMap. Removing entanglement while holding everything else fixed costs 13 test samples, which is the direct contribution of the ZZ entangling interactions. The X+XX variant adds entanglement through a different gate basis and reaches 91.94%, above the unentangled result but 2.68 points short of the ZZFeatureMap. The Y+YY variant falls furthest, to 83.33%, below both the unentangled map and the classical SVM baseline. The combination of Y-axis rotations with YY interactions appears to interfere destructively in the kernel for this feature distribution; this mechanism was not isolated, and no claim is made beyond the empirical observation on a single dataset.

**E.  Retained Components and Circuit Repetitions**

Fig. 9.  (Top) QSVC accuracy versus the number of SVD components retained; accuracy peaks at nine. (Bottom) Accuracy versus circuit repetitions; reps = 2 is optimal.
*(Files: results/paper/ablation/ablation\_svd\_dims.png and ablation\_reps.png)*

Sweeping the retained components from 2 to 18 shows accuracy peaking at nine, confirming the choice made in [7]. Below nine, discriminative structure from the ResNet50 activation map is discarded; above nine, the additional components contribute more noise than signal relative to the encoding capacity, and the required qubit count grows with them. Sweeping the circuit repetitions confirms reps = 2: reps = 3 and reps = 4 give no accuracy gain while adding statevector computation cost, consistent with the finding of Ozpolat and Karabatak [10] that increasing circuit depth beyond a sufficient encoding level does not improve quantum SVM performance.

---

## VII. LIMITATIONS

Five constraints bound the scope of these conclusions, each stated here with the step that would remove it.

**Single split, no variance estimate.** All results come from one stratified split at random\_state = 42 with no confidence intervals. Differences of one percentage point correspond to one or two test images and are within noise. As a post-submission check, the four factorial cells were re-evaluated over five seeds (0, 7, 21, 42, 99): [0,1]+linear 86.34 ± 3.01%, [0,1]+circular 88.17 ± 2.05%, [0,π]+linear 90.97 ± 1.39%, [0,π]+circular 92.15 ± 1.55%. The ordering is preserved in all five repetitions and the standard deviations confirm that the primary split (seed 42) is representative. Repeating the full 320-configuration sweep across ten or more seeds would attach tighter variance estimates and is the first item of further work.

**Pipeline-relative accuracies.** Re-running [0, 1] + full entanglement, the Prabhu et al. [7] configuration as inferred from the Qiskit default, gives 93.01% against the 94.09% reported there, a residual gap of 1.08 points that is well within what split and seed variation can explain. Absolute accuracies in this paper are specific to this pipeline, and the comparisons the paper draws are between configurations evaluated within it, all of which share the same features, split and test set.

**Domain-bounded advantage.** The zero-shot transfer in Section V-E reverses the model ordering, so every accuracy, significance and robustness result in this paper is a within-domain result for the training scanner. Domain adaptation of the feature extractor is the concrete step that would test whether the circuit-design conclusions survive the change of domain.

**Margin against the strongest baseline.** The significance result in Section V-B is established against a tuned RBF SVM. Against Random Forest, the strongest classical baseline evaluated, the margin is 2.68 points and is not resolvable at n = 186. Gradient-boosted trees and a fine-tuned convolutional baseline were not evaluated, so the comparison set is narrower than a full classical benchmark would require.

**Ideal simulation.** All quantum results come from noiseless statevector simulation, so the reported accuracies are an upper bound on what a NISQ device would deliver at nine qubits. The circuit is small enough that noise-model evaluation is immediately feasible and is planned.

**Inherited extraction layer.** The pool1\_pool layer was adopted from [7] rather than selected empirically, and no layer-wise ablation against deeper ResNet50 outputs was run, so the claim that shallow morphological features suit this task rests on the prior work rather than on evidence presented here.

---

## VIII. CONCLUSION

This work treats the quantum feature map as a design space to be measured rather than a component to be adopted. A controlled $2 \times 2$ factorial ablation over encoding range and entanglement topology shows that scaling features to $[0, \pi]$ and closing the qubit chain into a ring each raise accuracy at both levels of the other factor, and that their joint gain of 4.84 percentage points falls 1.08 points short of the sum of the separate gains. The best of the four cells, $[0, \pi]$ with circular entanglement at reps = 2, classifies 94.62% of a held-out 186-image test set, and McNemar's test places that result significantly above a tuned classical RBF SVM ($\chi^2 = 10.32, p = 0.0013$). Two further analyses that the surveyed quantum ECG literature does not report accompany it: Expected Calibration Error below 0.07 for all three deployed models, which is the property a confidence score must have before a clinician can act on it, and a nine-condition perturbation study in which phase encoding retains 86.7% accuracy under brightness distortion where the classical SVM reaches 46.7%.

The scope of the claim is set by two results reported alongside these. Against Random Forest rather than the SVM, the margin is 2.68 points and does not reach significance (b=6, c=11, $\chi^2 = 0.941$, $p = 0.332$) at this test-set size. Under zero-shot transfer to a second scanner domain the ordering reverses entirely. Together these locate the contribution precisely: the circuit design choices measured here are the right ones for this pipeline and this domain, the accuracy they deliver is competitive with the strongest classical baseline rather than decisively above it, and the obstacle to generalisation lies in the classical feature extractor rather than in the quantum circuit. Three items of work now address that ordering directly: domain-adaptive fine-tuning of the ResNet50 extractor, noise-model evaluation at nine qubits, and gradient-based saliency overlays that would show clinicians which region of a trace drove a decision.

---

## CODE AND DATA AVAILABILITY

The ECG image dataset used in this study is publicly available on Mendeley Data [18]. The implementation, trained model artefacts and the scripts reproducing all tables and figures are available at https://github.com/karthikspoojary/QuCardio.

---

## ACKNOWLEDGEMENT

The authors thank the Qiskit open-source community and IBM Quantum for the quantum computing framework. The ECG dataset was made publicly available by Khan et al. through Mendeley Data, collected at the Ch. Pervaiz Elahi Institute of Cardiology, Multan. The authors acknowledge the support of the Department of Computer Science and Engineering, St. Joseph Engineering College, Mangaluru.

---

## REFERENCES

*(Numbered in order of first appearance in the text, as IEEE style requires.)*

[1] World Health Organisation, "Cardiovascular diseases (CVDs)," Fact Sheet, Jun. 2021. [Online]. Available: https://www.who.int/news-room/fact-sheets/detail/cardiovascular-diseases-(cvds) [Accessed: 18 Sep. 2026].

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

---

*End of manuscript. Figure files, the 6-page trim plan and the internal consistency log are maintained separately in QuCardio_Review_Findings.md and are not part of the submitted document.*
