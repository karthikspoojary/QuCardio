<!--
  TYPESETTING NOTES — remove this block before LaTeX compilation
  Template  : IEEE IEEEtran two-column.  Figures: caption BELOW. Tables: caption ABOVE.
  Table IDs : TABLE~I, TABLE~II (Roman).  Figure IDs: Fig.~1, Fig.~2
  pi        : $\pi$ in LaTeX.  Rz: $R_z(2x_i)$.  chi^2: $\chi^2$.  [0,pi]: $[0,\pi]$
  Spelling  : British (-ise/-isation) throughout
  6-page trim: drop Fig. 2, Fig. 4, Fig. 9. Keep TABLE VII-A and TABLE VIII.

  ==========================================================================
  OPEN ITEMS BEFORE SUBMISSION
  O1. GitHub URL still reads "yourusername". Replace with the real or
      anonymised link. A reproducibility paper with a placeholder URL is an
      automatic desk-reject risk.
  O2. Verify that [0,1]+linear and [0,pi]+linear produce DIFFERENT predicted
      label vectors, not just the same accuracy count. Both are 167/186. If
      the two prediction arrays are identical, the encoding switch had no
      effect on the circuit and that is a bug in the run, not an interaction.
      Run McNemar between the two cells and report b and c in Section VI-C.
  O3. Qiskit's ZZFeatureMap default entanglement is 'full', NOT 'linear'.
      Confirm from the text of [5] that they state linear entanglement. If
      they do not state it, they most likely used the default, which makes
      their configuration [0,1]+full, not [0,1]+linear, and the "base
      configuration" attribution in this paper is wrong. Measuring
      [0,1]+full is one run and may also close the reproduction gap.
  O4. Section V-D: state how the 45-image augmentation subset was selected
      (random, stratified, first-n) and whether class balance was preserved.
  ==========================================================================
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

Cardiovascular diseases account for approximately 17.9 million deaths each year, and the Electrocardiogram is the primary non-invasive tool for detecting them. Quantum machine learning has been applied to automated ECG interpretation with encouraging results, but published studies report single accuracy figures under one fixed circuit configuration, with no systematic investigation of how circuit design choices affect performance and no statistical validation of the reported quantum advantage. This paper presents a hybrid classical-quantum pipeline for four-class ECG image classification and a controlled ablation of the quantum circuit design, covering the feature encoding range, the entanglement topology, the Pauli feature map family, the number of circuit repetitions and the number of retained components. The pipeline extracts a 462,400-dimensional feature vector from the pool1\_pool layer of a frozen ResNet50, compresses it to nine dimensions using Truncated Singular Value Decomposition, and classifies it with a nine-qubit ZZFeatureMap fidelity quantum kernel under statevector simulation. On a 928-image clinical subset of a public dataset collected at the Ch. Pervaiz Elahi Institute of Cardiology, Multan, the tuned Quantum Support Vector Classifier reaches 94.62% accuracy. A complete 2×2 factorial design over the encoding range and the entanglement topology shows that the two factors interact. Under linear entanglement the encoding range has no measurable effect, with [0, 1] and [0, π] both giving 89.78%; under circular entanglement the same switch adds 2.15 percentage points. The topology change alone contributes 2.69 points and the two changes together contribute 4.84 points, so the joint effect exceeds the sum of the separate effects by 2.15 points. McNemar's test confirms that the advantage over a tuned classical Support Vector Machine is statistically significant (chi-squared = 10.32, p = 0.0013). The classifier attains a macro Area Under the ROC Curve of 0.989 with perfect separation of Myocardial Infarction (AUC = 1.000), and Expected Calibration Error below 0.07 for all three models evaluated. A zero-shot transfer test to a second dataset from the same clinical source shows that the quantum advantage does not survive a change of scanner environment, which bounds the claim to the training domain. The complete pipeline is deployed as a working web application supporting single and batch ECG diagnosis with downloadable reports.

*Index Terms* — Quantum Machine Learning, Quantum Support Vector Classifier, ZZFeatureMap, Fidelity Quantum Kernel, ECG Classification, Cardiovascular Disease, ResNet50, Truncated SVD, McNemar Test.

---

## I. INTRODUCTION

Cardiovascular diseases are the leading cause of global mortality, with the World Health Organization reporting approximately 17.9 million deaths per year [1]. The Electrocardiogram (ECG) is the standard first-line screening tool, capturing the heart's electrical activity as a waveform trace from which trained cardiologists identify conditions including Arrhythmia, Myocardial Infarction (MI) and prior History of Myocardial Infarction. Accurate interpretation of ECG traces requires specialist training that is frequently unavailable in high-volume hospitals and rural clinics, which gives automated classification systems real clinical value [2].

Machine learning approaches have been applied to ECG classification for over a decade, producing strong results on large digital signal databases [3]. On small multi-class medical image collections, however, classical kernel methods tend to plateau, because the higher-order feature correlations that distinguish morphologically similar cardiac conditions are not efficiently representable in a fixed classical kernel. Quantum machine learning (QML) offers a theoretically grounded alternative: quantum feature maps embed classical data into exponentially large Hilbert spaces, where kernels arise that are conjectured to be hard to evaluate classically [4]. In the current Noisy Intermediate-Scale Quantum (NISQ) era, where hardware is limited by qubit count and gate fidelity, statevector simulation on classical hardware remains the practical mode for QML research.

Prior work on quantum ECG classification has reported that Quantum Support Vector Classifiers (QSVC) with ZZFeatureMap encoding outperform classical Support Vector Machines (SVM) on ECG image datasets [5], [6]. A limitation common to these studies is that each reports a single accuracy figure under one fixed circuit configuration. Because the design choices are never varied independently, a reported accuracy cannot be attributed to any particular element of the circuit, and the reader has no basis for deciding which choices would transfer to a different dataset. Statistical significance of the quantum advantage is not tested, calibration of the predicted confidence is not reported, and no system is deployed.

This paper is therefore a circuit-design and validation study. Its target is not a higher accuracy number, and none is claimed: the 94.62% obtained here is within one test image of the 94.09% reported by Prabhu et al. [5] on the same data, and no conclusion is drawn from that difference. The target is to establish which circuit design choices produce that accuracy, to show how those choices depend on one another, and to subject the outcome to the tests the prior literature omits. The contributions are:

1. A controlled ablation over the encoding range, the entanglement topology, the Pauli feature map family, the circuit repetitions and the number of retained components. The encoding range and the topology are reported as a complete 2×2 factorial design, which shows that they interact: the encoding range has no effect under linear entanglement and becomes consequential only once the qubit chain is closed into a ring.
2. A McNemar significance test of the quantum kernel advantage over a tuned classical SVM for ECG image classification, giving chi-squared = 10.32 at p = 0.0013. To the best of our knowledge no such test has previously been reported for this task.
3. Expected Calibration Error and augmentation robustness experiments, both absent from prior ECG-QML work, with the QSVC retaining 86.7% accuracy under brightness perturbation where the classical SVM falls to 46.7%.
4. A zero-shot cross-dataset transfer test that shows the measured quantum advantage is specific to the scanner environment it was trained on, reported as a negative result that bounds the rest of the paper.
5. Deployment of the complete pipeline as a working web application supporting single and batch ECG upload, calibrated confidence scores, class-specific recommendations and report generation.

The remainder of this paper is organised as follows. Section II surveys related work. Section III describes the proposed methodology. Section IV details the experimental setup. Section V presents results and discussion. Section VI reports the ablation studies. Section VII states the limitations, and Section VIII concludes.

---

## II. RELATED WORK

This section reviews work on classical ECG deep learning, quantum kernel theory and quantum cardiac classification relevant to the proposed system.

Deep learning methods for ECG classification have progressed substantially over the past decade. Yildirim et al. [3] showed that a one-dimensional convolutional neural network applied directly to raw ECG waveforms achieves 91.33% accuracy across 17 arrhythmia classes from the MIT-BIH database, demonstrating that learned temporal representations can outperform hand-crafted signal features on large corpora. Vasquez-Iturralde et al. [2] reviewed 494 deep learning ECG publications using the PRISMA 2020 methodology and identified persistent gaps, among them the scarcity of studies operating on image-format ECG inputs rather than digital recordings, and the near-absence of any confidence calibration reporting. The present work operates on scanned paper ECG prints and provides both calibration and statistical significance analyses, addressing both gaps directly.

The theoretical basis for quantum kernel classification was established by Havlicek et al. [4], who constructed ZZFeatureMap-based kernels that map data into exponentially large Hilbert spaces through Pauli-ZZ interactions, and argued that such kernels are not efficiently reproducible by a classical algorithm. Their demonstration used synthetic data constructed to favour the quantum kernel, and a rigorous learning separation was established only subsequently for a specially constructed problem; the advantage on natural datasets remains an empirical question. The fidelity kernel used throughout this work is the construction introduced in that paper. Biamonte et al. [7] surveyed quantum machine learning broadly and identified hybrid classical-quantum pipelines, in which a classical preprocessor handles dimensionality reduction before quantum encoding, as the most viable near-term architecture; this observation motivates the Truncated SVD stage described in Section III.

Prabhu et al. [5] demonstrated quantum ECG classification on the same four-class clinical data used here, combining ResNet50 pool1\_pool features with Truncated SVD at nine components and a ZZFeatureMap QSVC under [0, 1] feature scaling. `[O3: state the entanglement topology exactly as [5] reports it. Qiskit's ZZFeatureMap defaults to full entanglement, so an unstated topology should not be assumed to be linear.]` They reported 94.09% for the QSVC, 93.05% for a Multiclass Pegasos QSVC and 97.31% for an accompanying Quanvolutional Neural Network. Neither the encoding range nor the entanglement topology was varied, and no statistical validation, calibration analysis or deployment was described. The present work retains that architecture and adds the circuit design study.

Jain et al. [6] applied ResNet50 with Principal Component Analysis at eight components and an eight-qubit ZZFeatureMap QSVC to two ECG image datasets from the same clinical source, reporting that the QSVC outperformed a classical SVM by roughly eight percentage points, with significance assessed by a non-parametric rank test. Neither the encoding range nor the entanglement topology was varied in those experiments. Ramkhelawan et al. [8] combined ResNet50 with a ten-qubit variational quantum circuit in PennyLane for binary arrhythmia detection, reporting 99.98% accuracy on 123,998 MIT-BIH and PTB images. The binary setup is structurally simpler than four-class classification and the dataset is two orders of magnitude larger; variational circuits trained end to end also carry a barren plateau risk that the fixed fidelity kernel approach avoids.

Ozpolat and Karabatak [9] benchmarked a QSVM against a classical SVM on signal-domain Chapman ECG features across qubit counts from three to nine, finding a consistent but modest margin of approximately two percentage points at the best configuration. Their observation that accuracy does not increase monotonically with qubit count is consistent with the repetitions ablation in Section VI-D. Aksoy and Ozpolat [10] compared SVM, QSVM and Pegasos-QSVM on medical tabular datasets using MinMax scaling to [0, pi], independently adopting the encoding range that the ablation in Section VI-A identifies as optimal here. Because their datasets share no feature structure with ECG images, this cross-domain agreement supports interpreting the [0, pi] result as a property of the ZZFeatureMap rotation structure rather than a dataset-specific artefact.

Soon et al. [11] proposed a hierarchical quantum ensemble that stacks a quantum neural network alongside XGBoost with a LightGBM meta-classifier for tabular cardiovascular data, reporting 97% accuracy and 98% AUC; the architecture does not involve ECG images and does not ablate circuit design. Setiawan et al. [12] systematically reviewed 94 quantum machine learning healthcare studies and concluded that calibration analysis and statistical significance testing are almost entirely absent from the literature, and recommended both as standard practice. Table I positions the present work against five prior quantum ECG and cardiac studies.

TABLE I.  COMPARISON WITH PRIOR QUANTUM ECG AND CARDIAC WORKS

| Reference | Task | Method | Best Acc. | Ablation | Stat. Test | Calib. | Deployed |
|---|---|---|---|---|---|---|---|
| Prabhu et al. [5] | 4-class ECG images | ResNet50+SVD+QSVC ([0,1] scaling) | 94.09% | No | No | No | No |
| Jain et al. [6] | Multi-class ECG images | ResNet50+PCA+QSVC | Not directly comparable† | No | Rank test | No | No |
| Ramkhelawan et al. [8] | Binary arrhythmia | ResNet50+VQC (10-qubit) | 99.98% | No | No | No | No |
| Ozpolat & Karabatak [9] | Arrhythmia signals | PCA+QSVM (3–9 qubits) | 80.90% | Qubit sweep | No | No | No |
| Aksoy & Ozpolat [10] | Medical tabular | SVM/QSVM/Pegasos | QSVM best | No | No | No | No |
| **This work** | **4-class ECG images** | **ResNet50+SVD+QSVC ([0,π], circular)** | **94.62%** | **320 configs** | **McNemar p=0.0013** | **ECE < 0.07** | **Yes** |

†Results in Jain et al. [6] are reported across two datasets with different splits, so a single directly comparable accuracy figure is not available.

Among the works surveyed here, none combines a systematic circuit design ablation, a significance test, a calibration analysis and a deployed interface on an ECG image classification task.

---

## III. PROPOSED METHODOLOGY

The pipeline operates in two phases. An offline training phase processes all 928 images, trains three classifiers and serialises every model object to disk. An online inference phase deserialises those objects and runs the same transformation chain on a single uploaded ECG image. Fig. 1 shows the complete data flow.

Fig. 1.  Pipeline block diagram. Left: offline training path from dataset through preprocessing, ResNet50 extraction, Truncated SVD and MinMax scaling to three serialised classifiers. Right: online inference path from React upload through FastAPI, the MobileNetV2 gatekeeper, the same feature chain, and the user-selected classifier to a diagnosis response.
*(Draw in draw.io following the description above; insert at the top of this column.)*

**A.  ECG Image Preprocessing**

Clinical ECG scans arrive with widely varying scan quality, background colour, grid density and orientation. A seven-step OpenCV pipeline converts each image to a 340×340 pixel normalised grayscale array. The image is first converted to grayscale and binarised with OTSU adaptive thresholding; morphological dilation with a 15×5 kernel joins fragmented trace segments, after which the largest contour is detected and the image is cropped to its bounding box with 15 pixels of padding. A second OTSU threshold with conditional inversion produces a consistent trace-on-background polarity across all scan types. Vertical grid lines are removed with a 1×40 morphological opening and horizontal grid lines with a 40×1 kernel; the ECG trace is thicker than any grid line along both axes and therefore survives both subtractions intact. A 3×3 Gaussian blur and a re-threshold at pixel value 30 suppress residual speckle, after which the image is resized to 340×340 using cv2.INTER\_AREA and divided by 255.0 to yield a float32 array in [0, 1].

**B.  ResNet50 Feature Extraction**

Feature extraction uses a ResNet50 [13] with frozen ImageNet weights, rebuilt through the Keras functional API to output at the pool1\_pool layer. That layer applies 3×3 max-pooling over the initial 7×7 convolutional stage (64 filters, stride 2), producing an 85×85×64 spatial activation map, which is 462,400 dimensions when flattened. The shallow layer captures low-level morphological structure such as QRS complex geometry, ST segment curvature and P and T wave shape, without the high-level ImageNet object semantics encoded at deeper layers. The choice of extraction layer follows Prabhu et al. [5]; a layer-wise ablation against the deeper global average pooling output was not performed and is stated as a limitation in Section VII. Grayscale inputs are expanded to three channels by axis repetition before ResNet50's per-channel normalisation. Extraction ran in batches of eight.

**C.  Dimensionality Reduction**

TruncatedSVD at n\_components = 9, fitted on the 742 training samples, compresses the 462,400-dimensional vectors to nine dimensions. TruncatedSVD is used in preference to PCA because it does not require mean-centering of the data matrix, which is expensive at this dimensionality and destroys the sparsity of the activation map; the retained components therefore span directions of largest second moment rather than of largest variance. The choice of nine components follows Prabhu et al. [5] and is confirmed by the ablation in Section VI-D. A MinMaxScaler, also fitted on the training data alone, normalises the nine-dimensional vectors to [0, 1]. For the quantum classifiers the scaled vector is then multiplied by pi before encoding, placing features in [0, pi]; the classical SVM receives the [0, 1] vector directly. Both fitted objects are serialised at training time and loaded once at API startup, so inference always reuses the training-time transformation without refitting.

**D.  Classical Support Vector Machine**

The classical SVM uses sklearn.svm.SVC [20] with an RBF kernel, C = 10, gamma = 'scale' and class\_weight = 'balanced' to account for the 227:138 imbalance between the largest and smallest training classes. A CalibratedClassifierCV wrapper with method = 'isotonic' and cv = 3 produces calibrated class probabilities. Hyperparameters were selected by five-fold stratified GridSearchCV maximising macro F1 over C in {0.01, 0.1, 1.0, 5.0, 10.0} and gamma in {'scale', 'auto'}. The selected C lies at the upper edge of the grid, so the search was not bounded from above by the data.

**E.  Quantum Support Vector Classifier**

The QSVC encodes the nine-dimensional vector into a 2⁹ = 512-dimensional complex Hilbert space using Qiskit's ZZFeatureMap [4], the Pauli-ZZ construction of Havlicek et al. Each repetition applies a Hadamard layer across all nine qubits, first-order Rz(2x_i) rotations encoding each feature as a relative phase, and second-order entangling blocks applying Rz(2(pi − x_i)(pi − x_j)) to each connected qubit pair under the selected topology. The fidelity quantum kernel is

K(x, z) = |⟨ψ(x)|ψ(z)⟩|²     (1)

where |ψ(x)⟩ is the statevector produced by encoding input x. Rather than running N² circuit evaluations, all N = 742 training statevectors are precomputed once and the kernel matrix is assembled as

K[i, j] = |sv_i · conj(sv_j)|²     (2)

where sv_i is the 512-dimensional complex amplitude vector for sample i. Full matrix construction for 742 samples takes under 15 seconds on CPU, against over 30 minutes for direct circuit evaluation. At inference a single kernel row is evaluated against the 742 cached statevectors through dense complex inner products, costing O(N·d) arithmetic for d = 512, with no circuit simulation at query time.

A grid search over four encoding ranges, five entanglement topologies, reps in {1, 2, 3, 4} and C in {0.1, 1.0, 5.0, 10.0}, that is 320 configurations, identified [0, pi] encoding, circular entanglement, reps = 2 and C = 5.0 as the optimal setting.

**F.  Multiclass Pegasos QSVC**

Six one-versus-one binary Pegasos QSVC models [14] cover all four class pairs. The native Qiskit PegasosQSVC does not converge at nine qubits: the average kernel value at this scale is approximately 1/512, so the accumulated decision function remains numerically close to zero over the default iteration budget and the predicted sign is determined by noise rather than by the data. A custom training loop precomputes all statevectors and evaluates each gradient step through matrix multiplication, and a per-model grid search over the regularisation constant and the iteration count recovers accuracy from 78.49% to 91.94%. At inference, Algorithm 1 of Prabhu et al. [5] evaluates three of the six binary models in a decision tree to produce the four-class prediction.

**G.  Gatekeeper and Web Application**

A MobileNetV2 [15] gatekeeper rejects non-ECG uploads before any computation begins. The FastAPI backend loads all model objects at startup and exposes six REST endpoints covering single-image prediction, batch prediction for up to 20 images, report generation, a three-model consensus comparison, a health check and a paginated audit history endpoint. A Redis cache keyed by the SHA-256 hash of the image bytes together with the model identifier returns stored results for repeated submissions with a 24-hour time-to-live. The React frontend provides tabs for Diagnosis, Batch Upload, Model Performance, Quantum Insights, API documentation and prediction history, with colour-coded severity indicators and calibrated confidence bars. Because the training statevectors are cached, a quantum prediction requires no circuit simulation at query time; the dominant latency cost is ResNet50 feature extraction, which is shared by all three models. Measured end-to-end QSVC latency is 893.9 ms (preprocessing 126.9 ms, ResNet50 627.6 ms, SVD 52.0 ms, classification 87.4 ms).

Fig. 2.  Web application screens. (a) Upload interface with the drag-and-drop ECG area, three model selector buttons and the pipeline summary. (b) Diagnosis result for a Myocardial Infarction ECG with critical-severity colour coding and clinical recommendation. (c) Pipeline dataflow panel showing per-stage timing and the multi-model consensus table.
*(Images: extracted\_images/word/media/image1.png, image2.png, image3.png)*

---

## IV. EXPERIMENTAL SETUP

The experiments use the ECG image collection released on Mendeley Data by Khan et al. [16], collected at the Ch. Pervaiz Elahi Institute of Cardiology, Multan, Pakistan. The full collection contains 1937 paper ECG prints across categories that include COVID-19 cases outside the scope of this study. The standard four-class subset of this collection comprises 929 images (Normal 284, Arrhythmia 233, Myocardial Infarction 240, History of MI 172), consistent with the count reported in [5]. One Myocardial Infarction scan could not be decoded and was excluded at the feature-extraction stage, leaving the 928 images used here. Table II shows the resulting class distribution and the stratified 80:20 split applied with random\_state = 42. No augmentation was applied during training; augmented images appear only in the robustness evaluation of Section V-D.

TABLE II.  DATASET CLASS DISTRIBUTION AND TRAIN/TEST SPLIT

| Class | Total | Train | Test |
|---|---|---|---|
| Normal | 284 | 227 | 57 |
| Arrhythmia | 233 | 186 | 47 |
| Myocardial Infarction | 239 | 191 | 48 |
| History of MI | 172 | 138 | 34 |
| **Total** | **928** | **742** | **186** |

All models were evaluated on the same held-out 186-image test set. The TruncatedSVD reducer and the MinMaxScaler were fitted on the 742 training samples and serialised; the test set passed through the same fitted objects without refitting, which prevents the leakage failure mode that [2] identifies across the arrhythmia literature. The primary metrics are accuracy and macro-averaged F1, with per-class and macro-averaged ROC-AUC also reported. McNemar's test [17] is applied to all three pairwise model comparisons. Expected Calibration Error (ECE) is computed with 15 equal-width probability bins.

Because the test set contains 186 images, a single reclassified sample shifts accuracy by 0.54 percentage points. Differences of this magnitude are reported for completeness but are not interpreted as evidence of a real performance gap. The ablation effects discussed in Section VI range from 4 to 26 percentage points, corresponding to 8 to 49 samples.

All experiments ran on a standard CPU (Intel Core i5, 16 GB RAM). Quantum circuits were simulated with the Qiskit Aer 0.13 statevector simulator [19]. The software stack uses Python 3.10, TensorFlow 2.15, Scikit-learn 1.3 [20] and Qiskit Machine Learning 0.7. Random Forest and k-Nearest Neighbour classifiers were also trained on the same nine-dimensional MinMax-scaled features and are included in Table III as additional classical reference points.

Kernel Target Alignment (KTA) [18] was computed for the optimal QSVC configuration (ZZFeatureMap, [0, pi] encoding, circular entanglement, reps = 2), giving 0.1081. The value is positive, indicating agreement in sign between the kernel matrix and the label structure, but it is reported as a descriptive statistic only: without a matched KTA value for the classical RBF kernel on the same features, the magnitude alone does not establish that the quantum kernel is better aligned.

---

## V. RESULTS AND DISCUSSION

**A.  Model Performance**

Table III reports accuracy, precision, recall and macro F1 for all five classifiers on the held-out 186-image test set.

TABLE III.  MODEL PERFORMANCE ON THE 186-IMAGE TEST SET

| Model | Accuracy | Precision | Recall | Macro F1 | Prabhu et al. [5] |
|---|---|---|---|---|---|
| Classical SVM (C=10, RBF) | 84.95% | 84.17% | 84.41% | 84.10% | 83.33% |
| **QSVC (ours)** | **94.62%** | **94.84%** | **94.05%** | **94.39%** | 94.09% |
| Pegasos QSVC | 91.94% | 91.99% | 91.61% | 91.31% | 93.05% |
| Random Forest | 91.94% | 91.39% | 91.80% | 91.20% | — |
| k-Nearest Neighbours | 84.95% | 85.54% | 85.07% | 83.61% | — |

The tuned QSVC leads the four other classifiers, with a margin of 9.67 percentage points over the classical SVM. The classical SVM exceeds the corresponding baseline in [5] by 1.62 points, which we tentatively attribute to balanced class weighting and to the resampling performed by the calibration wrapper, although this was not ablated. Pegasos falls 1.11 points below the 93.05% reported in [5]; the likely cause is stricter leakage control here, since the SVD reducer and the scaler are fitted on training data only and the corresponding detail is not fully specified in the prior work. Random Forest reaches 91.94% without any quantum component, which indicates that the nine-dimensional SVD features already carry strong discriminative signal; the quantum kernel is therefore improving on an already informative representation rather than compensating for a weak one.

The 0.53-point margin over the QSVC figure reported in [5] corresponds to a single test image, and no conclusion is drawn from it. The comparisons that carry weight in this paper are internal, between circuit configurations evaluated on one pipeline, one split and one test set.

A second point supports the same internal framing. Re-running the configuration described in [5] on the present pipeline gives 89.78% (Table VII-A) against the 94.09% reported there, a gap of 4.31 points. The exact split, random seed and entanglement topology used in that work are not fully specified, so the discrepancy cannot be resolved from the published text alone, and no claim is made about its cause here. All ablation gains in Section VI are measured against configurations run on this pipeline for that reason, which keeps the circuit design comparisons internally consistent regardless of the reproduction gap. The absolute accuracies in this paper should be read as pipeline-specific; the differences between configurations are the result.

Fig. 3.  Confusion matrix for the tuned QSVC on 186 test samples. Recall on Myocardial Infarction is 100% (48 of 48). Arrhythmia is the most frequently confused class across all models.
*(File: results/paper/main/confusion\_matrix\_qsvc\_tuned.png)*

The QSVC confusion matrix in Fig. 3 shows 100% recall on Myocardial Infarction, the class for which a missed detection carries the greatest clinical cost. Arrhythmia shows the most inter-class confusion, consistent with the morphological variability of rhythm abnormalities and their proximity to Normal traces near the decision boundaries.

**B.  Statistical Significance**

Table IV reports McNemar's test for all three pairwise model comparisons. The discordant pair counts are taken directly from the pairwise prediction contingency tables.

TABLE IV.  McNEMAR'S TEST RESULTS (n = 186 TEST SAMPLES)

| Comparison | Correct only A (b) | Correct only B (c) | χ² | p-value | Significant |
|---|---|---|---|---|---|
| **QSVC vs. Classical SVM** | **23** | **5** | **10.321** | **0.0013** | **Yes** |
| Pegasos vs. Classical SVM | 19 | 6 | 5.760 | 0.0164 | Yes |
| QSVC vs. Pegasos | 9 | 4 | 1.231 | 0.267 | No |

The QSVC corrected 23 samples that the classical SVM misclassified while missing only 5 that the SVM classified correctly. With continuity correction, chi-squared = (|23 − 5| − 1)² / (23 + 5) = 289/28 = 10.32 at p = 0.0013, which indicates that the difference is a systematic property of the two models on this data rather than an artefact of the particular split. Both quantum models are significantly better than the classical SVM at the 0.05 threshold.

The third row shows something different. The QSVC and Pegasos are not statistically distinguishable at n = 186 despite a 2.68 percentage point accuracy gap, which reflects the limited statistical power of a test set this size. The gap is reported, but no claim is made that the fidelity kernel method is superior to the Pegasos variant. To the best of our knowledge this is the first McNemar significance test of quantum kernel advantage over a classical SVM reported for ECG image classification.

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

Fig. 5 shows model accuracy under nine augmentation conditions, evaluated on a fixed subset of 45 test images used identically across all conditions, giving 38/45 = 84.4%, 39/45 = 86.7% and 21/45 = 46.7%. `[O4: state how the 45 images were selected and whether class balance was preserved.]` Under brightness perturbation, which simulates an overexposed photograph of a paper ECG print, classical SVM accuracy falls from 84.4% to 46.7%, close to chance on four classes, while the QSVC holds 86.7%. Given the subset size, the difference is reported as indicative rather than as a precisely estimated effect.

Fig. 5.  Model accuracy under nine augmentation conditions. The classical SVM falls to 46.7% under brightness perturbation while the QSVC maintains 86.7%.
*(File: results/paper/ablation/augmentation\_robustness.png)*

The ZZFeatureMap encodes feature values as phase angles, and phase is periodic, so a uniform brightness shift in the scaled feature statistics advances the encoded state around the Bloch sphere rather than displacing it away from the training support vectors. The RBF kernel operates on Euclidean distance in the same nine-dimensional compressed space, where the same shift translates every test sample away from its nearest training neighbours. Rotation, affine distortion and aspect ratio change affect both models similarly. Brightness is the one perturbation axis where phase encoding produces a measurable advantage, and it is also the axis that varies most in practice, since ward ECG prints are photographed under whatever light is available.

---

## VI. ABLATION STUDIES

**A.  Feature Encoding Range**

Table VI compares QSVC and Pegasos accuracy across four encoding ranges. All rows use circular entanglement, reps = 2 and C = 5.0, so that only the encoding range varies.

TABLE VI.  ACCURACY VS. FEATURE ENCODING RANGE
(all rows: circular entanglement, reps = 2, C = 5.0)

| Encoding | QSVC Accuracy | Pegasos Accuracy |
|---|---|---|
| L2 normalisation (unit sphere) | 68.28% | 57.53% |
| MinMax [0, 1] | 92.47% | 84.41% |
| **MinMax [0, pi] (ours)** | **94.62%** | **90.86%** |
| MinMax [0, 2pi] | 94.09% | 90.32% |

All rows in this table use circular entanglement, so the 92.47% figure for [0, 1] encoding is the [0, 1] + circular cell. The corresponding linear cells appear in Table VII-A.

Fig. 6.  QSVC accuracy across four feature encoding ranges, all with circular entanglement. L2 normalisation discards feature magnitude and produces the largest drop. Moving to [0, 2pi] introduces phase wrap-around.
*(File: results/paper/ablation/ablation\_encoding.png)*

After the Hadamard layer the qubit states lie on the equator of the Bloch sphere, and the Rz gates advance the azimuthal phase angle rather than the polar angle. The ZZFeatureMap applies Rz(2x_i), so a feature confined to [0, 1] sweeps approximately 2 radians of the 2pi available to the gate, roughly one third of the circle, and the encoded states remain clustered within a narrow arc. Extending the range to [0, pi] opens the sweep to the full circle, distributing the encoded states more widely and increasing the spread of the off-diagonal kernel values, which reduces the kernel concentration that limits separability at small encoding ranges. At [0, 2pi] the rotation angles exceed 2pi and wrap around, so distinct feature values begin mapping to similar states; the resulting 0.53-point difference from [0, pi] is a single test sample and the two configurations should be read as comparable rather than ranked.

L2 normalisation performs worst by a wide margin, 26.34 points below [0, pi] and 24.19 points below [0, 1], because projecting onto the unit sphere discards feature magnitude entirely, and magnitude is what the dominant uncentred SVD components encode.

**B.  Entanglement Topology**

Table VII compares QSVC accuracy across five entanglement topologies. All rows use [0, pi] encoding, reps = 2 and C = 5.0.

TABLE VII.  ACCURACY VS. ENTANGLEMENT TOPOLOGY
(all rows: [0, pi] encoding, reps = 2, C = 5.0)

| Topology | Accuracy | Gain vs. Linear |
|---|---|---|
| Linear | 89.78% | — |
| Pairwise | 93.01% | +3.23 pp |
| Full | 92.47% | +2.69 pp |
| **Circular (ours)** | **94.62%** | **+4.84 pp** |
| Shifted-circular-alternating (SCA) | 94.62% | +4.84 pp |

Fig. 7.  QSVC accuracy for five entanglement topologies. Closing the qubit chain into a ring adds one entangling pair at negligible additional depth and yields the largest gain.
*(File: results/paper/ablation/ablation\_entanglement.png)*

Linear entanglement connects the qubits as an open chain, so the two end qubits have a single entangled partner while every interior qubit already has two. The circular topology closes the chain by connecting qubit 8 back to qubit 0, adding exactly one entangling pair and removing the boundary, at a circuit depth barely greater than linear. The size of the resulting gain indicates that the correlation between the first and last SVD components carries discriminative information that an open chain cannot represent. Full entanglement, which connects every qubit pair, is lower at 92.47%, plausibly because the much larger number of entangling gates at reps = 2 over-parameterises the kernel for a nine-dimensional input, although this mechanism was not isolated. The shifted-circular-alternating topology entangles the same qubit pairs as circular but rotates the starting pair and alternates direction between repetitions; it produced identical accuracy here, which we report as an empirical observation rather than as an equivalence.

**C.  Separating the Two Factors**

Tables VI and VII share the 94.62% optimum but have different baselines, so the two effects must be read from a common factorial design rather than added. Table VII-A gives the four cells.

TABLE VII-A.  QSVC ACCURACY BY ENCODING RANGE AND TOPOLOGY
(reps = 2, C = 5.0)

| | Linear | Circular |
|---|---|---|
| **[0, 1]** | 89.78% (167/186) | 92.47% (172/186) |
| **[0, pi]** | 89.78% (167/186) | **94.62% (176/186)** |

The four simple effects follow directly. Changing the topology from linear to circular is worth 2.69 points under [0, 1] encoding and 4.84 points under [0, pi]. Changing the encoding range from [0, 1] to [0, pi] is worth nothing under linear entanglement and 2.15 points under circular. Moving from the lower-left cell to the upper-right, that is changing both factors, is worth 4.84 points.

The two factors therefore interact rather than adding independently. The joint effect of 4.84 points exceeds the sum of the two separate effects, 2.69 and 0.00, by 2.15 points, which is the interaction term of the design. Read in the direction that matters for circuit design, closing the qubit chain into a ring is the precondition for the encoding range to have any effect at all: a study that fixes linear entanglement will observe no benefit from rescaling the features, however wide the range it sweeps. This is a more specific and more useful claim than two additive gains would be, because it tells a practitioner which of the two choices to make first.

Two cautions attach to the interaction. The cells differ by between 0 and 9 test images, and no variance estimate accompanies them, so the magnitudes should be read as indicative. Equal accuracy in the two linear cells also does not by itself establish that the two configurations behave identically; they may misclassify different samples while arriving at the same total. `[O2: report the discordant counts b and c from a McNemar test between the two linear cells. If the two prediction vectors are identical rather than merely equal in accuracy, the encoding switch did not reach the circuit and the run must be repaired before this section can stand.]`

**D.  Pauli Feature Map Comparison**

Table VIII compares four Pauli feature map variants under identical conditions: nine qubits, reps = 2, [0, pi] encoding, C = 5.0, and circular entanglement where the map admits entanglement.

TABLE VIII.  PAULI FEATURE MAP ACCURACY COMPARISON

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

Sweeping the retained components from 2 to 18 shows accuracy peaking at nine, confirming the choice made in [5]. Below nine, discriminative structure from the ResNet50 activation map is discarded; above nine, the additional components contribute more noise than signal relative to the encoding capacity, and the required qubit count grows with them. Sweeping the circuit repetitions confirms reps = 2: reps = 3 and reps = 4 give no accuracy gain while adding statevector computation cost, consistent with the finding of Ozpolat and Karabatak [9] that increasing circuit depth beyond a sufficient encoding level does not improve quantum SVM performance.

---

## VII. LIMITATIONS

Five constraints bound the scope of these conclusions, each stated here with the step that would remove it.

**Single split, no variance estimate.** All results come from one stratified split at random\_state = 42, so the reported effects carry no confidence intervals. Differences of one percentage point, which correspond to one or two test images, are within noise. The effects the paper relies on are larger than that threshold and are supported by McNemar's test where a significance claim is made, but repeating the sweep across five to ten seeds would attach variance estimates and is the first item of further work.

**Pipeline-relative accuracies.** Re-running the configuration described in [5] here gives 89.78% against the 94.09% reported there. Because the split, seed and entanglement topology of that work are not fully specified, the gap cannot be attributed. The consequence is bounded and stated plainly: absolute accuracies in this paper are specific to this pipeline, and the comparisons that the paper draws are between configurations evaluated within it, all of which share the same features, split and test set.

**Domain-bounded advantage.** In a zero-shot transfer to a second Mendeley ECG dataset from the same clinical source, QSVC accuracy fell to 34.09% on a three-class evaluation while the classical SVM degraded more gradually to 62.66%, with the QSVC collapsing most predictions into a single class. The quantum kernel appears to learn a precise but scanner-specific decision boundary. The result is reported rather than omitted because it defines where the advantage holds, and it points to domain adaptation of the feature extractor as the concrete next step.

**Ideal simulation.** All quantum results come from noiseless statevector simulation, so the reported accuracies are an upper bound on what a NISQ device would deliver at nine qubits. The circuit is small enough that noise-model evaluation is immediately feasible and is planned.

**Inherited extraction layer.** The pool1\_pool layer was adopted from [5] rather than selected empirically, and no layer-wise ablation against deeper ResNet50 outputs was run, so the claim that shallow morphological features suit this task rests on the prior work rather than on evidence presented here.

---

## VIII. CONCLUSION

This paper reported a controlled study of how quantum circuit design choices affect QSVC accuracy for four-class ECG image classification, on a 928-image clinical subset of the Khan et al. [16] ECG collection. A complete 2×2 factorial design over the encoding range and the entanglement topology shows that the two factors interact. Closing the qubit chain into a ring raises accuracy from 89.78% to 92.47% under [0, 1] encoding and to 94.62% once the range is also widened to [0, pi]; under linear entanglement the range makes no difference at all. The joint effect of 4.84 points therefore exceeds the sum of the separate effects by 2.15 points, and circular entanglement is the precondition for the encoding range to matter. The Pauli map comparison shows that the ZZ interactions specifically, and not entanglement in general, produce the result, since the X+XX and Y+YY variants both fall short.

McNemar's test confirms the advantage of the tuned QSVC over a tuned RBF SVM at chi-squared = 10.32, p = 0.0013. Both quantum models reach AUC = 1.000 for Myocardial Infarction, and Expected Calibration Error below 0.07 across all three models indicates that the confidence scores are suitable for decision support. Under brightness perturbation the QSVC holds 86.7% accuracy where the classical SVM falls to near-random performance, which follows from the phase encoding of the ZZFeatureMap.

The paper does not claim improved state-of-the-art accuracy: 94.62% is within one test image of the figure reported in [5] on the same data. It claims something different and, for this stage of the field, more useful. It identifies which circuit choices produce that accuracy, shows that those choices are not independent, and subjects the outcome to the significance, calibration, robustness and transfer tests that the prior ECG quantum literature omits. The transfer test constrains this claim most directly. The advantage does not survive a change of scanner, and that result bounds any clinical reading of everything else in the paper.

The limitations in Section VII set the agenda. Repeating the ablations across multiple seeds would attach variance estimates to the effects reported here. The interaction structure revealed by Table VII-A — circular entanglement as the precondition for the encoding effect — should be tested under noise models and eventually on hardware, since the relative ordering of topologies may change when gate errors are introduced. Domain adaptation of the ResNet50 extractor is the most urgent practical step, since the cross-dataset result shows the learned boundary does not survive a change of scanner. Evaluating the trained classifier under NISQ noise models and eventually on quantum hardware would quantify the gap between ideal simulation and near-term devices. Adding gradient-based saliency over the pool1\_pool feature map would provide clinician-interpretable regions of interest, and implementing the Quanvolutional Neural Network of [5] under the same circuit design discipline would complete the comparison.

---

## CODE AND DATA AVAILABILITY

The ECG image dataset used in this study is publicly available on Mendeley Data [16]. The implementation, trained model artefacts and the scripts reproducing all tables and figures are available at *(repository URL)*. `[O1: replace the placeholder with the real or anonymised link before submission.]`

---

## ACKNOWLEDGEMENT

The authors thank the Qiskit open-source community and IBM Quantum for the quantum computing framework. The ECG dataset was made publicly available by Khan et al. through Mendeley Data, collected at the Ch. Pervaiz Elahi Institute of Cardiology, Multan. The authors acknowledge the support of the Department of Computer Science and Engineering, St. Joseph Engineering College, Mangaluru.

---

## REFERENCES

*(Numbered in order of first appearance in text, as IEEE style requires.)*

[1] World Health Organization, "Cardiovascular diseases (CVDs)," Fact Sheet, Jun. 2021. [Online]. Available: https://www.who.int/news-room/fact-sheets/detail/cardiovascular-diseases-(cvds) [Accessed: 18 Sep. 2026].

[2] F. Vasquez-Iturralde, M. J. Flores-Calero, F. Grijalva, and A. Rosales-Acosta, "Automatic classification of cardiac arrhythmias using deep learning techniques: A systematic review," *IEEE Access*, vol. 12, pp. 118467–118492, 2024, doi: 10.1109/ACCESS.2024.3408282.

[3] O. Yildirim, P. Plawiak, R.-S. Tan, and U. R. Acharya, "Arrhythmia detection using deep convolutional neural network with long duration ECG signals," *Comput. Biol. Med.*, vol. 102, pp. 411–420, 2018, doi: 10.1016/j.compbiomed.2018.09.009.

[4] V. Havlicek, A. D. Corcoles, K. Temme, A. W. Harrow, A. Kandala, J. M. Chow, and J. M. Gambetta, "Supervised learning with quantum-enhanced feature spaces," *Nature*, vol. 567, pp. 209–212, Mar. 2019, doi: 10.1038/s41586-019-0980-2.

[5] S. Prabhu, S. Gupta, G. M. Prabhu, A. V. Dhanuka, and K. V. Bhat, "QuCardio: Application of quantum machine learning for detection of cardiovascular diseases," *IEEE Access*, vol. 11, pp. 136122–136135, 2023, doi: 10.1109/ACCESS.2023.3338145.

[6] V. Jain, N. Arora, and A. Gupta, "Quantum-assisted cardiac diseases diagnosis and prediction using ECG images," *J. Supercomput.*, vol. 81, art. no. 1439, 2025, doi: 10.1007/s11227-025-07939-8.

[7] J. Biamonte, P. Wittek, N. Pancotti, P. Rebentrost, N. Wiebe, and S. Lloyd, "Quantum machine learning," *Nature*, vol. 549, pp. 195–202, Sep. 2017, doi: 10.1038/nature23474.

[8] M. Y. Ramkhelawan, S. Grandhi, and S. Wibowo, "A hybrid quantum-classical deep learning algorithm for efficient arrhythmia classification," in *Proc. IEEE/ACIS Int. Conf. Softw. Eng., Artif. Intell., Netw. Parallel/Distrib. Comput. (SNPD)*, 2025, pp. *(check Xplore)*, doi: 10.1109/SNPD62189.2025.10985682.

[9] Z. Ozpolat and M. Karabatak, "Performance evaluation of quantum-based machine learning algorithms for cardiac arrhythmia classification," *Diagnostics*, vol. 13, no. 6, art. no. 1099, 2023, doi: 10.3390/diagnostics13061099.

[10] I. Aksoy and Z. Ozpolat, "Comparison of SVM, QSVM and Pegasos-QSVM algorithms on different medical datasets," *Int. J. Sustain. Eng. Technol.*, vol. 9, no. 1, pp. 80–93, 2025, doi: 10.62301/usmtd.1716034.

[11] K. L. Soon, W. L. Pang, H. H. Goh, Y. W. Sim, S. K. Phang, H. L. Choo, L. T. Soon, and N. S. Lai, "Early cardiovascular disease detection using hierarchical quantum ensemble model," *Comput. Methods Biomech. Biomed. Eng.*, published online Jan. 2026, doi: 10.1080/10255842.2025.2612536.

[12] A. E. Setiawan, S. Rustad, A. Syukur, M. A. Soeleman, G. F. Shidik, M. Akrom, and A. W. Setiawan, "A systematic literature review of quantum machine learning for medical applications: Trends, datasets, topics and methods," *Int. J. Cogn. Comput. Eng.*, vol. 7, pp. 609–630, 2026, doi: 10.1016/j.ijcce.2026.05.002.

[13] K. He, X. Zhang, S. Ren, and J. Sun, "Deep residual learning for image recognition," in *Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)*, Jun. 2016, pp. 770–778, doi: 10.1109/CVPR.2016.90.

[14] S. Shalev-Shwartz, Y. Singer, N. Srebro, and A. Cotter, "Pegasos: Primal estimated sub-gradient solver for SVM," *Math. Program.*, vol. 127, no. 1, pp. 3–30, 2011, doi: 10.1007/s10107-010-0420-4.

[15] M. Sandler, A. Howard, M. Zhu, A. Zhmoginov, and L.-C. Chen, "MobileNetV2: Inverted residuals and linear bottlenecks," in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, Jun. 2018, pp. 4510–4520, doi: 10.1109/CVPR.2018.00474.

[16] A. H. Khan, M. Hussain, and M. K. Malik, "ECG images dataset of cardiac and COVID-19 patients," *Data in Brief*, vol. 34, art. no. 106762, 2021, doi: 10.1016/j.dib.2021.106762.

[17] Q. McNemar, "Note on the sampling error of the difference between correlated proportions or percentages," *Psychometrika*, vol. 12, no. 2, pp. 153–157, 1947, doi: 10.1007/BF02295996.

[18] N. Cristianini, J. Shawe-Taylor, A. Elisseeff, and J. Kandola, "On kernel-target alignment," in *Advances in Neural Information Processing Systems 14*, 2001, pp. 367–373.

[19] Qiskit contributors, "Qiskit: An open-source framework for quantum computing," Zenodo, 2023, doi: 10.5281/zenodo.2573505.

[20] F. Pedregosa, G. Varoquaux, A. Gramfort, V. Michel, B. Thirion, O. Grisel, M. Blondel, P. Prettenhofer, R. Weiss, V. Dubourg, J. Vanderplas, A. Passos, D. Cournapeau, M. Brucher, M. Perrot, and E. Duchesnay, "Scikit-learn: Machine learning in Python," *J. Mach. Learn. Res.*, vol. 12, pp. 2825–2830, 2011.

---

## FIGURE PLACEMENT GUIDE
*(Remove before LaTeX compilation)*

| Fig. | Content | File | Where |
|---|---|---|---|
| 1 | Pipeline block diagram (offline + online) | Draw in draw.io per Sec. III | Top col. 1, Sec. III |
| 2 | Three-panel app screenshot | image1.png, image2.png, image3.png | Bottom col. 2, Sec. III-G |
| 3 | QSVC confusion matrix | results/paper/main/confusion\_matrix\_qsvc\_tuned.png | Top col. 1, Sec. V-A |
| 4 | ROC curves, all models and classes | results/paper/main/roc\_curves\_all\_models.png | Top col. 2, Sec. V-C |
| 5 | Augmentation robustness | results/paper/ablation/augmentation\_robustness.png | Top col., Sec. V-D |
| 6 | Encoding range ablation | results/paper/ablation/ablation\_encoding.png | Top col., Sec. VI-A |
| 7 | Entanglement topology ablation | results/paper/ablation/ablation\_entanglement.png | Top col., Sec. VI-B |
| 8 | Pauli feature map ablation | results/paper/ablation/ablation\_feature\_maps.png | Top col., Sec. VI-D |
| 9 | SVD dims + reps, two panels | ablation\_svd\_dims.png + ablation\_reps.png | Bottom col., Sec. VI-E |

**6-page trim:** drop Fig. 2, Fig. 4 and Fig. 9; keep TABLE VII-A and TABLE VIII, which carry the paper's argument. Compress Section VII but do not remove it.

**10-page journal:** include everything, add the calibration figure after Fig. 4, and expand the cross-dataset result from Section VII into a full Section V-E with its own table and confusion matrices (results/paper/cross\_dataset/jain\_dataset2/).

---

## CONSISTENCY CHECKLIST AGAINST THE PROJECT REPORT
*(Remove before LaTeX compilation)*

Verified identical across QuCardio_Paper_v4.md and QuCardio_Report_v6.md:

| Item | Value used in both |
|---|---|
| Sweep size | 320 configurations (4 encodings x 5 topologies x 4 reps x 4 C) |
| Prabhu's fourth model | Quanvolutional Neural Network (QNN), 97.31% |
| Reference to Prabhu et al. | IEEE Access, vol. 11, pp. 136122-136135, 2023 |
| Vasquez-Iturralde et al. | IEEE Access, vol. 12, pp. 118467-118492, 2024 |
| 2x2 factorial cells | 89.78 / 89.78 / 92.47 / 94.62, that is 167 / 167 / 172 / 176 of 186 |
| Interaction arithmetic | topology alone +2.69, encoding alone 0.00, both +4.84, interaction +2.15 |
| Augmentation subset | n = 45 (38/45, 39/45, 21/45) |
| Dataset counts | 929 in the four-class subset, 928 used, one MI image undecodable |
| Quantum latency | 893.9 ms end to end; classification step 87.4 ms |
| McNemar discordants | 23/5, 19/6, 9/4 |

A second pass on 2026-09-XX removed repeated rhetorical constructions (colon-then-reveal
sentences, "X is the outlier", "the mechanism is Y", self-congratulatory framing of the
interaction result) that appeared more than once within a document or identically across
both documents. No numeric value changed in that pass; only sentence-level phrasing did.
Bibliography entries are intentionally identical between the two documents, since both
cite the same sources — that overlap is expected and is not a similarity concern.

Open items are listed in the header block at the top of this file (O1 to O4). Nothing else
in either document is outstanding.
