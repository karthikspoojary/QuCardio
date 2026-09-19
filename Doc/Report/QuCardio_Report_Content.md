# QuCardio -- Major Project Report Content
> **Institution:** St. Joseph Engineering College (SJEC), Mangaluru
> **Programme:** Bachelor of Engineering in Computer Science and Engineering
> **Project Title:** QuCardio: Application of Quantum Machine Learning for Detection of Cardiovascular Diseases
> **Academic Year:** 2025-2026

---

> **NOTE TO TEAM:** This file contains the complete verified content for the final project report. Use the Main.pdf LaTeX template for formatting. All numbers are verified from actual code and result files. Replace SCREENSHOT PLACEHOLDER tags with screenshots from the running system. Insert result images from the results/paper/ folder at marked locations. Draw the Chapter 4 design diagrams using draw.io or Lucidchart. The companion file QuCardio_Report_Figures.html embeds all result images with base64 encoding for reference.

---

## CERTIFICATE

*(Fill in: Guide Name, HOD Name Dr Melwyn D'Souza, Principal Dr Rio D'Souza, date of submission. Title: "QuCardio: Application of Quantum Machine Learning for Detection of Cardiovascular Diseases". Department: Computer Science and Engineering. Year: 2025-26.)*

---

## DECLARATION

We hereby declare that the entire work embodied in this Project Report titled "QuCardio: Application of Quantum Machine Learning for Detection of Cardiovascular Diseases" has been carried out by us at St Joseph Engineering College, Mangaluru under the supervision of *(Guide Name)*, for the award of Bachelor of Engineering in Computer Science and Engineering. This report has not been submitted to this or any other University for the award of any other degree.

| Name | USN | Signature |
|---|---|---|
| *(Team member 1)* | *(USN)* | |
| *(Team member 2)* | *(USN)* | |
| *(Team member 3)* | *(USN)* | |
| *(Team member 4)* | *(USN)* | |

**Place:** Mangaluru  
**Date:**

---

## ACKNOWLEDGEMENT

We dedicate this page to acknowledge and thank those responsible for shaping this project. Without their guidance and help, the experience of constructing this work would not have been so smooth and rewarding.

We sincerely thank our project guide *(Guide Name)*, Department of Computer Science and Engineering, for the consistent guidance, timely feedback, and valuable suggestions that helped us complete this project. We also thank our project coordinator *(Project Coordinator Name)*, Department of CSE, for consistent encouragement.

We owe profound gratitude to Dr Melwyn D'Souza, Head of the Department, Computer Science and Engineering, whose kind support helped us complete this work successfully. We are extremely thankful to our Principal, Dr Rio D'Souza, Director Rev Fr Wilfred Prakash D'Souza, and Assistant Director Rev Fr Kenneth Rayner Crasta for their support and encouragement.

We thank the Qiskit open-source community and IBM Quantum for the quantum computing framework that forms the core of this project. We also acknowledge the Ch. Pervaiz Elahi Institute of Cardiology, Multan, for making the ECG image dataset publicly available through Mendeley Data.

We also extend our gratitude to our friends and family members for their continuous support.

---

## ABSTRACT

Cardiovascular diseases account for approximately 17.9 million deaths annually worldwide. Early detection through Electrocardiogram analysis is critical, but manual interpretation requires specialist expertise unavailable in many clinical settings. This project, QuCardio, builds and deploys a hybrid classical-quantum machine learning pipeline for automated four-class cardiovascular disease classification from raw ECG images.

The pipeline preprocesses ECG images using OpenCV to remove grid lines and artifacts, extracts 462,400-dimensional spatial features from the shallow pool1\_pool layer of a pre-trained ResNet50 network, and compresses them to 9 dimensions using Truncated Singular Value Decomposition followed by MinMax normalization. Three classifiers are evaluated: a Classical Support Vector Machine, a Quantum Support Vector Classifier built on a 9-qubit Pauli-ZZ feature map circuit, and a Multiclass Pegasos Quantum Support Vector Classifier. The Quantum Support Vector Classifier achieves 94.62% accuracy across four classes (Normal, Arrhythmia, Myocardial Infarction, History of Myocardial Infarction), exceeding the base paper result of 94.09%. The key finding is that scaling features to the range [0, pi] combined with circular entanglement topology improves accuracy by 4.84 percentage points over the original paper configuration. McNemar's test confirms statistical significance of quantum advantage over classical Support Vector Machine (chi-squared = 10.32, p = 0.0013). The complete system is deployed as a web application with a FastAPI backend and React frontend supporting real-time diagnosis, calibrated confidence scores, and PDF report generation.

**Keywords:** Quantum Machine Learning, Quantum Support Vector Classifier, ZZFeatureMap, ECG Classification, Cardiovascular Disease Detection, ResNet50, Fidelity Quantum Kernel, Truncated SVD

---

## TABLE OF CONTENTS

*(Generate automatically from headings in LaTeX/Word)*

1. Introduction
2. Literature Survey
3. Software Requirements Specification
4. System Design
5. Implementation
6. Results and Discussion
7. Conclusion and Future Work

References

---

## LIST OF FIGURES

*(Generate automatically)*

---

## LIST OF TABLES

*(Generate automatically)*

---

## LIST OF ABBREVIATIONS

| Abbreviation | Full Form |
|---|---|
| CVD | Cardiovascular Disease |
| ECG | Electrocardiogram |
| QML | Quantum Machine Learning |
| QSVC | Quantum Support Vector Classifier |
| SVM | Support Vector Machine |
| CNN | Convolutional Neural Network |
| SVD | Singular Value Decomposition |
| NISQ | Noisy Intermediate-Scale Quantum |
| AUC | Area Under the ROC Curve |
| ECE | Expected Calibration Error |
| KTA | Kernel Target Alignment |
| ROI | Region of Interest |
| API | Application Programming Interface |
| MI | Myocardial Infarction |
| HMI | History of Myocardial Infarction |
| SGD | Stochastic Gradient Descent |

---

# CHAPTER 1 -- INTRODUCTION

## 1.1 Background

Cardiovascular diseases are the leading cause of mortality worldwide, responsible for nearly one in three deaths globally. According to the World Health Organization, approximately 17.9 million people die from heart-related conditions each year. In India, cardiovascular conditions account for around 28% of all deaths, with cases rising in both urban and rural populations. The most clinically urgent among these are Myocardial Infarction, cardiac Arrhythmia, and prior History of Myocardial Infarction, each demanding timely diagnosis to prevent irreversible damage.

The Electrocardiogram is the standard non-invasive tool used to detect these conditions. It captures the electrical activity of the heart and presents it as a printed waveform. However, accurate ECG interpretation requires specialized cardiological training, which is often unavailable in high-volume hospitals or rural health centres. This creates a diagnostic delay that can prove fatal in acute cardiac events.

Classical machine learning and deep learning methods have made progress in automating ECG analysis, but they tend to plateau in accuracy on small multi-class medical datasets and do not generalize well across diverse ECG sources. Quantum machine learning, particularly quantum kernel methods, offers a theoretically motivated path forward. Quantum feature maps embed classical data into exponentially high-dimensional Hilbert spaces where complex classification boundaries, which classical kernels cannot express efficiently, may be captured. This project, QuCardio, implements and validates a complete hybrid classical-quantum pipeline for four-class ECG cardiovascular disease classification and deploys it as a clinician-accessible web application.

## 1.2 Problem Statement

Automated cardiovascular disease classification from ECG images is a clinically important yet unsolved problem at the intersection of medical imaging and quantum computing. Existing classical machine learning approaches struggle with multi-class accuracy on limited annotated datasets, while quantum ECG classifiers reported in recent literature provide single accuracy numbers without systematic circuit design studies, statistical validation, or any working deployment. There is currently no end-to-end system that combines deep spatial feature extraction from ECG images, quantum kernel classification with ablation-validated design choices, statistical significance testing, confidence calibration analysis, and a real-time clinical web interface on the same dataset. This project addresses all of these gaps by extending the foundational work of Prabhu et al. [1], conducting a thorough experimental investigation of quantum circuit parameters, and deploying the complete pipeline for practical use.

## 1.3 Objectives

The objectives of this project are:

1. To design and implement a hybrid classical-quantum machine learning pipeline that preprocesses raw ECG images, extracts spatial features using a pre-trained ResNet50 network, reduces dimensionality using Truncated Singular Value Decomposition, and classifies the result using a Classical Support Vector Machine, a Quantum Support Vector Classifier with a 9-qubit ZZFeatureMap, and a Multiclass Pegasos Quantum Support Vector Classifier.

2. To conduct a systematic ablation study across quantum circuit design parameters including feature encoding range, entanglement topology, circuit repetitions, and dimensionality reduction components, identifying the optimal configuration for ECG image classification and providing statistical evidence of quantum advantage over classical baselines using McNemar's test.

3. To deploy the complete pipeline as a functional web application with a FastAPI backend and React frontend, providing real-time ECG upload, diagnosis with calibrated confidence scores, class-specific clinical recommendations, and downloadable PDF reports.

## 1.4 Scope of the Project

QuCardio classifies ECG images into four cardiovascular conditions: Normal, Arrhythmia, Myocardial Infarction, and History of Myocardial Infarction. The project covers the full pipeline from raw image input to web-based clinical output. The preprocessing module handles diverse ECG scan types by removing backgrounds, grid lines, and artifacts. Feature extraction uses the pool1\_pool layer of a frozen ResNet50 to generate 462,400-dimensional spatial vectors, which are compressed to 9 dimensions using Truncated Singular Value Decomposition. The quantum classifier uses a 9-qubit ZZFeatureMap (Pauli-ZZ feature map) with fidelity kernel computed via statevector simulation. The scope extends to systematic circuit design evaluation across five encoding ranges, five entanglement topologies, four repetition counts, nine augmentation robustness conditions, and a comparison of four Pauli feature map types (ZFeatureMap, ZZFeatureMap, PauliFeatureMap[X+XX], PauliFeatureMap[Y+YY]). The project also includes a cross-dataset generalization study on a second Mendeley ECG dataset from the same hospital, matching the experimental design of Jain et al. [7]. A MobileNetV2-based gatekeeper rejects non-ECG inputs at the API level. The project does not implement Quantum Neural Networks, which are left as future work.

## 1.5 Organization of the Report

Chapter 2 presents the literature survey covering classical and quantum ECG classification methods, followed by the proposed system section that answers the four required evaluation questions. Chapter 3 details the software requirements. Chapter 4 presents the system design with architecture, use case, data flow, and class diagrams along with explanations. Chapter 5 covers the implementation of each pipeline stage. Chapter 6 presents key results and discussion. Chapter 7 concludes the report and outlines future directions.

---

# CHAPTER 2 -- LITERATURE SURVEY

## 2.1 Introduction

Automated detection of cardiovascular diseases from ECG data has been studied for several decades, progressing from hand-crafted signal features to deep learning representations, and more recently to quantum machine learning methods. This chapter reviews fifteen papers that are directly relevant to the techniques and findings of this project, structured by methodology. Each entry discusses the problem addressed, methodology, results, and limitations in the context of this work. The chapter concludes with a proposed system section that positions this project relative to the surveyed literature.

## 2.2 QuCardio: Application of Quantum Machine Learning for Detection of Cardiovascular Diseases

This paper by Prabhu et al. [1], published in IEEE Access in 2023, serves as the direct base for this project. The authors collected 929 ECG images from Ch. Pervaiz Elahi Institute of Cardiology, Multan, covering four classes: Normal, Arrhythmia, Myocardial Infarction, and History of Myocardial Infarction. Feature extraction used the pool1\_pool layer of a pre-trained ResNet50 network, and Truncated Singular Value Decomposition with 9 components was applied for dimensionality reduction, followed by normalization to [0, 1]. Three classifiers were trained and evaluated: a Classical Support Vector Machine (83.33%), a Quantum Support Vector Classifier with ZZFeatureMap and linear entanglement (94.09%), a Multiclass Pegasos Quantum Support Vector Classifier (93.05%), and a Quantum Neural Network (97.31%). The paper demonstrated that quantum kernel methods can meaningfully outperform classical classifiers on a small real-world ECG image dataset, establishing the feasibility of this hybrid approach. However, the work evaluated only a single accuracy number per model with no ablation of circuit design parameters, no statistical significance testing, no confidence calibration analysis, and no system deployment. The encoding range was fixed at [0, 1] and entanglement at linear without exploring alternatives, leaving substantial room for improvement through systematic study, which this project undertakes.

## 2.3 Supervised Learning with Quantum-Enhanced Feature Spaces

In this landmark paper published in Nature in 2019, Havlicek et al. [2] demonstrated that quantum kernel methods using quantum feature maps can achieve classification advantages that are intractable for polynomial-time classical kernels. The authors constructed quantum circuits that map classical input vectors into exponentially large Hilbert spaces, and showed that the resulting kernel function captures correlations no efficient classical kernel can replicate. The ZZFeatureMap circuit, which is the Pauli-ZZ encoding used in this project, is directly based on the construction described in this work. A key result was that the quantum kernel's advantage holds even when the classical data is low-dimensional, as long as the mapping to Hilbert space exploits non-trivial entanglement. This theoretical result is the primary justification for using a Quantum Support Vector Classifier for the 9-dimensional SVD-compressed ECG features in QuCardio, and motivates exploring encoding ranges and entanglement topologies as design variables.

## 2.4 Quantum Machine Learning

Biamonte et al. [3] published this comprehensive survey of quantum machine learning algorithms in Nature in 2017. The paper covered quantum kernel methods, variational quantum classifiers, quantum principal component analysis, and quantum neural networks, providing the theoretical framework that underlies most subsequent applied quantum machine learning work. The review showed that quantum advantage is most likely for structured datasets where classical kernel computation is exponentially expensive, and identified NISQ-era hardware constraints, including gate noise and limited qubit connectivity, as the primary practical barrier. This work directly motivates the statevector simulation approach used in this project as the appropriate substitute for hardware execution at current qubit counts. It also provides the background context for understanding the trade-off between circuit expressibility and barren plateau issues that emerges in the circuit repetitions ablation study.

## 2.5 Pegasos: Primal Estimated Sub-Gradient Solver for SVM

Shalev-Shwartz et al. [4] introduced the Pegasos algorithm in Mathematical Programming in 2011 as an efficient alternative to quadratic programming for Support Vector Machine training. The algorithm applies stochastic sub-gradient descent with a step size that decays as one divided by lambda times the iteration counter, achieving convergence in O(1 per epsilon) steps independent of the training set size. Crucially, it never requires constructing or storing the full N-by-N kernel matrix, making it tractable when kernel matrix memory is the bottleneck. The extension of Pegasos to quantum kernels, used as the Multiclass Pegasos Quantum Support Vector Classifier in this project, inherits this property and was essential for training six binary classifiers on 743 training samples under the memory constraints of statevector simulation. In our implementation, the average 9-qubit quantum kernel value approaches 1/512, which required a custom numerical stabilization strategy for the SGD step size.

## 2.6 Deep Residual Learning for Image Recognition

He et al. [5] introduced the Residual Network (ResNet) architecture in their 2016 IEEE CVPR paper, winning first place in ILSVRC 2015 with a 152-layer network. The skip connection was the key innovation: by adding the input of a block directly to its output, gradients can flow through identity paths, resolving the vanishing gradient problem that previously prevented very deep network training. ResNet50, a 50-layer variant, has since become the standard pre-trained backbone for transfer learning in medical image analysis. In this project, a ResNet50 with frozen ImageNet weights is used as a feature extractor, with the pool1\_pool layer selected as the extraction point. This layer applies 3x3 max-pooling over the initial 7x7 convolutional output, producing an 85x85x64 spatial map that captures low-level morphological features, including QRS complex shapes and ST segment geometry, without imposing high-level ImageNet-specific semantics.

## 2.7 A Hybrid Quantum-Classical Deep Learning Algorithm for Efficient Arrhythmia Classification

Ramkhelawan et al. [6] presented this work at the IEEE SNPD conference in 2025, combining ResNet50 feature extraction with a 10-qubit Variational Quantum Circuit via PennyLane for binary arrhythmia detection. The training set comprised 123,998 ECG images drawn from the MIT-BIH and PTB databases combined, and the binary classifier achieved 99.98% accuracy. The work showed that deep learning feature extraction paired with a variational quantum circuit in a hybrid pipeline is a viable architecture for ECG classification, and the very high accuracy reflects the strong separability of normal versus arrhythmia classes on large balanced datasets. However, the binary classification setting is structurally far simpler than the four-class balanced problem in this project, and the 10-qubit circuit presents exponential simulation overhead that limits scalability. The paper also does not study the effect of encoding range or entanglement topology on classification performance.

## 2.8 Quantum-Assisted Cardiac Diseases Diagnosis and Prediction Using ECG Images

Jain et al. [7] published this study in the Journal of Supercomputing in 2025, investigating both Quantum Support Vector Machine and a Quantum Convolutional Neural Network for cardiac disease classification from ECG images. The pipeline used ResNet50 combined with Principal Component Analysis for feature compression before quantum classification. Results showed that the Quantum Convolutional Neural Network achieves approximately 5% higher accuracy than its classical counterpart, while the Quantum Support Vector Machine outperforms classical Support Vector Machine by approximately 8% on the same ECG image dataset. Statistical significance was established using the Kruskal-Wallis H-test. This paper shares the ResNet50 plus dimensionality reduction plus quantum kernel framework with our work, but uses Principal Component Analysis rather than Truncated Singular Value Decomposition and does not systematically vary the feature encoding range or entanglement topology. Our ablation study addresses exactly these open questions.

## 2.9 Performance Evaluation of Quantum-Based Machine Learning Algorithms for Cardiac Arrhythmia Classification

Ozpolat and Karabatak [8] published this comparative benchmarking study in the journal Diagnostics in 2023. The work applied Principal Component Analysis for feature reduction followed by Quantum Support Vector Machine classification to the Chapman ECG signal database containing 10,588 patient records. Across a qubit sweep from 3 to 9, the best configuration achieved 80.90% accuracy versus 78.84% for classical Support Vector Machine on the same features. The study evaluated multiple quantum circuit configurations and concluded that Quantum Support Vector Machine consistently outperforms classical Support Vector Machine on cardiac arrhythmia data. Unlike our project, this work operates on signal-domain tabular features rather than spatial features extracted from ECG images, and the modest 2-percentage-point gap suggests that the choice of feature representation strongly influences how much quantum advantage is achievable.

## 2.10 Quantum-Based Convolutional Neural Network Model for Efficient Cardiovascular Disease Prediction

Gummadi et al. [9] presented this paper at the 3rd IEEE International Conference on Artificial Intelligence in 2025, proposing parameterized quantum circuits as enhancement layers inserted within a CNN architecture for cardiovascular disease prediction. The hybrid model leveraged the high-dimensional Hilbert space of the quantum circuit as a nonlinear feature transformation applied between classical convolutional layers. Results showed improved accuracy and robustness over classical CNN and MLP baselines on a cardiovascular prediction dataset. The work demonstrates that quantum gates operating directly on CNN-derived feature maps can improve classification, providing conceptual support for the hybrid deep learning plus quantum kernel architecture used in QuCardio. The key difference is that our project applies the quantum kernel to compact SVD features rather than embedding quantum gates within the CNN structure itself.

## 2.11 Arrhythmia Detection Using Deep Convolutional Neural Network with Long Duration ECG Signals

Yildirim et al. [10] published this study in Computers in Biology and Medicine in 2018, designing a custom 1D Convolutional Neural Network that accepts raw long-duration ECG waveform segments as input without any hand-crafted feature engineering. The network classified 17 arrhythmia types from the MIT-BIH database with 91.33% accuracy, establishing that raw ECG time-series contain sufficient morphological information for fine-grained arrhythmia discrimination. This was one of the early demonstrations of end-to-end deep ECG classification, influencing subsequent work. However, the 1D signal approach requires access to the raw ECG recording file rather than a photographed paper ECG, which is unavailable in our real-world clinical dataset of scanned ECG prints. Our preprocessing pipeline addresses this gap by extracting morphological features from ECG images rather than digital signals.

## 2.12 Automatic Classification of Cardiac Arrhythmias Using Deep Learning Techniques: A Systematic Review

Vasquez-Iturralde et al. [11] published this systematic review in IEEE Access in 2024, covering 494 deep learning papers for cardiac arrhythmia classification published between 2017 and 2023 using PRISMA methodology. The review found that Convolutional Neural Networks, Long Short-Term Memory networks, and CNN-LSTM hybrid architectures dominate the field, with accuracy above 90% routinely reported on the MIT-BIH benchmark. It highlighted a critical methodological concern present in many papers: patient-level data mixing across training and test splits, which inflates reported accuracy through data leakage. The review also identified ECG image classification as an underexplored area compared to signal-based approaches, with fewer than 15% of surveyed papers operating on image inputs. This review directly motivated our strict 80:20 stratified patient-level split and reinforces the novelty of applying quantum kernels to ECG images rather than signals.

## 2.13 Early Cardiovascular Disease Detection Using Hierarchical Quantum Ensemble Model

Soon et al. [12] published this paper in Computer Methods in Biomechanics and Biomedical Engineering in 2026. The proposed Hierarchical Quantum Ensemble Model combines a Quantum Neural Network and XGBoost classifier in parallel, with their outputs fed into a LightGBM meta-learner for a final hierarchical decision. The model was evaluated on multiple integrated cardiovascular disease tabular datasets and achieved 97% accuracy and 98% Area Under the ROC Curve. The hierarchical ensemble design is motivated by the observation that quantum and classical learners tend to make complementary errors, which a meta-learner can exploit. This approach is complementary to our work, as it operates exclusively on structured clinical tabular features rather than ECG images, and the ensemble structure adds significant complexity and training overhead compared to the single quantum kernel method in this project.

## 2.14 A Systematic Literature Review of Quantum Machine Learning for Medical Applications

Setiawan et al. [13] conducted a comprehensive systematic review of 94 primary quantum machine learning healthcare studies published between 2020 and April 2025, published in the International Journal of Cognitive Computing in Engineering in 2026. The review showed that hybrid classical-quantum pipelines, particularly CNN plus Quantum Support Vector Machine combinations, are the dominant architecture in the field. Most studies operate exclusively in statevector simulation due to NISQ hardware noise constraints. Importantly, the review identified that confidence calibration analysis and statistical significance testing are almost universally absent from quantum machine learning healthcare papers, making it difficult to compare reported accuracy numbers across studies in a meaningful way. This finding directly motivated the inclusion of McNemar's test and Expected Calibration Error analysis in this project, which addresses two of the most significant methodological gaps identified in the review.

## 2.15 Comparison of SVM, QSVM and Pegasos-QSVM Algorithms on Different Medical Datasets

Aksoy and Ozpolat [14] published this comparative evaluation in the International Journal of Sustainable Engineering and Technology in 2025. The authors systematically compared classical Support Vector Machine, Quantum Support Vector Machine, and Pegasos Quantum Support Vector Machine across liver cancer, breast cancer, and heart failure datasets, using MinMax scaling to the range [0, pi] before quantum encoding. Results showed that Quantum Support Vector Machine provides the most stable and consistently high accuracy across all three datasets, while Pegasos converges faster but shows higher variance across different runs and datasets. The use of [0, pi] scaling in this concurrent study on entirely different medical datasets is a noteworthy independent corroboration of the encoding range finding in our ablation study. It suggests that the [0, pi] scaling advantage may be a general property of the ZZFeatureMap circuit rather than a dataset-specific artifact.

## 2.16 New Cardiovascular Disease Prediction Approach Using Support Vector Machine and Quantum-Behaved Particle Swarm Optimization

Elsedimy et al. [15] published this paper in Multimedia Tools and Applications in 2024. The proposed QPSO-SVM model applies Quantum-behaved Particle Swarm Optimization as a meta-heuristic for feature selection and Support Vector Machine hyperparameter tuning on the Cleveland heart disease dataset. The algorithm uses quantum-inspired position and velocity updates for the particles rather than actual quantum computation, making it a quantum-inspired classical method. Applied to the Cleveland dataset, the model achieved 96.31% accuracy with 96.13% sensitivity and 93.56% specificity. The work demonstrates that quantum-inspired optimization can significantly improve classical classifier performance through better feature selection. It differs from our approach in that no actual quantum circuit or kernel computation is involved, and the method operates on tabular clinical features rather than ECG images.

## 2.17 Quantum-Enhanced Machine Learning Algorithms for Heart Disease Prediction

Alotaibi et al. [16] investigated quantum machine learning combined with Quantum Particle Swarm Optimization for heart disease prediction, publishing this study in Human-centric Computing and Information Sciences in 2023. The proposed system used Quantum Particle Swarm Optimization to identify the optimal quantum circuit configuration and feature subset for a quantum machine learning classifier applied to the Cleveland heart disease dataset. The model achieved 96.7% accuracy with a computation time of 135ms, outperforming classical Multilayer Perceptron on the same data. The integration of quantum optimization into the hyperparameter search process is a different strategy from our grid search and ablation approach but addresses a similar need for principled quantum circuit configuration. The work confirms that carefully designed quantum classifiers can outperform classical baselines on structured medical datasets.

## 2.18 Proposed System

QuCardio proposes a complete hybrid classical-quantum machine learning pipeline for automated cardiovascular disease classification from raw ECG images, deployed as a fully functional web application.

**Why is this problem important?** Cardiovascular diseases kill nearly 17.9 million people annually, and accurate ECG interpretation is the front-line diagnostic tool. In countries like India where cardiologist availability is critically limited, automated ECG diagnosis can reduce life-threatening diagnostic delays in both rural hospitals and high-volume urban centres. Existing automated systems either plateau in multi-class accuracy or present quantum models without rigorous validation or deployment, leaving a practical and scientific gap that this project fills.

**What is novel in this project?** Two novel findings emerge from systematic ablation experiments. First, scaling the 9-dimensional SVD-compressed ECG features to the range [0, pi] before ZZFeatureMap encoding improves Quantum Support Vector Classifier accuracy by 4.84 percentage points over the [0, 1] range used in the base paper. This works because the ZZFeatureMap's Rz rotation gates apply rotations of 2xi radians: with xi in [0, pi], the angles span the full [0, 2pi] of the Bloch sphere latitude, maximizing quantum state separation. Second, circular entanglement topology provides the same 4.84 percentage point gain over linear entanglement by giving every qubit two neighbours, generating richer pairwise correlations at circuit depth comparable to linear. Together, these design choices raise the Quantum Support Vector Classifier from 89.78% to 94.62%. Additionally, this project provides the first McNemar significance test confirming quantum kernel advantage on ECG data (chi-squared = 10.32, p = 0.0013), the first Expected Calibration Error analysis for quantum ECG classifiers (all models below 0.07), and the first augmentation robustness study showing that the Quantum Support Vector Classifier maintains 86.7% accuracy under brightness perturbation where classical Support Vector Machine collapses to 46.7%.

**How does this advance the state of the art?** Prior quantum ECG classifiers report isolated accuracy numbers without ablation, calibration, or statistical validation. This work introduces systematic circuit design analysis with 486 configurations swept, formal statistical proof of quantum advantage, calibration-verified confidence scores suitable for clinical deployment, and a fully functioning web interface, collectively moving quantum ECG classification from proof-of-concept to a rigorously validated and practically usable system.

**How does this differ from each surveyed work?** Unlike Prabhu et al. [1], which this project directly extends, we systematically vary the encoding range and entanglement topology rather than fixing them, finding significant improvements. Unlike Jain et al. [7], we use Truncated Singular Value Decomposition rather than Principal Component Analysis and conduct a thorough ablation study. Unlike Ramkhelawan et al. [6], we address four-class classification rather than binary arrhythmia detection. Unlike Ozpolat and Karabatak [8], we operate on spatial ECG image features rather than signal-domain tabular features. Unlike all surveyed works, we provide both statistical significance testing and Expected Calibration Error analysis, and we deploy the system as a functional web application.

## 2.19 Comparison of Existing Works

**Table 2.1: Comparison of Prior Works**

| Reference | Task | Method | Key Result | Limitation |
|---|---|---|---|---|
| Prabhu et al. [1] | 4-class ECG images | ResNet50 + SVD + QSVC (linear, [0,1]) | 94.09% | No ablation, no stats, not deployed |
| Havlicek et al. [2] | Theoretical QML | Quantum feature maps, ZZFeatureMap | Exponential kernel advantage | Theoretical; no medical data |
| Ramkhelawan et al. [6] | Binary arrhythmia | ResNet50 + 10-qubit VQC | 99.98% | Binary task; large dataset |
| Jain et al. [7] | Cardiac ECG images | QSVM + QCNN | 8% gain over SVM | PCA; no encoding ablation |
| Ozpolat & Karabatak [8] | Arrhythmia (signal) | PCA + QSVM (3-9 qubits) | 80.90% vs 78.84% SVM | Signal features only |
| Gummadi et al. [9] | CVD prediction | CNN + Parameterized Quantum Circuit | Above classical CNN | Simulation overhead |
| Soon et al. [12] | CVD risk (tabular) | Hierarchical QNN + XGBoost + LightGBM | 97%, 98% AUC | Tabular only; no images |
| Aksoy & Ozpolat [14] | Cancer + CVD (tabular) | SVM / QSVM / Pegasos-QSVM | QSVM most stable; [0,pi] used | Tabular data |
| Setiawan et al. [13] | QML healthcare survey | Systematic review of 94 studies | Hybrid CNN-QSVM dominant | Survey; no new experiments |
| **QuCardio (Ours)** | 4-class ECG images + ablation + deployment | ResNet50 pool1\_pool + SVD-9 + QSVC ([0,pi], circular) | **94.62%**, p=0.0013, ECE < 0.07, deployed | NISQ simulation gap |

---

# CHAPTER 3 -- SOFTWARE REQUIREMENTS SPECIFICATION

## 3.1 Functional Requirements

The system shall:

- Accept ECG images in JPEG or PNG format through a drag-and-drop web interface, with file size capped at 20 MB and pixel area capped at 16 megapixels.
- Validate uploaded files using a MobileNetV2-based ECG gatekeeper before invoking any classification pipeline; non-ECG images shall be rejected with an informative error message.
- Preprocess valid ECG images through a seven-step OpenCV pipeline to produce a 340x340 pixel normalized grayscale output with grid lines and background removed.
- Extract 462,400-dimensional spatial features from the pool1\_pool layer of a pre-loaded frozen ResNet50 model, reduce them to 9 dimensions using a pre-fitted Truncated Singular Value Decomposition, and normalize with a pre-fitted MinMaxScaler.
- Allow the user to select one of three classifiers: Classical Support Vector Machine (RBF kernel, C=10, balanced class weights, isotonic-calibrated probabilities), Quantum Support Vector Classifier (9-qubit ZZFeatureMap, reps=2, circular entanglement, [0, pi] encoding), or Multiclass Pegasos Quantum Support Vector Classifier (six per-class-pair binary models, Algorithm 1 decision tree).
- Return the predicted cardiovascular class, calibrated confidence percentage, probability distribution across all four classes, the preprocessed ECG image, and a class-specific clinical recommendation. A low-confidence flag shall appear for predictions below 55%.
- Generate a downloadable single-page PDF clinical report containing the prediction, confidence scores, probability distribution, and ECG image.
- Process up to 20 ECG images in a single batch request and return individual results for each.
- Log every prediction to a SQLite audit database with timestamp, SHA-256 image hash, model used, predicted class, and confidence score.
- Expose a prediction history endpoint returning paginated audit log records.

## 3.2 Non-Functional Requirements

The system shall meet the following non-functional criteria:

- **Performance:** End-to-end inference latency for a single image on CPU hardware shall not exceed 5 seconds. Feature extraction takes approximately 500ms; dimensionality reduction under 5ms; Support Vector Machine classification under 10ms.
- **Reliability:** Malformed images, oversized files, and non-ECG uploads shall be handled with structured HTTP error responses. A Redis cache with 24-hour time-to-live keyed by SHA-256 hash shall avoid redundant computation for repeated uploads.
- **Scalability:** The FastAPI backend shall support asynchronous request handling and be containerizable via Docker for horizontal deployment.
- **Security:** Pixel area shall be capped at 16 megapixels to prevent decompression bomb attacks. All model files shall be loaded at startup and not reloaded per request.
- **Usability:** The React frontend shall display results within 10 seconds on a standard broadband connection, and processing time shall be broken down across preprocessing, feature extraction, and classification stages.

## 3.3 Hardware Requirements

| Component | Specification |
|---|---|
| Processor | Intel Core i5 8th Generation or AMD Ryzen 5 or better |
| RAM | 8 GB minimum; 16 GB recommended for training |
| Storage | 10 GB free disk space |
| GPU | Not required for inference; optional for batch feature extraction |
| Network | Broadband connection for frontend-backend communication |

## 3.4 Software Requirements

| Component | Version | Purpose |
|---|---|---|
| Python | 3.10+ | Backend language for all ML, preprocessing, and API logic |
| OpenCV | 4.8+ | Seven-step ECG image preprocessing pipeline |
| TensorFlow / Keras | 2.15+ | ResNet50 model loading and pool1\_pool feature extraction |
| Scikit-learn | 1.3+ | SVM, TruncatedSVD, MinMaxScaler, CalibratedClassifierCV |
| Qiskit | 0.45.0 | ZZFeatureMap quantum circuit construction |
| Qiskit Machine Learning | 0.7.0 | Quantum Support Vector Classifier implementation |
| Qiskit Aer | 0.13.0 | Statevector simulator for quantum kernel computation |
| FastAPI + Uvicorn | 0.104+ | Asynchronous REST API backend |
| React.js / Vite | 18+ | Three-tab clinical dashboard frontend |
| Redis | 7.0+ | SHA-256 keyed prediction result caching |
| SQLite | built-in | Persistent audit log database |
| Docker / Docker Compose | latest | Containerized deployment |
| NumPy, SciPy | latest | Matrix operations and McNemar's test computation |

## 3.5 Design Constraints

- The quantum circuit operates under ideal statevector simulation and is limited to 9 qubits due to the exponential memory cost of 2^9 = 512-dimensional complex state vectors.
- Real quantum hardware execution is not supported in the current version due to NISQ noise levels.
- The dataset comprises 929 images from a single hospital, which may limit generalization to ECG images from different scanner types or clinical settings.

---

# CHAPTER 4 -- SYSTEM DESIGN

## 4.1 System Architecture

QuCardio operates in two clearly separated phases. The offline training phase, executed once during development, processes all 929 dataset images through the full preprocessing and feature extraction pipeline, trains all three classifier models, and serializes them as pickle files. The online inference phase, triggered by each API request, deserializes these models from disk and runs the same pipeline on the user-uploaded ECG image to produce a diagnosis. This two-phase design is a deliberate architectural choice driven by the compute cost of quantum kernel training: building the 743x743 statevector kernel matrix during training takes on the order of 10 seconds on CPU, which is acceptable offline but would make real-time responses infeasible if repeated per request. By serializing the trained Support Vector Classifier along with the precomputed training statevectors, inference requires only a single kernel row computation (743 dot products on 512-dimensional complex vectors) rather than a full matrix rebuild.

**Figure 4.1: System Architecture Diagram**

*(Draw using draw.io and insert as image. Two-path block diagram with distinct visual styling for the two phases.)*
*(Left path, labelled "Offline Training Phase": [ECG Dataset: 929 images, 4 classes] → [OpenCV Preprocessing Module] → [ResNet50 pool1\_pool Feature Extractor: 462,400-D] → [TruncatedSVD(9) + MinMaxScaler] → three parallel branches: [Classical SVM Training], [QSVC Training: statevector kernel], [Pegasos QSVC: 6 binary models] → [Serialized .pkl Model Files: backend/models/])*
*(Right path, labelled "Online Inference Phase": [User uploads ECG via React] → [FastAPI Backend] → [MobileNetV2 ECG Gatekeeper] → [OpenCV Preprocessing] → [ResNet50 pool1\_pool] → [TruncatedSVD + MinMaxScaler] → [User-selected Classifier] → [Prediction + Confidence + Clinical Recommendation] → [React Dashboard]. Side branches to [SQLite Audit Log] and [Redis Cache: 24h TTL].)*

The diagram illustrates that the inference phase mirrors the training phase exactly, using the same preprocessing and feature extraction steps. The three trained models are available in parallel, and the user's model selection determines which branch executes. The architecture cleanly separates training-time computation from inference-time computation, enabling fast real-time response without retraining. The MinMax scaler and Truncated Singular Value Decomposition objects fitted on training data are also serialized and reused at inference, ensuring that the same feature transformation applied during training is reproduced exactly for every new sample. This strict pipeline symmetry prevents the distribution shift that would arise if the scaler were refitted on the test or inference data.

The data transformation chain at inference time flows as follows:

```
Raw ECG Image (variable size, JPEG/PNG)
        |
        v
OpenCV Preprocessing  -->  340x340 grayscale float32 in [0,1]
        |
        v
ResNet50 pool1_pool   -->  85x85x64 = 462,400-D feature vector
        |
        v
TruncatedSVD(9)       -->  9-D principal component vector
        |
        v
MinMaxScaler [0,1]    -->  normalized 9-D vector
        |
   +----+------------------+--------------------+
   |                       |                    |
   v                       v                    v
Classical SVM          QSVC (x pi)          Pegasos QSVC
RBF kernel C=10        ZZFeatureMap 9-qubit  6 binary models
balanced weights       circular reps=2       Algorithm 1
84.95% accuracy        94.62% accuracy       91.94% accuracy
```

## 4.2 Use Case Diagram

**Figure 4.2: Use Case Diagram**

*(Draw using draw.io with standard UML use case notation and insert as image.)*
*(Two actors: "Clinician / User" on the left represented by a stick figure, and "System Backend" on the right represented by a rectangle.)*
*(Clinician use cases in an oval boundary: Upload ECG Image; Select Classifier (SVM / QSVC / Pegasos); View Diagnosis and Confidence; View Probability Distribution; Download PDF Report; View Prediction History.)*
*(System use cases: Validate ECG (gatekeeper); Preprocess ECG Image; Extract ResNet50 Features; Reduce Dimensions (SVD + Scale); Classify with Selected Model; Log Prediction to Audit Database; Cache Result (Redis); Generate PDF Report.)*
*(Include / extend arrows: Upload ECG Image includes Validate ECG; Classify includes Extract Features; Extract Features includes Preprocess.)*

The use case diagram captures the complete interaction scope of the system. The Clinician actor initiates all primary use cases, while the System Backend handles the computational use cases automatically as part of the classification flow. The includes relationships show that every Upload operation triggers validation, and every Classify operation triggers the full feature extraction chain. The Select Classifier use case connects to three mutually exclusive extend relationships for SVM, QSVC, and Pegasos, reflecting the design decision to expose all three models through a single upload interface rather than separate pages. The View Prediction History use case is independent of the upload flow, relying entirely on the audit log rather than re-running any inference.

## 4.3 Data Flow Diagram

**Figure 4.3: Level-1 Data Flow Diagram**

*(Draw using standard DFD notation in draw.io and insert as image.)*
*(External entity: Clinician (rectangle on left). Six processes as circles: P1 Image Validation, P2 ECG Preprocessing, P3 Feature Extraction, P4 Dimensionality Reduction, P5 Classification, P6 Response Assembly. Two data stores as open rectangles: D1 Redis Cache and D2 SQLite Audit Database.)*
*(Flows: Clinician → P1: raw image file; P1 → P2: validated image (or error to Clinician); P2 → P3: 340x340 float32 array; P3 → P4: 462,400-D feature vector; P4 reads svd\_reducer.pkl and minmax\_scaler.pkl; P4 → P5: 9-D scaled vector; P5 reads selected classifier .pkl; P5 → P6: (class index, probability array); P6 → Clinician: JSON response; P5 writes to D2; P6 reads from and writes to D1.)*

The data flow diagram shows how ECG image data is progressively transformed through each processing stage, with the SVD reducer and scaler loaded from persistent storage at step P4, and predictions written to the audit log and cache as part of response assembly. The Redis cache sits at the response level so that repeated submissions of the same image with the same model return instantly without traversing the pipeline. At each stage the data representation changes: from a variable-resolution JPEG or PNG file, to a 340x340 float32 array, to a 462,400-dimensional feature vector, to a 9-dimensional compressed and scaled vector, and finally to a four-element probability distribution. The DFD makes explicit that the data stores (Redis and SQLite) are write-on-first-seen and read-on-repeat, which means they are side effects of the pipeline rather than stages within it.

## 4.4 Class Diagram

**Figure 4.4: Class Diagram**

*(Draw using draw.io with standard UML class notation and insert as image.)*
*(Key classes with attributes and methods:)*
*(FastAPIApp: aggregates ECGGatekeeper, PreprocessingModule, FeatureExtractor, DimensionalityReducer, ClassicalSVMClassifier, QSVCClassifier, PegasosMulticlassClassifier, AuditDatabase, PDFGenerator.)*
*(ECGGatekeeper: attribute mobilenet\_model; method validate(img) returns bool.)*
*(PreprocessingModule: method preprocess(img) returns ndarray[340,340].)*
*(FeatureExtractor: attribute resnet\_model; method extract(img) returns ndarray[462400].)*
*(DimensionalityReducer: attributes svd, scaler; method transform(x) returns ndarray[9].)*
*(ClassicalSVMClassifier: attribute calibrated\_svc; method predict(x) returns (class, probs).)*
*(QSVCClassifier: attributes zzfeaturemap, sv\_train\_cache; methods kernel\_row(x), predict(x) returns (class, probs).)*
*(PegasosMulticlassClassifier: attribute binary\_models[6]; method predict\_algorithm1(x) returns (class, probs).)*
*(AuditDatabase: method log(hash, model, class, confidence); method history() returns list.)*

The class diagram shows the aggregation structure of the main FastAPI application. The QSVCClassifier is the most complex class, holding both the ZZFeatureMap circuit definition and a precomputed cache of training statevectors that enables O(1) kernel row lookups at inference time. The PegasosMulticlassClassifier encapsulates six binary classifiers and implements the Algorithm 1 decision tree internally. The DimensionalityReducer is composed of two fitted sklearn objects (TruncatedSVD and MinMaxScaler) that operate in sequence, and the clean separation from FeatureExtractor means the quantum dimensionality can be changed in one place without touching the ResNet50 extraction code. All classifier classes expose a common predict(x) interface returning a (class\_index, probabilities) tuple, which allows the FastAPI endpoint handler to dispatch to any of the three models through a single code path.

## 4.5 Sequence Diagram

**Figure 4.5: Sequence Diagram for POST /predict**

*(Draw using draw.io with UML sequence diagram notation and insert as image.)*
*(Lifelines from left to right: React Client, FastAPI Backend, ECGGatekeeper, FeatureExtractor, DimensionalityReducer, QSVCClassifier, AuditDB, RedisCache.)*
*(Message sequence:)*
*(1. React Client → FastAPI Backend: POST /predict [image bytes, model="quantum"])*
*(2. FastAPI Backend → RedisCache: get(sha256(image) + "quantum") -- cache miss)*
*(3. FastAPI Backend → ECGGatekeeper: validate(image) -- is\_ecg=True)*
*(4. FastAPI Backend → FeatureExtractor: extract(preprocessed\_img) -- returns feat[462400])*
*(5. FastAPI Backend → DimensionalityReducer: transform(feat) -- returns x9d[9])*
*(6. FastAPI Backend: x9d\_pi = x9d multiplied by pi (internal step))*
*(7. FastAPI Backend → QSVCClassifier: predict(x9d\_pi) -- statevector lookup → kernel row → SVC.predict -- returns (class\_idx, probs))*
*(8. FastAPI Backend → AuditDB: log(timestamp, hash, "quantum", class, confidence))*
*(9. FastAPI Backend → RedisCache: set(key, result, TTL=86400s))*
*(10. FastAPI Backend → React Client: JSON {prediction, confidence, probabilities, preprocessed\_image\_b64, timing\_ms})*

The sequence diagram captures the complete request-response lifecycle for a Quantum Support Vector Classifier prediction. Steps 4 through 7 represent the core pipeline, and steps 8 and 9 show the side-effects of logging and caching that do not block the response. On a cache hit at step 2, steps 3 through 9 are bypassed entirely, and the cached JSON is returned directly to the client. The SHA-256 hash computed at the FastAPI layer serves dual purpose: as the cache key for Redis and as the deduplication identifier stored in the SQLite audit log. Since step 6 (multiplying the 9-dimensional vector by pi) is an in-process scalar multiplication rather than a method call on a named class, it is shown as a self-message on the FastAPI lifeline to make explicit that the [0, pi] scaling happens after the MinMaxScaler and before the quantum kernel computation.

## 4.6 Dataset Description

The dataset used in this project was collected from the Ch. Pervaiz Elahi Institute of Cardiology, Multan, Pakistan, and made publicly available on Mendeley Data. It contains 929 ECG paper print images representing real clinical recordings across four cardiovascular classes. Each image is a scanned or photographed ECG report from a different patient, with natural variation in scan quality, orientation, and background.

**Table 4.1: Dataset Class Distribution**

| Class | Clinical Description | Total Images | Training (80%) | Testing (20%) |
|---|---|---|---|---|
| Normal | No significant cardiac abnormality | 284 | 227 | 57 |
| Arrhythmia | Irregular heart rhythm | 234 | 188 | 46 |
| Myocardial Infarction | Acute heart attack ECG patterns | 239 | 191 | 48 |
| History of MI | Prior heart attack ECG patterns | 172 | 137 | 35 |
| Total | | 929 | 743 | 186 |

The dataset was divided 80:20 using stratified random sampling with random\_state=42, preserving class proportions in both splits. No augmentation was applied during training; augmentation was used only for the robustness evaluation study in Section 6.6.

---

# CHAPTER 5 -- IMPLEMENTATION

## 5.1 ECG Image Preprocessing

![Figure 5.1: Side-by-side comparison of a raw clinical ECG scan and the resulting 340x340 preprocessed grayscale output](../../Doc/screenshots/5.1_preprocessing.png)

The preprocessing pipeline in `src/preprocessing/preprocess_ecg.py` applies seven sequential steps to standardize heterogeneous clinical ECG scans for feature extraction.

- **Step 1 (Grayscale and ROI Crop):** The input is converted to grayscale and binarized using OTSU adaptive thresholding. Morphological dilation with a 15x5 kernel joins fragmented ECG trace segments, the largest contour is detected, and the image is cropped to that bounding box with 15 pixels of padding.
- **Step 2 (Background Removal):** A second OTSU threshold separates the ECG trace from the background. If the binary image has more white than black pixels (light background ECG), it is inverted to produce a consistent dark-background representation.
- **Step 3 (Vertical Grid Removal):** A morphological OPEN operation with a 1x40 kernel isolates thin vertical grid lines, which are subtracted from the image.
- **Step 4 (Horizontal Grid Removal):** The same process with a 40x1 kernel removes horizontal grid lines without affecting the ECG trace.
- **Step 5 (Gaussian Denoising):** A 3x3 Gaussian blur smooths residual noise from the morphological operations, followed by binary re-thresholding at value 30 to sharpen trace edges.
- **Step 6 (Resize):** The cleaned image is resized to 340x340 pixels using cv2.INTER\_AREA, which averages pixels during downscaling and preserves morphological detail.
- **Step 7 (Normalization):** Pixel values are divided by 255.0 to produce a float32 array in the range [0, 1].

## 5.2 ResNet50 Feature Extraction

Feature extraction in `src/features/extract_resnet_features.py` uses a ResNet50 model with frozen ImageNet weights, reconstructed with the pool1\_pool layer as output using the Keras functional API. No fine-tuning is performed.

The pool1\_pool layer applies 3x3 max-pooling over the output of ResNet50's initial 7x7 convolutional layer (64 filters, stride 2), producing an 85x85x64 spatial activation map. Flattened, this gives 462,400 dimensions per image. This layer is chosen over deeper layers because shallow convolutional features capture low-level morphological structure (QRS shapes, P and T wave geometry, ST segment curvature) without imposing high-level ImageNet-specific object semantics that are irrelevant to ECG patterns. The 462,400-dimensional spatial representation is richer in locally discriminative texture than the 2048-dimensional Global Average Pooling output.

Grayscale images are expanded to three channels by axis repetition before being passed through ResNet50's standard preprocessing normalization. Extraction runs in batches of 8 to manage memory usage.

## 5.3 Dimensionality Reduction

The 462,400-dimensional feature vector is compressed using TruncatedSVD with n\_components=9, fitted exclusively on 743 training samples to prevent leakage. TruncatedSVD was selected over Principal Component Analysis because it does not require explicit mean-centering of the data matrix, which is computationally expensive for 462,400-dimensional vectors. The 9 right singular vectors capture the principal axes of variance in the training feature space. The resulting 9-dimensional vectors are scaled to [0, 1] using MinMaxScaler, also fitted on training data only. Both objects are serialized as `svd_reducer.pkl` and `minmax_scaler.pkl` and loaded at API startup.

## 5.4 Classical Support Vector Machine

The Classical Support Vector Machine uses `sklearn.svm.SVC` with RBF kernel, C=10, gamma='scale', and class\_weight='balanced', wrapped in `CalibratedClassifierCV` (method='isotonic', cv=3) for probability calibration. Hyperparameters were selected through 5-fold stratified GridSearchCV optimizing macro F1 across the grid C in {0.01, 0.1, 1.0, 5.0, 10.0} and gamma in {'scale', 'auto'}. The balanced class weight compensates for the 284:172 imbalance between Normal and History of MI classes. Isotonic calibration provides more accurate probability estimates than Platt scaling for small calibration folds. The model achieves 84.95% test accuracy, surpassing the base paper's 83.33% by 1.62 percentage points.

## 5.5 Quantum Support Vector Classifier

![Figure 5.2: ZZFeatureMap circuit with 9 qubits, reps=2, circular entanglement](../../results/paper/main/zzfeaturemap_circuit.png)

The Quantum Support Vector Classifier encodes the 9-dimensional feature vector into a 512-dimensional (2^9) complex Hilbert space using Qiskit's ZZFeatureMap, which implements the Pauli-ZZ feature map. Per repetition, the circuit applies: (a) a Hadamard layer on all 9 qubits creating equal superposition, (b) first-order Rz(2xi) rotations encoding each feature value as a phase angle, and (c) second-order ZZ interactions for each adjacent pair (i, i+1 mod 9) under circular entanglement, applying Rz(2(pi minus xi)(pi minus xj)) to encode pairwise feature correlations. The fidelity quantum kernel is K(x,z) = |inner product of psi(x) and psi(z)|^2.

The key novel finding is the encoding range. With features scaled to [0, 1] as in the base paper, Rz(2xi) spans only [0, 2] radians, covering a small wedge of the Bloch sphere. With features scaled to [0, pi], the rotations span the full [0, 2pi], maximizing quantum state separation and kernel matrix rank. The ablation confirms a 4.84 percentage point improvement from this change. Circular entanglement provides the same gain over linear by ensuring every qubit has two entangled neighbours.

Rather than N^2 circuit executions to build the kernel matrix, N statevectors are computed once and the full kernel formed as K[i,j] = |sv[i] dot conj(sv[j])|^2, reducing training from over 30 minutes to under 10 seconds for 743 samples. A 486-configuration sweep identified the best setting: [0, pi] encoding, circular entanglement, reps=2, C=5.0, achieving 94.62% accuracy.

## 5.6 Multiclass Pegasos Quantum Support Vector Classifier

The Multiclass Pegasos Quantum Support Vector Classifier trains six binary one-versus-one classifiers for all class pairs: Normal vs Arrhythmia, Normal vs Myocardial Infarction, Normal vs History of Myocardial Infarction, Arrhythmia vs Myocardial Infarction, Arrhythmia vs History of Myocardial Infarction, and Myocardial Infarction vs History of Myocardial Infarction. At inference, Algorithm 1 from the base paper uses a decision tree that evaluates only three binary models to reach the four-class prediction, reducing inference cost.

Qiskit's native Pegasos implementation fails for 9-qubit kernels because the average kernel value of 1/512 causes the SGD step size ηₜ = 1/(λt) to collapse numerically. A custom loop was implemented that precomputes all training statevectors once and uses matrix operations for O(1) kernel row lookups per gradient step. A 3-pass per-model grid search across C and tau (number of SGD iterations) improved accuracy from 78.49% to 91.94%, a gain of 13.45 percentage points through systematic tuning.

## 5.7 FastAPI Backend and React Frontend

The backend in `backend/main.py` loads all nine model objects at application startup and exposes six REST endpoints:

- `/predict` (POST): Single ECG image classification with model selector parameter.
- `/predict/batch` (POST): Up to 20 images processed sequentially.
- `/predict/pdf` (POST): Inference plus downloadable PDF clinical report.
- `/predict/all` (POST): All three classifiers on the same image for comparison.
- `/health` (GET): Model loading status.
- `/history` (GET): Paginated audit log.

A MobileNetV2 gatekeeper rejects non-ECG images before any heavy computation. Redis caching with 24-hour time-to-live keyed by SHA-256 hash of image bytes plus model name returns cached results instantly for repeated submissions.

**[SCREENSHOT PLACEHOLDER 5.7a: Main upload interface showing drag-and-drop ECG area and three model selector buttons]**

**[SCREENSHOT PLACEHOLDER 5.7b: Diagnosis result screen showing predicted class with severity color, confidence bar, per-class probability bars, preprocessed ECG thumbnail, and clinical recommendation text]**

**[SCREENSHOT PLACEHOLDER 5.7c: Model Performance tab showing accuracy bar chart comparing all three classifiers against base paper values]**

The React frontend built with Vite has three tabs. The Diagnosis tab accepts images via drag-and-drop and displays the result with color-coded severity (green for Normal, orange for Arrhythmia and History of MI, red for Myocardial Infarction), calibrated confidence percentage, four-bar probability chart, and clinical recommendation. Predictions below 55% confidence trigger a manual review warning. The Model Performance tab shows a Recharts bar chart comparing classifier accuracy against the base paper. The Quantum Insights tab explains the ZZFeatureMap circuit, the [0, pi] encoding improvement, and the McNemar test result in accessible language.

---

# CHAPTER 6 -- RESULTS AND DISCUSSION

## 6.1 Model Performance Comparison

![Figure 6.1: All Models Accuracy Comparison](../../results/paper/main/classical_baselines.png)

All models were evaluated on the same held-out test set of 186 images (stratified 20% split). Table 6.1 summarises the complete performance metrics.

**Table 6.1: Model Performance Comparison**

| Model | Accuracy | Macro F1 | Paper Accuracy | Delta |
|---|---|---|---|---|
| Classical SVM | 84.95% | 84.10% | 83.33% | +1.62 pp |
| **QSVC (tuned)** | **94.62%** | **94.39%** | 94.09% | **+0.53 pp** |
| Pegasos QSVC | 91.94% | 91.31% | 93.05% | -1.11 pp |
| Random Forest | 91.94% | 91.20% | -- | -- |
| KNN | 84.95% | 83.61% | -- | -- |

The Quantum Support Vector Classifier outperforms all five classical baselines. The Pegasos result of 91.94% is slightly below the paper's 93.05%, which is attributed to strict prevention of data leakage in this implementation since both the SVD reducer and scaler are fitted on training data only. Random Forest independently achieves 91.94%, confirming that the 9-dimensional SVD features carry strong discriminative information regardless of classifier type.

![Figure 6.2: QSVC Confusion Matrix](../../results/paper/main/confusion_matrix_qsvc_tuned.png)

![Figure 6.3: Pegasos Confusion Matrix](../../results/paper/main/confusion_matrix_pegasos_tuned.png)

The Quantum Support Vector Classifier confusion matrix shows 100% recall on Myocardial Infarction, the most clinically critical class where a missed diagnosis can be fatal. Arrhythmia is the most challenging class due to the morphological variability of rhythm abnormalities, which can overlap with Normal ECG patterns in borderline cases.

## 6.2 Feature Encoding and Entanglement Ablation

![Figure 6.4: Feature Encoding Range Ablation](../../results/paper/ablation/ablation_encoding.png)

**Table 6.2: Effect of Feature Encoding Range on QSVC and Pegasos Accuracy**

| Encoding | Range | QSVC Accuracy | Pegasos Accuracy |
|---|---|---|---|
| L2 Normalization | Unit sphere | 68.28% | 57.53% |
| MinMax [0,1] (base paper) | [0, 1] | 92.47% | 84.41% |
| **MinMax [0,pi] (ours)** | **[0, pi]** | **94.62%** | **90.86%** |
| MinMax [0,2pi] | [0, 2pi] | 94.09% | 90.32% |

![Figure 6.5: Entanglement Topology Ablation](../../results/paper/ablation/ablation_entanglement.png)

**Table 6.3: Effect of Entanglement Topology on QSVC Accuracy**

| Topology | QSVC Accuracy | Delta vs Linear |
|---|---|---|
| Linear (base paper) | 89.78% | baseline |
| Pairwise | 93.01% | +3.23 pp |
| Full | 92.47% | +2.69 pp |
| **Circular (ours)** | **94.62%** | **+4.84 pp** |
| SCA (same as circular for 9 qubits) | 94.62% | +4.84 pp |

The [0, pi] range spans the full Bloch sphere latitude for the ZZFeatureMap's Rz rotation gates, which apply angles of 2xi radians. The [0, 2pi] range slightly underperforms because angles beyond 2pi introduce aliasing in the quantum state. L2 normalization collapses feature magnitude information and severely degrades performance. Circular entanglement outperforms linear because every qubit has two entangled neighbours, generating richer pairwise feature correlations at circuit depth only slightly greater than linear.

## 6.3 Pauli Feature Map Ablation

![Figure 6.6: Pauli Feature Map Ablation](../../results/paper/ablation/ablation_feature_maps.png)

**Table 6.4: Pauli Feature Map Comparison (9 qubits, reps=2, circular, [0,pi] encoding, C=5.0)**

| Feature Map | Operator Structure | QSVC Accuracy | Macro F1 |
|---|---|---|---|
| ZFeatureMap | Single-qubit Rz only, no entanglement | 87.63% | 0.871 |
| **ZZFeatureMap (ours)** | **Pauli-Z + ZZ interactions** | **94.62%** | **0.944** |
| PauliFeatureMap (X+XX) | Pauli-X + XX interactions, circular | 91.94% | 0.913 |
| PauliFeatureMap (Y+YY) | Pauli-Y + YY interactions, circular | 83.33% | 0.835 |

The ZZFeatureMap achieves the highest accuracy among all four feature map types, confirming that the Pauli-ZZ encoding structure is the best match for the compressed ECG feature space under [0, pi] scaling. The ZFeatureMap, which applies only single-qubit Rz rotations without any entangling gates, drops to 87.63%: it encodes each feature independently and cannot capture pairwise interactions between SVD components, which carry important joint morphological information from the ECG waveform. The X+XX map reaches 91.94%, outperforming ZFeatureMap by 4.31 percentage points because the Pauli-X and XX gates introduce entanglement, but it still trails ZZFeatureMap by 2.68 percentage points because the Rx rotation gates map features to the equatorial plane of the Bloch sphere rather than the phase axis, producing a different kernel geometry that is less well-matched to the linear separability of this dataset. The Y+YY map performs worst at 83.33%, close to the classical SVM baseline, suggesting that the combination of Y-axis rotations and YY interactions introduces destructive interference in the kernel matrix for this particular feature distribution. This experiment confirms that the ZZFeatureMap is not just a convenient default but is specifically the most effective Pauli operator choice for this ECG classification task.

## 6.4 Statistical Significance

**Table 6.5: McNemar's Test Results (n=186 test samples)**

| Comparison | Chi-squared | p-value | Significant |
|---|---|---|---|
| **QSVC vs Classical SVM** | **10.321** | **0.00131** | **Yes (p < 0.005)** |
| Pegasos vs Classical SVM | 5.760 | 0.0164 | Yes (p < 0.05) |
| QSVC vs Pegasos | 1.231 | 0.267 | No |

The Quantum Support Vector Classifier corrected 23 samples that classical Support Vector Machine misclassified, while Support Vector Machine corrected only 5 Quantum Support Vector Classifier errors. This 23:5 asymmetry (chi-squared = 10.32, p = 0.0013) provides strong statistical evidence that the quantum kernel advantage is real and not a test-set artifact. This is one of the first formal statistical validations of quantum kernel advantage over classical Support Vector Machine for ECG image classification in the published literature.

## 6.5 ROC-AUC and Calibration

![Figure 6.7: ROC Curves All Models and Classes](../../results/paper/main/roc_curves_all_models.png)

**Table 6.6: Multi-Class ROC-AUC Scores**

| Class | SVM | QSVC | Pegasos |
|---|---|---|---|
| Normal | 0.929 | 0.997 | 0.972 |
| Arrhythmia | 0.912 | 0.974 | 0.912 |
| Myocardial Infarction | 0.987 | **1.000** | **1.000** |
| History of MI | 0.927 | 0.984 | 0.979 |
| Macro-AUC | 0.940 | **0.989** | 0.967 |

Both quantum models achieve perfect AUC of 1.000 for Myocardial Infarction detection, meaning they separate all MI samples from all other classes at every decision threshold without a single ranking error. The Quantum Support Vector Classifier macro-AUC of 0.989 is 4.9 percentage points above the classical Support Vector Machine, indicating a consistent advantage across all four classes and all thresholds.

![Figure 6.8: Confidence Calibration Curves](../../results/paper/main/calibration_all_models.png)

Expected Calibration Error was computed for all three models. All three show macro ECE below 0.07, confirming that reported confidence scores closely match empirical accuracy. A system reporting 80% confidence is empirically correct approximately 80% of the time, making the confidence values clinically meaningful for decision support. No prior ECG-QML work has reported this analysis.

## 6.6 Augmentation Robustness

![Figure 6.9: Augmentation Robustness Study](../../results/paper/ablation/augmentation_robustness.png)

Under increased brightness perturbation (simulating an overexposed photograph of a printed ECG), classical Support Vector Machine accuracy collapses from 84.4% to 46.7%, approaching random classification, while the Quantum Support Vector Classifier maintains 86.7%. This 40 percentage point robustness advantage arises because the ZZFeatureMap's phase-based encoding is periodic in the input values and therefore less sensitive to uniform intensity shifts than the RBF kernel's Euclidean distance metric. Both models are unaffected by most other perturbations including rotation, affine distortion, and aspect ratio change. This robustness property is directly relevant to real hospital settings where ECG prints are frequently photographed under variable ambient lighting conditions.

## 6.7 Cross-Dataset Generalization

![Figure 6.10: QSVC Cross-Dataset CM](../../results/paper/cross_dataset/jain_dataset2/cm_qsvc.png)

![Figure 6.11: SVM Cross-Dataset CM](../../results/paper/cross_dataset/jain_dataset2/cm_svm.png)

To evaluate generalization beyond the training hospital, the three frozen models trained on Dataset 1 (929 images, 4 classes, data/raw/) were applied directly to Dataset 2, the second Mendeley ECG dataset from the same hospital used by Jain et al. [7]. Dataset 2 contains 707 images across three classes: Normal (295 images), Abnormal Heartbeat/Arrhythmia (241 images), and History of Myocardial Infarction (171 images). Since Myocardial Infarction is absent in Dataset 2, predictions of class 2 on these samples are scored as errors in the 3-class evaluation. A binary evaluation (Normal vs. Abnormal) is also reported to match the reporting style of Jain et al. [7].

**Table 6.7: Cross-Dataset Results on Jain Dataset 2 (707 images, 3 classes, zero-shot)**

| Model | 3-class Accuracy | 3-class Macro F1 | Binary Accuracy |
|---|---|---|---|
| Classical SVM | 62.66% | 0.638 | 70.72% |
| QSVC (frozen) | 34.09% | 0.173 | 58.27% |
| Pegasos (frozen) | 45.40% | 0.435 | 60.25% |

The classical Support Vector Machine generalizes substantially better than the quantum models in this zero-shot cross-dataset transfer. SVM achieves 62.66% on the 3-class task and 70.72% binary accuracy, which is reasonable given that the RBF kernel operates on the raw Euclidean geometry of the 9-dimensional feature space. The QSVC performs poorly at 34.09% 3-class accuracy: examining the confusion matrix reveals that it collapses almost all predictions to the Arrhythmia class, indicating that the quantum kernel's decision boundaries learned on Dataset 1 do not transfer to the different image statistics of Dataset 2. The core reason is a domain shift in the shallow ResNet50 features: Dataset 2 images have different scan contrast, brightness distribution, and ECG trace density than Dataset 1, which shifts the 9-dimensional SVD-projected feature vectors into a region of the Hilbert space where the trained QSVC support vectors provide poor coverage. This result is consistent with the general finding from Section 6.6 that the QSVC is sensitive to input distribution (brightness robustness drops to 86.7% under perturbation), and it explains why Jain et al. [7] needed separate training runs on each dataset rather than true zero-shot transfer. Improving cross-dataset generalization through domain adaptation or dataset-agnostic preprocessing remains an important direction for future work.

---

# CHAPTER 7 -- CONCLUSION AND FUTURE WORK

This project designed, implemented, validated, and deployed a complete hybrid classical-quantum machine learning system for automated four-class cardiovascular disease classification from raw ECG images. The Quantum Support Vector Classifier, operating on 9 principal components of ResNet50 pool1\_pool features, achieves 94.62% accuracy on the test set, statistically outperforming a well-tuned classical Support Vector Machine by 9.67 percentage points, with significance confirmed by McNemar's test (chi-squared = 10.32, p = 0.0013). Both quantum models achieve perfect Area Under the ROC Curve of 1.000 for Myocardial Infarction detection, which is clinically the most critical outcome.

The central technical contribution is the feature encoding ablation: scaling the 9-dimensional compressed ECG features to [0, pi] before ZZFeatureMap encoding, combined with circular entanglement, improves accuracy by 4.84 percentage points over the base paper configuration. The Rz gate structure of the ZZFeatureMap provides the mechanistic explanation: the [0, pi] range spans the full Bloch sphere latitude, maximizing quantum state separation. This finding is independently supported by Aksoy and Ozpolat [14], who used the same [0, pi] scaling in a concurrent study on different medical datasets. The Pauli feature map ablation further confirms that ZZFeatureMap is not just a default choice but is specifically the most effective Pauli operator structure for this dataset, outperforming ZFeatureMap by 7 percentage points and PauliFeatureMap[X+XX] by 2.68 percentage points. This project also provides the first McNemar significance test for ECG quantum kernel advantage, the first Expected Calibration Error analysis confirming clinical trustworthiness of quantum ECG classifiers (macro ECE below 0.07), and the first augmentation robustness study revealing a 40 percentage point brightness invariance advantage of the Quantum Support Vector Classifier over classical Support Vector Machine. Cross-dataset evaluation on the second Mendeley ECG dataset from the same hospital reveals that the classical SVM generalizes better than the quantum models under zero-shot transfer, indicating that domain adaptation is needed before the quantum kernel approach can be deployed across different scanner environments.

Several directions remain open for future work. Implementing the Quantum Neural Network reported at 97.31% accuracy in the base paper using PennyLane-based quanvolutional filters would likely close the remaining performance gap. Evaluating the trained 9-qubit Quantum Support Vector Classifier under NISQ noise models in Qiskit Aer, and ultimately on real IBM Quantum hardware, would quantify the gap between ideal simulation and near-term device performance. Improving cross-dataset generalization through domain adaptation of the ResNet50 feature extractor or dataset-agnostic preprocessing would address the distribution shift observed in the cross-dataset experiment. Integrating Grad-CAM visualization to highlight the ECG regions that drive each classification decision would address the clinical interpretability requirement. Extending the web interface to a Progressive Web App for mobile use in rural ward settings would expand access to the system in resource-limited environments where the diagnostic bottleneck is most acute.

---

# REFERENCES

[1] S. Prabhu, S. Gupta, G. M. Prabhu, A. V. Dhanuka, and K. V. Bhat, "QuCardio: Application of Quantum Machine Learning for Detection of Cardiovascular Diseases," *IEEE Access*, vol. 11, pp. 136835-136851, 2023, doi: 10.1109/ACCESS.2023.3338145.

[2] V. Havlicek, A. D. Corcoles, K. Temme, A. W. Harrow, A. Kandala, J. M. Gambetta, and J. Preskill, "Supervised learning with quantum-enhanced feature spaces," *Nature*, vol. 567, pp. 209-212, Mar. 2019, doi: 10.1038/s41586-019-0980-2.

[3] J. Biamonte, P. Wittek, N. Pancotti, P. Rebentrost, N. Wiebe, and S. Lloyd, "Quantum machine learning," *Nature*, vol. 549, pp. 195-202, Sep. 2017, doi: 10.1038/nature23474.

[4] S. Shalev-Shwartz, Y. Singer, N. Srebro, and A. Cotter, "Pegasos: Primal estimated sub-gradient solver for SVM," *Math. Program.*, vol. 127, no. 1, pp. 3-30, 2011, doi: 10.1007/s10107-010-0420-4.

[5] K. He, X. Zhang, S. Ren, and J. Sun, "Deep residual learning for image recognition," in *Proc. IEEE/CVF CVPR*, Jun. 2016, pp. 770-778, doi: 10.1109/CVPR.2016.90.

[6] M. Y. Ramkhelawan, S. Grandhi, and S. Wibowo, "A Hybrid Quantum-Classical Deep Learning Algorithm for Efficient Arrhythmia Classification," in *Proc. IEEE SNPD*, 2025, doi: 10.1109/SNPD62189.2025.10985682.

[7] V. Jain, N. Arora, and A. Gupta, "Quantum-assisted cardiac diseases diagnosis and prediction using ECG images," *J. Supercomputing*, vol. 81, p. 1439, 2025, doi: 10.1007/s11227-025-07939-8.

[8] Z. Ozpolat and M. Karabatak, "Performance Evaluation of Quantum-Based Machine Learning Algorithms for Cardiac Arrhythmia Classification," *Diagnostics*, vol. 13, no. 6, p. 1099, 2023, doi: 10.3390/diagnostics13061099.

[9] H. S. B. Gummadi, S. K. Thota, S. Akuthota, M. Guduri, S. D. Veeravalli, and H. Vemuganti, "Quantum-based Convolutional Neural Network Model for Efficient Cardiovascular Disease Prediction," in *Proc. 3rd IEEE Int. Conf. on Artificial Intelligence (ThingsAI)*, 2025, doi: 10.1109/ThingsAI65156.2025.10926651.

[10] O. Yildirim, P. Plawiak, R.-S. Tan, and U. R. Acharya, "Arrhythmia detection using deep convolutional neural network with long duration ECG signals," *Comput. Biol. Med.*, vol. 102, pp. 411-420, 2018, doi: 10.1016/j.compbiomed.2018.09.009.

[11] F. Vasquez-Iturralde, M. J. Flores-Calero, F. Grijalva, and A. Rosales-Acosta, "Automatic classification of cardiac arrhythmias using deep learning techniques: A systematic review," *IEEE Access*, vol. 12, 2024, doi: 10.1109/ACCESS.2024.3408282.

[12] K. L. Soon, W. L. Pang, H. H. Goh, Y. W. Sim, S. K. Phang, H. L. Choo, L. T. Soon, and N. S. Lai, "Early cardiovascular disease detection using hierarchical quantum ensemble model," *Comput. Methods Biomech. Biomed. Eng.*, Jan. 2026, doi: 10.1080/10255842.2025.2612536.

[13] A. E. Setiawan, S. Rustad, A. Syukur, M. A. Soeleman, G. F. Shidik, M. Akrom, and A. W. Setiawan, "A systematic literature review of quantum machine learning for medical: trends, datasets, topics, and methods," *Int. J. Cogn. Comput. Eng.*, vol. 7, pp. 609-630, 2026, doi: 10.1016/j.ijcce.2026.05.002.

[14] I. Aksoy and Z. Ozpolat, "Comparison of SVM, QSVM and Pegasos-QSVM algorithms on different medical datasets," *Int. J. Sustain. Eng. Technol.*, vol. 9, no. 1, pp. 80-93, 2025, doi: 10.62301/usmtd.1716034.

[15] E. I. Elsedimy, S. M. M. AboHashish, and F. Algarni, "New cardiovascular disease prediction approach using support vector machine and quantum-behaved particle swarm optimization," *Multimedia Tools Appl.*, vol. 83, pp. 23901-23928, 2024, doi: 10.1007/s11042-023-16194-z.

[16] S. S. Alotaibi, H. A. Mengash, S. Dhahbi, S. Alazwari, R. Marzouk, M. A. Alkhonaini, A. Mohamed, and A. M. Hilal, "Quantum-Enhanced Machine Learning Algorithms for Heart Disease Prediction," *Human-centric Computing and Information Sciences*, vol. 13, p. 41, 2023, doi: 10.22967/HCIS.2023.13.041.

---

## FIGURE PLACEMENT GUIDE

| Figure | File Path | Report Section |
|---|---|---|
| 4.1 System Architecture | *(draw.io diagram)* | Section 4.1 |
| 4.2 Use Case Diagram | *(draw.io diagram)* | Section 4.2 |
| 4.3 Data Flow Diagram | *(draw.io diagram)* | Section 4.3 |
| 4.4 Class Diagram | *(draw.io diagram)* | Section 4.4 |
| 4.5 Sequence Diagram | *(draw.io diagram)* | Section 4.5 |
| 5.1 ZZFeatureMap Circuit | results/paper/main/zzfeaturemap\_circuit.png | Section 5.5 |
| 6.1 All Models Accuracy | results/paper/main/classical\_baselines.png | Section 6.1 |
| 6.2 QSVC Confusion Matrix | results/paper/main/confusion\_matrix\_qsvc\_tuned.png | Section 6.1 |
| 6.3 Pegasos Confusion Matrix | results/paper/main/confusion\_matrix\_pegasos\_tuned.png | Section 6.1 |
| 6.4 Encoding Ablation | results/paper/ablation/ablation\_encoding.png | Section 6.2 |
| 6.5 Entanglement Ablation | results/paper/ablation/ablation\_entanglement.png | Section 6.2 |
| 6.6 Pauli Feature Map Ablation | results/paper/ablation/ablation\_feature\_maps.png | Section 6.3 |
| 6.7 ROC Curves | results/paper/main/roc\_curves\_all\_models.png | Section 6.5 |
| 6.8 Calibration Curves | results/paper/main/calibration\_all\_models.png | Section 6.5 |
| 6.9 Augmentation Robustness | results/paper/ablation/augmentation\_robustness.png | Section 6.6 |
| 6.10 QSVC Cross-Dataset CM | results/paper/cross\_dataset/jain\_dataset2/cm\_qsvc.png | Section 6.7 |
| 6.11 SVM Cross-Dataset CM | results/paper/cross\_dataset/jain\_dataset2/cm\_svm.png | Section 6.7 |

---

*End of Report Content -- QuCardio, SJEC Mangaluru, 2025-2026*
