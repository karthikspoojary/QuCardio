# QuCardio — Quantum ECG Classification System

> **Classify cardiovascular conditions from ECG images using quantum machine learning — with statistical validation, confidence calibration, and a real-time clinical web interface.**

[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue)](https://www.python.org/)
[![Qiskit 0.45](https://img.shields.io/badge/Qiskit-0.45-purple)](https://qiskit.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green)](https://fastapi.tiangolo.com/)
[![React 18](https://img.shields.io/badge/React-18-cyan)](https://react.dev/)
[![QSVC Accuracy](https://img.shields.io/badge/QSVC%20Accuracy-94.62%25-brightgreen)]()

---

## What Is QuCardio?

QuCardio is a complete hybrid classical-quantum machine learning system that takes a photographed or scanned ECG paper print as input and returns a four-class cardiovascular diagnosis — **Normal**, **Arrhythmia**, **Myocardial Infarction**, or **History of MI** — with a calibrated confidence score and a class-specific clinical recommendation.

The system was built to answer a question that prior quantum ECG classifiers never asked: *which specific circuit design choices actually produce the result, and do they hold up under statistical testing?* A 320-configuration ablation study, a McNemar significance test, a calibration analysis, and a cross-dataset transfer test are all included alongside the deployed application.

---

## Key Results

| Model | Accuracy | Macro F1 | McNemar vs SVM |
|---|---|---|---|
| **QSVC** (ZZFeatureMap, circular, [0,π]) | **94.62%** | **0.944** | χ²=10.32, **p=0.0013** |
| Pegasos QSVC | 91.94% | 0.913 | p=0.016 |
| Classical SVM (RBF, C=10) | 84.95% | 0.841 | — |
| Random Forest | 91.94% | 0.912 | — |
| k-Nearest Neighbours | 84.95% | 0.836 | — |

- Both quantum models reach **AUC = 1.000** on Myocardial Infarction (zero ranking errors across 48 test cases)
- Expected Calibration Error **< 0.07** for all three models — confidence scores are reliable
- Under brightness perturbation (overexposed scan), QSVC holds **86.7%** while the classical SVM falls to **46.7%**
- End-to-end QSVC latency: **893.9 ms** (preprocessing 126.9 ms, ResNet50 627.6 ms, SVD 52.0 ms, classification 87.4 ms)

---

## How It Works — Pipeline at a Glance

```
ECG image (JPEG/PNG)
        │
        ▼
┌─────────────────────┐
│  MobileNetV2        │  ← Rejects non-ECG uploads before any heavy work
│  ECG Gatekeeper     │
└─────────────────────┘
        │ valid ECG
        ▼
┌─────────────────────┐
│  OpenCV             │  ← 7 steps: grayscale, OTSU threshold, dilation,
│  Preprocessing      │    contour crop, grid removal, Gaussian blur, resize
│  → 340×340 float32  │
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  ResNet50           │  ← Frozen ImageNet weights, taps pool1_pool layer
│  Feature Extraction │    Captures low-level morphology (QRS shape, ST segment)
│  → 462,400-D vector │
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  Truncated SVD      │  ← Fitted on training data only (no leakage)
│  + MinMax Scaler    │    Compresses to 9 dimensions, scaled to [0,1]
│  → 9-D vector       │
└─────────────────────┘
        │
        ├──────────────────────────────────────────────┐
        │ ×π for quantum                               │ [0,1] for SVM
        ▼                                              ▼
┌──────────────────────┐                  ┌─────────────────────┐
│  QSVC / Pegasos QSVC │                  │  Classical SVM      │
│  ZZFeatureMap 9-qubit│                  │  RBF, C=10          │
│  Fidelity kernel     │                  │  Isotonic calibrated│
└──────────────────────┘                  └─────────────────────┘
        │                                              │
        └──────────────────┬───────────────────────────┘
                           ▼
                  Class label + Confidence + Recommendation
```

---

## Project Structure

```
QuCardio/
├── src/
│   ├── preprocessing/
│   │   ├── preprocess_ecg.py          # 7-step OpenCV pipeline (training)
│   │   └── preprocess_inference.py    # Inference-time preprocessing
│   ├── features/
│   │   ├── extract_resnet_features.py # ResNet50 pool1_pool extraction (462K-D)
│   │   └── reduce_dimensions.py       # TruncatedSVD(9) + MinMaxScaler
│   ├── quantum/
│   │   ├── train_qsvc.py              # QSVC training (statevector kernel)
│   │   ├── tune_qsvc.py               # 320-config grid sweep
│   │   ├── train_pegasos.py           # Multiclass Pegasos QSVC
│   │   └── tune_pegasos.py            # Per-model Pegasos hyperparameter search
│   ├── classical/
│   │   ├── svm_baseline.py            # Tuned classical SVM
│   │   └── classical_baselines.py     # RF, KNN, LR, MLP baselines
│   ├── ablation/
│   │   ├── ablation_encoding.py       # Encoding range sweep
│   │   ├── ablation_entanglement.py   # Topology sweep
│   │   ├── ablation_feature_maps.py   # Pauli map comparison
│   │   ├── ablation_svd_dims.py       # Component count sweep
│   │   └── reps_ablation.py           # Circuit repetitions sweep
│   ├── analysis/
│   │   ├── augmentation_robustness.py # 9 perturbation conditions
│   │   ├── cross_dataset_jain_d2.py   # Zero-shot transfer test
│   │   └── confidence_intervals.py    # Bootstrap CIs
│   └── ood/
│       └── train_ecg_detector.py      # MobileNetV2 gatekeeper training
│
├── backend/
│   ├── main.py                        # FastAPI app (6 REST endpoints)
│   ├── database.py                    # SQLite audit log
│   ├── pdf_generator.py               # PDF clinical report generation
│   └── models/                        # Serialised model artefacts (git-ignored)
│
├── frontend/
│   └── src/
│       ├── App.jsx                    # Main React app (6 tabs)
│       └── BatchUploadTab.jsx         # Batch upload UI
│
├── results/
│   ├── paper/main/                    # Confusion matrices, ROC curves, metrics
│   ├── paper/ablation/                # All ablation JSON results and plots
│   ├── paper/statistics/              # McNemar, ECE, confidence intervals
│   └── paper/cross_dataset/           # Zero-shot transfer results
│
├── data/                              # Features, statevectors (git-ignored)
├── config.py                          # Central project configuration
├── docker-compose.yml                 # One-command deployment
├── Dockerfile                         # Backend container
└── requirements.txt
```

---

## Quick Start

### Option A — Docker (recommended)

```bash
git clone https://github.com/karthikspoojary/QuCardio.git
cd QuCardio
docker compose up
```

Open **http://localhost:5173** in your browser.

> **Note:** Trained model artefacts must be placed in `backend/models/` before starting.  
> See [Training the Models](#training-the-models) if you need to train from scratch.

### Option B — Local Development

**Prerequisites:** Python 3.10+, Node.js 18+, Redis (optional)

```bash
# 1. Clone and enter the project
git clone https://github.com/karthikspoojary/QuCardio.git
cd QuCardio

# 2. Create a Python virtual environment
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. Install frontend dependencies
cd frontend && npm install && cd ..

# 5. Start the backend (Terminal 1)
cd backend && uvicorn main:app --host 0.0.0.0 --port 8000

# 6. Start the frontend (Terminal 2)
cd frontend && npm run dev
```

Open **http://localhost:5173** in your browser.

---

## Dataset Setup

This project uses the **ECG Images Dataset** released on Mendeley Data by Khan et al., collected at the Ch. Pervaiz Elahi Institute of Cardiology, Multan, Pakistan.

**Download:** [Mendeley Data — ECG Images Dataset](https://data.mendeley.com/datasets/gwbz3fsgp8/2)

After downloading, place images into:

```
data/raw/
├── Normal/                   # 284 images
├── Arrhythmia/               # 233 images
├── Myocardial_Infarction/    # 240 images (one unreadable, 239 used)
└── History_of_MI/            # 172 images
```

---

## Training the Models

Run these scripts sequentially from the project root. Training requires the dataset in `data/raw/`.

```bash
# Step 1: Preprocess images → data/processed_340/
python src/preprocessing/preprocess_ecg.py

# Step 2: Extract ResNet50 features → data/resnet50_features.npz
python src/features/extract_resnet_features.py

# Step 3: Reduce dimensions → data/features_9d.npz
python src/features/reduce_dimensions.py

# Step 4: Train classical SVM → backend/models/svm_model.pkl
python src/classical/svm_baseline.py

# Step 5: Train QSVC (tuned) → backend/models/qsvc_model.pkl
python src/quantum/tune_qsvc.py

# Step 6: Train Pegasos QSVC → backend/models/pegasos_*.pkl
python src/quantum/train_pegasos.py && python src/quantum/tune_pegasos.py
```

---

## API Endpoints

The FastAPI backend exposes six endpoints at `http://localhost:8000`:

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/predict` | Single ECG — returns class, confidence, probabilities, preprocessed image |
| `POST` | `/predict/batch` | Up to 20 ECGs — returns individual results for each |
| `POST` | `/predict/pdf` | Single ECG — returns a downloadable PDF clinical report |
| `POST` | `/predict/all` | Single ECG — runs all three models for comparison |
| `GET` | `/health` | Backend status and model load state |
| `GET` | `/history` | Paginated audit log of past predictions |

Interactive API documentation is available at **http://localhost:8000/docs**.

---

## Web Interface

The React dashboard provides six tabs:

| Tab | Purpose |
|---|---|
| **Diagnosis** | Drag-and-drop ECG upload, model selector, result display with colour-coded severity |
| **Batch Upload** | Process up to 20 ECGs in one request |
| **Model Performance** | Accuracy, F1, precision, recall comparison across all three models |
| **Quantum Insights** | Plain-language explanation of the ZZFeatureMap circuit and ablation findings |
| **API Docs** | Inline REST endpoint documentation |
| **History** | Paginated audit log of past predictions |

Severity colour coding: 🟢 Normal · 🟡 Arrhythmia · 🟡 History of MI · 🔴 Myocardial Infarction

---

## Reproducibility

All results in the associated paper are reproducible from the scripts in this repository. Key result files:

| File | Contents |
|---|---|
| `results/paper/main/metrics_report.json` | Accuracy, F1, AUC for all models |
| `results/paper/statistics/mcnemar_test.json` | McNemar χ² and p-values |
| `results/paper/statistics/ece_scores.json` | Expected Calibration Error |
| `results/paper/ablation/ablation_encoding.json` | Encoding range sweep |
| `results/paper/ablation/ablation_entanglement.json` | Topology sweep |
| `results/paper/ablation/o2_linear_mcnemar.json` | Linear cell comparison (b=8, c=2) |
| `results/paper/ablation/o3_01_full.json` | [0,1]+full reproduction (93.01%) |

---

## Citation

If you use this work, please cite the associated paper:

> *Quantum Kernel Circuit Design for Four-Class ECG Image Classification: Encoding Range, Entanglement Topology, and Statistical Validation* — St. Joseph Engineering College, Mangaluru, 2025.

The base methodology extends the work of Prabhu et al.:
> S. Prabhu et al., "QuCardio: Application of quantum machine learning for detection of cardiovascular diseases," *IEEE Access*, vol. 11, pp. 136122–136135, 2023. doi: [10.1109/ACCESS.2023.3338145](https://doi.org/10.1109/ACCESS.2023.3338145)

---

## License

MIT — see `LICENSE` for details.
