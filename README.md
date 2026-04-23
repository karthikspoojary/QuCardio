# QuCardio — ECG Classification Pipeline

An end-to-end ECG image classification system implementing the methodology from the **QuCardio** paper (Prabhu et al.). This branch (`classical`) implements the full Classical ML Pipeline using ResNet50 feature extraction and a tuned Support Vector Machine (SVM), achieving **91.4% accuracy** — surpassing the paper's 83.33% baseline.

## 📁 Project Structure

```
QuCardio/
├── src/
│   ├── preprocessing/
│   │   └── preprocess_ecg.py       # Image preprocessing (crop, denoise, resize)
│   ├── features/
│   │   ├── extract_resnet_features.py  # ResNet50 pool1_pool feature extraction
│   │   └── reduce_dimensions.py        # Truncated SVD to 9D + MinMax scaling
│   └── classical/
│       └── svm_baseline.py         # Tuned Classical SVM (Grid Search)
├── backend/
│   ├── main.py                     # FastAPI backend server
│   └── models/                     # Trained .pkl model artifacts (git-ignored)
├── frontend/
│   └── src/
│       └── App.jsx                 # React dashboard UI
├── data/                           # Datasets (git-ignored, see below)
├── results/                        # Output metrics and confusion matrix
├── requirements.txt
└── README.md
```

## 🏆 Results

| Model | Paper Accuracy | Our Accuracy |
|---|---|---|
| Classical SVM (Default) | 83.33% | 62.37% |
| Classical SVM (Grid Search Tuned) | 83.33% | **91.40%** ✅ |

**Best Parameters Found:** `C=500, gamma='scale', kernel='rbf'`

## 🗂️ Dataset Setup

This project uses the **Mendeley ECG Image Dataset**. Download it and place images into:

```
data/
└── raw/
    ├── Normal/
    ├── Arrhythmia/
    ├── Myocardial_Infarction/
    └── History_of_MI/
```

> **Note:** The dataset is NOT included in this repository due to its size.

## ⚙️ Setup

```bash
# 1. Clone the repo
git clone <your-repo-url>
cd QuCardio

# 2. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. Install frontend dependencies
cd frontend && npm install && cd ..
```

## 🚀 Running the ML Pipeline (One-Time Setup)

Run these scripts sequentially from the project root. Make sure your dataset is in `data/raw/` first.

```bash
# Step 1: Preprocess images (~20-30 min)
python src/preprocessing/preprocess_ecg.py

# Step 2: Extract ResNet50 features (~45-60 min)
python src/features/extract_resnet_features.py

# Step 3: Apply SVD dimensionality reduction (~2-5 min)
python src/features/reduce_dimensions.py

# Step 4: Train and tune the Classical SVM (~5-10 min)
python src/classical/svm_baseline.py
```

## 🖥️ Running the Dashboard

After training, open **two terminals** from the project root:

**Terminal 1 — Backend:**
```bash
source venv/bin/activate
cd backend
uvicorn main:app --port 8000
```

**Terminal 2 — Frontend:**
```bash
cd frontend
npm run dev
```

Then open your browser at **http://localhost:5173**

## 🧠 Methodology (QuCardio Paper)

- **Preprocessing**: Border crop → Background removal → Vertical grid line removal → Resize to 340×340
- **Feature Extraction**: Pre-trained ResNet50, features from the `pool1_pool` layer (shallow features), flattened to a 1D vector
- **Dimensionality Reduction**: Truncated SVD to compress features to **9 components**, followed by MinMax scaling to `[0, 1]`
- **Classification**: RBF-kernel SVM with Grid Search hyperparameter tuning (`C`, `gamma`)
