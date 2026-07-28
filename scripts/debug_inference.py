"""
debug_inference.py
==================
End-to-end pipeline diagnostic.
Takes ONE known image from data/processed_340/ (training set) and:
  1. Loads it EXACTLY as the training code did
  2. Runs it through the backend's inference pipeline
  3. Shows what the SVM predicts → should match the true label

Run from project root:
  python scripts/debug_inference.py

Expected result: prediction matches the folder name (e.g. Normal → Normal)
If it predicts Arrhythmia for everything → the SVD/scaler mismatch is still present.
"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import joblib
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import Model

import config
from src.preprocessing.preprocess_ecg import preprocess_ecg

# ── Load models ───────────────────────────────────────────────────────────────
print("Loading ResNet50 pool1_pool ...")
base_model  = ResNet50(weights='imagenet', include_top=False)
pool1_out   = base_model.get_layer('pool1_pool').output
resnet      = Model(inputs=base_model.input, outputs=pool1_out)

model_dir   = Path('backend/models')
svd         = joblib.load(model_dir / 'svd_reducer.pkl')
scaler      = joblib.load(model_dir / 'minmax_scaler.pkl')
svm         = joblib.load(model_dir / 'svm_model.pkl')

CLASS_NAMES = ['Normal', 'Arrhythmia', 'Myocardial_Infarction', 'History_of_MI']
print("All models loaded.\n")

# ── Pick 2 images per class ───────────────────────────────────────────────────
data_dir = Path('data/processed_340')
test_cases = []
for cls in CLASS_NAMES:
    cls_dir = data_dir / cls
    imgs = sorted(cls_dir.glob('*.jpg'))[:2] + sorted(cls_dir.glob('*.png'))[:2]
    for p in imgs[:2]:
        test_cases.append((cls, p))

print(f"Testing {len(test_cases)} images (2 per class)\n")
print(f"{'True Class':<28} {'Predicted':<28} {'Confidence':>10}  Match?")
print("-" * 75)

correct = 0
for true_class, img_path in test_cases:
    true_label = CLASS_NAMES.index(true_class)

    # ── METHOD A: Exactly as training did ─────────────────────────────────
    # Training: cv2.imread(GRAYSCALE) → float32 (values 0–255, NO /255)
    img_train_style = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE).astype(np.float32)
    img_train_style = cv2.resize(img_train_style, (340, 340)) if img_train_style.shape != (340,340) else img_train_style

    X_a = np.expand_dims(img_train_style, axis=0)[..., np.newaxis]
    X_a = np.repeat(X_a, 3, axis=-1)
    feat_a = resnet.predict(preprocess_input(X_a), verbose=0).reshape(1, -1)
    svd_a  = svd.transform(feat_a)
    sc_a   = scaler.transform(svd_a)
    pred_a = int(svm.predict(sc_a)[0])
    prob_a = float(svm.predict_proba(sc_a)[0][pred_a])

    match = "✅" if pred_a == true_label else "❌"
    if pred_a == true_label:
        correct += 1
    print(f"{true_class:<28} {CLASS_NAMES[pred_a]:<28} {prob_a*100:>9.1f}%  {match}")

print("-" * 75)
print(f"Accuracy: {correct}/{len(test_cases)}  ({correct/len(test_cases)*100:.0f}%)")
print()

# ── METHOD B: Via preprocess_ecg() (backend's current path) ──────────────
# Training: OTSU-binarized JPG → read GRAYSCALE → float32 [0–255]
# Backend:  preprocess_ecg() → float32 [0,1] → *255 → float32 [0–255]
# These SHOULD be identical since processed_340 files ARE the OTSU output saved as uint8
print("\n── Comparing Training vs Backend preprocessing ──")
img_path = test_cases[0][1]
true_class = test_cases[0][0]

# Training style
img_train = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE).astype(np.float32)

# Backend style — re-run preprocess_ecg on the already-preprocessed file
# (This simulates uploading a processed_340 image directly)
img_backend = preprocess_ecg(str(img_path), output_size=(340, 340)) * 255.0

diff = np.abs(img_train - img_backend)
print(f"  Image: {img_path.name}  (true: {true_class})")
print(f"  Training style:  min={img_train.min():.0f}  max={img_train.max():.0f}  mean={img_train.mean():.1f}")
print(f"  Backend  style:  min={img_backend.min():.0f}  max={img_backend.max():.0f}  mean={img_backend.mean():.1f}")
print(f"  Max pixel diff:  {diff.max():.2f}   Mean diff: {diff.mean():.4f}")

if diff.max() < 5:
    print("  ✅ Pixel-level match — preprocessing pipelines are identical.")
else:
    print("  ⚠️  Pixel-level MISMATCH — backend preprocessing differs from training!")
    print("      → Features fed to ResNet50 will be different → wrong SVD projection → wrong class")

# ── Check SVD output range ────────────────────────────────────────────────────
print("\n── SVD output range check (should be ~[0,1] after MinMax) ──")
data = np.load('data/features_9d.npz')
X_tr = data['train_features']
X_te = data['test_features']
print(f"  Train 9D: min={X_tr.min():.4f}  max={X_tr.max():.4f}  (should be 0.0 and 1.0)")
print(f"  Test  9D: min={X_te.min():.4f}  max={X_te.max():.4f}  (test min/max may exceed [0,1])")

# Run one train sample through inference and check it matches stored value
feat_check = resnet.predict(preprocess_input(
    np.repeat(np.expand_dims(cv2.imread(str(test_cases[0][1]),
    cv2.IMREAD_GRAYSCALE).astype(np.float32), axis=0)[..., np.newaxis], 3, axis=-1)
), verbose=0).reshape(1, -1)
svd_check  = svd.transform(feat_check)
sc_check   = scaler.transform(svd_check)
print(f"\n  Inference 9D for {test_cases[0][1].name}:")
print(f"    {sc_check[0]}")
