"""
Final test: run the EXACT backend pipeline on processed_340 images.
These were the images used for training, so predictions must be correct.
"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import joblib
import tempfile
import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import Model
from pathlib import Path

print("Loading ResNet50 pool1_pool...")
base   = ResNet50(weights="imagenet", include_top=False)
pool1  = base.get_layer("pool1_pool").output
resnet = Model(inputs=base.input, outputs=pool1)
svd    = joblib.load("backend/models/svd_reducer.pkl")
scaler = joblib.load("backend/models/minmax_scaler.pkl")
svm    = joblib.load("backend/models/svm_model.pkl")
CLASS  = ["Normal","Arrhythmia","Myocardial_Infarction","History_of_MI"]
print("Models loaded.\n")

def predict_training_style(img_path):
    """Load EXACTLY as training: cv2.IMREAD_GRAYSCALE → float32 [0–255]"""
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE).astype("float32")
    X = np.repeat(img[None,:,:,None], 3, axis=-1)
    f = resnet.predict(preprocess_input(X), verbose=0).reshape(1, -1)
    return int(svm.predict(scaler.transform(svd.transform(f)))[0])

def predict_backend_style(img_path_bgr):
    """Exact backend pipeline on a BGR image (simulating uploaded file)"""
    img_bgr = cv2.imread(str(img_path_bgr))
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, binary_roi = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        lc = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(lc)
        pad = 10
        x=max(0,x-pad); y=max(0,y-pad)
        w=min(gray.shape[1]-x,w+2*pad); h=min(gray.shape[0]-y,h+2*pad)
        cropped = gray[y:y+h,x:x+w]
    else:
        cropped = gray
    _, cleaned = cv2.threshold(cropped, 200, 255, cv2.THRESH_BINARY)
    cleaned = cv2.bitwise_not(cleaned)
    kv = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 50))
    cleaned = cv2.subtract(cleaned, cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kv))
    resized = cv2.resize(cleaned, (340, 340), interpolation=cv2.INTER_AREA)
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as jtmp:
        jp = jtmp.name
    cv2.imwrite(jp, resized)
    img_f32 = cv2.imread(jp, cv2.IMREAD_GRAYSCALE).astype("float32")
    os.remove(jp)
    X = np.repeat(img_f32[None,:,:,None], 3, axis=-1)
    f = resnet.predict(preprocess_input(X), verbose=0).reshape(1, -1)
    sc = scaler.transform(svd.transform(f))
    return int(svm.predict(sc)[0]), float(svm.predict_proba(sc)[0].max())

# Test A: processed_340 files (identical to training data load)
print("=" * 70)
print("TEST A: processed_340 images (same files as training)")
print("  Training-style load should give correct predictions")
print("=" * 70)
data_dir = Path("data/processed_340")
raw_dir  = Path("data/raw")
correct_a = 0; total_a = 0

for cls in CLASS:
    for p in list((data_dir / cls).glob("*.jpg"))[:4]:
        pred = predict_training_style(p)
        match = "✅" if pred == CLASS.index(cls) else "❌"
        if pred == CLASS.index(cls): correct_a += 1
        total_a += 1
        print(f"  {match} [{cls[:12]}]  {p.name[:20]}  → {CLASS[pred]}")

print(f"\n  Training-style accuracy: {correct_a}/{total_a} = {correct_a/total_a*100:.0f}%\n")

# Test B: Upload raw images through backend pipeline
print("=" * 70)
print("TEST B: raw images through backend pipeline (simulates upload)")
print("=" * 70)
correct_b = 0; total_b = 0

for cls in CLASS:
    for p in list((raw_dir / cls).glob("*.jpg"))[:4]:
        pred, conf = predict_backend_style(p)
        match = "✅" if pred == CLASS.index(cls) else "❌"
        if pred == CLASS.index(cls): correct_b += 1
        total_b += 1
        print(f"  {match} [{cls[:12]}]  {p.name[:20]}  → {CLASS[pred]}  ({conf*100:.0f}%)")

print(f"\n  Backend accuracy on raw images: {correct_b}/{total_b} = {correct_b/total_b*100:.0f}%")
print()
if correct_b / total_b >= 0.75:
    print("✅ Backend pipeline is working! Arrhythmia bug is FIXED.")
else:
    print("❌ Still failing — deeper investigation needed.")
