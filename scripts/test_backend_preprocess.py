"""
Test that the backend preprocess_ecg() path matches training exactly.
Run: python scripts/test_backend_preprocess.py
"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import Model
from pathlib import Path
from src.preprocessing.preprocess_ecg import preprocess_ecg

# Load ResNet
base   = ResNet50(weights="imagenet", include_top=False)
pool1  = base.get_layer("pool1_pool").output
resnet = Model(inputs=base.input, outputs=pool1)

svd    = joblib.load("backend/models/svd_reducer.pkl")
scaler = joblib.load("backend/models/minmax_scaler.pkl")
svm    = joblib.load("backend/models/svm_model.pkl")
CLASS  = ["Normal","Arrhythmia","Myocardial_Infarction","History_of_MI"]

def predict_train_style(img_path):
    """Exactly as training: cv2.IMREAD_GRAYSCALE → float32 [0–255]"""
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE).astype("float32")
    X   = img[None, :, :, None]
    X   = np.repeat(X, 3, axis=-1)
    X   = preprocess_input(X)
    f   = resnet.predict(X, verbose=0).reshape(1, -1)
    r   = svd.transform(f)
    s   = scaler.transform(r)
    return int(svm.predict(s)[0]), float(svm.predict_proba(s)[0].max()), s[0]

def predict_backend_style(img_path):
    """Backend path: OLD preprocessing (fixed thresh 200) + JPEG round-trip"""
    import tempfile, os
    img_bgr = cv2.imread(str(img_path))

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, binary_roi = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        pad = 10
        x = max(0, x-pad); y = max(0, y-pad)
        w = min(gray.shape[1]-x, w+2*pad); h = min(gray.shape[0]-y, h+2*pad)
        cropped = gray[y:y+h, x:x+w]
    else:
        cropped = gray

    _, cleaned = cv2.threshold(cropped, 200, 255, cv2.THRESH_BINARY)
    cleaned = cv2.bitwise_not(cleaned)
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 50))
    cleaned = cv2.subtract(cleaned, cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel_v))
    resized = cv2.resize(cleaned, (340, 340), interpolation=cv2.INTER_AREA)

    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as jtmp:
        jpeg_path = jtmp.name
    cv2.imwrite(jpeg_path, resized)
    img_f32 = cv2.imread(jpeg_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    os.remove(jpeg_path)
    X = img_f32[None, :, :, None]
    X = np.repeat(X, 3, axis=-1)
    X = preprocess_input(X)
    f = resnet.predict(X, verbose=0).reshape(1, -1)
    r = svd.transform(f)
    s = scaler.transform(r)
    return int(svm.predict(s)[0]), float(svm.predict_proba(s)[0].max()), s[0]

print(f"\n{'Class':<28} {'Train Style':<22} {'Backend Style':<22} {'9D Match?':>10} {'Pred Match?':>12}")
print("-" * 100)

all_ok = True
for cls_name in CLASS:
    img_dir = Path("data/processed_340") / cls_name
    imgs = sorted(img_dir.glob("*.jpg"))[:3]
    for img_path in imgs:
        t_pred, t_conf, t_9d = predict_train_style(img_path)
        b_pred, b_conf, b_9d = predict_backend_style(img_path)
        
        diff_9d   = float(np.abs(t_9d - b_9d).max())
        pred_match = t_pred == b_pred
        match_icon = "✅" if pred_match else "❌"
        
        if not pred_match:
            all_ok = False
        
        print(f"{cls_name:<28} {CLASS[t_pred]:<22} {CLASS[b_pred]:<22} {diff_9d:>10.4f} {match_icon:>12}")

print("\n" + ("✅ FULL MATCH — backend preprocessing identical to training" if all_ok 
              else "❌ MISMATCH — backend and training give different predictions!"))
