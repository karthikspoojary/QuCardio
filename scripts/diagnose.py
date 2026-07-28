import numpy as np
import joblib
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("QuCardio Inference Diagnostic")
print("=" * 60)

# ── 1. Check features_9d.npz ──────────────────────────────────────────────────
print("\n[1] features_9d.npz")
try:
    d = np.load("data/features_9d.npz")
    X_train = d["train_features"]
    X_test  = d["test_features"]
    y_train = d["y_train"]
    y_test  = d["y_test"]
    print(f"    train_features: {X_train.shape}  min={X_train.min():.4f}  max={X_train.max():.4f}")
    print(f"    test_features:  {X_test.shape}  min={X_test.min():.4f}  max={X_test.max():.4f}")
    print(f"    y_train unique: {sorted(set(y_train.tolist()))}")
except Exception as e:
    print(f"    ERROR: {e}")

# ── 2. Check sv_train.npz ──────────────────────────────────────────────────────
print("\n[2] sv_train.npz")
try:
    sv = np.load("data/sv_train.npz", allow_pickle=True)
    sv_arr = sv["sv_train"]
    print(f"    sv_train: {sv_arr.shape}  dtype={sv_arr.dtype}")
except Exception as e:
    print(f"    ERROR: {e}")

# ── 3. Check resnet50_features.npz ────────────────────────────────────────────
print("\n[3] resnet50_features.npz")
try:
    r = np.load("data/resnet50_features.npz")
    tr = r["train_features"]
    te = r["test_features"]
    print(f"    train_features: {tr.shape}")
    print(f"    test_features:  {te.shape}")
except Exception as e:
    print(f"    ERROR: {e}")

# ── 4. Load SVD + Scaler and test on known 9D feature ──────────────────────────
print("\n[4] SVD + Scaler sanity check")
try:
    svd    = joblib.load("backend/models/svd_reducer.pkl")
    scaler = joblib.load("backend/models/minmax_scaler.pkl")
    svm    = joblib.load("backend/models/svm_model.pkl")

    # Take first train sample from features_9d (already reduced+scaled)
    x_known = X_train[0:1]  # shape (1, 9)
    pred = int(svm.predict(x_known)[0])
    prob = svm.predict_proba(x_known)[0]
    
    CLASS = ["Normal", "Arrhythmia", "Myocardial_Infarction", "History_of_MI"]
    print(f"    First train sample: {x_known}")
    print(f"    True label (y_train[0]): {CLASS[int(y_train[0])]}")
    print(f"    SVM predict: {CLASS[pred]}  (confidence={prob[pred]*100:.1f}%)")
    print(f"    All probs: { {CLASS[i]: round(float(prob[i]),3) for i in range(4)} }")
    
    if int(y_train[0]) == pred:
        print("    ✅ SVM correctly classifies first training sample")
    else:
        print("    ❌ SVM misclassifies even its own training sample → SVD/scaler mismatch")
except Exception as e:
    print(f"    ERROR: {e}")

# ── 5. Check actual image → pipeline manually ──────────────────────────────────
print("\n[5] Manual pipeline on a processed_340 image")
try:
    import cv2
    from pathlib import Path
    import tensorflow as tf
    from tensorflow.keras.applications import ResNet50
    from tensorflow.keras.applications.resnet50 import preprocess_input
    from tensorflow.keras.models import Model
    
    # Load ResNet
    base = ResNet50(weights="imagenet", include_top=False)
    pool1 = base.get_layer("pool1_pool").output
    resnet = Model(inputs=base.input, outputs=pool1)
    
    svd_r   = joblib.load("backend/models/svd_reducer.pkl")
    scaler_r = joblib.load("backend/models/minmax_scaler.pkl")
    svm_r   = joblib.load("backend/models/svm_model.pkl")
    
    CLASS = ["Normal", "Arrhythmia", "Myocardial_Infarction", "History_of_MI"]

    results = []
    for cls_name in CLASS:
        img_dir = Path("data/processed_340") / cls_name
        imgs = list(img_dir.glob("*.jpg"))[:1]
        if not imgs:
            imgs = list(img_dir.glob("*.png"))[:1]
        if not imgs:
            continue
        img_path = imgs[0]

        # Training-style load
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        img_f = img.astype("float32")  # [0,255] — exactly as training

        X = img_f[None, :, :, None]         # (1,340,340,1)
        X = np.repeat(X, 3, axis=-1)        # (1,340,340,3)
        X = preprocess_input(X)

        feat = resnet.predict(X, verbose=0).reshape(1, -1)
        reduced = svd_r.transform(feat)
        scaled  = scaler_r.transform(reduced)
        pred    = int(svm_r.predict(scaled)[0])
        prob    = float(svm_r.predict_proba(scaled)[0][pred])
        
        match = "✅" if pred == CLASS.index(cls_name) else "❌"
        print(f"    {match} {cls_name:<28} → predicted: {CLASS[pred]:<28} ({prob*100:.1f}%)")
        results.append(pred == CLASS.index(cls_name))

    acc = sum(results) / len(results) if results else 0
    print(f"\n    Mini-test accuracy: {sum(results)}/{len(results)} = {acc*100:.0f}%")
    if acc == 1.0:
        print("    ✅ Pipeline is working correctly on known images")
    else:
        print("    ❌ Pipeline fails on known images — SVD/scaler mismatch!")
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"    ERROR: {e}")

print("\n" + "=" * 60)
