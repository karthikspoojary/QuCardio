import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

def test_gatekeeper():
    model_path = 'models/ecg_detector.keras'
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return
        
    print("Loading Gatekeeper...")
    model = load_model(model_path)
    
    # Test on a real ECG
    ecg_img_path = 'data/ECG_DATA/train/Normal/Normal(1).jpg'
    
    # Test on a non-ECG (synthetic text)
    non_ecg_path = 'data/NON_ECG_DATA/non_ecg_text_001.jpg'
    
    for path, label in [(ecg_img_path, 'REAL ECG'), (non_ecg_path, 'NON-ECG (Text)')]:
        img = cv2.imread(path)
        if img is None:
            print(f"Could not read {path}")
            continue
            
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (224, 224))
        img_prep = tf.keras.applications.mobilenet_v2.preprocess_input(img_resized.astype(np.float32)[np.newaxis, ...])
        
        prob = model.predict(img_prep, verbose=0)[0][0]
        pred_label = "ECG" if prob >= 0.5 else "NON-ECG"
        print(f"Testing {label}: Prediction={pred_label} (Confidence={prob:.4f})")

if __name__ == '__main__':
    test_gatekeeper()
