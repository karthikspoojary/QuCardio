from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
import joblib
import base64
import os
import sys

import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import Model

# Add root directory to sys.path so we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing.preprocess_ecg import preprocess_ecg

app = FastAPI(title="QuCardio ML Backend", version="1.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Global model state ---
resnet_pool1 = None
svd_reducer = None
minmax_scaler = None
svm_model = None

CLASS_MAPPING = {0: 'Normal', 1: 'Arrhythmia', 2: 'Myocardial_Infarction', 3: 'History_of_MI'}

# Clinical descriptions for each class
CLASS_INFO = {
    'Normal': {
        'description': 'No significant abnormalities detected in this ECG.',
        'severity': 'normal',
        'action': 'Routine follow-up recommended.'
    },
    'Arrhythmia': {
        'description': 'Irregular heart rhythm pattern detected.',
        'severity': 'warning',
        'action': 'Consult a cardiologist for further evaluation.'
    },
    'Myocardial_Infarction': {
        'description': 'Patterns consistent with acute myocardial infarction.',
        'severity': 'critical',
        'action': 'Seek immediate emergency medical attention.'
    },
    'History_of_MI': {
        'description': 'ECG patterns suggest a prior myocardial infarction.',
        'severity': 'warning',
        'action': 'Cardiologist consultation and monitoring advised.'
    }
}


@app.on_event("startup")
async def load_models():
    global resnet_pool1, svd_reducer, minmax_scaler, svm_model
    print("Loading models...")
    try:
        base_model = ResNet50(weights='imagenet', include_top=False)
        try:
            pool1_layer = base_model.get_layer('pool1_pool').output
        except ValueError:
            pool1_layer = base_model.layers[4].output
        resnet_pool1 = Model(inputs=base_model.input, outputs=pool1_layer)

        model_dir = os.path.join(os.path.dirname(__file__), 'models')
        svd_reducer = joblib.load(os.path.join(model_dir, 'svd_reducer.pkl'))
        minmax_scaler = joblib.load(os.path.join(model_dir, 'minmax_scaler.pkl'))
        svm_model = joblib.load(os.path.join(model_dir, 'svm_model.pkl'))
        print("Models loaded successfully!")
    except Exception as e:
        print(f"Warning: Could not load models: {e}")


@app.get("/")
def read_root():
    return {
        "status": "ok",
        "message": "QuCardio API is running",
        "version": "1.1.0",
        "models_loaded": svm_model is not None
    }


@app.get("/health")
def health_check():
    return {
        "resnet_loaded": resnet_pool1 is not None,
        "svd_loaded": svd_reducer is not None,
        "scaler_loaded": minmax_scaler is not None,
        "svm_loaded": svm_model is not None,
    }


@app.post("/predict")
async def predict_ecg(file: UploadFile = File(...), model: str = "classical"):
    if not resnet_pool1 or not svm_model:
        raise HTTPException(
            status_code=503,
            detail="Models not loaded. Please run the training pipeline first."
        )

    try:
        # --- Read uploaded image ---
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file. Please upload a PNG or JPG.")

        # --- Preprocess ---
        temp_path = "temp_uploaded_ecg.png"
        cv2.imwrite(temp_path, img)
        preprocessed_normalized = preprocess_ecg(temp_path, output_size=(340, 340))
        os.remove(temp_path)

        # --- Encode preprocessed image as base64 for the frontend ---
        preprocessed_display = (preprocessed_normalized * 255).astype(np.uint8)
        # Apply CLAHE for better contrast in preview
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        preprocessed_display = clahe.apply(preprocessed_display)
        _, buffer = cv2.imencode('.png', preprocessed_display)
        preprocessed_b64 = base64.b64encode(buffer).decode('utf-8')

        # --- Feature extraction ---
        # Scale to [0, 255] for ResNet preprocess_input
        preprocessed_float = preprocessed_normalized * 255.0

        X = np.expand_dims(preprocessed_float, axis=0)         # (1, 340, 340)
        X = X[..., np.newaxis]                                  # (1, 340, 340, 1)
        X_rgb = np.repeat(X, 3, axis=-1)                        # (1, 340, 340, 3)
        X_prep = preprocess_input(X_rgb)

        features_3d = resnet_pool1.predict(X_prep, verbose=0)
        features_1d = features_3d.reshape(features_3d.shape[0], -1)

        # --- Dimensionality reduction ---
        reduced = svd_reducer.transform(features_1d)
        scaled = minmax_scaler.transform(reduced)

        # --- SVM Prediction ---
        prediction_idx = svm_model.predict(scaled)[0]
        probabilities = svm_model.predict_proba(scaled)[0]

        predicted_class = CLASS_MAPPING.get(prediction_idx, "Unknown")
        confidence = float(probabilities[prediction_idx])

        # Format probabilities
        prob_dict = {CLASS_MAPPING[i]: float(probabilities[i]) for i in range(len(probabilities))}

        # Confidence level flag
        if confidence >= 0.80:
            confidence_level = "high"
        elif confidence >= 0.55:
            confidence_level = "medium"
        else:
            confidence_level = "low"

        return {
            "prediction": predicted_class,
            "confidence": confidence,
            "confidence_level": confidence_level,
            "probabilities": prob_dict,
            "class_info": CLASS_INFO.get(predicted_class, {}),
            "model_used": "Classical SVM (RBF, C=500)" if model == "classical" else "Quantum SVC (Coming Soon)",
            "preprocessed_image": f"data:image/png;base64,{preprocessed_b64}"
        }

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
