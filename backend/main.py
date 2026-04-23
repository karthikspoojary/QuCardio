from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
import joblib
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import Model
import io
import sys
import os

# Add root directory to sys.path so we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing.preprocess_ecg import preprocess_ecg

app = FastAPI(title="QuCardio ML Backend")

# Allow CORS for the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Global variables to hold loaded models
resnet_pool1 = None
svd_reducer = None
minmax_scaler = None
svm_model = None

CLASS_NAMES = ['Normal', 'Arrhythmia', 'Myocardial_Infarction', 'History_of_MI']
CLASS_MAPPING = {0: 'Normal', 1: 'Arrhythmia', 2: 'Myocardial_Infarction', 3: 'History_of_MI'}

@app.on_event("startup")
async def load_models():
    """Load models on startup so inference is fast"""
    global resnet_pool1, svd_reducer, minmax_scaler, svm_model
    print("Loading models...")
    try:
        # 1. Load ResNet50
        base_model = ResNet50(weights='imagenet', include_top=False)
        try:
            pool1_layer = base_model.get_layer('pool1_pool').output
        except ValueError:
            pool1_layer = base_model.layers[4].output
        resnet_pool1 = Model(inputs=base_model.input, outputs=pool1_layer)
        
        # 2. Load Scikit-Learn Models
        model_dir = os.path.join(os.path.dirname(__file__), 'models')
        svd_reducer = joblib.load(os.path.join(model_dir, 'svd_reducer.pkl'))
        minmax_scaler = joblib.load(os.path.join(model_dir, 'minmax_scaler.pkl'))
        svm_model = joblib.load(os.path.join(model_dir, 'svm_model.pkl'))
        print("Models loaded successfully!")
    except Exception as e:
        print(f"Warning: Could not load some models: {e}. Make sure to train the pipeline first.")

@app.get("/")
def read_root():
    return {"status": "ok", "message": "QuCardio API is running"}

@app.post("/predict")
async def predict_ecg(file: UploadFile = File(...)):
    if not resnet_pool1 or not svm_model:
        raise HTTPException(status_code=500, detail="Models are not loaded. Did you run the training scripts?")
        
    try:
        # Read the uploaded file
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
            
        # 1. Preprocess
        # Save temp file for the preprocessing function since it reads from path
        temp_path = "temp_uploaded.png"
        cv2.imwrite(temp_path, img)
        preprocessed = preprocess_ecg(temp_path, output_size=(340, 340))
        
        # Scale back to [0, 255] for ResNet preprocess_input
        preprocessed = preprocessed * 255.0
        
        os.remove(temp_path)
        
        # 2. ResNet50 Feature Extraction
        # Add batch dimension
        X = np.expand_dims(preprocessed, axis=0) # (1, 340, 340)
        
        # Add channel dimension
        X_resized = X[..., np.newaxis] # (1, 340, 340, 1)
        
        # Convert to RGB
        X_rgb = np.repeat(X_resized, 3, axis=-1) # (1, 340, 340, 3)
        
        # Preprocess for ImageNet
        X_prep = preprocess_input(X_rgb)
        
        # Extract pool1_pool
        features_3d = resnet_pool1.predict(X_prep, verbose=0)
        
        # Flatten
        features_1d = features_3d.reshape(features_3d.shape[0], -1)
        
        # 3. SVD Reduction
        reduced = svd_reducer.transform(features_1d)
        
        # 4. Scale
        scaled = minmax_scaler.transform(reduced)
        
        # 5. Predict with SVM
        prediction = svm_model.predict(scaled)[0]
        probabilities = svm_model.predict_proba(scaled)[0]
        
        predicted_class = CLASS_MAPPING.get(prediction, "Unknown")
        confidence = probabilities[prediction]
        
        # Return all probabilities nicely formatted
        prob_dict = {CLASS_MAPPING[i]: float(probabilities[i]) for i in range(len(probabilities))}
        
        return {
            "prediction": predicted_class,
            "confidence": float(confidence),
            "probabilities": prob_dict
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
