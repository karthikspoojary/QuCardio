"""
QuCardio Backend — FastAPI
==========================
Inference pipeline:
  ECG image → OLD fixed-threshold pipeline [thresh=200, grid removal, JPEG round-trip, 340×340]
           → ResNet50 pool1_pool [462,400-D]
           → TruncatedSVD [9-D]
           → MinMax scale [0,1]
           → Classical SVM  (model=classical)
           → QSVC            (model=quantum)
           → Pegasos QSVC   (model=pegasos)

CRITICAL — Inference uses the OLD fixed-threshold pipeline (thresh=200, NOT OTSU) because
data/processed_340/ was built with that old pipeline. The current preprocess_ecg() in
src/preprocessing/preprocess_ecg.py uses OTSU adaptive thresholding, which produces
different pixel values → completely different 462K-D pool1_pool features → wrong SVD
projection → always predicts Arrhythmia regardless of input.

DO NOT replace the inline preprocessing below with preprocess_ecg() — it will break
all three models. To retrain on OTSU-preprocessed images, rebuild processed_340/ first,
then refit SVD + scaler + all three classifiers from scratch.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import numpy as np
import cv2
import joblib
import base64
import os
import sys
import time
import tempfile
import hashlib
from concurrent.futures import ThreadPoolExecutor
from typing import List
import json

try:
    import redis as redis_lib
    _REDIS_AVAILABLE = True
except ImportError:
    _REDIS_AVAILABLE = False

import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import Model

# ── Qiskit (for QSVC / Pegasos kernel computation at inference) ──────────────
from qiskit.circuit.library import ZZFeatureMap
from qiskit.quantum_info import Statevector

# Add project root so we can import config and src modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.preprocessing.preprocess_ecg import preprocess_ecg
from src.preprocessing.preprocess_inference import preprocess_for_inference
import backend.database as db
from backend.pdf_generator import generate_pdf_report

app = FastAPI(title="QuCardio ML Backend", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────────────────────
# Global model state
# ─────────────────────────────────────────────────────────────────────────────
resnet_pool1     = None   # ResNet50 up to pool1_pool
svd_reducer      = None   # TruncatedSVD(9)
minmax_scaler    = None   # MinMaxScaler fitted on SVD output
svm_model        = None   # Classical SVM (CalibratedClassifierCV)
qsvc_model       = None   # QSVC  (SVC precomputed kernel, tuned)
qsvc_feat_variant = "minmax_0pi"  # feature transform for QSVC: minmax_01|l2_norm|minmax_0pi
qsvc_reps        = 2              # ZZFeatureMap reps for QSVC
qsvc_entangle    = "circular"     # entanglement for QSVC
pegasos_models   = {}     # {(c1,c2): PegasosSVMKernel}
feature_map_qsvc = None   # ZZFeatureMap used for QSVC
feature_map_pegasos = None # ZZFeatureMap used for Pegasos
sv_train_qsvc    = None   # Cached training statevectors for QSVC
sv_train_pegasos = None   # Cached training statevectors for Pegasos
X_train_9d       = None   # 9-D scaled training features (needed for Pegasos kernel)
ecg_gatekeeper_model = None # MobileNetV2 Binary ECG Detector (OOD Gatekeeper)
redis_client     = None   # Optional Redis cache (None = disabled)
REDIS_CACHE_TTL  = 86400  # 24 hours
MAX_BATCH_FILES  = 20     # Max files per batch request
MAX_FILE_BYTES   = 20 * 1024 * 1024   # 20 MB hard limit per upload
MAX_IMAGE_PX     = 8000   # Reject images wider/taller than this (compression-bomb guard)
VALID_MODELS     = {"classical", "quantum", "pegasos"}

CLASS_PAIRS = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]

CLASS_MAPPING = {
    0: 'Normal',
    1: 'Arrhythmia',
    2: 'Myocardial_Infarction',
    3: 'History_of_MI',
}

CLASS_INFO = {
    'Normal': {
        'description': 'No significant abnormalities detected in this ECG.',
        'severity': 'normal',
        'action': 'Annual cardiac check-up recommended. Maintain a healthy lifestyle.',
        'recommendation': 'No abnormalities detected. Annual cardiac checkup recommended.',
    },
    'Arrhythmia': {
        'description': 'Irregular heart rhythm pattern detected in this ECG.',
        'severity': 'warning',
        'action': 'Consult a cardiologist within 7 days. 24-hour Holter monitoring may be advised.',
        'recommendation': 'Arrhythmia detected. Recommend 24-hour Holter monitoring and cardiology consultation within 7 days.',
    },
    'Myocardial_Infarction': {
        'description': 'ECG patterns consistent with acute myocardial infarction detected.',
        'severity': 'critical',
        'action': 'SEEK EMERGENCY CARE IMMEDIATELY. Call emergency services now.',
        'recommendation': 'ACUTE MYOCARDIAL INFARCTION DETECTED. SEEK EMERGENCY CARE IMMEDIATELY. Call 112 / 911.',
    },
    'History_of_MI': {
        'description': 'ECG patterns suggest a prior myocardial infarction event.',
        'severity': 'warning',
        'action': 'Follow up with cardiologist. Continue prescribed medications and monitoring.',
        'recommendation': 'History of MI pattern detected. Follow up with cardiologist. Continue prescribed medications.',
    },
}

MODEL_LABELS = {
    'classical': 'Classical SVM (RBF, C=10, balanced)',
    'quantum':   'QSVC (ZZFeatureMap 9-qubit, reps=2, circular, C=5.0 — 94.62%)',
    'pegasos':   'Pegasos QSVC (Custom SGD, per-model tuned — 91.94%)',
}

# ─────────────────────────────────────────────────────────────────────────────
# Startup — load every model once
# ─────────────────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def load_models():
    global resnet_pool1, svd_reducer, minmax_scaler
    global svm_model, qsvc_model, pegasos_models
    global feature_map_qsvc, feature_map_pegasos, sv_train_qsvc, sv_train_pegasos, X_train_9d
    global qsvc_feat_variant, qsvc_reps, qsvc_entangle, redis_client

    print("\n" + "="*55)
    print("QuCardio Backend v3.0 — Loading all models")
    print("="*55)
    
    print("[0/6] Initializing Audit Database...")
    db.init_db()

    # ── 0.5 Redis (optional) ──────────────────────────────
    if _REDIS_AVAILABLE:
        redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379")
        try:
            redis_client = redis_lib.from_url(redis_url, decode_responses=True, socket_connect_timeout=2)
            redis_client.ping()
            print(f"[0.5] Redis cache connected ✅  ({redis_url})")
        except Exception as _e:
            redis_client = None
            print(f"[0.5] Redis not available ({_e}) — caching disabled (OK for local dev)")
    else:
        print("[0.5] redis-py not installed — caching disabled")

    # ── 1. ResNet50 → pool1_pool ──────────────────────────────
    print("[1/6] Loading ResNet50 (pool1_pool)...")
    base_model = ResNet50(weights='imagenet', include_top=False)
    try:
        pool1_layer = base_model.get_layer('pool1_pool').output
    except ValueError:
        pool1_layer = base_model.layers[4].output
    resnet_pool1 = Model(inputs=base_model.input, outputs=pool1_layer)
    print(f"      Output shape: {resnet_pool1.output_shape}")

    model_dir = os.path.join(os.path.dirname(__file__), 'models')

    # ── 1.5 Gatekeeper Model (OOD) ───────────────────────────
    # train_ecg_detector.py saves to project-root models/ecg_detector.keras.
    # Check both locations so the backend works regardless of where it was saved.
    _project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _gk_candidates = [
        os.path.join(model_dir, 'ecg_detector.keras'),                  # backend/models/
        os.path.join(_project_root, 'models', 'ecg_detector.keras'),    # project root models/
    ]
    gatekeeper_path = next((p for p in _gk_candidates if os.path.exists(p)), None)
    if gatekeeper_path:
        print(f"[1.5/6] Loading ECG Gatekeeper ({os.path.relpath(gatekeeper_path)}) ...")
        from tensorflow.keras.models import load_model
        global ecg_gatekeeper_model
        ecg_gatekeeper_model = load_model(gatekeeper_path)
        print("      Gatekeeper loaded ✅")
    else:
        print("[1.5/6] ⚠️ ECG Gatekeeper not found — falling back to heuristic checks")
        print("        (searched: backend/models/ and project-root models/)")

    # ── 2. SVD + Scaler ──────────────────────────────────────
    print("[2/6] Loading SVD reducer + MinMax scaler...")
    svd_reducer   = joblib.load(os.path.join(model_dir, 'svd_reducer.pkl'))
    minmax_scaler = joblib.load(os.path.join(model_dir, 'minmax_scaler.pkl'))

    # ── 3. Classical SVM ─────────────────────────────────────
    print("[3/6] Loading Classical SVM...")
    svm_model = joblib.load(os.path.join(model_dir, 'svm_model.pkl'))
    print(f"      SVM loaded ✅  (CalibratedClassifierCV, RBF C=10)")

    # ── 4. QSVC + meta ───────────────────────────────────────
    print("[4/6] Loading QSVC (tuned — 94.62%)...")
    qsvc_model = joblib.load(os.path.join(model_dir, 'qsvc_model.pkl'))
    # Load tuning meta (feature variant, reps, entanglement) saved by tune_qsvc.py
    meta_path = os.path.join(model_dir, 'qsvc_meta.json')
    if os.path.exists(meta_path):
        import json as _json
        with open(meta_path) as _f:
            _meta = _json.load(_f)
        qsvc_feat_variant = _meta.get("feature_variant", "minmax_0pi")
        qsvc_reps         = int(_meta.get("reps", 2))
        qsvc_entangle     = _meta.get("entanglement", "circular")
        print(f"      QSVC loaded ✅  feat={qsvc_feat_variant}  reps={qsvc_reps}  entangle={qsvc_entangle}")
    else:
        qsvc_feat_variant = "minmax_0pi"
        qsvc_reps         = 2
        qsvc_entangle     = "circular"
        print(f"      QSVC loaded ✅  (using default: minmax_0pi, reps=2, circular)")

    # ── 5. Pegasos 6 binary models ───────────────────────────
    print("[5/6] Loading Pegasos QSVC (6 binary models)...")
    for c1, c2 in CLASS_PAIRS:
        path = os.path.join(model_dir, f'pegasos_{c1}_{c2}.pkl')
        if os.path.exists(path):
            pegasos_models[(c1, c2)] = joblib.load(path)
            print(f"      pegasos_{c1}_{c2} ✅")
        else:
            print(f"      ⚠️  pegasos_{c1}_{c2}.pkl NOT FOUND — Pegasos disabled")
    print(f"      {len(pegasos_models)}/6 Pegasos models loaded")

    # ── 6. ZZFeatureMap + cached training statevectors ────────
    print("[6/6] Loading ZZFeatureMaps + training statevectors...")
    feature_map_qsvc = ZZFeatureMap(          # Used for QSVC (tuned best config)
        feature_dimension=config.FEATURE_DIMENSION,
        reps=qsvc_reps,
        entanglement=qsvc_entangle
    )
    feature_map_pegasos = ZZFeatureMap(       # Used for Pegasos (linear, reps=2)
        feature_dimension=config.FEATURE_DIMENSION,
        reps=2,
        entanglement='linear'
    )
    print(f"      ZZFeatureMap (QSVC): reps={qsvc_reps} entanglement={qsvc_entangle}")
    print(f"      ZZFeatureMap (Peg): reps=2 entanglement=linear")

    # Load cached training statevectors for QSVC kernel computation
    sv_train_qsvc_path = config.FEATURES_DIR / 'sv_train_qsvc.npz'
    if sv_train_qsvc_path.exists():
        sv_train_qsvc = np.load(sv_train_qsvc_path, allow_pickle=True)['sv_train']
        print(f"      sv_train_qsvc loaded ✅")
    else:
        print(f"      ⚠️  sv_train_qsvc.npz not found")
        
    sv_train_pegasos_path = config.FEATURES_DIR / 'sv_train_pegasos.npz'
    if sv_train_pegasos_path.exists():
        sv_train_pegasos = np.load(sv_train_pegasos_path, allow_pickle=True)['sv_train']
        print(f"      sv_train_pegasos loaded ✅")
    else:
        print(f"      ⚠️  sv_train_pegasos.npz not found")

    # Load training 9D features (needed to slice kernel rows for Pegasos)
    features_9d_path = config.FEATURES_DIR / 'features_9d.npz'
    if features_9d_path.exists():
        feat_data = np.load(features_9d_path)
        X_train_9d = feat_data['train_features']    # already MinMax-scaled
        print(f"      X_train_9d loaded ✅  shape={X_train_9d.shape}")
    else:
        print(f"      ⚠️  features_9d.npz not found — Pegasos inference disabled")

    print("\n✅ All models ready!")
    print("="*55 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Quantum helpers
# ─────────────────────────────────────────────────────────────────────────────

def compute_single_statevector(x_9d, fmap):
    """Compute ZZFeatureMap statevector for one 9-D sample. Returns (512,) complex."""
    bound = fmap.assign_parameters(x_9d)
    return Statevector(bound).data  # shape (512,)


def quantum_kernel_row(x_9d_single, sv_train_matrix, fmap):
    """
    K(x, x_train_i) = |<ψ(x)|ψ(x_i)>|²  for all training samples.
    x_9d_single     : (9,)      — one test sample (already MinMax-scaled)
    sv_train_matrix : (N, 512)  — cached training statevectors
    Returns         : (N,)      — kernel row
    """
    sv_x = compute_single_statevector(x_9d_single, fmap)  # (512,)
    return (np.abs(sv_train_matrix.conj() @ sv_x) ** 2).real  # (N,)


def pegasos_predict_node(k_row_full, c1, c2):
    """
    Run one Pegasos binary model and return the predicted original class label.
    k_row_full : (N_train,)  full kernel row for test sample vs all training
    c1, c2     : class indices for the binary model
    Returns    : c1 or c2
    """
    m   = pegasos_models[(c1, c2)]
    idx = m.train_indices_
    k_sub = k_row_full[idx].reshape(1, -1)     # (1, |binary_train|)
    pred  = m.predict(k_sub)[0]                # -1 or +1
    return c1 if pred == -1 else c2


def pegasos_multiclass_predict(k_row_full):
    """
    Algorithm 1 from paper — 3-model decision tree for 4-class Pegasos.
    k_row_full : (N_train,) kernel row
    Returns    : predicted class index (0-3)
    """
    pred_01 = pegasos_predict_node(k_row_full, 0, 1)
    pred_23 = pegasos_predict_node(k_row_full, 2, 3)

    if pred_01 == 0:
        return pegasos_predict_node(k_row_full, 0, 2) if pred_23 == 2 \
               else pegasos_predict_node(k_row_full, 0, 3)
    else:
        return pegasos_predict_node(k_row_full, 1, 2) if pred_23 == 2 \
               else pegasos_predict_node(k_row_full, 1, 3)


# ─────────────────────────────────────────────────────────────────────────────
# Health / root endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/")
def read_root():
    return {
        "status":           "ok",
        "message":          "QuCardio API v3.0 is running",
        "version":          "3.0.0",
        "models_loaded": {
            "svm":         svm_model     is not None,
            "qsvc":        qsvc_model    is not None,
            "pegasos":     len(pegasos_models) == 6,
            "sv_train_qsvc":    sv_train_qsvc      is not None,
            "sv_train_pegasos": sv_train_pegasos   is not None,
            "X_train_9d":  X_train_9d    is not None,
        },
    }


@app.get("/health")
def health_check():
    return {
        "resnet_loaded":     resnet_pool1   is not None,
        "svd_loaded":        svd_reducer    is not None,
        "scaler_loaded":     minmax_scaler  is not None,
        "svm_loaded":        svm_model      is not None,
        "qsvc_loaded":       qsvc_model     is not None,
        "pegasos_models":    len(pegasos_models),
        "sv_train_qsvc_loaded":   sv_train_qsvc       is not None,
        "sv_train_pegasos_loaded":sv_train_pegasos    is not None,
        "X_train_9d_loaded": X_train_9d     is not None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Predict endpoint
# ─────────────────────────────────────────────────────────────────────────────

def is_valid_ecg_image(img_gray: np.ndarray, img_bgr: np.ndarray | None = None) -> tuple[bool, str]:
    """
    Heuristic ECG validator.
    Returns (is_valid, reason_if_not).

    ECG images have five key structural properties:
      - Landscape aspect ratio (wider than tall — standard ECG paper format)
      - Near-pure white paper background (>85% bright pixels, very low color saturation)
      - Dark waveform lines spanning ≥65% of rows and ≥85% of columns (full-area traces)
      - Thin oscillating lines with HIGH oscillation frequency per column (≥500 mean transitions)
      - High-frequency energy in the 2-D FFT (waveform = distributed HF content)
    Screenshots fail on color-saturation, coverage, and/or oscillation-intensity checks.
    """
    h, w = img_gray.shape

    # ── 1. Minimum resolution ──────────────────────────────────────────────────
    if h < 100 or w < 100:
        return False, "Image too small (min 100×100 required)."

    # ── 2. Aspect ratio: ECG recordings are landscape (wider than tall) ────────
    # Standard 12-lead ECG paper is always wider than tall.
    # Pure screenshots / portrait photos fail this.
    aspect = w / h
    if aspect < 1.1:
        return False, (
            f"Image aspect ratio {aspect:.2f} is too square or portrait "
            f"(need width ≥ 1.1× height). ECG recordings are always landscape-oriented."
        )

    # ── 3. Color saturation check — ECG paper has no vivid colors ─────────────
    # Webpage screenshots, UI mockups, and colored charts contain vivid colored pixels
    # (buttons, headers, highlighted text). ECG paper has almost zero saturation.
    # ECG dataset: max 2.9% of pixels with HSV saturation > 50.
    # Screenshots with colored UI elements: typically 10–40% saturated pixels.
    if img_bgr is not None:
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        sat_ratio = float(np.sum(hsv[:, :, 1] > 50) / hsv[:, :, 1].size)
        if sat_ratio > 0.08:   # ECG max observed: 0.029; allow 8% for safety margin
            return False, (
                f"Image contains {sat_ratio*100:.1f}% vivid-color pixels "
                f"(ECG paper has <3%). Colored UI elements, charts, or screenshots detected."
            )

    # ── 4. Mostly light background ─────────────────────────────────────────────
    avg = img_gray.mean()
    if avg < 100:
        return False, "Image too dark — ECG images must have a light paper background."

    # ── 5. At least 85% bright-white pixels (the paper) ───────────────────────
    # ECG dataset: light_ratio (pixels > 180) ranges 0.936–0.967 across all 928 images.
    # Webpage screenshots with text/UI elements typically score 0.50–0.80.
    light_ratio = np.sum(img_gray > 180) / img_gray.size
    if light_ratio < 0.85:
        return False, (
            f"Only {light_ratio*100:.1f}% of pixels are bright-white "
            f"(need ≥85%). ECG paper background is almost entirely white. "
            f"Image likely contains colored headers, sidebars, or non-ECG content."
        )

    # ── 6. Dark pixel count within reasonable bounds ───────────────────────────
    dark_mask  = img_gray < 80
    dark_ratio = dark_mask.sum() / img_gray.size
    if dark_ratio < 0.005:
        return False, "No signal lines detected — image appears blank or invalid."
    if dark_ratio > 0.3:
        return False, "Image has too many dark areas to be an ECG."

    # ── 7. STRUCTURAL: dark pixels must span across many ROWS ─────────────────
    # ECG traces (even multi-lead) cross every row of the image.
    # Text / screenshots have dark pixels only in a narrow horizontal band
    # (where the glyphs sit) — leaving large blank regions above/below.
    rows_with_dark = np.any(dark_mask, axis=1).sum()   # rows that contain ≥1 dark px
    row_coverage   = rows_with_dark / h
    if row_coverage < 0.65:
        return False, (
            f"Dark signal lines only cover {row_coverage*100:.0f}% of image rows "
            f"(need ≥65%). ECG recordings have continuous waveforms in every row. "
            f"Image appears to have large blank/whitespace regions (text document or form)."
        )

    # ── 8. STRUCTURAL: dark pixels must span across many COLUMNS ───────────────
    # ECG waveforms span the full width; isolated labels / QR codes do not.
    cols_with_dark = np.any(dark_mask, axis=0).sum()
    col_coverage   = cols_with_dark / w
    if col_coverage < 0.85:
        return False, (
            f"Dark signal lines only cover {col_coverage*100:.0f}% of image columns "
            f"(need ≥85%). ECG waveforms span the full recording width. "
            f"Image appears to have content only in a portion of the frame."
        )

    # ── 9. COLUMN-PROFILE OSCILLATION — presence + intensity ──────────────────
    # ECG traces are thin jagged lines: each column of the image alternates rapidly
    # between dark (signal) and light (background) pixels as you scan downward.
    # Text glyphs are wide blobs with few transitions per column.
    #
    # Two-part check:
    #   a) At least 30% of columns must have ≥3 transitions  (presence check)
    #   b) Among those oscillating columns, the MEAN transition count must be ≥500.
    #      ECG dataset minimum observed: 4882 transitions/col (mean ~5961).
    #      Text / form pages at full resolution can reach 2000–8000 but are already
    #      rejected by the col_coverage ≥ 85% check above; the 500 floor handles
    #      genuine low-resolution or single-lead ECG scans.
    binary_dark = (img_gray < 100).astype(np.uint8)          # 1=dark, 0=light
    col_transitions = np.diff(binary_dark, axis=0)            # shape (h-1, w)
    transitions_per_col = np.abs(col_transitions).sum(axis=0) # (w,) — transitions per column

    osc_mask      = transitions_per_col >= 3
    osc_cols      = int(osc_mask.sum())
    osc_col_ratio = osc_cols / w
    if osc_col_ratio < 0.30:                                  # need 30% of columns oscillating
        return False, (
            f"Only {osc_col_ratio*100:.0f}% of image columns show ECG-like oscillation "
            f"(need ≥30%). Image appears to be text, a screenshot, or a document."
        )

    # Intensity check: mean transitions among oscillating columns must be sufficiently high.
    # We require a minimum of 500 transitions for high-res scans, OR at least 20% of the 
    # image height for lower-res scans (e.g. 340x340 images can only mathematically have 340 transitions).
    min_required_intensity = min(500.0, h * 0.20)
    mean_osc_intensity = float(transitions_per_col[osc_mask].mean())
    if mean_osc_intensity < min_required_intensity:
        return False, (
            f"Oscillating columns average only {mean_osc_intensity:.1f} transitions "
            f"(need ≥{min_required_intensity:.1f}). Signal oscillation frequency is far too low for an ECG waveform — "
            f"image appears to be text, a document, or a non-ECG graphic."
        )

    # ── 10. THIN-LINE CHECK: ECG signal lines are narrow (1–4 px thick) ───────
    # For each dark column, compute the mean run-length of consecutive dark pixels.
    # ECG lines: short runs (1–6 px). Text characters: long runs (8–30+ px).
    dark_col_mask = np.any(binary_dark, axis=0)               # columns with any dark pixel
    if dark_col_mask.sum() > 0:
        run_lengths = []
        for c in np.where(dark_col_mask)[0][::max(1, w // 60)]:  # sample ~60 columns
            col = binary_dark[:, c]
            runs = []
            run = 0
            for px in col:
                if px:
                    run += 1
                else:
                    if run > 0:
                        runs.append(run)
                        run = 0
            if run > 0:
                runs.append(run)
            if runs:
                run_lengths.extend(runs)
        if run_lengths:
            mean_run = float(np.mean(run_lengths))
            if mean_run > 18:
                return False, (
                    f"Dark pixel runs average {mean_run:.1f}px thick "
                    f"(ECG signal lines are thin, ≤18px). "
                    f"Image appears to contain text or large graphical elements."
                )

    # ── 11. FFT HIGH-FREQUENCY ENERGY CHECK ───────────────────────────────────
    # ECG waveforms are thin, oscillating signals distributed across the whole image →
    # they contribute strong energy to the HIGH-frequency bands of the 2-D FFT.
    # Screenshots / text documents are dominated by LOW-frequency content
    # (large uniform regions, thick letter strokes, wide margins).
    #
    # Method: compute the 2-D FFT magnitude spectrum, split into a central
    # low-frequency disc (radius = min(h,w)//4) and everything outside it.
    # HF ratio = energy_outside_disc / total_energy.
    # ECG images: HF ratio ≈ 0.55–0.90  (waveform = many small oscillations)
    # Screenshots: HF ratio ≈ 0.20–0.50  (dominated by large smooth regions)
    fft_mag   = np.abs(np.fft.fftshift(np.fft.fft2(img_gray.astype(np.float32))))
    cy, cx    = h // 2, w // 2
    radius    = min(h, w) // 4
    ys, xs    = np.ogrid[:h, :w]
    lf_disc   = (ys - cy) ** 2 + (xs - cx) ** 2 <= radius ** 2  # True = low-freq centre
    total_energy = fft_mag.sum()
    if total_energy > 0:
        hf_ratio = fft_mag[~lf_disc].sum() / total_energy
        if hf_ratio < 0.50:
            return False, (
                f"FFT high-frequency energy ratio is {hf_ratio:.2f} "
                f"(need ≥0.50). Image is dominated by low-frequency content — "
                f"not consistent with ECG waveform structure."
            )

    return True, "OK"

@app.post("/predict")
async def predict_ecg(model: str = "classical", file: UploadFile = File(...)):
    """
    model param : "classical" | "quantum" | "pegasos"
    """
    if resnet_pool1 is None or svd_reducer is None:
        raise HTTPException(
            status_code=503,
            detail="Models not loaded. Start the server and wait for startup to complete.",
        )

    try:
        # ── Validate model_key early — return 400 for unknown models ──────────
        model_key = model.lower().strip()
        if model_key not in VALID_MODELS:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown model '{model}'. Valid options: {sorted(VALID_MODELS)}"
            )

        # ── Read uploaded bytes with size guard (compression-bomb protection) ──
        contents = await file.read(MAX_FILE_BYTES + 1)
        if len(contents) > MAX_FILE_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"File too large (>{MAX_FILE_BYTES // (1024*1024)} MB). Upload a smaller ECG image."
            )
        img_hash  = hashlib.sha256(contents).hexdigest()

        # ── Redis cache check ─────────────────────────────────────────────────
        cache_key = f"qucardio:{model_key}:{img_hash}"
        if redis_client is not None:
            _cached = redis_client.get(cache_key)
            if _cached:
                cached_result = json.loads(_cached)
                # Log cache hits to audit DB so every prediction is recorded
                try:
                    db.log_prediction(
                        patient_id=getattr(file, "filename", "unknown"),
                        image_hash=img_hash,
                        model_used=MODEL_LABELS.get(model_key, model_key),
                        predicted_class=cached_result.get("prediction", "Unknown"),
                        confidence=cached_result.get("confidence", 0.0),
                        probabilities=cached_result.get("probabilities", {}),
                    )
                except Exception:
                    pass  # Never let audit DB failure break the cache hit
                print(f"  [predict] CACHE HIT model={model_key}")
                return cached_result

        # ── Decode image with dimension guard (compression-bomb protection) ────
        nparr     = np.frombuffer(contents, np.uint8)
        img_gray  = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        img_color = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img_gray is None:
            raise HTTPException(status_code=400, detail="Invalid image file format.")

        h, w = img_gray.shape
        if h > MAX_IMAGE_PX or w > MAX_IMAGE_PX:
            raise HTTPException(
                status_code=413,
                detail=f"Image dimensions {w}×{h} exceed maximum {MAX_IMAGE_PX}px. Please resize before uploading."
            )

        # ── Step 0: Validate if it's an ECG ──
        if ecg_gatekeeper_model is not None:
            # Use ML Gatekeeper
            img_rgb = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, (224, 224))
            img_prep = tf.keras.applications.mobilenet_v2.preprocess_input(img_resized.astype(np.float32)[np.newaxis, ...])
            prob_ecg = float(ecg_gatekeeper_model.predict(img_prep, verbose=0)[0][0])
            if prob_ecg < 0.5:
                raise HTTPException(status_code=422, detail=f"Image rejected by AI Gatekeeper (Not an ECG). Confidence: {(1-prob_ecg)*100:.1f}%")
        else:
            # Fallback to heuristics
            is_ecg, reason = is_valid_ecg_image(img_gray, img_bgr=img_color)
            if not is_ecg:
                raise HTTPException(status_code=422, detail=f"Invalid ECG image: {reason}")

        # Need BGR array for preprocess_for_inference (it expects BGR input)
        img_bgr = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

        # ── Step 1: Preprocessing (old fixed-threshold pipeline) ──────────────
        # Delegates to src/preprocessing/preprocess_inference.py which contains
        # the full pipeline documentation and is shared with all cross-dataset
        # evaluation scripts (ST-5B, 5C, 5D, 5E).
        t_pre = time.perf_counter()
        img_for_resnet = preprocess_for_inference(img_bgr)  # float32 [0–255] (340,340)

        display_uint8 = img_for_resnet.astype(np.uint8)
        _, buf = cv2.imencode('.png', display_uint8)
        preprocessed_b64 = base64.b64encode(buf).decode('utf-8')

        t_pre_ms = (time.perf_counter() - t_pre) * 1000

        # ── Step 2: ResNet50 pool1_pool feature extraction ────────────────────
        # Matches extract_resnet_features.py exactly:
        #   img = cv2.imread(processed_340/cls/x.jpg, IMREAD_GRAYSCALE) → float32 [0–255]
        #   batch_rgb = np.repeat(batch[...,newaxis], 3, axis=-1)
        #   batch_prep = preprocess_input(batch_rgb)
        t_feat = time.perf_counter()
        X        = np.expand_dims(img_for_resnet, axis=0)          # (1, 340, 340)
        X        = X[..., np.newaxis]                               # (1, 340, 340, 1)
        X_rgb    = np.repeat(X, 3, axis=-1)                         # (1, 340, 340, 3)
        X_prep   = preprocess_input(X_rgb)
        feat_3d  = resnet_pool1.predict(X_prep, verbose=0)          # (1, 85, 85, 64)
        feat_1d  = feat_3d.reshape(1, -1)                           # (1, 462400)
        t_feat_ms = (time.perf_counter() - t_feat) * 1000

        # ── Step 3: SVD + MinMax ──────────────────────────────────────────────
        t_svd = time.perf_counter()
        reduced = svd_reducer.transform(feat_1d)                   # (1, 9)
        scaled  = minmax_scaler.transform(reduced)                 # (1, 9) in [0,1]
        t_svd_ms = (time.perf_counter() - t_svd) * 1000

        # ── Step 4: Classification ────────────────────────────────────────────
        t_clf = time.perf_counter()

        if model_key == 'quantum':
            # ── QSVC: apply feature transform → compute kernel row → SVC.predict ──
            if qsvc_model is None:
                raise HTTPException(status_code=503, detail="QSVC model not loaded.")
            if sv_train_qsvc is None:
                raise HTTPException(status_code=503,
                    detail="Training statevectors (sv_train_qsvc.npz) not found.")

            # Apply the same feature transform used during QSVC training
            # Best config: minmax_0pi → multiply MinMax-scaled [0,1] features by π
            # This maps feature values to rotation angles [0, π] — optimal for ZZFeatureMap
            if qsvc_feat_variant == "minmax_0pi":
                x_qsvc = scaled[0] * np.pi
            elif qsvc_feat_variant == "l2_norm":
                from sklearn.preprocessing import normalize as _norm
                x_qsvc = _norm(scaled, norm="l2")[0]
            else:  # minmax_01 — use as-is
                x_qsvc = scaled[0]

            # K_test_row shape: (1, N_train) — compare one test sample vs all train
            k_row     = quantum_kernel_row(x_qsvc, sv_train_qsvc, feature_map_qsvc)  # (N_train,)
            K_row_2d  = k_row.reshape(1, -1).astype(np.float32)    # (1, N_train)
            prediction_idx = int(qsvc_model.predict(K_row_2d)[0])

            # QSVC decision function returns a (4,) OvR vector by default in sklearn
            decision = qsvc_model.decision_function(K_row_2d)[0]  # (4,)
            # Temperature-scaled softmax (T=4.0):
            # Quantum kernel decision values are compressed into a small range (~0–0.3).
            # Raw softmax produces near-uniform probabilities (low confidence) even when
            # the prediction is correct.  Multiplying by T before softmax sharpens the
            # distribution — same argmax (prediction unchanged), more meaningful confidence.
            T = 4.0
            scaled_d = decision * T
            exp_d = np.exp(scaled_d - scaled_d.max())
            probabilities = (exp_d / exp_d.sum()).tolist()

        elif model_key == 'pegasos':
            # ── Pegasos: kernel row → Algorithm 1 decision tree ──────────────
            if len(pegasos_models) < 6:
                raise HTTPException(status_code=503,
                    detail="Pegasos models not fully loaded (need 6 binary models).")
            if sv_train_pegasos is None:
                raise HTTPException(status_code=503,
                    detail="Training statevectors (sv_train_pegasos.npz) not found.")

            k_row          = quantum_kernel_row(scaled[0], sv_train_pegasos, feature_map_pegasos)  # (N_train,)
            prediction_idx = int(pegasos_multiclass_predict(k_row))

            # Pegasos has no probability output — aggregate decision scores from the
            # 6 binary models into a (4,) class score vector, then sharpen.
            scores = np.zeros(4, dtype=np.float64)
            for (c1, c2), m in pegasos_models.items():
                idx   = m.train_indices_
                k_sub = k_row[idx].reshape(1, -1)
                d     = float(m.decision_function(k_sub)[0])
                scores[c2] += d       # positive decision → votes for c2
                scores[c1] -= d       # negative decision → votes for c1
            # Shift to non-negative, then square to amplify the winning class
            # (squaring preserves ranking while widening the gap between top and rest).
            scores -= scores.min()
            scores = scores ** 2
            scores += 1e-6
            probabilities = (scores / scores.sum()).tolist()

        else:
            # ── Classical SVM (default) ───────────────────────────────────────
            if svm_model is None:
                raise HTTPException(status_code=503, detail="SVM model not loaded.")
            prediction_idx = int(svm_model.predict(scaled)[0])
            probabilities  = svm_model.predict_proba(scaled)[0].tolist()

        t_clf_ms = (time.perf_counter() - t_clf) * 1000
        total_ms = t_pre_ms + t_feat_ms + t_svd_ms + t_clf_ms

        # ── Format response ───────────────────────────────────────────────────
        predicted_class = CLASS_MAPPING.get(prediction_idx, "Unknown")
        confidence      = float(probabilities[prediction_idx])
        prob_dict       = {CLASS_MAPPING[i]: float(probabilities[i])
                           for i in range(len(probabilities))}

        if   confidence >= 0.80: confidence_level = "high"
        elif confidence >= 0.55: confidence_level = "medium"
        else:                    confidence_level = "low"
        
        # Log prediction to audit database
        log_id = db.log_prediction(
            patient_id=getattr(file, "filename", "unknown"),
            image_hash=img_hash,
            model_used=MODEL_LABELS.get(model_key, model_key),
            predicted_class=predicted_class,
            confidence=confidence,
            probabilities=prob_dict
        )

        # Debug print so terminal shows what's happening
        print(f"  [predict] model={model_key}  →  {predicted_class}  "
              f"({confidence*100:.1f}%)  total={total_ms:.0f}ms  log_id={log_id}")

        response = {
            "prediction":         predicted_class,
            "confidence":         confidence,
            "confidence_level":   confidence_level,
            "probabilities":      prob_dict,
            "class_info":         CLASS_INFO.get(predicted_class, {}),
            "model_used":         MODEL_LABELS.get(model_key, model_key),
            "preprocessed_image": f"data:image/png;base64,{preprocessed_b64}",
            "svd_features":       scaled[0].tolist(),
            "processing_time": {
                "preprocessing_ms":      round(t_pre_ms,  1),
                "feature_extraction_ms": round(t_feat_ms, 1),
                "svd_reduction_ms":      round(t_svd_ms,  1),
                "classification_ms":     round(t_clf_ms,  1),
                "total_ms":              round(total_ms,  1),
            },
            "cache_hit": False,
        }

        # Store in Redis — exclude the large preprocessed_image to keep memory bounded
        if redis_client is not None:
            try:
                cacheable = {k: v for k, v in response.items() if k != "preprocessed_image"}
                cacheable["cache_hit"] = True
                # preprocessed_image is NOT included — cache hits get None for this field
                # (the frontend handles missing preprocessed_image gracefully)
                redis_client.setex(cache_key, REDIS_CACHE_TTL, json.dumps(cacheable))
            except Exception as _ce:
                print(f"  [cache] write failed: {_ce}")

        return response

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch")
async def predict_batch(model: str = "classical", files: List[UploadFile] = File(...)):
    """
    Batch processing endpoint. Max 20 files per request.
    Files are processed sequentially to avoid exhausting RAM with many heavy models.
    """
    if len(files) > MAX_BATCH_FILES:
        raise HTTPException(
            status_code=400,
            detail=f"Too many files: {len(files)}. Maximum is {MAX_BATCH_FILES} per batch."
        )

    results = []
    for file in files:
        try:
            result = await predict_ecg(model=model, file=file)
            results.append({"filename": file.filename, "status": "success", "result": result})
        except HTTPException as e:
            results.append({"filename": file.filename, "status": "error", "detail": e.detail})
        except Exception as e:
            results.append({"filename": file.filename, "status": "error", "detail": str(e)})

    return {"batch_results": results}


@app.post("/predict/pdf")
async def predict_pdf(
    background_tasks: BackgroundTasks,
    model: str = "classical",
    file: UploadFile = File(...)
):
    """
    Generate a 1-page PDF clinical report for a single ECG.
    The temporary PDF is deleted from disk after the response is sent.
    """
    try:
        result = await predict_ecg(model=model, file=file)
        pdf_path = generate_pdf_report(result, filename=file.filename or "report")
        # Schedule cleanup so /tmp/qucardio_reports doesn't fill the disk
        background_tasks.add_task(os.remove, pdf_path)
        return FileResponse(
            path=pdf_path,
            media_type="application/pdf",
            filename=f"QuCardio_Report_{file.filename}.pdf"
        )
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
