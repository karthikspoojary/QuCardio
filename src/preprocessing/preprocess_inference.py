"""
preprocess_for_inference — ECG Preprocessing Pipeline (Fixed-Threshold)
========================================================================
This module replicates the preprocessing pipeline used when building
data/processed_340/ (the training data for all three QuCardio models).

The training data was preprocessed with:
  threshold=200, bitwise_not, 1×50 vertical grid removal, resize 340×340,
  save as JPEG then reload as grayscale.

This pipeline is kept frozen so that inference features align exactly with
training features.  The SVD reducer and scaler were both fitted on features
produced by this pipeline and must not change.

USED BY
-------
  - backend/main.py  /predict endpoint
  - src/analysis/cross_dataset_ecgdata.py
  - src/analysis/cross_dataset_jain_d2.py
  - src/analysis/augmentation_robustness.py
"""

import os
import tempfile
import numpy as np
import cv2


def preprocess_for_inference(source, output_size=(340, 340)) -> np.ndarray:
    """
    Apply the fixed-threshold preprocessing pipeline to an ECG image.

    Parameters
    ----------
    source : str | Path | np.ndarray
        Either a file path (str/Path) or a pre-loaded BGR uint8 numpy array.
    output_size : tuple
        Output (width, height). Default (340, 340).

    Returns
    -------
    np.ndarray
        float32 array, shape output_size, values in [0, 255].
        This matches the format of images in data/processed_340/ exactly.
        Pass to ResNet50 as-is (preprocess_input is applied separately).
    """
    # ── Load ──────────────────────────────────────────────────────────────────
    if isinstance(source, (str, os.PathLike)):
        bgr = cv2.imread(str(source))
        if bgr is None:
            raise ValueError(f"Cannot read image: {source}")
    else:
        bgr = source  # assume pre-loaded BGR uint8 array

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # ── Step 1: ROI crop (fixed threshold 200) ────────────────────────────────
    _, binary_roi = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    contours, _   = cv2.findContours(binary_roi, cv2.RETR_EXTERNAL,
                                     cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest  = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        pad = 10
        x = max(0, x - pad)
        y = max(0, y - pad)
        w = min(gray.shape[1] - x, w + 2 * pad)
        h = min(gray.shape[0] - y, h + 2 * pad)
        cropped = gray[y:y + h, x:x + w]
    else:
        cropped = gray

    # ── Step 2: Background removal (thresh 200 → bitwise_not) ────────────────
    _, cleaned = cv2.threshold(cropped, 200, 255, cv2.THRESH_BINARY)
    cleaned    = cv2.bitwise_not(cleaned)

    # ── Step 3: Remove vertical grid lines (1×50 morphological open) ─────────
    kernel_v   = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 50))
    vert_lines = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel_v)
    cleaned    = cv2.subtract(cleaned, vert_lines)

    # ── Step 4: Resize ────────────────────────────────────────────────────────
    resized = cv2.resize(cleaned, output_size, interpolation=cv2.INTER_AREA)

    # ── Step 5: JPEG round-trip ───────────────────────────────────────────────
    # Training saved each processed image as .jpg then reloaded via IMREAD_GRAYSCALE,
    # introducing mild JPEG compression artefacts. We replicate this exactly.
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as jtmp:
        jpeg_path = jtmp.name
    cv2.imwrite(jpeg_path, resized)          # default JPEG quality ≈ 95
    result = cv2.imread(jpeg_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    os.remove(jpeg_path)

    return result  # float32, shape (340, 340), values [0, 255]
