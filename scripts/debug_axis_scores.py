#!/usr/bin/env python3
"""
Debug: measure axis_angle_fraction and periodicity on processed training crops
to verify they pass the 0.42 threshold after preprocessing.
"""
import cv2
import numpy as np
import glob
import sys
sys.path.insert(0, ".")
from src.ood.ood_guards import axis_angle_fraction, ecg_periodicity_score

# 1. Check processed 340x340 training crops
files = glob.glob("data/processed_340/**/*.jpg", recursive=True)
print(f"Processed 340x340 crops — {len(files)} files")
axis_scores = []
per_scores = []
for f in files:
    img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
    if img is None:
        continue
    af = axis_angle_fraction(img)
    ps, _ = ecg_periodicity_score(img)
    axis_scores.append(af)
    per_scores.append(ps)

print(f"  axis_frac : min={min(axis_scores):.4f} mean={np.mean(axis_scores):.4f} max={max(axis_scores):.4f}")
print(f"  periodicity: min={min(per_scores):.4f} mean={np.mean(per_scores):.4f} max={max(per_scores):.4f}")
over_thresh = [f for f, a in zip(files, axis_scores) if a > 0.42]
print(f"  axis > 0.42: {len(over_thresh)} / {len(files)}")
if over_thresh:
    print("  First offenders:")
    for f in over_thresh[:5]:
        idx = files.index(f)
        print(f"    {f[-60:]}  axis={axis_scores[idx]:.4f}")

under_thresh = [f for f, p in zip(files, per_scores) if p < 0.14]
print(f"  periodicity < 0.14: {len(under_thresh)} / {len(files)}")

# 2. Simulate what preprocess_for_inference does — check a raw upload
# by running our preprocessing on a sample raw file
print()
raw_files = glob.glob("data/ECG_Image_data/**/*.jpg", recursive=True)
if not raw_files:
    raw_files = glob.glob("data/**/*.jpg", recursive=True)
raw_samples = [f for f in raw_files if "processed_340" not in f][:5]

if raw_samples:
    print(f"Raw ECG files — testing {len(raw_samples)} samples:")
    sys.path.insert(0, "backend")
    try:
        from preprocess_inference import preprocess_for_inference
    except ImportError:
        try:
            from src.preprocessing.preprocess_inference import preprocess_for_inference
        except ImportError:
            preprocess_for_inference = None

    for f in raw_samples:
        img_raw = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        if img_raw is None:
            continue
        af_raw = axis_angle_fraction(img_raw)
        ps_raw, _ = ecg_periodicity_score(img_raw)
        print(f"  RAW  axis={af_raw:.3f} per={ps_raw:.3f}  {f[-50:]}")
        if preprocess_for_inference is not None:
            img_bgr = cv2.cvtColor(img_raw, cv2.COLOR_GRAY2BGR)
            proc = preprocess_for_inference(img_bgr)
            proc_u8 = proc.astype(np.uint8)
            af_proc = axis_angle_fraction(proc_u8)
            ps_proc, _ = ecg_periodicity_score(proc_u8)
            print(f"  PROC axis={af_proc:.3f} per={ps_proc:.3f}")
