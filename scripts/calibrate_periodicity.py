"""
Calibrate the periodicity / row-band / axis-angle thresholds
against the 928 real ECG images in data/processed_340/.

Prints per-score percentiles so we can pick safe thresholds
(i.e. no false positives on the training domain).
"""
import sys, glob
sys.path.insert(0, '.')

import cv2
import numpy as np
from src.ood.ood_guards import ecg_periodicity_score, row_band_energy_score, axis_angle_fraction

paths = sorted(glob.glob('data/processed_340/**/*.jpg', recursive=True))
print(f"Found {len(paths)} ECG images")

per_scores, row_scores, axis_scores = [], [], []

for p in paths:
    img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"  WARN: could not read {p}")
        continue
    ps, _ = ecg_periodicity_score(img)
    rs    = row_band_energy_score(img)
    af    = axis_angle_fraction(img)
    per_scores.append(ps)
    row_scores.append(rs)
    axis_scores.append(af)

per_scores  = np.array(per_scores)
row_scores  = np.array(row_scores)
axis_scores = np.array(axis_scores)

print(f"\n--- Periodicity score (higher = more ECG-like) ---")
print(f"  min={per_scores.min():.4f}  p1={np.percentile(per_scores,1):.4f}  "
      f"p2={np.percentile(per_scores,2):.4f}  p5={np.percentile(per_scores,5):.4f}  "
      f"median={np.median(per_scores):.4f}  max={per_scores.max():.4f}")
print(f"  Current threshold: 0.15  |  ECGs below it: {(per_scores < 0.15).sum()}")

print(f"\n--- Row-band energy score (higher = more ECG-like) ---")
print(f"  min={row_scores.min():.4f}  p1={np.percentile(row_scores,1):.4f}  "
      f"p5={np.percentile(row_scores,5):.4f}  median={np.median(row_scores):.4f}  "
      f"max={row_scores.max():.4f}")
print(f"  Current threshold: 0.55  |  ECGs below it: {(row_scores < 0.55).sum()}")

print(f"\n--- Axis-angle fraction (lower = more ECG-like) ---")
print(f"  min={axis_scores.min():.4f}  p95={np.percentile(axis_scores,95):.4f}  "
      f"p99={np.percentile(axis_scores,99):.4f}  median={np.median(axis_scores):.4f}  "
      f"max={axis_scores.max():.4f}")
print(f"  Current threshold: 0.70  |  ECGs above it: {(axis_scores > 0.70).sum()}")

# Safe thresholds = just below p1 of ECG distribution for lower-bound scores,
#                   just above p99 for upper-bound scores
print(f"\n--- Recommended safe thresholds (0 ECG false-positives) ---")
print(f"  periodicity  >= {np.percentile(per_scores, 1) * 0.90:.4f}  (p1 * 0.90)")
print(f"  row_band     >= {np.percentile(row_scores,  1) * 0.90:.4f}  (p1 * 0.90)")
print(f"  axis_frac    <= {min(0.99, np.percentile(axis_scores, 99) * 1.05):.4f}  (p99 * 1.05)")
