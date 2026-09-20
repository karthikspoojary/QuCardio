"""
Check periodicity / axis-angle scores on the hard-negative line-art images
to confirm they are separated from the ECG range.
"""
import sys, glob
sys.path.insert(0, '.')
import cv2
import numpy as np
from src.ood.ood_guards import ecg_periodicity_score, axis_angle_fraction

neg_paths = sorted(glob.glob('data/NON_ECG_DATA/*.jpg'))
print(f"Found {len(neg_paths)} hard-negative images")

per_scores, axis_scores = [], []
for p in neg_paths[:50]:   # sample first 50 for speed
    img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    if img is None:
        continue
    ps, _ = ecg_periodicity_score(img)
    af    = axis_angle_fraction(img)
    per_scores.append(ps)
    axis_scores.append(af)

per_scores  = np.array(per_scores)
axis_scores = np.array(axis_scores)

print(f"\n--- Periodicity score on line-art negatives ---")
print(f"  min={per_scores.min():.4f}  median={np.median(per_scores):.4f}  max={per_scores.max():.4f}")
print(f"  ECG range: [0.14, 0.88], ECG p1=0.177")
print(f"  Line-art above 0.14 threshold: {(per_scores > 0.14).sum()} of {len(per_scores)}")

print(f"\n--- Axis-angle fraction on line-art negatives ---")
print(f"  min={axis_scores.min():.4f}  median={np.median(axis_scores):.4f}  max={axis_scores.max():.4f}")
print(f"  ECG range: [0.22, 0.41], ECG p99=0.377")
print(f"  Line-art above 0.40 threshold: {(axis_scores > 0.40).sum()} of {len(axis_scores)}")
