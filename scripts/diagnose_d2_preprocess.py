"""
Diagnose preprocessing output on Dataset 2 images.
Compares old (fixed thresh=200) vs new (OTSU+polarity) preprocessing visually.
"""
import sys, os
sys.path.append('.')
import cv2
import numpy as np
from pathlib import Path

root = Path('data/ECG Dataset for Heart Condition Classification/ECG Dataset/ECG Dataset')
imgs = sorted(list(root.rglob('*.jpg')) + list(root.rglob('*.JPG')) + list(root.rglob('*.png')))[:6]

for p in imgs:
    bgr = cv2.imread(str(p))
    if bgr is None: continue
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    mean_val = gray.mean()
    otsu_thresh, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bright_pct = (gray > 200).sum() / gray.size * 100
    dark_pct   = (gray < 50).sum()  / gray.size * 100
    print('%s | mean=%.1f | otsu_thresh=%.0f | bright%%=%.1f | dark%%=%.1f | would_invert=%s'
          % (p.name[:35], mean_val, otsu_thresh, bright_pct, dark_pct, mean_val <= 127))
