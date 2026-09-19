"""
Verify gatekeeper passes Dataset 2 images with the updated 30% row_coverage threshold.
"""
import sys
sys.path.append('.')
import cv2
import numpy as np
from pathlib import Path

root = Path('data/ECG Dataset for Heart Condition Classification/ECG Dataset/ECG Dataset')
imgs = sorted(list(root.rglob('*.jpg')) + list(root.rglob('*.JPG')) + list(root.rglob('*.png')))[:20]

passed = failed = 0
for p in imgs:
    bgr = cv2.imread(str(p))
    if bgr is None: continue
    img_gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = img_gray.shape
    aspect = w / h

    def fail(reason):
        global failed
        failed += 1
        print('FAIL %s | %s' % (p.name[:35], reason))

    if aspect < 1.1:
        fail('aspect=%.2f' % aspect); continue
    avg = img_gray.mean()
    if avg < 100:
        fail('avg=%.1f' % avg); continue
    light_ratio = np.sum(img_gray > 180) / img_gray.size
    if light_ratio < 0.85:
        fail('light_ratio=%.1f%%' % (light_ratio*100)); continue
    dark_mask = img_gray < 80
    dark_ratio = dark_mask.sum() / img_gray.size
    if dark_ratio < 0.005:
        fail('no signal'); continue
    if dark_ratio > 0.3:
        fail('dark_ratio=%.1f%%' % (dark_ratio*100)); continue
    rows_with_dark = np.any(dark_mask, axis=1).sum()
    row_coverage = rows_with_dark / h
    if row_coverage < 0.30:   # UPDATED threshold
        fail('row_coverage=%.0f%%' % (row_coverage*100)); continue
    cols_with_dark = np.any(dark_mask, axis=0).sum()
    col_coverage = cols_with_dark / w
    if col_coverage < 0.80:   # UPDATED threshold
        fail('col_coverage=%.0f%%' % (col_coverage*100)); continue
    passed += 1
    print('PASS %s | row=%.0f%% col=%.0f%% dark=%.1f%%' % (
          p.name[:35], row_coverage*100, col_coverage*100, dark_ratio*100))

print('\nPassed: %d / %d' % (passed, passed+failed))
