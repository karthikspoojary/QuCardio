"""
Deep inspection: what do the processed_340 training images actually look like?
Are they truly binary OTSU images or do they have greyscale content?
"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
from pathlib import Path
from src.preprocessing.preprocess_ecg import preprocess_ecg

data_dir = Path("data/processed_340")
CLASS = ["Normal", "Arrhythmia", "Myocardial_Infarction", "History_of_MI"]

print("Inspecting training images (processed_340):")
print(f"{'File':<40} {'min':>6} {'max':>6} {'mean':>8} {'unique vals':>12}")
print("-" * 80)

for cls in CLASS[:2]:
    for img_path in list((data_dir / cls).glob("*.jpg"))[:3]:
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        unique_vals = len(np.unique(img))
        print(f"{img_path.name:<40} {img.min():>6} {img.max():>6} {img.mean():>8.1f} {unique_vals:>12}")

print()
print("Inspecting preprocess_ecg() output on raw images:")
raw_dir = Path("data/raw")
for cls in CLASS[:2]:
    for img_path in list((raw_dir / cls).glob("*.jpg"))[:3]:
        preprocessed = preprocess_ecg(str(img_path), output_size=(340, 340))
        img_u8 = (preprocessed * 255).astype(np.uint8)
        unique_vals = len(np.unique(img_u8))
        print(f"{img_path.name:<40} {img_u8.min():>6} {img_u8.max():>6} {img_u8.mean():>8.1f} {unique_vals:>12}")

print()
print("Key question: Are the processed_340 images JPEG-saved OTSU output from the raw images?")
print("Let's check if processed_340/Normal/X.jpg was made from raw/Normal/X.jpg")
print()

# Pick the first normal image that exists in both
raw_cls = raw_dir / "Normal"
proc_cls = data_dir / "Normal"

for raw_img in list(raw_cls.glob("*.jpg"))[:5]:
    proc_img = proc_cls / raw_img.name
    if not proc_img.exists():
        continue
    
    # Training image
    train = cv2.imread(str(proc_img), cv2.IMREAD_GRAYSCALE).astype(np.float32)
    
    # preprocess_ecg on raw → uint8
    pre  = preprocess_ecg(str(raw_img), output_size=(340, 340))
    pre_u8 = (pre * 255).astype(np.uint8)
    
    diff = np.abs(train - pre_u8.astype(np.float32))
    print(f"{raw_img.name}:")
    print(f"  processed_340 (train input): min={train.min():.0f} max={train.max():.0f} mean={train.mean():.1f}  unique={len(np.unique(train))}")
    print(f"  preprocess_ecg output:       min={pre_u8.min():.0f} max={pre_u8.max():.0f} mean={pre_u8.mean():.1f}  unique={len(np.unique(pre_u8))}")
    print(f"  max pixel diff: {diff.max():.1f}  mean diff: {diff.mean():.2f}")
    print()
    break
