"""
Quick test: apply _thin_to_training_style to a few existing PTB-XL images
and check that pixel density drops from ~12% to ~2-3%.
"""
import sys, cv2, numpy as np
sys.path.append('.')
from src.analysis.convert_ptbxl_matplotlib import _thin_to_training_style, IMG_SIZE_PX
from pathlib import Path
import shutil, tempfile

test_imgs = list(Path('data/ptb-xl-images/Normal').glob('*.png'))[:5]

for src in test_imgs:
    # Copy to temp so we don't destroy originals
    tmp = Path(tempfile.mktemp(suffix='.png'))
    shutil.copy(src, tmp)

    # Before
    before = cv2.imread(str(tmp), cv2.IMREAD_GRAYSCALE)
    wf_before = (before < 128).mean() * 100

    # Apply thinning
    _thin_to_training_style(tmp)

    # After
    after = cv2.imread(str(tmp), cv2.IMREAD_GRAYSCALE)
    wf_after = (after < 128).mean() * 100
    tmp.unlink()

    print(f'{src.name}: waveform {wf_before:.1f}% → {wf_after:.1f}%  (target: ~2-3%)')

# Also check mean pixel value after preprocess_for_inference
print()
print('=== After thinning + preprocess_for_inference ===')
sys.path.append('.')
from src.preprocessing.preprocess_inference import preprocess_for_inference

# Pick one, apply thinning to temp copy, then preprocess
src = test_imgs[0]
tmp = Path(tempfile.mktemp(suffix='.png'))
shutil.copy(src, tmp)
_thin_to_training_style(tmp)
proc = preprocess_for_inference(tmp)
tmp.unlink()
print(f'  After thinning + preprocess: mean={proc.mean():.1f} (target: ~12)')

# Compare with training image
proc_train = preprocess_for_inference('data/raw/Normal/Normal(1).jpg')
print(f'  Training image after preprocess: mean={proc_train.mean():.1f}')
