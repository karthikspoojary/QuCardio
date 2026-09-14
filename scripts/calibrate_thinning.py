"""Calibrate the thinning erosion kernel to match training image density."""
import sys, cv2, numpy as np, shutil, tempfile
sys.path.append('.')
from pathlib import Path
from src.preprocessing.preprocess_inference import preprocess_for_inference

# Target: training images have ~2-3% waveform density BEFORE preprocess,
# and mean ~12 AFTER preprocess_for_inference.
# Let's find the right erosion kernel size.

test_imgs = list(Path('data/ptb-xl-images/Normal').glob('*.png'))[:10]

print('Training reference:')
for p in ['data/raw/Normal/Normal(1).jpg', 'data/raw/Normal/Normal(2).jpg', 'data/raw/Normal/Normal(3).jpg']:
    raw = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    proc = preprocess_for_inference(p)
    print(f'  {p}: raw dark%={((raw<128).mean()*100):.1f}  proc mean={proc.mean():.1f}')

print()
IMG_SIZE_PX = 340

for kernel_size in [(1,1), (2,1), (1,2), (2,2), (3,3)]:
    results = []
    for src in test_imgs:
        img = cv2.imread(str(src), cv2.IMREAD_GRAYSCALE)
        if img.shape != (IMG_SIZE_PX, IMG_SIZE_PX):
            img = cv2.resize(img, (IMG_SIZE_PX, IMG_SIZE_PX))
        _, binary = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV)
        if kernel_size == (1,1):
            thinned = binary  # no erosion
        else:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
            thinned = cv2.erode(binary, kernel, iterations=1)
        result = np.full((IMG_SIZE_PX, IMG_SIZE_PX), 255, dtype=np.uint8)
        result[thinned > 0] = 0
        # Then check after preprocess_for_inference
        import tempfile as tf, os
        with tf.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            tmppath = tmp.name
        cv2.imwrite(tmppath, result)
        proc = preprocess_for_inference(tmppath)
        os.remove(tmppath)
        results.append(proc.mean())
    avg_mean = np.mean(results)
    wf_density = np.mean([(cv2.imread(str(s), cv2.IMREAD_GRAYSCALE) < 128).mean()*100 for s in test_imgs])
    print(f'  kernel={str(kernel_size):<8}: proc_mean={avg_mean:.1f}  (target ~12)')
