"""Check current state of already-thinned images."""
import sys, cv2, numpy as np
sys.path.append('.')
from pathlib import Path
from src.preprocessing.preprocess_inference import preprocess_for_inference

imgs = list(Path('data/ptb-xl-images/Normal').glob('*.png'))[:10]
means = []
for p in imgs:
    proc = preprocess_for_inference(str(p))
    means.append(proc.mean())
    print(f'{p.name}: proc_mean={proc.mean():.1f}')

print(f'\nAvg proc_mean: {np.mean(means):.1f}  (training target ~12)')

# Training reference
train_means = []
for p in list(Path('data/raw/Normal').glob('*.jpg'))[:10]:
    proc = preprocess_for_inference(str(p))
    train_means.append(proc.mean())
print(f'Training avg proc_mean: {np.mean(train_means):.1f}')
