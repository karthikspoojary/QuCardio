"""
Retroactively apply _thin_to_training_style to all existing PTB-XL images in-place.
This is much faster than re-running the full conversion (~2 min vs ~18 min).
Run this once after the thinning fix was added to convert_ptbxl_matplotlib.py.
"""
import sys
sys.path.append('.')
from pathlib import Path
from tqdm import tqdm
from src.analysis.convert_ptbxl_matplotlib import _thin_to_training_style

PTBXL_IMAGES_DIR = Path('data/ptb-xl-images')
FOLDERS = ['Normal', 'Arrhythmia', 'Myocardial_Infarction', 'History_of_MI']

all_pngs = []
for folder in FOLDERS:
    all_pngs.extend((PTBXL_IMAGES_DIR / folder).glob('*.png'))

print(f'Applying thinning to {len(all_pngs)} images in-place ...')
for p in tqdm(all_pngs):
    _thin_to_training_style(p)

print('Done. Re-run cross_dataset_ptbxl.py to evaluate.')
