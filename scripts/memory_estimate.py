"""
Memory cost calculator for various training sizes.
Also check what the test set looks like.
"""
import numpy as np

FEATURE_DIM = 462400  # ResNet50 pool1_pool flattened
FLOAT32 = 4  # bytes

print('=== RAM needed for raw ResNet features ===')
for n in [500, 800, 1000, 1200, 2000, 4184]:
    gb = n * FEATURE_DIM * FLOAT32 / (1024**3)
    print(f'  n={n:5d}: {gb:.2f} GB raw  →  9D after SVD = {n*9*4/(1024**2):.2f} MB')

print()
print('=== Quantum kernel RAM (K = N×N float32) ===')
for n in [500, 800, 1000, 1200, 2000]:
    mb = n * n * FLOAT32 / (1024**2)
    print(f'  n={n:5d}: K matrix = {mb:.0f} MB')

print()
print('=== Statevector time estimate (CPU, ~1.2 sec/statevector for 9-qubit reps=2) ===')
sv_per_sec = 1.2  # approximate from our run: 4184 images in ~53s of actual SV compute
for n in [500, 1000, 2000, 4184]:
    minutes = (n / sv_per_sec) / 60
    print(f'  n={n:5d} statevectors: ~{minutes:.0f} min')

print()
# SVD streaming: incremental SVD needs only batch at a time
# After SVD reduction, working memory is tiny
print('=== Streaming SVD (IncrementalPCA / TruncatedSVD in batches) ===')
batch = 16
batch_ram = batch * FEATURE_DIM * FLOAT32 / (1024**2)
print(f'  Per batch ({batch} images): {batch_ram:.0f} MB peak  (then project to 9D and discard)')
print(f'  Total accumulated 9D for 1000 images: {1000*9*4/(1024**2):.1f} MB')
print(f'  → Entire pipeline fits in <300 MB RAM regardless of dataset size')
