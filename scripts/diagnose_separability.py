"""
Check if the problem is the image rendering style or the ResNet features themselves.
Test: does a random forest on raw ResNet features (no SVD) do better?
Also check: what do the rendered PTB-XL images actually look like to ResNet?
"""
import sys, numpy as np
sys.path.append('.')
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

d = np.load('results/ptbxl_retrained/cache_n1000/train_raw.npz')
X_raw, y = d['X'], d['y']

print('=== The core problem: class separability ===')
print(f'  4 classes, BALANCED train (250 each) → random chance = 25.0%')
print(f'  Majority class baseline on TEST = 45.8% (1918/4184 are Normal)')
print(f'  Our best CV score = 33-34% on BALANCED train')
print(f'  This means classes are barely separable in ResNet pool1_pool features')
print()

# Check: what does a Random Forest get on MORE SVD dims?
print('=== Random Forest on different feature counts ===')
for n in [9, 32, 100, 500]:
    svd = TruncatedSVD(n_components=n, random_state=42)
    X_n = svd.fit_transform(X_raw)
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    cv = cross_val_score(rf, X_n, y, cv=5, scoring='accuracy')
    print(f'  n={n:4d} dims: RF CV={cv.mean()*100:.1f}% ± {cv.std()*100:.1f}%')

# Check: what does the pool1_pool layer actually encode for these images?
# Look at per-feature variance across classes
print()
print('=== ResNet pool1_pool feature analysis ===')
print(f'  Feature dim: {X_raw.shape[1]}')
print(f'  Mean feature value: {X_raw.mean():.4f}')
print(f'  Std across all: {X_raw.std():.4f}')
print(f'  Fraction of near-zero features (<0.01 std): {(X_raw.std(axis=0) < 0.01).mean()*100:.1f}%')

# Fisher score per feature (ratio of between-class to within-class variance)
means = {i: X_raw[y==i].mean(axis=0) for i in range(4)}
global_mean = X_raw.mean(axis=0)
between = sum(250 * (means[i] - global_mean)**2 for i in range(4)) / 4
within  = sum(X_raw[y==i].var(axis=0) for i in range(4)) / 4
fisher  = between / (within + 1e-10)
top_k = np.argsort(fisher)[::-1][:20]
print(f'  Top 20 Fisher-score feature indices: {top_k[:5]} ...')
print(f'  Max Fisher score: {fisher.max():.4f}')
print(f'  Median Fisher score: {np.median(fisher):.6f}')
print(f'  Fisher > 0.1: {(fisher > 0.1).sum()} features')
print(f'  Fisher > 0.01: {(fisher > 0.01).sum()} features')

# Try SVM on top Fisher features
print()
print('=== SVM on top Fisher-discriminative features (no SVD) ===')
from sklearn.svm import SVC
for k in [50, 100, 500, 1000]:
    top_feat = np.argsort(fisher)[::-1][:k]
    X_k = X_raw[:, top_feat]
    svm = SVC(kernel='rbf', C=5.0, gamma='scale')
    cv = cross_val_score(svm, X_k, y, cv=5, scoring='accuracy')
    print(f'  top {k:5d} Fisher features: SVM CV={cv.mean()*100:.1f}% ± {cv.std()*100:.1f}%')
