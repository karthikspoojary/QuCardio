"""
Diagnose why PTB-XL retrained results are still poor (~27-29%).
Key questions:
1. SVD explains only 15% variance — is 9D enough?
2. Are the 4 classes separable at all in 9D PTB-XL feature space?
3. What does a simple nearest-centroid classifier get?
4. What does a linear SVM get without the RBF kernel?
5. Is the QSVC actually severely overfitting (99.4% train vs 27% test)?
6. What does the class distribution look like in 9D?
"""
import sys, numpy as np, joblib
sys.path.append('.')
import config
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import NearestCentroid
from sklearn.svm import SVC
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.dummy import DummyClassifier

# Load cached features
d = np.load('results/ptbxl_retrained/cache_n1000/train_raw.npz')
X_train_raw, y_train = d['X'], d['y']
d2 = np.load('results/ptbxl_retrained/cache_n1000/test_9d.npz')
X_test_9d_1000svd, y_test = d2['X'], d2['y']

print('=== Dataset sizes ===')
print(f'  Train: {X_train_raw.shape}')
print(f'  Test:  {X_test_9d_1000svd.shape}')
print(f'  Train class dist: {dict((i,int((y_train==i).sum())) for i in range(4))}')
print(f'  Test  class dist: {dict((i,int((y_test==i).sum())) for i in range(4))}')

# Majority class baseline
dummy = DummyClassifier(strategy='most_frequent')
dummy.fit(X_train_raw[:, :9], y_train)  # features don't matter for majority
print(f'\n  Majority class baseline (test): {accuracy_score(y_test, dummy.predict(X_test_9d_1000svd))*100:.1f}%')

# --- Test different SVD dimensions ---
print('\n=== SVD dimension sweep (SVM rbf C=5) ===')
svd_full = TruncatedSVD(n_components=50, random_state=42)
scaler_full = MinMaxScaler()
X_train_50 = scaler_full.fit_transform(svd_full.fit_transform(X_train_raw))

for n in [9, 16, 32, 50]:
    X_tr = X_train_50[:, :n]
    svd_n = TruncatedSVD(n_components=n, random_state=42)
    sc_n  = MinMaxScaler()
    X_tr_n = sc_n.fit_transform(svd_n.fit_transform(X_train_raw))
    ev = svd_n.explained_variance_ratio_.sum()
    svm = SVC(kernel='rbf', C=5.0, gamma='scale')
    cv_scores = cross_val_score(svm, X_tr_n, y_train, cv=5, scoring='accuracy')
    print(f'  n={n:3d}: explained_var={ev*100:.1f}%  CV-5 acc={cv_scores.mean()*100:.1f}% ± {cv_scores.std()*100:.1f}%')

# --- Test different C values for SVM with n=9 ---
print('\n=== SVM C sweep (SVD n=9) ===')
svd9 = TruncatedSVD(n_components=9, random_state=42)
sc9  = MinMaxScaler()
X_tr9 = sc9.fit_transform(svd9.fit_transform(X_train_raw))
X_te9_refit = sc9.transform(svd9.transform(np.load('results/ptbxl_retrained/cache_n1000/train_raw.npz')['X']))  # train features projected

# Need test features re-projected with this SVD
# Load test raw... actually we don't have it cached. Use 5-fold CV on train instead.
for C in [0.1, 0.5, 1.0, 5.0, 10.0]:
    for kernel in ['linear', 'rbf']:
        svm = SVC(kernel=kernel, C=C)
        cv = cross_val_score(svm, X_tr9, y_train, cv=5, scoring='accuracy')
        print(f'  kernel={kernel:6s} C={C:5.1f}: CV={cv.mean()*100:.1f}% ± {cv.std()*100:.1f}%')

# --- Nearest centroid (no hyperparameters) ---
print('\n=== Nearest centroid (n=9 features) ===')
nc = NearestCentroid()
cv = cross_val_score(nc, X_tr9, y_train, cv=5, scoring='accuracy')
print(f'  CV-5: {cv.mean()*100:.1f}% ± {cv.std()*100:.1f}%')

# --- Class separability: inter-class distance vs intra-class std ---
print('\n=== Class separability in PTB-XL 9D space ===')
centroids = {i: X_tr9[y_train==i].mean(axis=0) for i in range(4)}
intra_stds = {i: X_tr9[y_train==i].std() for i in range(4)}
print('  Intra-class std:')
for i in range(4):
    print(f'    {config.CLASS_NAMES[i]}: {intra_stds[i]:.4f}')
print('  Inter-class centroid distances:')
for i in range(4):
    for j in range(i+1, 4):
        d = np.linalg.norm(centroids[i] - centroids[j])
        print(f'    {config.CLASS_NAMES[i][:8]} vs {config.CLASS_NAMES[j][:8]}: {d:.4f}  (ratio to std: {d/((intra_stds[i]+intra_stds[j])/2):.2f}x)')
