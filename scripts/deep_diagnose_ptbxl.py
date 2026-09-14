"""
Deep diagnosis: look at what the SVM actually predicts and where
the PTB-XL 9D features land relative to the training manifold.
"""
import sys, numpy as np, joblib
sys.path.append('.')
import config

# Load data
train_raw = np.load('data/resnet50_features.npz')
svd    = joblib.load('backend/models/svd_reducer.pkl')
scaler = joblib.load('backend/models/minmax_scaler.pkl')
svm    = joblib.load('backend/models/svm_model.pkl')

X_train = scaler.transform(svd.transform(train_raw['train_features']))
y_train = train_raw['y_train']
X_test  = scaler.transform(svd.transform(train_raw['test_features']))
y_test  = train_raw['y_test']

ptbxl = np.load('results/ptbxl_cross_dataset/ptbxl_features_9d.npz')
X_ptbxl = ptbxl['X_9d']
y_ptbxl = ptbxl['y_true']

# SVM on train/test set
from sklearn.metrics import accuracy_score
print('SVM on original train set:', accuracy_score(y_train, svm.predict(X_train)))
print('SVM on original test set: ', accuracy_score(y_test,  svm.predict(X_test)))
print('SVM on PTB-XL:            ', accuracy_score(y_ptbxl, svm.predict(X_ptbxl)))

print()
# What does SVM predict on PTB-XL per class?
y_pred_ptbxl = svm.predict(X_ptbxl)
print('=== SVM prediction distribution on PTB-XL ===')
print('Predicted class counts:', {i: int((y_pred_ptbxl==i).sum()) for i in range(4)})
print('True class counts:     ', {i: int((y_ptbxl==i).sum()) for i in range(4)})

print()
# Where are PTB-XL features in 9D space?
print('=== 9D Feature per-class stats ===')
for ci, name in enumerate(config.CLASS_NAMES):
    mask_tr  = y_train == ci
    mask_ptb = y_ptbxl == ci
    if mask_ptb.sum() == 0:
        continue
    print(f'\n  {name}:')
    print(f'    Train  n={mask_tr.sum()}: mean={X_train[mask_tr].mean(axis=0).round(3)}')
    print(f'    PTB-XL n={mask_ptb.sum()}: mean={X_ptbxl[mask_ptb].mean(axis=0).round(3)}')

    # L2 distance between centroids
    c_tr  = X_train[mask_tr].mean(axis=0)
    c_ptb = X_ptbxl[mask_ptb].mean(axis=0)
    dist = np.linalg.norm(c_tr - c_ptb)
    print(f'    Centroid L2 distance: {dist:.4f}')

print()
# What is the within-class L2 distance in training?
print('=== Within-class std (training) ===')
for ci, name in enumerate(config.CLASS_NAMES):
    mask = y_train == ci
    std = X_train[mask].std()
    print(f'  {name}: std={std:.4f}')

print()
# Key question: are PTB-XL features BETWEEN training classes or outside them?
# Compute distance from each PTB-XL point to each training class centroid
centroids = {ci: X_train[y_train == ci].mean(axis=0) for ci in range(4)}
print('=== PTB-XL nearest centroid accuracy ===')
nearest = []
for x in X_ptbxl:
    dists = [np.linalg.norm(x - centroids[ci]) for ci in range(4)]
    nearest.append(np.argmin(dists))
nearest = np.array(nearest)
print(f'  Nearest-centroid accuracy: {accuracy_score(y_ptbxl, nearest)*100:.2f}%')
print(f'  Predicted class counts: {dict((i, int((nearest==i).sum())) for i in range(4))}')
