"""
Test domain adaptation: shift+scale PTB-XL 9D features to match training distribution
before classifying. This is a post-hoc correction for the systematic SVD Dim-0 offset.
"""
import sys, numpy as np, joblib
sys.path.append('.')
import config
from sklearn.metrics import accuracy_score, f1_score

# Load
train_raw = np.load('data/resnet50_features.npz')
svd    = joblib.load('backend/models/svd_reducer.pkl')
scaler = joblib.load('backend/models/minmax_scaler.pkl')
svm    = joblib.load('backend/models/svm_model.pkl')

X_train = scaler.transform(svd.transform(train_raw['train_features']))
y_train = train_raw['y_train']

ptbxl = np.load('results/ptbxl_cross_dataset/ptbxl_features_9d.npz')
X_ptbxl = ptbxl['X_9d'].copy()
y_ptbxl = ptbxl['y_true']

print('=== Raw PTB-XL per-dim stats ===')
for d in range(9):
    print(f'  Dim {d}: ptbxl_mean={X_ptbxl[:,d].mean():.3f}  train_mean={X_train[:,d].mean():.3f}')

# Approach: z-score PTB-XL to training stats (shift+scale per dim)
X_ptbxl_adapted = X_ptbxl.copy()
for d in range(9):
    tr_mean = X_train[:, d].mean()
    tr_std  = X_train[:, d].std()
    pt_mean = X_ptbxl[:, d].mean()
    pt_std  = X_ptbxl[:, d].std()
    if pt_std > 1e-6:
        X_ptbxl_adapted[:, d] = (X_ptbxl[:, d] - pt_mean) / pt_std * tr_std + tr_mean
    else:
        X_ptbxl_adapted[:, d] = tr_mean

print()
print('=== After z-score alignment ===')
for d in range(9):
    print(f'  Dim {d}: adapted_mean={X_ptbxl_adapted[:,d].mean():.3f}  train_mean={X_train[:,d].mean():.3f}')

# Evaluate
y_pred_raw     = svm.predict(X_ptbxl)
y_pred_adapted = svm.predict(X_ptbxl_adapted)

print()
print(f'SVM accuracy — raw PTB-XL features:     {accuracy_score(y_ptbxl, y_pred_raw)*100:.2f}%')
print(f'SVM accuracy — adapted PTB-XL features: {accuracy_score(y_ptbxl, y_pred_adapted)*100:.2f}%')
print()
print(f'F1-macro — raw:     {f1_score(y_ptbxl, y_pred_raw, average="macro", zero_division=0):.4f}')
print(f'F1-macro — adapted: {f1_score(y_ptbxl, y_pred_adapted, average="macro", zero_division=0):.4f}')

print()
print('Predicted class distribution (adapted):')
print({i: int((y_pred_adapted==i).sum()) for i in range(4)})
print('True class distribution:')
print({i: int((y_ptbxl==i).sum()) for i in range(4)})
