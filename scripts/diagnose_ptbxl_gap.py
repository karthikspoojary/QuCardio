import numpy as np
import joblib

# Check actual array shapes in both files
d1 = np.load('data/features_9d.npz')
print('=== data/features_9d.npz ===')
for k in d1.keys():
    arr = d1[k]
    print(f'  {k}: shape={arr.shape}, dtype={arr.dtype}')

print()
d2 = np.load('data/resnet50_features.npz')
print('=== data/resnet50_features.npz ===')
for k in d2.keys():
    arr = d2[k]
    print(f'  {k}: shape={arr.shape}, dtype={arr.dtype}')

print()
d3 = np.load('results/ptbxl_cross_dataset/ptbxl_features_9d.npz')
print('=== results/ptbxl_cross_dataset/ptbxl_features_9d.npz ===')
for k in d3.keys():
    arr = d3[k]
    print(f'  {k}: shape={arr.shape}, dtype={arr.dtype}')
    if k == 'X_9d':
        print(f'    mean={arr.mean():.4f}, std={arr.std():.4f}, min={arr.min():.4f}, max={arr.max():.4f}')

# Now compute 9D from the resnet50 raw features
svd    = joblib.load('backend/models/svd_reducer.pkl')
scaler = joblib.load('backend/models/minmax_scaler.pkl')
raw = np.load('data/resnet50_features.npz')
X_train_9d = scaler.transform(svd.transform(raw['train_features']))
X_test_9d  = scaler.transform(svd.transform(raw['test_features']))
y_train = raw['y_train']
y_test  = raw['y_test']

X_ptbxl = d3['X_9d']

print()
print('=== 9D Feature Distribution ===')
print(f'Train (742)   mean={X_train_9d.mean():.4f} std={X_train_9d.std():.4f} range=[{X_train_9d.min():.3f},{X_train_9d.max():.3f}]')
print(f'Test  (186)   mean={X_test_9d.mean():.4f}  std={X_test_9d.std():.4f}  range=[{X_test_9d.min():.3f},{X_test_9d.max():.3f}]')
print(f'PTB-XL(4184)  mean={X_ptbxl.mean():.4f}  std={X_ptbxl.std():.4f}  range=[{X_ptbxl.min():.3f},{X_ptbxl.max():.3f}]')

print()
print('=== Per-dim comparison ===')
print(f'{"Dim":<5} {"Tr mean":>9} {"Te mean":>9} {"PTB mean":>9} {"PTB std":>9} {"Tr std":>9}')
for d in range(9):
    print(f'{d:<5} {X_train_9d[:,d].mean():>9.4f} {X_test_9d[:,d].mean():>9.4f} {X_ptbxl[:,d].mean():>9.4f} {X_ptbxl[:,d].std():>9.4f} {X_train_9d[:,d].std():>9.4f}')

print()
print('=== OOD: PTB-XL dims outside [train_min, train_max] ===')
for d in range(9):
    lo = X_train_9d[:, d].min()
    hi = X_train_9d[:, d].max()
    ood = ((X_ptbxl[:, d] < lo) | (X_ptbxl[:, d] > hi)).mean()
    print(f'  Dim {d}: train=[{lo:.3f},{hi:.3f}]  OOD={ood*100:.1f}%')
