import numpy as np
d = np.load('data/sv_train_pegasos.npz', allow_pickle=True)
print('keys:', list(d.keys()))
sv = d['sv_train']
print('shape:', sv.shape)
print('first vec norm:', np.linalg.norm(sv[0]))
print('dtype:', sv.dtype)

# Also check features_9d to see what encoding they were built from
d2 = np.load('data/features_9d.npz')
X_tr = d2['train_features']
print('\nfeatures_9d train range: min={:.4f} max={:.4f}'.format(X_tr.min(), X_tr.max()))

# Check which feature map the cached SVs came from by testing kernel value range
K_diag = np.abs(np.dot(sv, sv.conj().T))**2
print('kernel diagonal (should be ~1.0):', K_diag.diagonal()[:5])
print('kernel off-diag range:', K_diag[0,1:10])
