import numpy as np
d = np.load('data/features_9d.npz')
print('Keys:', list(d.keys()))
for k in d.keys():
    print('  %s: shape=%s dtype=%s' % (k, d[k].shape, d[k].dtype))
