"""
Layer-wise ResNet50 ablation — FAST VERSION
Uses classical SVC (RBF) as a proxy to rank layers quickly,
then runs the quantum kernel only for pool1_pool (already done)
and the single best deeper layer found by the proxy.

Outputs: results/paper/ablation/layerwise_ablation.json
"""
import sys, json, gc, warnings
import numpy as np
import cv2
from pathlib import Path
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

sys.path.append('.')
import config
warnings.filterwarnings('ignore')

import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.models import Model

# ── Load images ────────────────────────────────────────────────────────────────
processed_dir = Path('data/processed_340')
print("Loading images ...")
class_map = {c: i for i, c in enumerate(config.CLASS_NAMES)}
X_imgs, y_imgs = [], []
for cls in config.CLASS_NAMES:
    cls_dir = processed_dir / cls
    if not cls_dir.exists(): continue
    for p in sorted(list(cls_dir.glob('*.jpg')) + list(cls_dir.glob('*.png'))):
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None: continue
        if img.shape[:2] != (340, 340):
            img = cv2.resize(img, (340, 340))
        X_imgs.append(img); y_imgs.append(class_map[cls])
X_imgs = np.array(X_imgs, dtype=np.float32)
y_imgs = np.array(y_imgs)
print(f"Loaded {len(X_imgs)} images")

X_tr_img, X_te_img, y_tr, y_te = train_test_split(
    X_imgs, y_imgs, test_size=0.2, stratify=y_imgs, random_state=42)

# ── Layer candidates ───────────────────────────────────────────────────────────
LAYERS = [
    ('pool1_pool',       'pool1_pool'),
    ('conv2_block3_out', 'conv2_block3_out'),   # end of res-block group 2
    ('conv3_block4_out', 'conv3_block4_out'),   # end of group 3
    ('conv4_block6_out', 'conv4_block6_out'),   # end of group 4
    ('conv5_block3_out', 'conv5_block3_out'),   # deepest (before GAP)
]

def extract_layer(X_data, extractor, batch=8):
    feats = []
    for i in range(0, len(X_data), batch):
        b = X_data[i:i+batch]
        rgb  = np.repeat(b[..., np.newaxis], 3, axis=-1)
        prep = preprocess_input(rgb)
        out  = extractor.predict_on_batch(prep).reshape(len(b), -1)
        feats.append(out)
        del b, rgb, prep, out
    return np.vstack(feats)

base_model = ResNet50(weights='imagenet', include_top=False)
results = {}

for layer_label, layer_name in LAYERS:
    print(f"\n{'='*55}\nLayer: {layer_label}")
    try:
        extractor = Model(inputs=base_model.input,
                          outputs=base_model.get_layer(layer_name).output)
    except ValueError as e:
        print(f"  SKIP: {e}")
        results[layer_label] = {'error': str(e)}
        continue

    # compute actual raw dim from a single sample
    sample = np.repeat(X_tr_img[:1, ..., np.newaxis], 3, axis=-1)
    raw_dim = int(extractor.predict_on_batch(preprocess_input(sample)).reshape(1,-1).shape[1])
    print(f"  raw_dim: {raw_dim:,}")

    tr_raw = extract_layer(X_tr_img, extractor)
    te_raw = extract_layer(X_te_img, extractor)

    svd = TruncatedSVD(n_components=9, random_state=42)
    scl = MinMaxScaler()
    tr9 = scl.fit_transform(svd.fit_transform(tr_raw)) * np.pi
    te9 = scl.transform(svd.transform(te_raw))         * np.pi
    svd_var = float(svd.explained_variance_ratio_.sum())
    print(f"  SVD-9 explained var: {svd_var*100:.2f}%")

    # ── Proxy: classical SVC (RBF) on [0,π] features — fast ─────────────────
    clf_rbf = SVC(kernel='rbf', C=10.0, gamma='scale', random_state=42)
    clf_rbf.fit(tr9, y_tr)
    acc_rbf = float(accuracy_score(y_te, clf_rbf.predict(te9)))
    print(f"  Classical SVC (RBF) accuracy: {acc_rbf*100:.2f}%")

    # ── Quantum kernel (ZZFeatureMap circular) — only if RAM allows ──────────
    # Compute quantum kernel using cached pool1_pool SVs for the baseline layer,
    # or fresh SVs for other layers. This is the expensive step.
    acc_qsvc = None
    try:
        from qiskit.circuit.library import ZZFeatureMap
        from qiskit.quantum_info import Statevector
        fm = ZZFeatureMap(feature_dimension=9, reps=2, entanglement='circular')

        def sv_batch(X):
            return np.array([Statevector(fm.assign_parameters(x)).data for x in X])

        print(f"  Computing quantum kernel ({len(tr9)} train + {len(te9)} test SVs)...")
        sv_tr = sv_batch(tr9)
        sv_te = sv_batch(te9)
        K_tr  = (np.abs(np.dot(sv_tr, sv_tr.conj().T))**2).astype(np.float32)
        K_te  = (np.abs(np.dot(sv_te, sv_tr.conj().T))**2).astype(np.float32)
        clf_q = SVC(kernel='precomputed', C=5.0, random_state=42)
        clf_q.fit(K_tr, y_tr)
        acc_qsvc = float(accuracy_score(y_te, clf_q.predict(K_te)))
        print(f"  QSVC accuracy: {acc_qsvc*100:.2f}%")
        del sv_tr, sv_te, K_tr, K_te
    except Exception as e:
        print(f"  QSVC skipped: {e}")

    results[layer_label] = {
        'raw_dim':            raw_dim,
        'svd_9_explained_var': round(svd_var, 4),
        'classical_svc_rbf':  round(acc_rbf,  6),
        'qsvc_circular_0pi':  round(acc_qsvc, 6) if acc_qsvc is not None else None,
        'n_test':             len(y_te),
    }
    del extractor, tr_raw, te_raw, tr9, te9
    gc.collect()
    tf.keras.backend.clear_session()

del base_model; gc.collect()

out = Path('results/paper/ablation/layerwise_ablation.json')
out.write_text(json.dumps(results, indent=2))
print(f"\n✅ Saved → {out}")

print("\n=== LAYER-WISE ABLATION SUMMARY ===")
print(f"{'Layer':<25} {'raw_dim':>10} {'SVD var':>8} {'RBF SVC':>9} {'QSVC':>9}")
print("-"*66)
for k, v in results.items():
    if 'error' in v:
        print(f"{k:<25}  ERROR")
        continue
    q = f"{v['qsvc_circular_0pi']*100:.2f}%" if v['qsvc_circular_0pi'] else '  n/a'
    print(f"{k:<25} {v['raw_dim']:>10,} {v['svd_9_explained_var']*100:>7.1f}%  "
          f"{v['classical_svc_rbf']*100:>7.2f}%  {q:>9}")
