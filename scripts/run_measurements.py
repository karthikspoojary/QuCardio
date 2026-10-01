"""
Three measurements in one script:
  1. Per-class precision/recall/F1 for QSVC, Pegasos QSVC, Classical SVM
  2. Kernel off-diagonal variance per encoding range (concentration measurement)
  3. Layer-wise ResNet50 ablation: pool1_pool vs deeper layers + SVD → SVC accuracy

Outputs:
  results/paper/main/per_class_metrics.json
  results/paper/ablation/kernel_concentration.json
  results/paper/ablation/layerwise_ablation.json
"""
import sys, json, warnings
import numpy as np
import joblib
from pathlib import Path
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                              classification_report)
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MinMaxScaler

sys.path.append('.')
import config
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# PART 1 — Per-class P/R/F1
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("PART 1 — Per-class precision / recall / F1")
print("="*60)

d       = np.load(config.FEATURES_DIR / 'features_9d.npz')
X_train = d['train_features']
X_test  = d['test_features']
y_train = d['y_train']
y_test  = d['y_test']

# QSVC
qsvc   = joblib.load(config.MODELS_DIR / 'qsvc_model.pkl')
sv_tr  = np.load(config.FEATURES_DIR / 'sv_train_qsvc.npz',  allow_pickle=True)['sv_train']
sv_te  = np.load(config.FEATURES_DIR / 'sv_test_qsvc.npz',   allow_pickle=True)['sv_test']
K_qsvc = (np.abs(np.dot(sv_te, sv_tr.conj().T))**2).astype(np.float32)
y_qsvc = qsvc.predict(K_qsvc)

# Classical SVM
svm    = joblib.load(config.MODELS_DIR / 'svm_model.pkl')
y_svm  = svm.predict(X_test)

# Pegasos
from src.quantum.train_pegasos import CLASS_PAIRS
sv_ptr = np.load(config.FEATURES_DIR / 'sv_train_pegasos.npz', allow_pickle=True)['sv_train']
sv_pte = np.load(config.FEATURES_DIR / 'sv_test_pegasos.npz',  allow_pickle=True)['sv_test']
K_peg  = (np.abs(np.dot(sv_pte, sv_ptr.conj().T))**2).astype(np.float64)
models_p = {}
for c1, c2 in CLASS_PAIRS:
    p = config.MODELS_DIR / f'pegasos_{c1}_{c2}.pkl'
    if p.exists():
        models_p[(c1,c2)] = joblib.load(p)

y_peg = []
for i in range(len(y_test)):
    def node(c1, c2):
        m = models_p[(c1,c2)]; idx = m.train_indices_
        k = K_peg[i, idx].reshape(1,-1); pr = m.predict(k)[0]
        return c1 if pr == -1 else c2
    p01 = node(0,1); p23 = node(2,3)
    y_peg.append(node(0,2) if p01==0 and p23==2 else
                 node(0,3) if p01==0 else
                 node(1,2) if p23==2 else node(1,3))
y_peg = np.array(y_peg)

def per_class_dict(y_true, y_pred, names):
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(names))), zero_division=0)
    return {names[i]: {
        'precision': round(float(p[i]), 4),
        'recall':    round(float(r[i]), 4),
        'f1':        round(float(f[i]), 4),
        'support':   int(s[i]),
    } for i in range(len(names))}

per_class = {
    'QSVC':          per_class_dict(y_test, y_qsvc, config.CLASS_NAMES),
    'Pegasos_QSVC':  per_class_dict(y_test, y_peg,  config.CLASS_NAMES),
    'Classical_SVM': per_class_dict(y_test, y_svm,  config.CLASS_NAMES),
}

for name, preds in [('QSVC', y_qsvc), ('Pegasos QSVC', y_peg), ('Classical SVM', y_svm)]:
    print(f'\n  {name}')
    print(classification_report(y_test, preds,
          target_names=config.CLASS_NAMES, zero_division=0, digits=4))

Path('results/paper/main/per_class_metrics.json').write_text(
    json.dumps(per_class, indent=2))
print("Saved → results/paper/main/per_class_metrics.json")


# ─────────────────────────────────────────────────────────────────────────────
# PART 2 — Kernel off-diagonal variance per encoding range
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("PART 2 — Kernel concentration (off-diagonal variance)")
print("="*60)

from qiskit.circuit.library import ZZFeatureMap
from qiskit.quantum_info import Statevector
from tqdm import tqdm

def compute_svs(X, fm, desc=''):
    svs = []
    for x in tqdm(X, desc=f'  SV {desc}', leave=False):
        svs.append(Statevector(fm.assign_parameters(x)).data)
    return np.array(svs)

def build_kernel(sv1, sv2):
    return (np.abs(np.dot(sv1, sv2.conj().T))**2).astype(np.float32)

fm_circ = ZZFeatureMap(feature_dimension=9, reps=2, entanglement='circular')

# We use the training set (N=742) to measure kernel distribution
# For speed: sample 200 train points
rng     = np.random.default_rng(42)
idx200  = rng.choice(len(X_train), 200, replace=False)
X_sub   = X_train[idx200]

encodings = {
    'raw (no scaling)':   X_sub,
    '[0, 1]':             X_sub,              # already [0,1] from features_9d
    '[0, π]':             X_sub * np.pi,
    '[0, 2π]':            X_sub * 2 * np.pi,
}

concentration = {}
for enc_name, X_enc in encodings.items():
    print(f'\n  Encoding: {enc_name}')
    sv  = compute_svs(X_enc, fm_circ, enc_name)
    K   = build_kernel(sv, sv)          # 200×200 kernel matrix

    # Off-diagonal entries only
    mask     = ~np.eye(200, dtype=bool)
    off_diag = K[mask]

    mean_K   = float(off_diag.mean())
    var_K    = float(off_diag.var())
    std_K    = float(off_diag.std())
    min_K    = float(off_diag.min())
    max_K    = float(off_diag.max())

    # Effective rank of kernel matrix (sum of eigenvalues / max eigenvalue)
    eigvals  = np.linalg.eigvalsh(K.astype(np.float64))
    eigvals  = eigvals[eigvals > 0]
    eff_rank = float((eigvals.sum()**2) / (eigvals**2).sum())

    print(f'    off-diag mean: {mean_K:.6f}   std: {std_K:.6f}   var: {var_K:.6f}')
    print(f'    off-diag min: {min_K:.6f}   max: {max_K:.6f}')
    print(f'    effective rank: {eff_rank:.2f}')

    concentration[enc_name] = {
        'n_samples':    200,
        'off_diag_mean': round(mean_K, 8),
        'off_diag_std':  round(std_K, 8),
        'off_diag_var':  round(var_K, 8),
        'off_diag_min':  round(min_K, 6),
        'off_diag_max':  round(max_K, 6),
        'effective_rank': round(eff_rank, 3),
    }

Path('results/paper/ablation/kernel_concentration.json').write_text(
    json.dumps(concentration, indent=2))
print("\nSaved → results/paper/ablation/kernel_concentration.json")


# ─────────────────────────────────────────────────────────────────────────────
# PART 3 — Layer-wise ResNet50 ablation
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("PART 3 — Layer-wise ResNet50 ablation")
print("="*60)

import gc, cv2
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.models import Model

# Layers to probe: (label, keras_layer_name, expected_output_shape)
LAYER_CANDIDATES = [
    ('pool1_pool',   'pool1_pool'),         # current: 85×85×64  = 462,400
    ('conv2_block1_out', 'conv2_block1_out'),  # 43×43×256 = 473,344  — after 1st bottleneck
    ('conv3_block1_out', 'conv3_block1_out'),  # 22×22×512 = 247,808
    ('conv4_block1_out', 'conv4_block1_out'),  # 11×11×1024= 123,904
    ('conv5_block3_out', 'conv5_block3_out'),  # 6×6×2048  = 73,728   — deepest
]

# Load images once (from processed_340 if it exists, else skip)
processed_dir = Path('data/processed_340')
if not processed_dir.exists():
    print("  data/processed_340 not found — skipping layer ablation")
    results_layer = {'error': 'data/processed_340 not found'}
else:
    print("  Loading images ...")
    class_names = config.CLASS_NAMES
    class_map   = {c: i for i, c in enumerate(class_names)}
    X_imgs, y_imgs = [], []
    for cls in class_names:
        cls_dir = processed_dir / cls
        if not cls_dir.exists():
            continue
        for p in sorted(cls_dir.glob('*.jpg')) + sorted(cls_dir.glob('*.png')):
            img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if img is None: continue
            if img.shape[:2] != (340, 340):
                img = cv2.resize(img, (340, 340))
            X_imgs.append(img)
            y_imgs.append(class_map[cls])
    X_imgs = np.array(X_imgs, dtype=np.float32)
    y_imgs = np.array(y_imgs)
    print(f"  Loaded {len(X_imgs)} images")

    # Train/test split matching the fixed split (random_state=42, stratified)
    from sklearn.model_selection import train_test_split
    X_tr_img, X_te_img, y_tr_l, y_te_l = train_test_split(
        X_imgs, y_imgs, test_size=0.2, stratify=y_imgs, random_state=42)

    base_model = ResNet50(weights='imagenet', include_top=False)
    results_layer = {}

    for layer_label, layer_name in LAYER_CANDIDATES:
        print(f"\n  Layer: {layer_label}")
        try:
            layer_out = base_model.get_layer(layer_name).output
        except ValueError:
            print(f"    Layer '{layer_name}' not found — skipping")
            results_layer[layer_label] = {'error': f'layer {layer_name} not found'}
            continue

        extractor = Model(inputs=base_model.input, outputs=layer_out)
        feat_dim  = int(np.prod(extractor.output_shape[1:]))
        print(f"    raw dim = {feat_dim}")

        def extract(X_data):
            feats = []
            for i in range(0, len(X_data), 8):
                batch = X_data[i:i+8]
                rgb   = np.repeat(batch[..., np.newaxis], 3, axis=-1)
                prep  = preprocess_input(rgb)
                out   = extractor.predict_on_batch(prep)
                feats.append(out.reshape(len(batch), -1))
                del batch, rgb, prep, out
            return np.vstack(feats)

        tr_raw = extract(X_tr_img)
        te_raw = extract(X_te_img)
        gc.collect()

        # SVD → 9D → MinMax[0,π] → SVC
        svd  = TruncatedSVD(n_components=9, random_state=42)
        scl  = MinMaxScaler()
        tr_9 = scl.fit_transform(svd.fit_transform(tr_raw)) * np.pi
        te_9 = scl.transform(svd.transform(te_raw))         * np.pi

        # Build quantum kernel (circular, reps=2) — same as final model
        fm = ZZFeatureMap(feature_dimension=9, reps=2, entanglement='circular')
        sv_tr = compute_svs(tr_9, fm, f'{layer_label} train')
        sv_te = compute_svs(te_9, fm, f'{layer_label} test')
        K_tr  = build_kernel(sv_tr, sv_tr)
        K_te  = build_kernel(sv_te, sv_tr)

        clf  = SVC(kernel='precomputed', C=5.0, random_state=42)
        clf.fit(K_tr, y_tr_l)
        y_pr = clf.predict(K_te)
        acc  = float(accuracy_score(y_te_l, y_pr))
        n    = len(y_te_l)
        print(f"    accuracy = {acc*100:.2f}%  ({int(round(acc*n))}/{n})")

        results_layer[layer_label] = {
            'raw_dim':  feat_dim,
            'svd_9d':   9,
            'accuracy': round(acc, 6),
            'acc_pct':  round(acc*100, 2),
            'n_test':   n,
        }
        del extractor, tr_raw, te_raw, tr_9, te_9, sv_tr, sv_te, K_tr, K_te
        gc.collect()
        tf.keras.backend.clear_session()

    del base_model; gc.collect()
    Path('results/paper/ablation/layerwise_ablation.json').write_text(
        json.dumps(results_layer, indent=2))
    print("\nSaved → results/paper/ablation/layerwise_ablation.json")

print("\n✅ All three parts complete.")
