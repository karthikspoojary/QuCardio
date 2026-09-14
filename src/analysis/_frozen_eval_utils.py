"""
_frozen_eval_utils.py — Shared helpers for ST-5B/5C/5D/5E frozen-model evaluation
===================================================================================
All ST-5x scripts evaluate models that were TRAINED on data/raw/ (742-sample split).
The frozen pipeline is:
  image → preprocess_for_inference() → ResNet50 pool1_pool → frozen svd_reducer.pkl
        → frozen minmax_scaler.pkl → frozen classifiers (SVM, QSVC, Pegasos)

DO NOT refit svd_reducer or minmax_scaler. DO NOT use preprocess_ecg (OTSU).
"""

import sys
import numpy as np
import joblib
import cv2
from pathlib import Path
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
import config

from src.preprocessing.preprocess_inference import preprocess_for_inference

# Lazy-loaded globals so ResNet50 is only built once per process
_resnet_model = None
_svm_model    = None
_qsvc_model   = None
_peg_models   = None
_sv_train_qsvc   = None
_sv_train_pegasos = None


def get_resnet():
    global _resnet_model
    if _resnet_model is None:
        import tensorflow as tf
        from tensorflow.keras.applications import ResNet50
        from tensorflow.keras.models import Model
        base = ResNet50(weights='imagenet', include_top=False)
        try:
            out = base.get_layer('pool1_pool').output
        except ValueError:
            out = base.layers[4].output
        _resnet_model = Model(inputs=base.input, outputs=out)
        print("  ResNet50 pool1_pool model loaded.")
    return _resnet_model


def extract_features(image_paths, batch_size=8):
    """
    Load images → preprocess_for_inference() → ResNet50 pool1_pool → flatten.
    Returns (N, 462400) float32 array.
    Skips any image that fails to load (warns and continues).
    """
    from tensorflow.keras.applications.resnet50 import preprocess_input
    resnet = get_resnet()
    feature_dim = 85 * 85 * 64   # pool1_pool output flattened
    n = len(image_paths)
    features = np.empty((n, feature_dim), dtype=np.float32)
    valid_mask = np.ones(n, dtype=bool)

    for i in tqdm(range(0, n, batch_size), desc="  Extracting features"):
        batch_paths = image_paths[i:i + batch_size]
        imgs = []
        batch_idx = []
        for j, p in enumerate(batch_paths):
            try:
                img = preprocess_for_inference(p)   # float32 (340,340)
            except Exception as e:
                print(f"  WARN: skipping {Path(p).name}: {e}")
                valid_mask[i + j] = False
                continue
            # RGB-ify and preprocess for ImageNet
            rgb = np.repeat(img[..., np.newaxis], 3, axis=-1)
            imgs.append(rgb)
            batch_idx.append(i + j)

        if not imgs:
            continue
        batch_arr = preprocess_input(np.stack(imgs))   # (B,340,340,3)
        feats_3d  = resnet.predict_on_batch(batch_arr) # (B,85,85,64)
        for k, global_idx in enumerate(batch_idx):
            features[global_idx] = feats_3d[k].reshape(-1)

    # Drop failed images
    features  = features[valid_mask]
    return features, valid_mask


def extract_and_project_streaming(image_paths, batch_size=16):
    """
    Load images → ResNet50 pool1_pool → IMMEDIATELY project with frozen SVD & Scaler.
    Avoids allocating a massive 7.5 GB (N, 462400) raw feature matrix in RAM.
    RAM usage stays strictly under ~150 MB regardless of dataset size.
    Returns (N_valid, 9) float32 array and valid_mask bool array.
    """
    from tensorflow.keras.applications.resnet50 import preprocess_input
    resnet = get_resnet()
    svd    = joblib.load(config.MODELS_DIR / 'svd_reducer.pkl')
    scaler = joblib.load(config.MODELS_DIR / 'minmax_scaler.pkl')

    n = len(image_paths)
    valid_mask = np.ones(n, dtype=bool)
    projected_batches = []

    for i in tqdm(range(0, n, batch_size), desc="  Extracting & Projecting (9-D)"):
        batch_paths = image_paths[i:i + batch_size]
        imgs = []
        batch_valid = []
        for j, p in enumerate(batch_paths):
            try:
                img = preprocess_for_inference(p)   # float32 (340,340)
                rgb = np.repeat(img[..., np.newaxis], 3, axis=-1)
                imgs.append(rgb)
                batch_valid.append(True)
            except Exception as e:
                print(f"  WARN: skipping {Path(p).name}: {e}")
                valid_mask[i + j] = False
                batch_valid.append(False)

        if not imgs:
            continue

        batch_arr = preprocess_input(np.stack(imgs))   # (B, 340, 340, 3)
        feats_3d  = resnet.predict_on_batch(batch_arr) # (B, 85, 85, 64)
        feats_flat = feats_3d.reshape(len(imgs), -1)   # temporary batch in RAM
        proj_9d   = scaler.transform(svd.transform(feats_flat)) # (B, 9)
        projected_batches.append(proj_9d)

    if projected_batches:
        X_9d = np.vstack(projected_batches)
    else:
        X_9d = np.empty((0, 9), dtype=np.float32)

    return X_9d, valid_mask


def project_features(raw_features):
    """
    Apply FROZEN svd_reducer.pkl + minmax_scaler.pkl → (N, 9) minmax_01.
    """
    svd    = joblib.load(config.MODELS_DIR / 'svd_reducer.pkl')
    scaler = joblib.load(config.MODELS_DIR / 'minmax_scaler.pkl')
    reduced = svd.transform(raw_features)
    scaled  = scaler.transform(reduced)
    return scaled   # (N, 9)  minmax_01


def load_svm():
    global _svm_model
    if _svm_model is None:
        _svm_model = joblib.load(config.MODELS_DIR / 'svm_model.pkl')
    return _svm_model


def predict_svm(X_9d):
    return load_svm().predict(X_9d)


def load_qsvc_sv():
    global _qsvc_model, _sv_train_qsvc
    if _qsvc_model is None:
        _qsvc_model    = joblib.load(config.MODELS_DIR / 'qsvc_model.pkl')
        sv_path = config.FEATURES_DIR / 'sv_train_qsvc.npz'
        _sv_train_qsvc = np.load(sv_path, allow_pickle=True)['sv_train']
    return _qsvc_model, _sv_train_qsvc


def predict_qsvc(X_9d):
    """X_9d: (N,9) minmax_01 — applies ×π internally before kernel."""
    from qiskit.circuit.library import ZZFeatureMap
    from qiskit.quantum_info import Statevector

    qsvc, sv_train = load_qsvc_sv()
    X_enc = X_9d * np.pi   # → minmax_0pi

    print("  QSVC: computing test statevectors …")
    sv_test = []
    fm = ZZFeatureMap(feature_dimension=9, reps=2, entanglement='circular')
    for x in tqdm(X_enc, desc="  SV", leave=False):
        sv_test.append(Statevector(fm.assign_parameters(x)).data)
    sv_test = np.array(sv_test)

    K_test = (np.abs(np.dot(sv_test, sv_train.conj().T)) ** 2).astype(np.float32)
    return qsvc.predict(K_test)


def load_pegasos():
    global _peg_models, _sv_train_pegasos
    if _peg_models is None:
        from src.quantum.train_pegasos import CLASS_PAIRS
        _peg_models = {}
        for c1, c2 in CLASS_PAIRS:
            p = config.MODELS_DIR / f'pegasos_{c1}_{c2}.pkl'
            if p.exists():
                _peg_models[(c1, c2)] = joblib.load(p)
        sv_path = config.FEATURES_DIR / 'sv_train_pegasos.npz'
        _sv_train_pegasos = np.load(sv_path, allow_pickle=True)['sv_train']
    return _peg_models, _sv_train_pegasos


def predict_pegasos(X_9d):
    """X_9d: (N,9) minmax_01 — Pegasos uses linear entanglement kernel."""
    from qiskit.circuit.library import ZZFeatureMap
    from qiskit.quantum_info import Statevector

    models, sv_train = load_pegasos()
    if len(models) < 6:
        print("  Pegasos: <6 models available — skipping")
        return None

    print("  Pegasos: computing test statevectors …")
    sv_test = []
    fm = ZZFeatureMap(feature_dimension=9, reps=2, entanglement='linear')
    for x in tqdm(X_9d, desc="  SV", leave=False):
        sv_test.append(Statevector(fm.assign_parameters(x)).data)
    sv_test = np.array(sv_test)

    K_te_full = (np.abs(np.dot(sv_test, sv_train.conj().T)) ** 2).astype(np.float64)
    N = len(X_9d)

    y_pred = []
    for i in range(N):
        def node(c1, c2):
            m   = models[(c1, c2)]
            idx = m.train_indices_
            k   = K_te_full[i, idx].reshape(1, -1)
            pr  = m.predict(k)[0]
            return c1 if pr == -1 else c2
        p01 = node(0, 1); p23 = node(2, 3)
        if p01 == 0:
            y_pred.append(node(0, 2) if p23 == 2 else node(0, 3))
        else:
            y_pred.append(node(1, 2) if p23 == 2 else node(1, 3))
    return np.array(y_pred)


def save_confusion_matrix(y_true, y_pred, labels, title, out_path, cmap='Blues'):
    """Save a labelled confusion matrix PNG."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix, accuracy_score

    cm  = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    acc = accuracy_score(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(max(5, len(labels)), max(4, len(labels) - 1)))
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap,
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_title(f'{title}\nAccuracy: {acc*100:.2f}%', fontsize=11)
    ax.set_xlabel('Predicted', fontsize=10)
    ax.set_ylabel('True', fontsize=10)
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=150, bbox_inches='tight')
    plt.close()
