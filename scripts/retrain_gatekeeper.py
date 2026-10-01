"""
Retrain the ECG gatekeeper (MobileNetV2) with hard negatives added.

Data sources:
  Positives  : data/ECG_DATA/  (up to 2000, same as original)
  Negatives  : data/NON_ECG_DATA/  (now includes hard_neg_chart_* and hard_neg_uml_*)

Key changes vs original train_ecg_detector.py:
  - pos:neg ratio capped at 1:2 (more negatives = fewer false accepts)
  - Same preprocessing path as production inference (fixed-threshold pipeline)
  - Saves to models/ecg_detector.keras (project root) AND backend/models/ecg_detector.keras
  - Reports confusion matrix on the held-out validation set

Output: models/ecg_detector.keras, backend/models/ecg_detector.keras
"""
import os, glob, sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import cv2

tf.get_logger().setLevel('ERROR')

# ── Data loading ──────────────────────────────────────────────────────────────
def load_data(ecg_dir="data/ECG_DATA", non_ecg_dir="data/NON_ECG_DATA",
              max_ecg=2000, pos_neg_ratio=2):
    """
    Returns train/val splits with pos:neg ≤ 1:pos_neg_ratio.
    pos_neg_ratio=2 means at most 2 negatives per positive → fewer false accepts.
    """
    ecg_paths = sorted(glob.glob(os.path.join(ecg_dir, "**", "*.jpg"), recursive=True))
    non_ecg_paths = sorted(glob.glob(os.path.join(non_ecg_dir, "*.jpg")))

    rng = np.random.default_rng(42)
    rng.shuffle(ecg_paths)
    ecg_paths = ecg_paths[:max_ecg]

    # cap negatives to pos_neg_ratio × positives
    max_neg = len(ecg_paths) * pos_neg_ratio
    rng.shuffle(non_ecg_paths)
    non_ecg_paths = non_ecg_paths[:max_neg]

    print(f"  Positives (ECG)   : {len(ecg_paths)}")
    print(f"  Negatives (non-ECG): {len(non_ecg_paths)}")
    hard = sum(1 for p in non_ecg_paths if 'hard_neg' in os.path.basename(p))
    print(f"  Of which hard-neg  : {hard}")

    paths  = ecg_paths + non_ecg_paths
    labels = [1] * len(ecg_paths) + [0] * len(non_ecg_paths)
    return train_test_split(paths, labels, test_size=0.15, random_state=42,
                            stratify=labels)


def preprocess_path(path, label):
    """Exact same fixed-threshold pipeline used in backend inference."""
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [224, 224])
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    return img, label


def build_model(freeze_base=True):
    base = MobileNetV2(input_shape=(224,224,3), include_top=False, weights='imagenet')
    base.trainable = not freeze_base
    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    out = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=base.input, outputs=out)
    model.compile(
        optimizer=Adam(1e-3),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    return model, base


def main():
    print("="*55)
    print("Gatekeeper Retraining with Hard Negatives")
    print("="*55)

    tr_paths, val_paths, tr_labels, val_labels = load_data()
    print(f"\nTrain: {len(tr_paths)}   Val: {len(val_paths)}")

    bs = 32
    tr_ds = (tf.data.Dataset.from_tensor_slices((tr_paths, tr_labels))
             .shuffle(len(tr_paths), seed=42)
             .map(preprocess_path, num_parallel_calls=tf.data.AUTOTUNE)
             .batch(bs).prefetch(tf.data.AUTOTUNE))

    val_ds = (tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
              .map(preprocess_path, num_parallel_calls=tf.data.AUTOTUNE)
              .batch(bs).prefetch(tf.data.AUTOTUNE))

    os.makedirs("models", exist_ok=True)
    save_path = "models/ecg_detector.keras"

    model, base = build_model(freeze_base=True)
    print(f"\nPhase 1: Train head only (base frozen) — 8 epochs")
    model.fit(
        tr_ds, validation_data=val_ds, epochs=8,
        callbacks=[
            ModelCheckpoint(save_path, save_best_only=True, monitor="val_auc",
                            mode="max", verbose=1),
            EarlyStopping(patience=3, restore_best_weights=True,
                          monitor="val_auc", mode="max"),
        ],
        verbose=1
    )

    print(f"\nPhase 2: Fine-tune top 30 layers — 5 epochs")
    for layer in base.layers[-30:]:
        layer.trainable = True
    model.compile(
        optimizer=Adam(1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    model.fit(
        tr_ds, validation_data=val_ds, epochs=5,
        callbacks=[
            ModelCheckpoint(save_path, save_best_only=True, monitor="val_auc",
                            mode="max", verbose=1),
            EarlyStopping(patience=3, restore_best_weights=True,
                          monitor="val_auc", mode="max"),
            ReduceLROnPlateau(patience=2, factor=0.5, monitor="val_auc", mode="max"),
        ],
        verbose=1
    )

    # ── Evaluate on validation set ────────────────────────────────────────────
    print("\nValidation evaluation:")
    from tensorflow.keras.models import load_model as lm
    best = lm(save_path)

    preds, trues = [], []
    for imgs, labs in val_ds:
        p = best.predict(imgs, verbose=0).flatten()
        preds.extend((p >= 0.5).astype(int).tolist())
        trues.extend(labs.numpy().tolist())

    cm = confusion_matrix(trues, preds)
    print("Confusion matrix (rows=true, cols=pred):")
    print("              pred-NEG  pred-ECG")
    print(f"  true-NEG      {cm[0,0]:5d}     {cm[0,1]:5d}")
    print(f"  true-ECG      {cm[1,0]:5d}     {cm[1,1]:5d}")

    tn, fp, fn, tp = cm.ravel()
    print(f"\n  Sensitivity (recall on ECG) : {tp/(tp+fn):.3f}")
    print(f"  Specificity (recall on non) : {tn/(tn+fp):.3f}")
    print(f"  False-accept rate           : {fp/(tn+fp):.3f}")

    print(classification_report(trues, preds,
          target_names=["Non-ECG","ECG"], digits=4))

    # Copy to backend/models/ for the live server
    backend_path = "backend/models/ecg_detector.keras"
    import shutil
    shutil.copy2(save_path, backend_path)
    print(f"\nCopied → {backend_path}")
    print(f"Saved  → {save_path}")
    print("✅ Gatekeeper retraining complete.")


if __name__ == "__main__":
    main()
