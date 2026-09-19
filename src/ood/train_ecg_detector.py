import os
import glob
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split

def load_data(ecg_dir="data/ECG_DATA", non_ecg_dir="data/NON_ECG_DATA"):
    # 1 for ECG, 0 for Non-ECG
    ecg_paths = glob.glob(os.path.join(ecg_dir, "**", "*.jpg"), recursive=True)
    non_ecg_paths = glob.glob(os.path.join(non_ecg_dir, "*.jpg"))
    
    # We might have too many ECG images (~8000), to balance it a bit we can sample
    # Let's use up to 2000 ECG images and all non-ECG
    # Sort before shuffling: glob.glob ordering is filesystem-dependent, so the same
    # seed produces the same subset only if the input list is in a deterministic order.
    ecg_paths = sorted(ecg_paths)
    np.random.seed(42)  # Seed for deterministic subset selection
    np.random.shuffle(ecg_paths)
    ecg_paths = ecg_paths[:2000]
    
    paths = ecg_paths + non_ecg_paths
    labels = [1] * len(ecg_paths) + [0] * len(non_ecg_paths)
    
    return train_test_split(paths, labels, test_size=0.15, random_state=42, stratify=labels)

def parse_image(filename, label):
    image = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [224, 224])
    # MobileNetV2 expects [-1, 1] input, we can do it via preprocess_input or manual
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    return image, label

def build_model():
    base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
    base_model.trainable = False  # Freeze base model
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.2)(x)
    predictions = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer=Adam(learning_rate=0.001), 
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])
    return model

def main():
    train_paths, val_paths, train_labels, val_labels = load_data()
    print(f"Training on {len(train_paths)} images, Validating on {len(val_paths)} images")
    
    batch_size = 32
    
    train_ds = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
    train_ds = train_ds.shuffle(len(train_paths)).map(parse_image, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    val_ds = tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
    val_ds = val_ds.map(parse_image, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    model = build_model()
    
    os.makedirs("models", exist_ok=True)
    callbacks = [
        ModelCheckpoint("models/ecg_detector.keras", save_best_only=True, monitor="val_accuracy"),
        EarlyStopping(patience=3, restore_best_weights=True)
    ]
    
    model.fit(train_ds, validation_data=val_ds, epochs=10, callbacks=callbacks)
    
    # ModelCheckpoint already saved the best model to models/ecg_detector.keras
    # EarlyStopping restored the best weights into memory.
    # We do NOT want to blindly save the final epoch over the best checkpoint.
    print("Training complete. Best ECG detector saved to models/ecg_detector.keras")

if __name__ == "__main__":
    main()
