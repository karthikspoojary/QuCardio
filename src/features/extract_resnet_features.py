import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
import tensorflow as tf
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.models import Model
import gc
def load_dataset(data_dir, image_size=(340, 340)):
    """Load preprocessed ECG images"""
    data_path = Path(data_dir)
    
    class_names = ['Normal', 'Arrhythmia', 'Myocardial_Infarction', 'History_of_MI']
    class_mapping = {'Normal': 0, 'Arrhythmia': 1, 
                     'Myocardial_Infarction': 2, 'History_of_MI': 3}
    
    X = []
    y = []
    loaded_classes = [] 
    
    for cls in class_names:
        cls_dir = data_path / cls
        if not cls_dir.exists():
            continue
        
        images = list(cls_dir.glob('*.jpg')) + list(cls_dir.glob('*.png'))
        if images:
            loaded_classes.append(cls)
            
        for img_path in tqdm(images, desc=f"Loading {cls}"):
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            
            if img.shape != image_size:
                img = cv2.resize(img, image_size)
            
            X.append(img)
            y.append(class_mapping[cls])
    
    X = np.array(X)
    y = np.array(y)
    
    # Normalize
    X = X.astype(np.float32)
    
    return X, y, loaded_classes


def extract_resnet50_features(X_train, X_test):
    """Highly Memory-optimized feature extraction with manual batching"""
    print("\\n" + "="*60)
    print("RESNET50 FEATURE EXTRACTION (pool1_pool)")
    print("="*60)
    
    print("Loading PRE-TRAINED ResNet50...")
    base_model = ResNet50(weights='imagenet', include_top=False)
    
    try:
        pool1_layer = base_model.get_layer('pool1_pool').output
    except ValueError:
        pool1_layer = base_model.layers[4].output
        
    resnet_pool1 = Model(inputs=base_model.input, outputs=pool1_layer)
    print(f"        Output shape per image: {resnet_pool1.output_shape}")
    
    # We know the pool1 output shape for 340x340 input is (85, 85, 64)
    # 85 * 85 * 64 = 462400
    feature_dim = 85 * 85 * 64
    batch_size = 8
    
    def extract_in_batches(X_data, desc="Processing"):
        """Process data in small batches and directly insert into pre-allocated flat array"""
        num_samples = len(X_data)
        # Pre-allocate 2D array to avoid massive memory spikes during reshape
        features_flat = np.empty((num_samples, feature_dim), dtype=np.float32)
        
        for i in tqdm(range(0, num_samples, batch_size), desc=desc):
            end = min(i + batch_size, num_samples)
            batch = X_data[i:end]
            
            # 1. Add channel dim and repeat to RGB
            batch_rgb = np.repeat(batch[..., np.newaxis], 3, axis=-1)
            
            # 2. Preprocess for ImageNet
            batch_prep = preprocess_input(batch_rgb)
            
            # 3. Predict (returns 3D tensors: batch x 85 x 85 x 64)
            batch_features_3d = resnet_pool1.predict_on_batch(batch_prep)
            
            # 4. Flatten and assign directly to pre-allocated array (no massive copy)
            features_flat[i:end] = batch_features_3d.reshape(batch_features_3d.shape[0], -1)
            
            # Force cleanup
            del batch, batch_rgb, batch_prep, batch_features_3d
            
        return features_flat

    print(f"\\nStep 1: Processing Train Set ({len(X_train)} images)...")
    train_features = extract_in_batches(X_train, desc="Train Batches")
    gc.collect()

    print(f"\\nStep 2: Processing Test Set ({len(X_test)} images)...")
    test_features = extract_in_batches(X_test, desc="Test Batches")
    gc.collect()
    
    print(f"\\n✅ Feature extraction complete!")
    print(f"   Train features shape: {train_features.shape}")
    print(f"   Test features shape:  {test_features.shape}")
    
    return train_features, test_features

def main():
    """Main pipeline"""
    import time
    total_start = time.time()
    
    # Load data
    print("Loading preprocessed images...")
    X, y, class_names = load_dataset('data/processed_340', image_size=(340, 340))
    
    if len(X) == 0:
        print("No images found in data/processed_340! Did you run preprocessing?")
        return
        
    print(f"\\n✅ Dataset loaded:")
    print(f"   Total images: {len(X)}")
    print(f"   Classes: {class_names}")
    print(f"   Shape: {X.shape}")
    
    # Split 80:20 (as per QuCardio paper)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\\n✅ Split complete (80:20):")
    print(f"   Train: {len(X_train)} images")
    print(f"   Test:  {len(X_test)} images")
    
    # Extract ResNet50 features
    train_features, test_features = extract_resnet50_features(X_train, X_test)
    
    # Save features
    print("\\nSaving extracted features...")
    Path('data').mkdir(exist_ok=True)
    np.savez_compressed(
        'data/resnet50_features.npz',
        train_features=train_features,
        test_features=test_features,
        y_train=y_train,
        y_test=y_test,
        class_names=class_names
    )
    
    total_elapsed = time.time() - total_start
    
    print(f"\\n{'='*60}")
    print(f"✅ ALL DONE!")
    print(f"{'='*60}")
    print(f"Total time: {total_elapsed/60:.1f} minutes")
    print(f"\\nFeatures saved to: data/resnet50_features.npz")

if __name__ == "__main__":
    main()
