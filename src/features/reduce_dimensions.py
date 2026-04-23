import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path

def reduce_features():
    """
    Reduce features to 9 using SVD
    
    QuCardio Paper (Section IV-A1):
    - "Dimensionality reduced using Truncated SVD"
    - "Yields top 9 most relevant features"
    - "Feature matrix scaled to range 0 to 1"
    """
    if not Path('data/resnet50_features.npz').exists():
        print("data/resnet50_features.npz not found! Run feature extraction first.")
        return
        
    print("Loading ResNet50 features...")
    data = np.load('data/resnet50_features.npz')
    
    train_features = data['train_features']
    test_features = data['test_features']
    y_train = data['y_train']
    y_test = data['y_test']
    
    print(f"Original train shape: {train_features.shape}")
    
    # Apply SVD (reduce to 9 features)
    print("\\nApplying Truncated SVD (reduction to 9 dimensions)...")
    svd = TruncatedSVD(n_components=9, random_state=42)
    
    train_reduced = svd.fit_transform(train_features)
    test_reduced = svd.transform(test_features)
    
    print(f"Reduced train shape: {train_reduced.shape}")
    print(f"Explained variance: {svd.explained_variance_ratio_.sum():.2%}")
    
    # Scale to [0, 1]
    print("\\nScaling to [0, 1]...")
    scaler = MinMaxScaler()
    
    train_scaled = scaler.fit_transform(train_reduced)
    test_scaled = scaler.transform(test_reduced)
    
    # Save the reducers for inference later
    print("\\nSaving SVD and Scaler models for backend usage...")
    import joblib
    Path('backend/models').mkdir(parents=True, exist_ok=True)
    joblib.dump(svd, 'backend/models/svd_reducer.pkl')
    joblib.dump(scaler, 'backend/models/minmax_scaler.pkl')
    
    # Save reduced features
    print("\\nSaving reduced features...")
    np.savez_compressed(
        'data/features_9d.npz',
        train_features=train_scaled,
        test_features=test_scaled,
        y_train=y_train,
        y_test=y_test
    )
    
    print(f"\\n✅ Done! Features reduced to 9 dimensions")
    print(f"   Saved to: data/features_9d.npz")
    print(f"   Models saved to: backend/models/")

if __name__ == "__main__":
    import time
    start = time.time()
    reduce_features()
    print(f"\\n⏱️ Time: {time.time() - start:.1f} seconds")
