import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

def preprocess_ecg(image_path, output_size=(340, 340)):
    """
    Complete preprocessing pipeline (from QuCardio paper Section III-B)
    Time: ~2 seconds per image
    """
    # Read image
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Cannot read: {image_path}")
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Step 1: Crop to ROI (remove borders)
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        padding = 10
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(gray.shape[1] - x, w + 2*padding)
        h = min(gray.shape[0] - y, h + 2*padding)
        cropped = gray[y:y+h, x:x+w]
    else:
        cropped = gray
    
    # Step 2: Remove background (threshold)
    _, cleaned = cv2.threshold(cropped, 200, 255, cv2.THRESH_BINARY)
    cleaned = cv2.bitwise_not(cleaned)
    
    # Step 3: Remove vertical grid lines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 50))
    vertical_lines = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.subtract(cleaned, vertical_lines)
    
    # Step 4: Resize
    resized = cv2.resize(cleaned, output_size, interpolation=cv2.INTER_AREA)
    
    # Step 5: Normalize to [0, 1]
    normalized = resized.astype(np.float32) / 255.0
    
    return normalized

def preprocess_all_images():
    """
    Preprocess entire dataset
    Time: ~20-30 minutes for 929 images
    """
    input_dir = Path('data/raw')
    
    # Process for QSVC (340x340)
    output_dir_340 = Path('data/processed_340')
    output_dir_340.mkdir(parents=True, exist_ok=True)
    
    # Process for QNN (64x64)
    output_dir_64 = Path('data/processed_64')
    output_dir_64.mkdir(parents=True, exist_ok=True)
    
    classes = ['Normal', 'Arrhythmia', 'Myocardial_Infarction', 'History_of_MI']
    
    total_images = 0
    for cls in classes:
        cls_dir = input_dir / cls
        if not cls_dir.exists():
            continue
        
        images = list(cls_dir.glob('*.jpg')) + list(cls_dir.glob('*.png'))
        total_images += len(images)
        
        # Create output directories
        (output_dir_340 / cls).mkdir(exist_ok=True)
        (output_dir_64 / cls).mkdir(exist_ok=True)
        
        print(f"\\nProcessing {cls}...")
        for img_path in tqdm(images, desc=cls):
            try:
                # Process for QSVC (340x340)
                img_340 = preprocess_ecg(img_path, output_size=(340, 340))
                out_path_340 = output_dir_340 / cls / img_path.name
                cv2.imwrite(str(out_path_340), (img_340 * 255).astype(np.uint8))
                
                # Process for QNN (64x64)
                img_64 = preprocess_ecg(img_path, output_size=(64, 64))
                out_path_64 = output_dir_64 / cls / img_path.name
                cv2.imwrite(str(out_path_64), (img_64 * 255).astype(np.uint8))
                
            except Exception as e:
                print(f"Error: {img_path.name} - {e}")
    
    print(f"\\n✅ Preprocessing complete!")
    print(f"Total images processed: {total_images}")
    print(f"Output directories:")
    print(f"  - {output_dir_340}")
    print(f"  - {output_dir_64}")

if __name__ == "__main__":
    import time
    start = time.time()
    preprocess_all_images()
    elapsed = time.time() - start
    print(f"\\n⏱️ Total time: {elapsed/60:.1f} minutes")
