import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

def preprocess_ecg(image_path, output_size=(340, 340)):
    """
    Robust preprocessing pipeline (QuCardio paper Section III-B)
    Uses OTSU adaptive thresholding to handle diverse image styles 
    (outside/internet images with different backgrounds, colors, paper types).
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Cannot read: {image_path}")

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # --- Step 1: Adaptive crop to ROI (remove white borders) ---
    # OTSU automatically computes threshold — works on any paper color
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Morphological dilation to connect ECG signal fragments into one big blob
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
    dilated = cv2.dilate(binary, kernel, iterations=2)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Find the largest contour — this is the ECG signal area
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        padding = 15
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(gray.shape[1] - x, w + 2 * padding)
        h = min(gray.shape[0] - y, h + 2 * padding)
        cropped = gray[y:y + h, x:x + w]
    else:
        cropped = gray

    # --- Step 2: Adaptive background removal ---
    # OTSU again on the cropped region for accurate binarization
    _, cleaned = cv2.threshold(cropped, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # If the ECG signal is bright (white on black background), invert it
    # Heuristic: if more white pixels than black, the signal is inverted
    white_pixels = np.sum(cleaned == 255)
    black_pixels = np.sum(cleaned == 0)
    if white_pixels > black_pixels:
        cleaned = cv2.bitwise_not(cleaned)

    # --- Step 3: Remove vertical grid lines ---
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    vertical_lines = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel_v)
    cleaned = cv2.subtract(cleaned, vertical_lines)

    # --- Step 4: Remove horizontal grid lines ---
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    horizontal_lines = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel_h)
    cleaned = cv2.subtract(cleaned, horizontal_lines)

    # --- Step 5: Light denoise to smooth the ECG trace ---
    cleaned = cv2.GaussianBlur(cleaned, (3, 3), 0)
    _, cleaned = cv2.threshold(cleaned, 30, 255, cv2.THRESH_BINARY)

    # --- Step 6: Resize ---
    resized = cv2.resize(cleaned, output_size, interpolation=cv2.INTER_AREA)

    # --- Step 7: Normalize to [0, 1] ---
    normalized = resized.astype(np.float32) / 255.0

    return normalized


def preprocess_all_images():
    """
    Preprocess entire dataset
    Time: ~20-30 minutes for 929 images
    """
    input_dir = Path('data/raw')

    output_dir_340 = Path('data/processed_340')
    output_dir_340.mkdir(parents=True, exist_ok=True)

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

        (output_dir_340 / cls).mkdir(exist_ok=True)
        (output_dir_64 / cls).mkdir(exist_ok=True)

        print(f"\nProcessing {cls}...")
        for img_path in tqdm(images, desc=cls):
            try:
                img_340 = preprocess_ecg(img_path, output_size=(340, 340))
                out_path_340 = output_dir_340 / cls / img_path.name
                cv2.imwrite(str(out_path_340), (img_340 * 255).astype(np.uint8))

                img_64 = preprocess_ecg(img_path, output_size=(64, 64))
                out_path_64 = output_dir_64 / cls / img_path.name
                cv2.imwrite(str(out_path_64), (img_64 * 255).astype(np.uint8))

            except Exception as e:
                print(f"Error: {img_path.name} - {e}")

    print(f"\n✅ Preprocessing complete!")
    print(f"Total images processed: {total_images}")
    print(f"Output directories:")
    print(f"  - {output_dir_340}")
    print(f"  - {output_dir_64}")


if __name__ == "__main__":
    import time
    start = time.time()
    preprocess_all_images()
    elapsed = time.time() - start
    print(f"\n⏱️ Total time: {elapsed / 60:.1f} minutes")
