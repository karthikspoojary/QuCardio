import os
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from PIL import Image, ImageDraw, ImageFont
import random

NUM_IMAGES_WEB = 300
NUM_IMAGES_TEXT = 200
SAVE_DIR = "data/NON_ECG_DATA"

def download_image(i):
    url = f"https://picsum.photos/224/224?random={i}"
    save_path = os.path.join(SAVE_DIR, f"non_ecg_web_{i:03d}.jpg")
    try:
        urllib.request.urlretrieve(url, save_path)
    except Exception as e:
        print(f"Failed to download web image {i}: {e}")

def create_synthetic_text_image(i):
    # Create an image with a light background to simulate paper/screenshots
    bg_color = (random.randint(220, 255), random.randint(220, 255), random.randint(220, 255))
    img = Image.new('RGB', (224, 224), color=bg_color)
    d = ImageDraw.Draw(img)
    
    # Draw some horizontal-ish text lines (blocks of dark pixels)
    # This simulates text documents which might fool basic heuristics
    for _ in range(random.randint(5, 15)):
        y = random.randint(10, 200)
        x_start = random.randint(10, 50)
        x_end = random.randint(150, 210)
        thickness = random.randint(2, 6)
        color = (random.randint(0, 50), random.randint(0, 50), random.randint(0, 50))
        # Draw a simulated line of text (dashed or continuous)
        if random.random() > 0.5:
            # continuous block (like redacted text)
            d.rectangle([x_start, y, x_end, y + thickness], fill=color)
        else:
            # dashed (like words)
            curr_x = x_start
            while curr_x < x_end:
                word_len = random.randint(5, 30)
                d.rectangle([curr_x, y, min(curr_x + word_len, x_end), y + thickness], fill=color)
                curr_x += word_len + random.randint(3, 10)
                
    save_path = os.path.join(SAVE_DIR, f"non_ecg_text_{i:03d}.jpg")
    img.save(save_path)

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    print(f"Creating {NUM_IMAGES_TEXT} synthetic text document images...")
    for i in range(NUM_IMAGES_TEXT):
        create_synthetic_text_image(i)
        
    print(f"Downloading {NUM_IMAGES_WEB} non-ECG random images from web...")
    with ThreadPoolExecutor(max_workers=10) as executor:
        list(executor.map(download_image, range(NUM_IMAGES_WEB)))
        
    print("Done preparing non-ECG OOD data.")

if __name__ == "__main__":
    main()
