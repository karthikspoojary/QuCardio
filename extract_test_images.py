"""
Extract the exact 186 test-set images used during training
and copy them to data/test_samples/ for demo/dashboard use.
"""

import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split

DATA_DIR     = Path('data/processed_340')
OUTPUT_DIR   = Path('data/test_samples')
CLASS_NAMES  = ['Normal', 'Arrhythmia', 'Myocardial_Infarction', 'History_of_MI']
CLASS_MAP    = {'Normal': 0, 'Arrhythmia': 1, 'Myocardial_Infarction': 2, 'History_of_MI': 3}
TEST_SIZE    = 0.2
RANDOM_STATE = 42

def main():
    all_paths, all_labels = [], []

    for cls in CLASS_NAMES:
        cls_dir = DATA_DIR / cls
        if not cls_dir.exists():
            print(f"  Skipping missing: {cls_dir}")
            continue
        images = sorted(list(cls_dir.glob('*.jpg')) + list(cls_dir.glob('*.png')))
        for img_path in images:
            all_paths.append(img_path)
            all_labels.append(CLASS_MAP[cls])

    print(f"Total images found : {len(all_paths)}")

    _, test_paths, _, _ = train_test_split(
        all_paths, all_labels,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=all_labels
    )

    print(f"Test set size      : {len(test_paths)} images")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for cls in CLASS_NAMES:
        (OUTPUT_DIR / cls).mkdir(exist_ok=True)

    counts = {cls: 0 for cls in CLASS_NAMES}
    for img_path in test_paths:
        cls_name = img_path.parent.name
        shutil.copy2(img_path, OUTPUT_DIR / cls_name / img_path.name)
        counts[cls_name] += 1

    print(f"\nDone! Images copied to: {OUTPUT_DIR}/")
    print(f"\nBreakdown by class:")
    for cls, count in counts.items():
        print(f"   {cls:30s} -> {count} images")
    print(f"\nTotal: {sum(counts.values())} test images ready for demo.")

if __name__ == "__main__":
    main()
