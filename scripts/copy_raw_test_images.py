"""
Copy the exact 186 raw (original, unprocessed) test-set images to
data/test_data/ using the same train_test_split seed as training.

The split is re-derived from processed_340 filenames (same as training),
then the corresponding raw images are copied by matching filenames.
"""

import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split

PROCESSED_DIR = Path('data/processed_340')   # used to derive the split
RAW_DIR       = Path('data/raw')             # source of originals
OUTPUT_DIR    = Path('data/test_data')       # destination

CLASS_NAMES  = ['Normal', 'Arrhythmia', 'Myocardial_Infarction', 'History_of_MI']
CLASS_MAP    = {'Normal': 0, 'Arrhythmia': 1, 'Myocardial_Infarction': 2, 'History_of_MI': 3}
TEST_SIZE    = 0.2
RANDOM_STATE = 42

# ── Raw class sub-folder names may differ from class names ─────────────────
# Map CLASS_NAME → actual subfolder name in data/raw/
RAW_SUBFOLDER = {
    'Normal':                'Normal',
    'Arrhythmia':            'Arrhythmia',
    'Myocardial_Infarction': 'Myocardial_Infarction',
    'History_of_MI':         'History_of_MI',
}

def main():
    all_paths, all_labels = [], []

    for cls in CLASS_NAMES:
        cls_dir = PROCESSED_DIR / cls
        if not cls_dir.exists():
            print(f"  Skipping missing processed dir: {cls_dir}")
            continue
        images = sorted(list(cls_dir.glob('*.jpg')) + list(cls_dir.glob('*.png')))
        for img_path in images:
            all_paths.append(img_path)
            all_labels.append(CLASS_MAP[cls])

    print(f"Total processed images found : {len(all_paths)}")

    _, test_paths, _, _ = train_test_split(
        all_paths, all_labels,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=all_labels
    )

    print(f"Test set size               : {len(test_paths)} images")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for cls in CLASS_NAMES:
        (OUTPUT_DIR / cls).mkdir(exist_ok=True)

    counts  = {cls: 0 for cls in CLASS_NAMES}
    missing = []

    for proc_path in test_paths:
        cls_name  = proc_path.parent.name
        raw_fname = proc_path.name
        raw_sub   = RAW_SUBFOLDER.get(cls_name, cls_name)
        raw_path  = RAW_DIR / raw_sub / raw_fname

        if not raw_path.exists():
            # Try common extensions
            found = False
            for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                alt = raw_path.with_suffix(ext)
                if alt.exists():
                    raw_path = alt
                    found = True
                    break
            if not found:
                missing.append(raw_path)
                continue

        shutil.copy2(raw_path, OUTPUT_DIR / cls_name / raw_fname)
        counts[cls_name] += 1

    print(f"\nDone! Raw images copied to: {OUTPUT_DIR}/")
    print(f"\nBreakdown by class:")
    for cls, count in counts.items():
        print(f"   {cls:30s} -> {count} images")
    print(f"\nTotal: {sum(counts.values())} test images copied.")

    if missing:
        print(f"\n⚠️  {len(missing)} raw images NOT found (listed below):")
        for p in missing[:20]:
            print(f"   {p}")

if __name__ == "__main__":
    main()
