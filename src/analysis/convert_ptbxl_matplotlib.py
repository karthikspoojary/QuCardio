"""
convert_ptbxl_matplotlib.py — PTB-XL → 340×340 ECG images (matplotlib, no ecg-image-kit)
===========================================================================================
Reads PTB-XL WFDB records, renders 12-lead ECG charts with matplotlib, and saves
labelled images to data/ptb-xl-images/{Normal,Arrhythmia,Myocardial_Infarction,History_of_MI}/

FILTER: only strat_fold ∈ {9, 10}  → ~2,178 human-validated test records
  (folds 9 & 10 are PTB-XL's recommended held-out set — never used in their training)

LABEL MAPPING (PTB-XL superclasses → QuCardio 4-class schema):
  NORM → Normal             (0)
  MI   → Myocardial_Infarction  (1)   # rename to MI folder for clarity
  CD   → Arrhythmia         (2)   # Conduction Disturbance ≈ rhythm disorder
  STTC → History_of_MI      (3)   # ST/T-wave change — matches "History of MI" clinically
  HYP  → skipped            (HYP = hypertrophy — no direct 4-class match)
  Records with multiple superclasses: first-match priority NORM > MI > CD > STTC

SIGNAL: uses records100 (100 Hz × 10 s = 1,000 samples/lead)
  → renders identical waveform detail to 500 Hz for image classification

OUTPUT:
  data/ptb-xl-images/
    Normal/               PNG images
    Arrhythmia/
    Myocardial_Infarction/
    History_of_MI/
  data/ptb-xl-images/manifest.json   (record_id, label, source_file)

PREREQUISITES:
  pip install wfdb
  AWS S3 download (run once — ~500 MB):
    aws s3 sync --no-sign-request s3://physionet-open/ptb-xl/1.0.3/records100/ data/ptb-xl/records100/
    aws s3 cp --no-sign-request s3://physionet-open/ptb-xl/1.0.3/ptbxl_database.csv data/ptb-xl/
    aws s3 cp --no-sign-request s3://physionet-open/ptb-xl/1.0.3/scp_statements.csv data/ptb-xl/

USAGE:
  python src/analysis/convert_ptbxl_matplotlib.py
  python src/analysis/convert_ptbxl_matplotlib.py --ptbxl-dir data/ptb-xl --out-dir data/ptb-xl-images
"""

import argparse
import json
import sys
import ast
from pathlib import Path

import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use('Agg')   # non-interactive backend — no display needed
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm

# ---------------------------------------------------------------------------
# PTB-XL superclass → QuCardio 4-class mapping
# Priority order when a record has multiple superclasses: NORM > MI > CD > STTC
# HYP is skipped (no clinical equivalent in our 4-label schema)
# ---------------------------------------------------------------------------
SUPERCLASS_PRIORITY = ['NORM', 'MI', 'CD', 'STTC']   # HYP absent → skipped

SUPERCLASS_TO_LABEL = {
    'NORM': ('Normal',                0),
    'MI':   ('Myocardial_Infarction', 1),
    'CD':   ('Arrhythmia',            2),
    'STTC': ('History_of_MI',         3),
}

# Lead names in PTB-XL (standard order from wfdb)
LEAD_NAMES = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF',
              'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

# Output image size — must match QuCardio pipeline (340 × 340 px)
IMG_SIZE_PX = 340
DPI = 100   # 340/100 = 3.40 inch figure → exactly 340 px when saved at DPI=100


def parse_scp_codes(scp_str: str) -> dict:
    """
    Parse the scp_codes column (stored as string repr of a dict).
    Returns {code_str: likelihood_float}, e.g. {'NORM': 100.0, 'SR': 0.0}.
    """
    try:
        return ast.literal_eval(scp_str)
    except Exception:
        return {}


def get_superclass(scp_codes: dict, scp_df: pd.DataFrame) -> tuple[str, str] | None:
    """
    Map an scp_codes dict to (folder_name, superclass_str) using priority order.
    Returns None if no mappable superclass is found (e.g. HYP-only records).
    """
    # Build set of superclasses present in this record
    # Note: scp_statements.csv uses 'diagnostic_class' (or 'superclass' in some mirrors)
    col_name = 'diagnostic_class' if 'diagnostic_class' in scp_df.columns else ('superclass' if 'superclass' in scp_df.columns else None)
    present_superclasses = set()
    for code in scp_codes:
        if code in scp_df.index and col_name is not None:
            sc = scp_df.loc[code, col_name]
            if pd.notna(sc) and sc in SUPERCLASS_TO_LABEL:
                present_superclasses.add(sc)

    # Select highest-priority superclass
    for sc in SUPERCLASS_PRIORITY:
        if sc in present_superclasses:
            folder, _ = SUPERCLASS_TO_LABEL[sc]
            return folder, sc

    return None


def _thin_to_training_style(png_path: Path) -> None:
    """
    Post-process a rendered ECG PNG so its pixel density matches the training images.

    Training images (data/processed_340/) have ~2–3% waveform pixels — they are
    sparse thin lines from scanned paper ECG prints.  matplotlib renders have ~12%
    because the anti-aliased 0.4pt line produces a fat brush at 100 DPI.

    Approach:
      1. Load the PNG (white bg, black waveform)
      2. Threshold to binary (black line = 1, white bg = 0)
      3. Skeletonize: erode the line to 1-px width (matches scanner output)
      4. JPEG round-trip: replicate the compression artefact in preprocess_ecg.py
         (the training pipeline saved every processed image as JPEG then reloaded)
      5. Overwrite the PNG with the thinned version

    Result: ~2–3% waveform density, matching training image pixel statistics.
    """
    img = cv2.imread(str(png_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return

    # Resize to exact 340×340 if needed (bbox_inches='tight' may vary slightly)
    if img.shape != (IMG_SIZE_PX, IMG_SIZE_PX):
        img = cv2.resize(img, (IMG_SIZE_PX, IMG_SIZE_PX), interpolation=cv2.INTER_AREA)

    # Identify waveform pixels: black ink on white background (value < 128 = ink)
    _, binary = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV)   # ink = 255

    # Thin with a 3×3 erosion — one pass is enough to strip anti-aliasing fuzz
    # without removing the actual 1-px skeleton of the waveform
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    thinned = cv2.erode(binary, kernel, iterations=1)

    # Reconstruct: white background + thin black waveform
    result = np.full((IMG_SIZE_PX, IMG_SIZE_PX), 255, dtype=np.uint8)
    result[thinned > 0] = 0     # black ink where waveform exists

    # JPEG round-trip — replicates what preprocess_ecg.py did to training images
    # (saves then reloads, introducing mild quantization that the SVD was fitted on)
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as jtmp:
        jpeg_path = jtmp.name
    cv2.imwrite(jpeg_path, result)
    result_jpeg = cv2.imread(jpeg_path, cv2.IMREAD_GRAYSCALE)
    os.remove(jpeg_path)

    # Save back as PNG (the file extension stays .png; the pixel values now match
    # the JPEG-round-tripped training images that ResNet50 + SVD were fitted on)
    cv2.imwrite(str(png_path), result_jpeg)


def render_ecg_image(signal: np.ndarray, record_name: str, label: str,
                     out_path: Path, fs: int = 100) -> bool:
    """
    Render a 12-lead ECG signal as a 340×340 PNG compatible with the QuCardio pipeline.

    Layout: 2 columns × 6 rows (lead I–VI left, lead V1–V6 right).
    Background: white. Waveform: black. No axes, no labels, no grid text.

    After matplotlib rendering, applies _thin_to_training_style() to reduce the
    waveform pixel density from ~12% (matplotlib anti-aliased) to ~2–3%
    (matching the Kaggle scanned ECG training images).

    Parameters
    ----------
    signal : np.ndarray, shape (n_samples, 12)
    record_name : str
    label : str — class folder name
    out_path : Path — where to save the PNG
    fs : int — sampling frequency (100 for records100)

    Returns
    -------
    bool — True if saved successfully
    """
    n_samples, n_leads = signal.shape
    if n_leads != 12:
        return False

    t = np.arange(n_samples) / fs   # time axis in seconds

    fig_size = IMG_SIZE_PX / DPI    # 3.40 inches
    fig = plt.figure(figsize=(fig_size, fig_size), facecolor='white')

    # 2-column, 6-row grid with very tight spacing
    gs = gridspec.GridSpec(6, 2, figure=fig,
                           hspace=0.05, wspace=0.02,
                           left=0.01, right=0.99,
                           top=0.99, bottom=0.01)

    # Lead layout: left column = leads 0–5 (I,II,III,aVR,aVL,aVF)
    #              right column = leads 6–11 (V1–V6)
    for row in range(6):
        for col in range(2):
            lead_idx = col * 6 + row
            ax = fig.add_subplot(gs[row, col])
            lead_signal = signal[:, lead_idx]

            ax.plot(t, lead_signal, color='black', linewidth=0.4, antialiased=True)
            ax.set_xlim(t[0], t[-1])

            # No axes decorations — clean waveform only
            ax.axis('off')

    plt.savefig(str(out_path), dpi=DPI, bbox_inches='tight',
                pad_inches=0, facecolor='white')
    plt.close(fig)

    # Post-process: thin waveform to match training image pixel density
    _thin_to_training_style(out_path)
    return True


def convert_ptbxl(ptbxl_dir: Path, out_dir: Path,
                  fold_set: list | None = None) -> dict:
    """
    Main conversion pipeline.

    1. Load ptbxl_database.csv and scp_statements.csv
    2. Filter to strat_fold ∈ fold_set (default [9, 10])
    3. Map each record to a 4-class label (skip HYP-only)
    4. Render matplotlib ECG image → out_dir/<label>/<record>.png
    5. Write manifest.json

    Returns summary dict.
    """
    if fold_set is None:
        fold_set = [9, 10]
    import wfdb   # imported here so the module can be imported without wfdb installed

    # ── Load metadata ────────────────────────────────────────────────────────
    csv_path = ptbxl_dir / 'ptbxl_database.csv'
    scp_path = ptbxl_dir / 'scp_statements.csv'

    if not csv_path.exists():
        raise FileNotFoundError(
            f"ptbxl_database.csv not found at {csv_path}\n"
            "Download it:\n"
            "  aws s3 cp --no-sign-request "
            "s3://physionet-open/ptb-xl/1.0.3/ptbxl_database.csv data/ptb-xl/"
        )
    if not scp_path.exists():
        raise FileNotFoundError(
            f"scp_statements.csv not found at {scp_path}\n"
            "Download it:\n"
            "  aws s3 cp --no-sign-request "
            "s3://physionet-open/ptb-xl/1.0.3/scp_statements.csv data/ptb-xl/"
        )

    db = pd.read_csv(csv_path, index_col='ecg_id')
    scp_df = pd.read_csv(scp_path, index_col=0)

    # ── Filter to requested folds ────────────────────────────────────────────
    db_folds = db[db['strat_fold'].isin(fold_set)].copy()
    print(f"Records in folds {fold_set}: {len(db_folds)}")

    # ── Create output directories ────────────────────────────────────────────
    for folder, _ in SUPERCLASS_TO_LABEL.values():
        (out_dir / folder).mkdir(parents=True, exist_ok=True)

    # ── Process each record ──────────────────────────────────────────────────
    manifest = []
    counts = {folder: 0 for folder, _ in SUPERCLASS_TO_LABEL.values()}
    skipped_hf = 0       # no matching superclass (HYP-only or unknown)
    skipped_io = 0       # file read / signal errors
    already_done = 0

    records100_dir = ptbxl_dir / 'records100'
    if not records100_dir.exists():
        raise FileNotFoundError(
            f"records100/ not found at {records100_dir}\n"
            "Download it:\n"
            "  aws s3 sync --no-sign-request "
            "s3://physionet-open/ptb-xl/1.0.3/records100/ data/ptb-xl/records100/"
        )

    for ecg_id, row in tqdm(db_folds.iterrows(), total=len(db_folds),
                             desc="Rendering ECG images"):
        # Parse SCP codes
        scp_codes = parse_scp_codes(row['scp_codes'])

        # Map to 4-class label
        result = get_superclass(scp_codes, scp_df)
        if result is None:
            skipped_hf += 1
            continue
        folder_name, superclass = result

        # Build WFDB record path
        # ptbxl_database.csv has 'filename_lr' (records100) and 'filename_hr' (records500)
        fname_col = 'filename_lr'
        if fname_col not in row.index or pd.isna(row[fname_col]):
            skipped_io += 1
            continue

        # filename_lr looks like "records100/00000/00001_lr"
        # wfdb.rdrecord expects the path WITHOUT extension
        record_rel = str(row[fname_col])           # e.g. "records100/00000/00001_lr"
        record_path = str(ptbxl_dir / record_rel)  # absolute path for wfdb

        out_fname = f"{ecg_id:05d}.png"
        out_path = out_dir / folder_name / out_fname

        if out_path.exists():
            already_done += 1
            counts[folder_name] += 1
            manifest.append({
                'ecg_id': int(ecg_id),
                'label': folder_name,
                'superclass': superclass,
                'file': str(out_path.relative_to(out_dir)),
            })
            continue

        # Load signal
        try:
            record = wfdb.rdrecord(record_path)
            signal = record.p_signal   # shape (n_samples, 12), float64, mV
        except Exception as e:
            print(f"  WARN: cannot read {record_path}: {e}")
            skipped_io += 1
            continue

        if signal is None or signal.shape[1] != 12:
            skipped_io += 1
            continue

        # Render image
        ok = render_ecg_image(signal, f"{ecg_id:05d}", folder_name, out_path)
        if not ok:
            skipped_io += 1
            continue

        counts[folder_name] += 1
        manifest.append({
            'ecg_id': int(ecg_id),
            'label': folder_name,
            'superclass': superclass,
            'file': str(out_path.relative_to(out_dir)),
        })

    # ── Write manifest ───────────────────────────────────────────────────────
    manifest_path = out_dir / 'manifest.json'
    with open(manifest_path, 'w') as f:
        json.dump({'records': manifest, 'counts': counts,
                   'skipped_no_label': skipped_hf,
                   'skipped_io': skipped_io,
                   'already_done': already_done}, f, indent=2)

    total = sum(counts.values())
    print("\n" + "=" * 60)
    print("PTB-XL IMAGE CONVERSION COMPLETE")
    print("=" * 60)
    print(f"  Total images rendered : {total}")
    print(f"  Normal                : {counts['Normal']}")
    print(f"  Myocardial_Infarction : {counts['Myocardial_Infarction']}")
    print(f"  Arrhythmia            : {counts['Arrhythmia']}")
    print(f"  History_of_MI         : {counts['History_of_MI']}")
    print(f"  Skipped (HYP/unmapped): {skipped_hf}")
    print(f"  Skipped (IO errors)   : {skipped_io}")
    print(f"  Already done (cached) : {already_done}")
    print(f"  Manifest saved        : {manifest_path}")
    print("=" * 60)

    return counts


def main():
    parser = argparse.ArgumentParser(
        description='Convert PTB-XL WFDB records to 340×340 ECG images'
    )
    parser.add_argument('--ptbxl-dir', default='data/ptb-xl',
                        help='Root directory containing ptbxl_database.csv and records100/')
    parser.add_argument('--out-dir', default='data/ptb-xl-images',
                        help='Output directory for rendered images (suffix -train or -all added when --folds != test)')
    parser.add_argument('--folds', default='test',
                        choices=['test', 'train', 'all'],
                        help=(
                            'test = folds 9+10 (default, held-out eval set); '
                            'train = folds 1–8 (for retraining); '
                            'all = all 10 folds'
                        ))
    args = parser.parse_args()

    ptbxl_dir = Path(args.ptbxl_dir)

    if args.folds == 'test':
        fold_set = [9, 10]
        out_dir  = Path(args.out_dir)
    elif args.folds == 'train':
        fold_set = list(range(1, 9))
        out_dir  = Path(args.out_dir + '-train')
    else:
        fold_set = list(range(1, 11))
        out_dir  = Path(args.out_dir + '-all')

    print("=" * 60)
    print("PTB-XL → matplotlib ECG image converter")
    print(f"  Input : {ptbxl_dir}")
    print(f"  Output: {out_dir}")
    print(f"  Folds : {fold_set}")
    print(f"  Labels: NORM→Normal, MI→MI, CD→Arrhythmia, STTC→History_of_MI")
    print("=" * 60)

    convert_ptbxl(ptbxl_dir, out_dir, fold_set=fold_set)


if __name__ == '__main__':
    main()
