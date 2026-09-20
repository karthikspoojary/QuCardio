"""
Generate hard-negative images for gatekeeper retraining.
Produces images that pass the current heuristic (thin lines on white, landscape)
but are NOT ECGs: line charts and UML-style sequence diagrams.

Output: data/NON_ECG_DATA/hard_neg_*.jpg  (~600 images)

Run time: ~1 minute on CPU.
"""
import os, random, math
import numpy as np
import cv2
from pathlib import Path

OUT_DIR = Path("data/NON_ECG_DATA")
OUT_DIR.mkdir(parents=True, exist_ok=True)
rng = random.Random(42)
np.random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# Helper: save as JPEG
# ─────────────────────────────────────────────────────────────────────────────
def save(img_bgr, name):
    path = OUT_DIR / name
    cv2.imwrite(str(path), img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])


def white_canvas(w=680, h=340):
    """Landscape canvas matching processed_340 aspect ratio."""
    return np.full((h, w, 3), 248, dtype=np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# Category 1: Line / area charts (300 images)
# ─────────────────────────────────────────────────────────────────────────────
COLORS_BGE = [
    (200, 80,  40),   # blue
    (40,  120, 200),  # orange-red
    (30,  160, 60),   # green
    (160, 40,  160),  # purple
    (0,   180, 180),  # teal
]

def make_line_chart(idx):
    w, h = rng.choice([(680,340),(800,320),(720,360),(900,300)])
    img = white_canvas(w, h)

    # --- axes ---
    margin_l, margin_r, margin_t, margin_b = 60, 20, 20, 40
    ax_x0, ax_y0 = margin_l, margin_t
    ax_x1, ax_y1 = w - margin_r, h - margin_b

    # axis lines
    cv2.line(img, (ax_x0, ax_y0), (ax_x0, ax_y1), (80,80,80), 1)
    cv2.line(img, (ax_x0, ax_y1), (ax_x1, ax_y1), (80,80,80), 1)

    # grid lines
    n_grid_y = rng.randint(3,7)
    for k in range(1, n_grid_y+1):
        y = ax_y0 + int((ax_y1-ax_y0)*k/(n_grid_y+1))
        cv2.line(img, (ax_x0, y), (ax_x1, y), (200,200,200), 1)
    n_grid_x = rng.randint(4,10)
    for k in range(1, n_grid_x+1):
        x = ax_x0 + int((ax_x1-ax_x0)*k/(n_grid_x+1))
        cv2.line(img, (x, ax_y0), (x, ax_y1), (200,200,200), 1)

    # tick marks
    for k in range(1, n_grid_y+1):
        y = ax_y0 + int((ax_y1-ax_y0)*k/(n_grid_y+1))
        cv2.line(img, (ax_x0-4, y), (ax_x0, y), (80,80,80), 1)
    for k in range(1, n_grid_x+1):
        x = ax_x0 + int((ax_x1-ax_x0)*k/(n_grid_x+1))
        cv2.line(img, (x, ax_y1), (x, ax_y1+4), (80,80,80), 1)

    # one or two data series
    n_series = rng.randint(1, 3)
    n_pts = rng.randint(8, 40)
    xs = np.linspace(ax_x0, ax_x1, n_pts, dtype=int)
    chart_h = ax_y1 - ax_y0

    for s in range(n_series):
        col = COLORS_BGE[s % len(COLORS_BGE)]
        # smooth random walk
        vals = np.cumsum(np.random.randn(n_pts) * chart_h * 0.06)
        vals = vals - vals.min()
        if vals.max() > 0:
            vals = vals / vals.max() * chart_h * 0.75
        ys = (ax_y1 - 10 - vals).astype(int)
        ys = np.clip(ys, ax_y0+2, ax_y1-2)

        pts = np.stack([xs, ys], axis=1).reshape(-1,1,2)
        cv2.polylines(img, [pts], False, col, rng.choice([1,2]))

        # optional filled area
        if rng.random() > 0.5:
            poly = np.vstack([[xs[0], ax_y1], np.stack([xs,ys],1), [xs[-1], ax_y1]])
            overlay = img.copy()
            cv2.fillPoly(overlay, [poly], (*col[:3], 40))
            cv2.addWeighted(overlay, 0.18, img, 0.82, 0, img)

        # optional dots
        if rng.random() > 0.5:
            for x, y in zip(xs[::max(1,n_pts//12)], ys[::max(1,n_pts//12)]):
                cv2.circle(img, (int(x), int(y)), 3, col, -1)

    save(img, f"hard_neg_chart_{idx:04d}.jpg")


print("Generating 300 line charts ...")
for i in range(300):
    make_line_chart(i)
print("  done")


# ─────────────────────────────────────────────────────────────────────────────
# Category 2: UML-style sequence / flow diagrams (300 images)
# ─────────────────────────────────────────────────────────────────────────────
def make_uml_diagram(idx):
    w, h = rng.choice([(680,340),(800,380),(700,360),(750,350)])
    img = white_canvas(w, h)

    n_actors = rng.randint(3, 7)
    actor_xs = np.linspace(60, w-60, n_actors, dtype=int)

    # --- actor boxes ---
    box_w, box_h = rng.randint(60,90), 24
    for ax in actor_xs:
        x0 = max(0, ax - box_w//2)
        x1 = min(w-1, ax + box_w//2)
        cv2.rectangle(img, (x0, 10), (x1, 10+box_h), (60,60,60), 1)

    # --- lifelines (vertical dashed) ---
    top_y   = 10 + box_h
    bot_y   = h - 20
    dash_on = 6
    for ax in actor_xs:
        y = top_y
        while y < bot_y:
            cv2.line(img, (ax, y), (ax, min(y+dash_on, bot_y)), (140,140,140), 1)
            y += dash_on * 2

    # --- messages (horizontal arrows between lifelines) ---
    n_msgs = rng.randint(5, 14)
    msg_ys = sorted(rng.sample(range(top_y+15, bot_y-10), min(n_msgs, bot_y-top_y-25)))
    for my in msg_ys:
        src, dst = rng.sample(list(actor_xs), 2)
        col = (50, 50, 50)
        cv2.arrowedLine(img, (src, my), (dst, my), col, 1, tipLength=0.04)

        # activation box on receiver
        if rng.random() > 0.4:
            ry0, ry1 = my, min(my + rng.randint(12,30), bot_y)
            rx = dst - 4
            cv2.rectangle(img, (rx, ry0), (rx+8, ry1), (100,100,100), 1)

    # --- note boxes ---
    if rng.random() > 0.5:
        nx = rng.randint(30, w-120)
        ny = rng.randint(top_y+30, bot_y-40)
        nw, nh = rng.randint(70,120), rng.randint(25,40)
        # folded corner
        pts = np.array([[nx,ny],[nx+nw-12,ny],[nx+nw,ny+12],[nx+nw,ny+nh],[nx,ny+nh]])
        cv2.polylines(img, [pts.reshape(-1,1,2)], True, (100,100,100), 1)
        cv2.line(img, (nx+nw-12,ny), (nx+nw-12,ny+12), (100,100,100), 1)
        cv2.line(img, (nx+nw-12,ny+12), (nx+nw,ny+12), (100,100,100), 1)

    save(img, f"hard_neg_uml_{idx:04d}.jpg")


print("Generating 300 UML-style diagrams ...")
for i in range(300):
    make_uml_diagram(i)
print("  done")


# ─────────────────────────────────────────────────────────────────────────────
# Count total
# ─────────────────────────────────────────────────────────────────────────────
hard_negs = list(OUT_DIR.glob("hard_neg_*.jpg"))
total     = len(list(OUT_DIR.glob("*.jpg")))
print(f"\nHard negatives written : {len(hard_negs)}")
print(f"Total NON_ECG_DATA     : {total}")
print("Done → data/NON_ECG_DATA/")
