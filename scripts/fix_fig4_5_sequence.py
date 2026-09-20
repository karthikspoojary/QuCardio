"""
Fig. 4.5 — UML Sequence Diagram  (QuCardio)   IEEE style — v5 final
=====================================================================
Key fix: smaller canvas (22"×18") at DPI=200 so text appears large.
  • FS_LIFE=14, FS_MSG=13, FS_STEP=13 — all clearly readable
  • Lifeline spacing 2.60" — fits 8 lifelines in 22" with margins
  • Step numbers LEFT of RC box — never overlap any arrow
  • ALT branches well-separated — 0.85" gap between condition label & arrow
  • All arrows touch activation bars exactly
  • Step 10 JSON label fits above footer boxes with 1.20" gap
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.font_manager as fm

SERIF = "Times New Roman"
if SERIF not in [f.name for f in fm.fontManager.ttflist]:
    SERIF = "DejaVu Serif"
plt.rcParams.update({"font.family": SERIF})

# ── Canvas ─────────────────────────────────────────────────────────────────────
# 26"×22" at 200 DPI = 5200×4400 px. Wider canvas gives more room per lifeline.
FW, FH = 26.0, 22.0
DPI    = 200
fig, ax = plt.subplots(figsize=(FW, FH), dpi=DPI)
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)  # axes fills entire figure
ax.axis("off")
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

C_BLK   = "#000000"
C_GREY  = "#555555"
C_LGREY = "#BBBBBB"
C_WHITE = "#FFFFFF"
C_FILL  = "#EFEFEF"
C_AFILL = "#D0D0D0"
C_ALT_B = "#003300"
C_ALT_F = "#F0F8F0"

FS_TITLE = 18
FS_LIFE  = 14
FS_MSG   = 13
FS_ALT   = 12
FS_STEP  = 13

LW_BOX  = 1.6
LW_MSG  = 1.3
LW_DASH = 0.9

# ── Lifeline geometry ──────────────────────────────────────────────────────────
# "ECGGatekeeper" at FS_LIFE=14 bold needs ≥ 2.20" wide box.
# Use 2.40" so ALL names (incl. ECGGatekeeper, QSVCClassifier) fit comfortably.
LL_W  = 2.40   # wider — fits ECGGatekeeper, QSVCClassifier without clipping
LL_H  = 0.80   # slightly taller for 2-line names
ACT_W = 0.18   # activation bar width

# 8 lifelines across FW=26": left margin 2.6", right margin 1.0"
# Available = 26 - 2.6 - 1.0 = 22.4", 7 gaps → step = 22.4/7 ≈ 3.20"
LMAR  = 2.60   # left edge of RC box centre
AVAIL = FW - LMAR - 1.00
LL_SEP = AVAIL / 7   # ~3.20" per gap

RC_X  = LMAR
FA_X  = RC_X + LL_SEP
RED_X = RC_X + 2*LL_SEP
GK_X  = RC_X + 3*LL_SEP
FE_X  = RC_X + 4*LL_SEP
DR_X  = RC_X + 5*LL_SEP
QS_X  = RC_X + 6*LL_SEP
AD_X  = RC_X + 7*LL_SEP

LIFELINES = [
    ("React\nClient",           RC_X),
    ("FastAPI\nBackend",        FA_X),
    ("RedisCache",              RED_X),
    ("ECGGatekeeper",           GK_X),
    ("FeatureExtractor",        FE_X),
    ("Dimensionality\nReducer", DR_X),
    ("QSVCClassifier",          QS_X),
    ("AuditDB",                 AD_X),
]

# ── Vertical layout — all Y computed up-front ──────────────────────────────────
TITLE_Y = FH - 0.34
TOP_Y   = FH - 0.86    # centre of header lifeline boxes

Y1 = TOP_Y - LL_H/2 - 0.94   # Step 1 arrow — 0.94" below header box bottom

# ALT frame — 2 branches × 1.55" each (extra room so return arrow clears divider)
ALT_TOP  = Y1 - 0.38
ALT_BRCH = 1.55
ALT_H    = 2*ALT_BRCH + 0.10
ALT_BOT  = ALT_TOP - ALT_H
MID_ALT  = ALT_TOP - ALT_BRCH

# Each branch: condition label at top edge, call arrow 0.90" below,
# return arrow 0.52" below call.  Branch height = 1.40" total.
# [cache miss] label bottom ≈ ALT_TOP - 0.04 - 0.22 = ALT_TOP - 0.26
# call arrow at ALT_TOP - 0.90 → gap = 0.64" — clearly separated
Y2a = ALT_TOP - 0.90   # cache miss call
Y2b = Y2a     - 0.52   # cache miss return — stays above MID_ALT by 0.38"
Y2c = MID_ALT - 0.90   # cache hit arrow — 0.90" below divider

# Steps 3-10 below ALT — generous uniform spacing
SEP = 0.88    # call-to-call
RET = 0.48    # return below call

Y3a = ALT_BOT - SEP
Y3b = Y3a - RET
Y4a = Y3b - SEP
Y4b = Y4a - RET
Y5a = Y4b - SEP
Y5b = Y5a - RET
Y6  = Y5b - SEP
SLH = 0.24
Y7a = Y6  - SEP - SLH
Y7b = Y7a - RET
Y8  = Y7b - SEP
Y9  = Y8  - SEP
Y10 = Y9  - SEP

# Footer lifeline boxes — 1.40" below Y10 (room for JSON label at 0.15" + box)
BOT_Y   = Y10 - 1.40
BOT_TOP = BOT_Y + LL_H/2
BOT_BOT = BOT_Y - LL_H/2

Y_MIN = BOT_BOT - 0.50
ax.set_xlim(0, FW)
ax.set_ylim(Y_MIN, FH)

# ── Lifeline dashed stems ──────────────────────────────────────────────────────
for _, x in LIFELINES:
    ax.plot([x, x], [TOP_Y - LL_H/2, BOT_TOP],
            linestyle="--", color=C_LGREY, linewidth=LW_DASH, zorder=1)

# ── Lifeline boxes ─────────────────────────────────────────────────────────────
def draw_ll_box(x, yc):
    ax.add_patch(patches.Rectangle(
        (x - LL_W/2, yc - LL_H/2), LL_W, LL_H,
        linewidth=LW_BOX, edgecolor=C_BLK, facecolor=C_FILL, zorder=5))

def draw_ll_text(x, yc, name):
    lines = name.split("\n")
    if len(lines) == 2:
        ax.text(x, yc + 0.17, lines[0], ha="center", va="center",
                fontsize=FS_LIFE, fontweight="bold", color=C_BLK, zorder=6)
        ax.text(x, yc - 0.17, lines[1], ha="center", va="center",
                fontsize=FS_LIFE, fontweight="bold", color=C_BLK, zorder=6)
    else:
        ax.text(x, yc, name, ha="center", va="center",
                fontsize=FS_LIFE, fontweight="bold", color=C_BLK, zorder=6)

for name, x in LIFELINES:
    draw_ll_box(x, TOP_Y);  draw_ll_text(x, TOP_Y, name)
    draw_ll_box(x, BOT_Y);  draw_ll_text(x, BOT_Y, name)

# ── Activation bars ────────────────────────────────────────────────────────────
def act_bar(x, ytop, ybot):
    if ytop <= ybot:
        return
    ax.add_patch(patches.Rectangle(
        (x - ACT_W/2, ybot), ACT_W, ytop - ybot,
        linewidth=0.8, edgecolor=C_BLK, facecolor=C_AFILL, zorder=3))

act_bar(RC_X, TOP_Y - LL_H/2, BOT_TOP)
act_bar(FA_X, TOP_Y - LL_H/2, BOT_TOP)
act_bar(RED_X, Y2a + 0.14, Y2b - 0.08)
act_bar(GK_X,  Y3a + 0.14, Y3b - 0.08)
act_bar(FE_X,  Y4a + 0.14, Y4b - 0.08)
act_bar(DR_X,  Y5a + 0.14, Y5b - 0.08)
act_bar(QS_X,  Y7a + 0.14, Y7b - 0.08)
# AuditDB Step-8 bar made taller (0.44") so the arriving arrow has a visible target
act_bar(AD_X,  Y8  + 0.22, Y8  - 0.22)
# RedisCache Step-9 bar also taller (0.44") for the same reason
act_bar(RED_X, Y9  + 0.22, Y9  - 0.22)

# ── Arrow helpers ──────────────────────────────────────────────────────────────
def arrow_line(x1, x2, y, ret=False, color=None):
    c  = color or (C_GREY if ret else C_BLK)
    ls = "--" if ret else "-"
    ax.plot([x1, x2], [y, y], linestyle=ls, color=c, linewidth=LW_MSG, zorder=4)
    dx = 0.01 if x2 > x1 else -0.01
    ax.annotate("", xy=(x2, y), xytext=(x2 - dx*14, y),
                arrowprops=dict(arrowstyle="->", color=c, lw=LW_MSG), zorder=5)

def msg(x1, x2, y, text, ret=False, color=None, above=True):
    c  = color or (C_GREY if ret else C_BLK)
    fw = "normal" if ret else "bold"
    arrow_line(x1, x2, y, ret=ret, color=color)
    mid = (x1 + x2) / 2
    dy  = +0.13 if above else -0.13
    ax.text(mid, y + dy, text,
            ha="center", va="center",
            fontsize=FS_MSG, fontweight=fw, color=c,
            bbox=dict(boxstyle="square,pad=0.08", fc=C_WHITE, ec="none", alpha=0.97),
            zorder=7, clip_on=False)

def step_lbl(y, num):
    """Step number strictly left of RC box — never overlaps any arrow."""
    ax.text(RC_X - LL_W/2 - 0.12, y, num,
            ha="right", va="center",
            fontsize=FS_STEP, fontweight="bold", color=C_BLK, zorder=8)

# ── ALT frame ──────────────────────────────────────────────────────────────────
ALT_L = FA_X  + ACT_W/2 - 0.06
ALT_R = RED_X + ACT_W/2 + 0.70
ax.add_patch(patches.Rectangle(
    (ALT_L, ALT_BOT), ALT_R - ALT_L, ALT_H,
    linewidth=1.5, edgecolor=C_ALT_B, facecolor=C_ALT_F,
    zorder=1, alpha=0.30))
tag_w, tag_h = 0.36, 0.22
ax.add_patch(patches.Rectangle(
    (ALT_L, ALT_TOP - tag_h), tag_w, tag_h,
    linewidth=1.1, edgecolor=C_ALT_B, facecolor=C_ALT_F, zorder=3))
ax.text(ALT_L + tag_w/2, ALT_TOP - tag_h/2, "alt",
        ha="center", va="center",
        fontsize=FS_ALT, fontweight="bold", color=C_ALT_B, zorder=8)
ax.plot([ALT_L, ALT_R], [MID_ALT, MID_ALT],
        linestyle="--", color=C_ALT_B, linewidth=1.0, zorder=2)
# Branch labels well above arrows (0.62" clearance each)
ax.text(ALT_L + tag_w + 0.10, ALT_TOP - 0.04,
        "[cache miss]",
        fontsize=FS_ALT, style="italic", color=C_ALT_B, va="top", ha="left",
        bbox=dict(boxstyle="square,pad=0.05", fc=C_ALT_F, ec="none"), zorder=9)
ax.text(ALT_L + 0.08, MID_ALT - 0.04,
        "[cache hit]",
        fontsize=FS_ALT, style="italic", color=C_ALT_B, va="top", ha="left",
        bbox=dict(boxstyle="square,pad=0.05", fc=C_ALT_F, ec="none"), zorder=9)

# ── Step 1: POST /predict ──────────────────────────────────────────────────────
step_lbl(Y1, "1.")
x1s = RC_X + ACT_W/2
x1e = FA_X - ACT_W/2
arrow_line(x1s, x1e, Y1)
ax.text((x1s + x1e)/2, Y1 + 0.15,
        'POST /predict  (image_bytes, model="quantum")',
        ha="center", va="bottom", fontsize=FS_MSG, fontweight="bold", color=C_BLK,
        bbox=dict(boxstyle="square,pad=0.08", fc=C_WHITE, ec="none", alpha=0.97),
        zorder=7, clip_on=False)

# ── Step 2: Cache check ────────────────────────────────────────────────────────
step_lbl(Y2a, "2.")
msg(FA_X+ACT_W/2, RED_X-ACT_W/2, Y2a, 'get( sha256(image) + "quantum" )')
msg(RED_X-ACT_W/2, FA_X+ACT_W/2, Y2b, "cache miss", ret=True, color=C_GREY)
arrow_line(RED_X-ACT_W/2, FA_X+ACT_W/2, Y2c, ret=True, color=C_ALT_B)
ax.text((FA_X+RED_X)/2, Y2c+0.14,
        "cached result  (skip to step 10)",
        ha="center", va="bottom", fontsize=FS_ALT, style="italic", color=C_ALT_B,
        bbox=dict(boxstyle="square,pad=0.06", fc=C_ALT_F, ec="none"), zorder=7,
        clip_on=False)

# ── Step 3: ECG Gatekeeper ────────────────────────────────────────────────────
step_lbl(Y3a, "3.")
msg(FA_X+ACT_W/2, GK_X-ACT_W/2, Y3a, "validate( image )")
msg(GK_X-ACT_W/2, FA_X+ACT_W/2, Y3b, "is_ecg = True", ret=True)

# ── Step 4: Feature extraction ────────────────────────────────────────────────
step_lbl(Y4a, "4.")
msg(FA_X+ACT_W/2, FE_X-ACT_W/2, Y4a, "extract( preprocessed_img )")
msg(FE_X-ACT_W/2, FA_X+ACT_W/2, Y4b, "feat[462400]", ret=True)

# ── Step 5: Dimensionality reduction ──────────────────────────────────────────
step_lbl(Y5a, "5.")
msg(FA_X+ACT_W/2, DR_X-ACT_W/2, Y5a, "transform( feat )")
msg(DR_X-ACT_W/2, FA_X+ACT_W/2, Y5b, "x9d[9]", ret=True)

# ── Step 6: Self-loop — scale to π ────────────────────────────────────────────
step_lbl(Y6, "6.")
SLX = FA_X + ACT_W/2
sl_r = SLX + 0.64
ax.plot([SLX, sl_r, sl_r, SLX],
        [Y6+SLH, Y6+SLH, Y6-SLH, Y6-SLH],
        color=C_BLK, linewidth=LW_MSG, zorder=4)
ax.annotate("", xy=(SLX, Y6-SLH), xytext=(SLX+0.20, Y6-SLH),
            arrowprops=dict(arrowstyle="->", color=C_BLK, lw=LW_MSG))
ax.text(sl_r + 0.10, Y6,
        r"x9d_pi = x9d $\times$ $\pi$   [0, $\pi$]",
        ha="left", va="center", fontsize=FS_MSG, fontweight="bold", color=C_BLK,
        bbox=dict(boxstyle="square,pad=0.08", fc=C_WHITE, ec="none", alpha=0.97),
        zorder=7, clip_on=False)

# ── Step 7: QSVC predict ──────────────────────────────────────────────────────
step_lbl(Y7a, "7.")
msg(FA_X+ACT_W/2, QS_X-ACT_W/2, Y7a, "predict( x9d_pi )")
msg(QS_X-ACT_W/2, FA_X+ACT_W/2, Y7b, "(class_idx, probs)", ret=True)

# ── Step 8: Audit log (sync — solid arrow) ────────────────────────────────────
step_lbl(Y8, "8.")
msg(FA_X+ACT_W/2, AD_X-ACT_W/2, Y8,
    'log(timestamp, hash, "quantum", class, conf)  [sync]', color=C_BLK)

# ── Step 9: Cache set ─────────────────────────────────────────────────────────
step_lbl(Y9, "9.")
msg(FA_X+ACT_W/2, RED_X-ACT_W/2, Y9, "set(key, result, TTL=86400s)")

# ── Step 10: JSON response ────────────────────────────────────────────────────
step_lbl(Y10, "10.")
x10s = FA_X - ACT_W/2
x10e = RC_X + ACT_W/2
arrow_line(x10s, x10e, Y10, ret=True, color=C_GREY)
ax.text((x10e+x10s)/2, Y10 - 0.15,
        "JSON {prediction, confidence, probabilities, image_b64, timing_ms}",
        ha="center", va="top", fontsize=FS_MSG, color=C_GREY,
        bbox=dict(boxstyle="square,pad=0.08", fc=C_WHITE, ec="none", alpha=0.97),
        zorder=7, clip_on=False)

# ── Title ──────────────────────────────────────────────────────────────────────
ax.text(FW/2, TITLE_Y,
        "Fig. 4.5  UML Sequence Diagram — POST /predict Endpoint (QuCardio)",
        ha="center", va="center",
        fontsize=FS_TITLE, fontweight="bold", color=C_BLK, zorder=9)

OUT = "/home/karthik/projects/QuCardio/Doc/report/extracted_images/fig4_5_sequence.png"
plt.savefig(OUT, bbox_inches="tight", pad_inches=0.08, dpi=DPI, facecolor="white")
print(f"Saved: {OUT}")
