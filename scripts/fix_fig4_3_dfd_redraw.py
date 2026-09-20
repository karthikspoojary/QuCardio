import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

fig, ax = plt.subplots(figsize=(20, 14), dpi=200)
W, H = 2000, 1400
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

P_EDGE  = "#1A56A0"; P_FILL = "white"; P_TXT = "#1A365D"
STORE_E = "#4A5568"; STORE_F = "#EDF2F7"
ENT_E   = "black";  ENT_F   = "white"
CACHE_C = "#6B46C1"
ERR_C   = "#C53030"
HIT_C   = "#276749"
PKL_E   = "#888888"; PKL_F = "#F7FAFC"

# ── IEEE-readable font sizes ─────────────────────────────────────────
FS_PROC  = 15
FS_STORE = 14
FS_ENT   = 16
FS_FLOW  = 12.5
FS_PKL   = 11

def proc(cx, cy, r, id_lbl, name1, name2=""):
    ax.add_patch(plt.Circle((cx, cy), r, lw=2.8,
                 edgecolor=P_EDGE, facecolor=P_FILL, zorder=3))
    if name2:
        ax.text(cx, cy + 15, id_lbl,  ha="center", va="center",
                fontsize=FS_PROC, fontweight="bold", color=P_TXT, zorder=4)
        ax.text(cx, cy - 1,  name1, ha="center", va="center",
                fontsize=FS_PROC - 1, fontweight="bold", color=P_TXT, zorder=4)
        ax.text(cx, cy - 18, name2, ha="center", va="center",
                fontsize=FS_PROC - 1, fontweight="bold", color=P_TXT, zorder=4)
    else:
        ax.text(cx, cy + 10, id_lbl, ha="center", va="center",
                fontsize=FS_PROC, fontweight="bold", color=P_TXT, zorder=4)
        ax.text(cx, cy - 9,  name1, ha="center", va="center",
                fontsize=FS_PROC - 1, fontweight="bold", color=P_TXT, zorder=4)

def store(x, y, w, h, code, label):
    sep = 56
    ax.add_patch(mpatches.FancyBboxPatch((x, y), w, h,
        boxstyle="square,pad=0", lw=2.2, edgecolor=STORE_E, facecolor=STORE_F, zorder=2))
    ax.plot([x+sep, x+sep], [y, y+h], color=STORE_E, lw=2.2, zorder=3)
    ax.text(x + sep/2, y + h/2, code,
            ha="center", va="center", fontsize=FS_STORE, fontweight="bold",
            color=STORE_E, zorder=4)
    ax.text(x + sep + (w - sep)/2, y + h/2, label,
            ha="center", va="center", fontsize=FS_STORE, fontweight="bold",
            color=STORE_E, zorder=4)

def entity(x, y, w, h, lbl):
    ax.add_patch(mpatches.FancyBboxPatch((x, y), w, h,
        boxstyle="square,pad=0", lw=3.2,
        edgecolor=ENT_E, facecolor=ENT_F, zorder=3))
    ax.text(x + w/2, y + h/2, lbl,
            ha="center", va="center",
            fontsize=FS_ENT, fontweight="bold", zorder=4)

def pkl_box(x, y, w, h, lbl):
    ax.add_patch(mpatches.FancyBboxPatch((x, y), w, h,
        boxstyle="square,pad=0", lw=1.6,
        edgecolor=PKL_E, facecolor=PKL_F, zorder=3))
    ax.text(x + w/2, y + h/2, lbl,
            ha="center", va="center",
            fontsize=FS_PKL, style="italic", color="#4A5568", zorder=4)

def arrow(x1, y1, x2, y2, lbl="", lx=None, ly=None,
          color="black", dash=False, lw=2.0, above=True):
    ls = (0, (5, 4)) if dash else "solid"
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="->", color=color, lw=lw, linestyle=ls))
    if lbl:
        tx = lx if lx is not None else (x1 + x2) / 2
        ty = ly if ly is not None else ((y1 + y2) / 2 + (15 if above else -18))
        ax.text(tx, ty, lbl, ha="center", va="center",
                fontsize=FS_FLOW, style="italic", color=color,
                bbox=dict(boxstyle="square,pad=0.25", fc="white", ec="none", alpha=0.95),
                zorder=6)

# ── Node positions ────────────────────────────────────────────────────
# Entity (Clinician) — left side, vertically centred
EX, EY, EW, EH = 30, 570, 170, 100

# Process nodes arranged in a ring
R = 96  # process circle radius (larger for bigger text)
P1 = (500,  1150, R)
P2 = (870,  1150, R)
P3 = (1280, 1150, R)
P4 = (1580,  790, R)
P5 = (1280,  380, R)
P6 = (650,   380, R)

# Data stores
D1 = (480, 700, 320, 60)     # Redis Cache
D2 = (1020, 100, 360, 60)    # SQLite Audit DB

# PKL annotation boxes
PKL_SVD = (1680, 760, 270, 56)
PKL_CLF = (1400, 345, 220, 52)

# ── Draw nodes ────────────────────────────────────────────────────────
entity(EX, EY, EW, EH, "Clinician")
proc(*P1, "P1", "Image", "Validation")
proc(*P2, "P2", "ECG", "Preprocessing")
proc(*P3, "P3", "Feature", "Extraction")
proc(*P4, "P4", "Dimensionality", "Reduction")
proc(*P5, "P5", "Classification")
proc(*P6, "P6", "Response", "Assembly")
store(*D1, "D1", "Redis Cache")
store(*D2, "D2", "SQLite Audit DB")
pkl_box(*PKL_SVD, "svd_reducer.pkl\nminmax_scaler.pkl")
pkl_box(*PKL_CLF, "classifier.pkl")

# ── Arrows ────────────────────────────────────────────────────────────
D1_CX = D1[0] + D1[2]/2    # 640
D1_TOP = D1[1] + D1[3]     # 760
D1_BOT = D1[1]              # 700

# 1. Clinician → P1  (raw image file)
arrow(EX + EW, EY + EH - 20,
      P1[0] - P1[2], P1[1],
      "raw image file", lx=330, ly=1060)

# 2. P1 → Clinician  (error if invalid)
arrow(P1[0] - P1[2] + 20, P1[1] - 55,
      EX + EW, EY + 65,
      "error (if invalid)", lx=240, ly=870, color=ERR_C)

# 3. P1 → D1  cache check (dashed purple) — comes down from P1 to D1 top
arrow(P1[0] + 25, P1[1] - P1[2],
      D1_CX - 30, D1_TOP,
      "cache check\n(on request)",
      lx=595, ly=965, color=CACHE_C, dash=True)

# 4. D1 → P6  cache HIT (green) — exits D1 right, enters P6 top
arrow(D1[0] + D1[2], D1[1] + D1[3]/2,
      P6[0] - 35, P6[1] + P6[2],
      "cached result\n(HIT: bypass P2-P5)",
      lx=870, ly=600, color=HIT_C)

# 5. P6 → D1  cache MISS write-back (purple) — exits P6 bottom-left, enters D1 left
arrow(P6[0] - P6[2] + 20, P6[1] - 20,
      D1[0], D1[1] + D1[3]/2,
      "cache result\n(on MISS)",
      lx=440, ly=540, color=CACHE_C)

# 6. P1 → P2  (validated image)
arrow(P1[0] + P1[2], P1[1],
      P2[0] - P2[2], P2[1],
      "validated image", ly=P1[1] + 24)

# 7. P2 → P3  (340×340 float32 array)
arrow(P2[0] + P2[2], P2[1],
      P3[0] - P3[2], P3[1],
      "340x340 float32 array", ly=P2[1] + 24)

# 8. P3 → P4  (462400-D feature vector — diagonal down-right)
arrow(P3[0] + P3[2] - 20, P3[1] - 40,
      P4[0] - P4[2] + 20, P4[1] + 40,
      "462400-D feature vector", lx=1490, ly=1010)

# 9. SVD pkl → P4  (dashed grey, right side)
ax.annotate("", xy=(P4[0] + P4[2] - 8, P4[1] + 16),
    xytext=(PKL_SVD[0], PKL_SVD[1] + PKL_SVD[3]/2),
    arrowprops=dict(arrowstyle="->", color=PKL_E, lw=1.5, linestyle=(0,(4,3))))

# 10. P4 → P5  (9-D scaled vector — diagonal down-left)
arrow(P4[0] - 30, P4[1] - P4[2] + 20,
      P5[0] + 30, P5[1] + P5[2] - 20,
      "9-D scaled vector [0, \u03c0]", lx=1490, ly=600)

# 11. Classifier pkl → P5  (dashed grey, right side)
ax.annotate("", xy=(P5[0] + P5[2] - 8, P5[1] + 18),
    xytext=(PKL_CLF[0], PKL_CLF[1] + PKL_CLF[3]/2),
    arrowprops=dict(arrowstyle="->", color=PKL_E, lw=1.5, linestyle=(0,(4,3))))

# 12. P5 → P6  (class index, probability array)
arrow(P5[0] - P5[2], P5[1],
      P6[0] + P6[2], P6[1],
      "class index, probability array", ly=P5[1] + 24)

# 13. P5 → D2  (prediction log — down)
arrow(P5[0] - 20, P5[1] - P5[2],
      D2[0] + D2[2]/2 + 20, D2[1] + D2[3],
      "prediction log", ly=250)

# 14. P6 → Clinician  (JSON response)
arrow(P6[0] - P6[2], P6[1],
      EX + EW, EY + EH/2,
      "JSON response", lx=330, ly=460)

plt.tight_layout(pad=0.3)
OUT = "/home/karthik/projects/QuCardio/Doc/report/extracted_images/fig4_3_dfd.png"
plt.savefig(OUT, bbox_inches="tight", dpi=200)
print("Saved:", OUT)
