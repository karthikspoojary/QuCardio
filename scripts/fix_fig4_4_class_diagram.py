"""
Fig. 4.4 — UML Class Diagram  (QuCardio)   IEEE style  v6
==========================================================
Key insight: canvas must be SMALL so font points appear large in the image.
  • FW=22", FH=auto (tight) at DPI=200 → output ~4400 × ~3000 px
  • Font sizes: FS_CLS=16, FS_BODY=13, FS_STEREO=13 — clearly readable
  • All box geometry in inches proportional to the smaller canvas
  • Arrows touch box borders exactly — ln() endpoints are box edges, no offset
  • T3 bus routed outside T2 boxes via right-margin vertical drop
  • No blank whitespace — ylim clamped to content
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon
import matplotlib.font_manager as fm

SERIF = "Times New Roman"
if SERIF not in [f.name for f in fm.fontManager.ttflist]:
    SERIF = "DejaVu Serif"
MONO = "DejaVu Sans Mono"
plt.rcParams.update({"font.family": SERIF})

# ── Canvas ────────────────────────────────────────────────────────────────────
# FW=22", DPI=200 → 4400 px wide — matches original canvas size so text
# appears at the same physical size as before.
FW  = 22.0    # figure width in inches — keep original for text readability
DPI = 200
fig, ax = plt.subplots(figsize=(FW, 16.0), dpi=DPI)
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
ax.set_xlim(0, FW)
ax.axis("off")
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

# ── Colours ───────────────────────────────────────────────────────────────────
C_BLK  = "#000000";  C_DARK = "#111111"
EC_BLUE  = "#0D2B5E"; FC_BLUE  = "#E8EEF8"
EC_GRN   = "#0A3D1F"; FC_GRN   = "#E6F4EB"
EC_PURP  = "#2D1169"; FC_PURP  = "#EDE8FA"
EC_ORNG  = "#5A2800"; FC_ORNG  = "#FAF0E6"
EC_RED   = "#5A0000"; FC_RED   = "#FAE8E8"
EC_GREY  = "#1E1E1E"; FC_GREY  = "#F2F2F2"
CC       = "#333333"

# ── Font sizes (points) ───────────────────────────────────────────────────────
FS_TITLE  = 20
FS_CLS    = 16      # class name
FS_STEREO = 13      # <<stereotype>>
FS_BODY   = 13      # fields & methods
FS_CONN   = 12
FS_LEGEND = 12

# ── Box geometry (inches, matching FW=22) ─────────────────────────────────────
HDR_S  = 0.58   # header WITH stereotype (2 lines)
HDR_N  = 0.42   # header WITHOUT stereotype (1 line)
LINE   = 0.27   # height per body text line
PAD    = 0.15   # top/bottom padding in each section
SEP    = 0.07   # gap between divider and text

def box_height(nf, nm, has_stereo=False):
    hdr = HDR_S if has_stereo else HDR_N
    return hdr + PAD + nf * LINE + SEP + PAD + nm * LINE + PAD

def draw_class(x, y, w, title, fields, methods, fc, ec, stereo=None):
    h   = box_height(len(fields), len(methods), stereo is not None)
    hdr = HDR_S if stereo else HDR_N

    # Outer white box
    ax.add_patch(mpatches.Rectangle(
        (x, y), w, h, lw=1.6, edgecolor=ec, facecolor="white", zorder=2))
    # Header fill
    ax.add_patch(mpatches.Rectangle(
        (x, y + h - hdr), w, hdr,
        lw=0, facecolor=fc, zorder=3))
    ax.add_patch(mpatches.Rectangle(
        (x, y + h - hdr), w, hdr,
        lw=1.6, edgecolor=ec, facecolor="none", zorder=4))

    if stereo:
        # Stereotype: upper 35% of header; title: lower 55%
        ax.text(x + w/2, y + h - hdr * 0.22, stereo,
                ha="center", va="center",
                fontsize=FS_STEREO, style="italic", color=ec, zorder=5,
                clip_on=False)
        ax.text(x + w/2, y + h - hdr * 0.68, title,
                ha="center", va="center",
                fontsize=FS_CLS, fontweight="bold", color=C_DARK, zorder=5,
                clip_on=False)
    else:
        ax.text(x + w/2, y + h - hdr/2, title,
                ha="center", va="center",
                fontsize=FS_CLS, fontweight="bold", color=C_DARK, zorder=5,
                clip_on=False)

    # Fields section
    ftop = y + h - hdr
    fh   = PAD + len(fields) * LINE + SEP
    fbot = ftop - fh
    ax.plot([x, x+w], [ftop, ftop], color=ec, lw=0.8, zorder=3)
    cy = ftop - PAD - LINE/2
    for f in fields:
        ax.text(x + 0.12, cy, f, ha="left", va="center",
                fontsize=FS_BODY, family=MONO, color=C_DARK, zorder=4,
                clip_on=False)
        cy -= LINE

    # Methods section
    ax.plot([x, x+w], [fbot, fbot], color=ec, lw=0.8, zorder=3)
    cy = fbot - PAD - LINE/2
    for m in methods:
        ax.text(x + 0.12, cy, m, ha="left", va="center",
                fontsize=FS_BODY, family=MONO, color=C_DARK, zorder=4,
                clip_on=False)
        cy -= LINE
    return h   # return box height so callers can compute top edge

def diamond(cx, cy, s=0.14, ec=EC_BLUE):
    pts = [(cx, cy+s), (cx+s*0.6, cy), (cx, cy-s), (cx-s*0.6, cy)]
    ax.add_patch(Polygon(pts, closed=True,
                         lw=1.4, edgecolor=ec, facecolor="white", zorder=6))

def ln(x1, y1, x2, y2, c=CC, lw=1.4):
    ax.plot([x1, x2], [y1, y2], color=c, lw=lw, zorder=1)

def arr(x1, y1, x2, y2, c=CC, lw=1.4):
    """Draw a line + filled arrowhead that touches exactly (x2,y2)."""
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=c, lw=lw,
                                mutation_scale=14,
                                shrinkA=0, shrinkB=0), zorder=5)

def clbl(x, y, txt, c=CC):
    ax.text(x, y, txt, ha="center", va="center", fontsize=FS_CONN,
            style="italic", color=c,
            bbox=dict(boxstyle="square,pad=0.10", fc="white", ec="none", alpha=0.95),
            zorder=8)

# ══════════════════════════════════════════════════════════════════════════════
# LAYOUT — bottom-up
# ══════════════════════════════════════════════════════════════════════════════
LM   = 0.32
RM   = 0.32
GAPV = 0.62   # vertical gap between tiers

# ── TIER 4: AuditDatabase  |  PDFGenerator ───────────────────────────────────
T4_BOT = 0.52
# AuditDatabase: longest line is "+ log_prediction(patient_id, hash," ~ 36 chars
# at 13pt mono ≈ 0.095"/char → 3.42", but "      model, cls, conf): int" needs
# the indent too. Use 5.80" to ensure NO overflow for any line + int suffix.
T4_W   = 5.80

adb_h = draw_class(LM, T4_BOT, T4_W, "AuditDatabase",
    ["- db_path: str",
     "- conn: sqlite3.Connection"],
    ["+ init_db(): void",
     "+ log_prediction(patient_id, hash,",
     "      model, cls, conf): int",
     "+ history(page: int): list"],
    FC_ORNG, EC_ORNG)
ADB_CX  = LM + T4_W/2
ADB_TOP = T4_BOT + adb_h

pdf_h = draw_class(FW - RM - T4_W, T4_BOT, T4_W, "PDFGenerator",
    [],
    ["+ generate_pdf_report(",
     "      result: dict,",
     "      filename: str): bytes"],
    FC_RED, EC_RED)
PDF_CX  = FW - RM - T4_W/2
PDF_TOP = T4_BOT + pdf_h
T4_TOP  = max(ADB_TOP, PDF_TOP)

# ── TIER 3: Three classifiers ─────────────────────────────────────────────────
T3_BOT = T4_TOP + GAPV
T3_GAP = 0.44
# "PegasosMulticlassClassifier" at FS_CLS=16 needs ≥ 4.3"
T3_W_AUTO = (FW - LM - RM - 2*T3_GAP) / 3
T3_W = max(T3_W_AUTO, 4.30)
T3_TOT = 3*T3_W + 2*T3_GAP
T3_X0 = (FW - T3_TOT) / 2   # centre the row

svm_h = draw_class(T3_X0, T3_BOT, T3_W, "ClassicalSVMClassifier",
    ["- calibrated_svc:",
     "    CalibratedClassifierCV",
     "- kernel: str = 'rbf'",
     "- C: float = 10.0"],
    ["+ predict(x: ndarray[9]):",
     "    tuple[int, ndarray[4]]",
     "+ predict_proba(x: ndarray[9]):",
     "    ndarray[4]"],
    FC_GRN, EC_GRN)
SVM_CX  = T3_X0 + T3_W/2
SVM_TOP = T3_BOT + svm_h

qsvc_x = T3_X0 + T3_W + T3_GAP
qsvc_h = draw_class(qsvc_x, T3_BOT, T3_W, "QSVCClassifier",
    ["- zzfeaturemap: ZZFeatureMap",
     "    (9q, reps=2, circular)",
     "- sv_train_cache:",
     "    ndarray[742, 512]"],
    ["+ kernel_row(x: ndarray[9]):",
     "    ndarray[742]",
     "+ predict(x: ndarray[9]):",
     "    tuple[int, ndarray[4]]"],
    FC_PURP, EC_PURP, stereo="<<Quantum>>")
QSVC_CX  = qsvc_x + T3_W/2
QSVC_TOP = T3_BOT + qsvc_h

peg_x = T3_X0 + 2*T3_W + 2*T3_GAP
peg_h = draw_class(peg_x, T3_BOT, T3_W, "PegasosMulticlassClassifier",
    ["- binary_models:",
     "    dict[(c1,c2)->PegasosQSVC]",
     "- CLASS_PAIRS: list (6 pairs)"],
    ["+ pegasos_multiclass_predict(",
     "    k_row: ndarray): int",
     "+ predict_node(k_row,",
     "    c1, c2): int"],
    FC_PURP, EC_PURP, stereo="<<Quantum>>")
PEG_CX  = peg_x + T3_W/2
PEG_TOP = T3_BOT + peg_h
T3_TOP  = max(SVM_TOP, QSVC_TOP, PEG_TOP)

# Bus line sits just above T3 boxes
BUS3_Y = T3_TOP + 0.24

# ── TIER 2: Five pipeline components ──────────────────────────────────────────
T2_BOT = T3_TOP + GAPV + 0.30
T2_GAP = 0.10   # narrower gap so each box gets more width within FW=22"
# With FW=22, LM=RM=0.32, 4×T2_GAP=0.40: each box = (22-0.64-0.40)/5 = 4.192"
# "preprocess_for_inference" bold 16pt: ~3.1" text + 2×0.20" pad → 3.50" — fits
# "    ndarray[340, 340]" mono 13pt: ~1.95" starting at x+0.12 — fits in 4.19"
T2_W_AUTO = (FW - LM - RM - 4*T2_GAP) / 5
T2_W = max(T2_W_AUTO, 4.10)   # auto-computed ≈4.19"; floor 4.10 for safety
T2_TOT = 5*T2_W + 4*T2_GAP
T2_X0 = (FW - T2_TOT) / 2

gk_h = draw_class(T2_X0, T2_BOT, T2_W, "ECGGatekeeper",
    ["- mobilenet_model: Model",
     "- threshold: float = 0.5"],
    ["+ validate(img: ndarray):",
     "    bool"],
    FC_BLUE, EC_BLUE)
GK_CX  = T2_X0 + T2_W/2
GK_TOP = T2_BOT + gk_h

pre_x = T2_X0 + T2_W + T2_GAP
pre_h = draw_class(pre_x, T2_BOT, T2_W, "preprocess_for_inference",
    [],
    ["+ preprocess_for_inference(",
     "    img_bgr: ndarray):",
     "    ndarray[340, 340]"],
    FC_BLUE, EC_BLUE, stereo="<<module>>")
PRE_CX  = pre_x + T2_W/2
PRE_TOP = T2_BOT + pre_h

fe_x = T2_X0 + 2*T2_W + 2*T2_GAP
fe_h = draw_class(fe_x, T2_BOT, T2_W, "FeatureExtractor",
    ["- resnet_pool1: Model",
     "    (ResNet50 pool1_pool)"],
    ["+ extract(img: ndarray):",
     "    ndarray[462400]"],
    FC_BLUE, EC_BLUE)
FE_CX  = fe_x + T2_W/2
FE_TOP = T2_BOT + fe_h

dr_x = T2_X0 + 3*T2_W + 3*T2_GAP
dr_h = draw_class(dr_x, T2_BOT, T2_W, "DimensionalityReducer",
    ["- svd: TruncatedSVD(9)",
     "- scaler: MinMaxScaler"],
    ["+ transform(x: ndarray):",
     "    ndarray[9]  in [0, 1]"],
    FC_BLUE, EC_BLUE)
DR_CX  = dr_x + T2_W/2
DR_TOP = T2_BOT + dr_h

gc_x = T2_X0 + 4*T2_W + 4*T2_GAP
gc_h = draw_class(gc_x, T2_BOT, T2_W, "GradCAMModel",
    ["- resnet_gradcam: Model",
     "    (conv5 + pool1)"],
    ["+ compute_gradcam(",
     "    img: ndarray):",
     "    ndarray[340, 340]"],
    FC_GREY, EC_GREY)
GC_CX  = gc_x + T2_W/2
GC_TOP = T2_BOT + gc_h
T2_RIGHT = gc_x + T2_W   # right edge of rightmost T2 box

T2_TOP = max(GK_TOP, PRE_TOP, FE_TOP, DR_TOP, GC_TOP)
# Bus sits 0.28" above T2 boxes — diamonds are centred here
BUS2_Y = T2_TOP + 0.28

# ── TIER 1: FastAPIApp ────────────────────────────────────────────────────────
T1_BOT = T2_TOP + GAPV + 0.16
FA_W   = FW - LM - RM

fa_h = draw_class(LM, T1_BOT, FA_W, "FastAPIApp",
    ["- app: FastAPI",
     "- redis_client: Redis  (optional, TTL = 86400s)",
     "- VALID_MODELS: set = { 'classical', 'quantum', 'pegasos' }"],
    ["+ startup_event(): void",
     "+ predict(file: UploadFile, model: str): PredictResponse",
     "+ predict_batch(files: List[UploadFile], model: str): BatchResponse",
     "+ predict_pdf(file: UploadFile, model: str, ...): FileResponse",
     "+ history(page: int): HistoryResponse"],
    FC_BLUE, EC_BLUE)
FA_CX  = LM + FA_W/2
FA_BOT = T1_BOT           # bottom edge of FA box
FA_TOP = T1_BOT + fa_h    # top edge of FA box

# ══════════════════════════════════════════════════════════════════════════════
# CONNECTORS
# The key rule: every line endpoint must be exactly ON a box edge.
# Box edges: left=x, right=x+w, bottom=y, top=y+h
# ══════════════════════════════════════════════════════════════════════════════

T2_CXs = [GK_CX, PRE_CX, FE_CX, DR_CX, GC_CX]
T3_CXs = [SVM_CX, QSVC_CX, PEG_CX]
DS = 0.14   # diamond half-size (must match diamond() s parameter)

# ── FA → T2 aggregation bus ───────────────────────────────────────────────────
# Vertical drop from FA bottom down to bus line
ln(FA_CX, FA_BOT, FA_CX, BUS2_Y, c=EC_BLUE)
# Horizontal bus across all T2 centres
ln(T2_CXs[0], BUS2_Y, T2_CXs[-1], BUS2_Y, c=EC_BLUE)
for cx in T2_CXs:
    # Diamond centred on bus
    diamond(cx, BUS2_Y, s=DS, ec=EC_BLUE)
    # Stub: T2 box top → bottom point of diamond (no gap)
    ln(cx, T2_TOP, cx, BUS2_Y - DS, c=EC_BLUE)

# ── FA → T3 dependency bus ────────────────────────────────────────────────────
# Dog-leg: FA bottom → right of all T2 boxes → down to BUS3_Y → T3 tops
DROP_X = T2_RIGHT + 0.30
ln(FA_CX, FA_BOT, DROP_X, FA_BOT, c=CC)    # horizontal at FA bottom
ln(DROP_X, FA_BOT, DROP_X, BUS3_Y, c=CC)   # vertical drop (outside T2)
ln(SVM_CX, BUS3_Y, DROP_X, BUS3_Y, c=CC)   # horizontal bus at T3 level
for cx in T3_CXs:
    ln(cx, T3_TOP, cx, BUS3_Y, c=CC)        # stub: T3 top → bus (no gap)

# ── FA → AuditDatabase  ──────────────────────────────────────────────────────
# Route: straight vertical drop in ADB_CX column (far left), FA_BOT → ADB_TOP.
# arrow enters the box top-centre, clearly visible and unambiguous.
# Label placed to the right of the line, in the gap between ADB and SVM columns.
ax.annotate("", xy=(ADB_CX, ADB_TOP), xytext=(ADB_CX, FA_BOT),
            arrowprops=dict(arrowstyle="-|>", color=EC_ORNG, lw=1.6,
                            mutation_scale=14, shrinkA=0, shrinkB=0), zorder=6)
# Label: just right of the orange arrow line, at T3-T4 mid-gap level
_ADB_MID_Y = T3_BOT - GAPV / 2           # midpoint of T3→T4 gap (empty space)
clbl(ADB_CX + 0.40, _ADB_MID_Y, "uses\n(audit log)", EC_ORNG)

# ── FA → PDFGenerator  ────────────────────────────────────────────────────────
# Route: straight vertical drop in PDF_CX column (far right), FA_BOT → PDF_TOP.
ax.annotate("", xy=(PDF_CX, PDF_TOP), xytext=(PDF_CX, FA_BOT),
            arrowprops=dict(arrowstyle="-|>", color=EC_RED, lw=1.6,
                            mutation_scale=14, shrinkA=0, shrinkB=0), zorder=6)
# Label: just left of the red arrow line, at T3-T4 mid-gap level
_PDF_MID_Y = T3_BOT - GAPV / 2           # midpoint of T3→T4 gap (empty space)
clbl(PDF_CX - 0.40, _PDF_MID_Y, "uses\n(pdf report)", EC_RED)

# ══════════════════════════════════════════════════════════════════════════════
# TITLE
# ══════════════════════════════════════════════════════════════════════════════
TITLE_Y = FA_TOP + 0.20
ax.text(FW/2, TITLE_Y,
        "Fig. 4.4  UML Class Diagram — QuCardio Core System",
        ha="center", va="bottom",
        fontsize=FS_TITLE, fontweight="bold", color=C_BLK, zorder=9)

# Clamp ylim to actual content (no blank top whitespace)
Y_TOP = TITLE_Y + 0.36
ax.set_ylim(T4_BOT - 0.54, Y_TOP)

# ══════════════════════════════════════════════════════════════════════════════
# LEGEND
# ══════════════════════════════════════════════════════════════════════════════
LEGEND = [
    (EC_BLUE, FC_BLUE,  "FastAPI / Pipeline"),
    (EC_GRN,  FC_GRN,   "Classical Classifier"),
    (EC_PURP, FC_PURP,  "Quantum Classifiers"),
    (EC_ORNG, FC_ORNG,  "AuditDatabase"),
    (EC_RED,  FC_RED,   "PDFGenerator"),
    (EC_GREY, FC_GREY,  "GradCAMModel"),
]
LITEM_W = (FW - LM - RM) / len(LEGEND)
LY = T4_BOT - 0.28
for k, (ec, fc, txt) in enumerate(LEGEND):
    rx = LM + k * LITEM_W
    ax.add_patch(mpatches.Rectangle(
        (rx, LY - 0.10), 0.22, 0.20,
        lw=1.2, edgecolor=ec, facecolor=fc, zorder=3))
    ax.text(rx + 0.30, LY, txt, ha="left", va="center",
            fontsize=FS_LEGEND, color=C_DARK, zorder=4)

OUT = "/home/karthik/projects/QuCardio/Doc/report/extracted_images/fig4_4_class_diagram.png"
plt.savefig(OUT, bbox_inches="tight", pad_inches=0.08, dpi=DPI, facecolor="white")
print(f"Saved: {OUT}")
