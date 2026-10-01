"""
Generate Figure 4.4 (Class Diagram) and Figure 4.5 (Sequence Diagram)
for the QuCardio BE project report.

Matches the style of the existing PNGs exactly:
  - Blue headers for FastAPI / Pipeline classes
  - Green for ClassicalSVMClassifier
  - Purple for Quantum classifiers
  - Orange for AuditDatabase
  - Red/salmon for PDFGenerator
  - Two-section UML boxes (attributes | methods)
  - Helvetica-like font (DejaVu Sans)
  - Titled with "Fig. X.X  UML ... — QuCardio ..."

Run:
    venv/bin/python3 scripts/generate_uml_diagrams.py
Outputs:
    Doc/report/extracted_images/fig4_4_class_diagram.png
    Doc/report/extracted_images/fig4_5_sequence.png
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import numpy as np
import os

OUT_DIR = "Doc/report/extracted_images"
os.makedirs(OUT_DIR, exist_ok=True)

# ─── colour palette (matches existing PNGs) ───────────────────────────────────
C_BLUE    = "#4A90D9"   # FastAPI / Pipeline header
C_BLUE_BG = "#D6E8F7"
C_GREEN   = "#4CAF50"   # Classical classifier header
C_GREEN_BG= "#E8F5E9"
C_PURPLE  = "#7E57C2"   # Quantum classifier header
C_PURPLE_BG="#EDE7F6"
C_ORANGE  = "#FF9800"   # AuditDatabase header
C_ORANGE_BG="#FFF3E0"
C_RED     = "#E53935"   # PDFGenerator header
C_RED_BG  = "#FFEBEE"
C_BODY    = "#FAFAFA"
C_BORDER  = "#90A4AE"
C_TEXT    = "#1A1A2E"
C_SUBTEXT = "#37474F"
MONO      = "DejaVu Sans Mono"
SANS      = "DejaVu Sans"

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def uml_box(ax, x, y, w, title, attrs, methods,
            hdr_color=C_BLUE, hdr_bg=C_BLUE_BG, stereotype=None,
            fs=7.2, lh=0.32):
    """
    Draw a UML class box.
    x, y   = bottom-left corner (in data coords)
    w      = width
    Returns total height used.
    """
    n_attr   = len(attrs)
    n_meth   = len(methods)
    hdr_h    = 0.52 if stereotype else 0.38
    attr_h   = max(n_attr * lh + 0.12, 0.22)
    meth_h   = max(n_meth * lh + 0.12, 0.22)
    total_h  = hdr_h + attr_h + meth_h

    # --- header ---
    ax.add_patch(mpatches.FancyBboxPatch(
        (x, y + attr_h + meth_h), w, hdr_h,
        boxstyle="square,pad=0", linewidth=0.8,
        edgecolor=hdr_color, facecolor=hdr_color))
    ty = y + attr_h + meth_h + hdr_h / 2
    if stereotype:
        ax.text(x + w/2, ty + 0.10, f"«{stereotype}»",
                ha="center", va="center", fontsize=fs-1, color="white",
                fontfamily=SANS, style="italic")
        ax.text(x + w/2, ty - 0.10, title,
                ha="center", va="center", fontsize=fs+0.5, color="white",
                fontfamily=SANS, fontweight="bold")
    else:
        ax.text(x + w/2, ty, title,
                ha="center", va="center", fontsize=fs+0.5, color="white",
                fontfamily=SANS, fontweight="bold")

    # --- attributes section ---
    ax.add_patch(mpatches.FancyBboxPatch(
        (x, y + meth_h), w, attr_h,
        boxstyle="square,pad=0", linewidth=0.8,
        edgecolor=hdr_color, facecolor=hdr_bg))
    for i, a in enumerate(attrs):
        ax.text(x + 0.06, y + meth_h + attr_h - 0.10 - i*lh, a,
                va="top", fontsize=fs, fontfamily=MONO, color=C_SUBTEXT)

    # --- divider ---
    ax.plot([x, x+w], [y + meth_h, y + meth_h],
            color=hdr_color, lw=0.6, alpha=0.6)

    # --- methods section ---
    ax.add_patch(mpatches.FancyBboxPatch(
        (x, y), w, meth_h,
        boxstyle="square,pad=0", linewidth=0.8,
        edgecolor=hdr_color, facecolor=C_BODY))
    for i, m in enumerate(methods):
        ax.text(x + 0.06, y + meth_h - 0.10 - i*lh, m,
                va="top", fontsize=fs, fontfamily=MONO, color=C_SUBTEXT)

    return total_h


def arrow(ax, x1, y1, x2, y2, style="->", color=C_BORDER, lw=1.0, label=None):
    """Draw an arrow between two points."""
    arrowprops = dict(
        arrowstyle=style, color=color, lw=lw,
        connectionstyle="arc3,rad=0.0"
    )
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=arrowprops)
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx, my + 0.05, label, ha="center", va="bottom",
                fontsize=6.5, color=C_SUBTEXT, fontfamily=SANS)


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4.4 — CLASS DIAGRAM
# ══════════════════════════════════════════════════════════════════════════════

def make_class_diagram():
    fig, ax = plt.subplots(figsize=(18, 12))
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 12)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    fig.suptitle("Fig. 4.4  UML Class Diagram — QuCardio Core System",
                 fontsize=12, fontweight="bold", fontfamily=SANS, y=0.98)

    # ── FastAPIApp (top centre) ────────────────────────────────────────────────
    uml_box(ax, 5.5, 8.2, 7.0,
            "FastAPIApp",
            ["- app: FastAPI",
             "- redis_client: Redis  (optional, TTL = 86400s)",
             "- VALID_MODELS: set = { 'classical', 'quantum', 'pegasos' }"],
            ["+ startup_event(): void",
             "+ predict(file: UploadFile, model: str): PredictResponse",
             "+ predict_batch(files: List[UploadFile], model: str): BatchResponse",
             "+ predict_pdf(file: UploadFile, model: str, ...): FileResponse",
             "+ history(page: int): HistoryResponse"],
            hdr_color=C_BLUE, hdr_bg=C_BLUE_BG)

    # ── Row 2: pipeline classes ────────────────────────────────────────────────
    # ECGGatekeeper
    uml_box(ax, 0.3, 4.2, 3.1,
            "ECGGatekeeper",
            ["- mobilenet_model: Model",
             "- mahal_guard: MahalanobisGuard",
             "- threshold: float = 0.5"],
            ["+ validate(img: ndarray): bool",
             "+ check_periodicity(img: ndarray): bool",
             "+ check_axis_angle(img: ndarray): bool",
             "+ check_mahalanobis(x9d): (bool, float)"],
            hdr_color=C_BLUE, hdr_bg=C_BLUE_BG)

    # preprocess_for_inference
    uml_box(ax, 3.7, 4.2, 3.2,
            "preprocess_for_inference",
            [],
            ["+ preprocess_for_inference(",
             "    img_bgr: ndarray):",
             "    ndarray[340, 340]"],
            hdr_color=C_BLUE, hdr_bg=C_BLUE_BG, stereotype="module")

    # FeatureExtractor
    uml_box(ax, 7.2, 4.2, 3.1,
            "FeatureExtractor",
            ["- resnet_pool1: Model",
             "  (ResNet50 pool1_pool)"],
            ["+ extract(img: ndarray):",
             "    ndarray[462400]"],
            hdr_color=C_BLUE, hdr_bg=C_BLUE_BG)

    # DimensionalityReducer
    uml_box(ax, 10.6, 4.2, 3.3,
            "DimensionalityReducer",
            ["- svd: TruncatedSVD(9)",
             "- scaler: MinMaxScaler"],
            ["+ transform(x: ndarray):",
             "    ndarray[9]  in [0, 1]"],
            hdr_color=C_BLUE, hdr_bg=C_BLUE_BG)

    # GradCAMModel
    uml_box(ax, 14.2, 4.2, 3.5,
            "GradCAMModel",
            ["- resnet_gradcam: Model",
             "  (conv5 + pool1)"],
            ["+ compute_gradcam(",
             "    img: ndarray):",
             "    ndarray[340, 340]"],
            hdr_color=C_BLUE, hdr_bg=C_BLUE_BG)

    # ── Row 3: classifiers + support ──────────────────────────────────────────
    # ClassicalSVMClassifier
    uml_box(ax, 0.3, 0.4, 3.3,
            "ClassicalSVMClassifier",
            ["- calibrated_svc: CalibratedClassifierCV",
             "- kernel: str = 'rbf'",
             "- C: float = 10.0"],
            ["+ predict(x: ndarray[9]):",
             "    tuple[int, ndarray[4]]",
             "+ predict_proba(x: ndarray[9]):",
             "    ndarray[4]"],
            hdr_color=C_GREEN, hdr_bg=C_GREEN_BG)

    # QSVCClassifier
    uml_box(ax, 3.9, 0.4, 3.5,
            "QSVCClassifier",
            ["- zzfeaturemap: ZZFeatureMap",
             "  (9q, reps=2, circular)",
             "- sv_train_cache:",
             "  ndarray[742, 512]"],
            ["+ kernel_row(x: ndarray[9]):",
             "    ndarray[742]",
             "+ predict(x: ndarray[9]):",
             "    tuple[int, ndarray[4]]"],
            hdr_color=C_PURPLE, hdr_bg=C_PURPLE_BG, stereotype="Quantum")

    # PegasosMulticlassClassifier
    uml_box(ax, 7.7, 0.4, 3.8,
            "PegasosMulticlassClassifier",
            ["- binary_models:",
             "  dict[(c1,c2)->PegasosQSVC]",
             "- CLASS_PAIRS: list (6 pairs)"],
            ["+ pegasos_multiclass_predict(",
             "    k_row: ndarray): int",
             "+ predict_node(k_row,",
             "    c1, c2): int"],
            hdr_color=C_PURPLE, hdr_bg=C_PURPLE_BG, stereotype="Quantum")

    # AuditDatabase
    uml_box(ax, 11.8, 0.4, 3.1,
            "AuditDatabase",
            ["- db_path: str",
             "- conn: sqlite3.Connection"],
            ["+ init_db(): void",
             "+ log_prediction(patient_id, hash,",
             "    model, cls, conf): int",
             "+ history(page: int): list"],
            hdr_color=C_ORANGE, hdr_bg=C_ORANGE_BG)

    # PDFGenerator
    uml_box(ax, 15.2, 0.4, 2.5,
            "PDFGenerator",
            ["- template_env: Jinja2Env",
             "- css_styles: str"],
            ["+ generate_pdf_report(",
             "    result: dict,",
             "    filename: str): bytes"],
            hdr_color=C_RED, hdr_bg=C_RED_BG)

    # ── AGGREGATION ARROWS (FastAPIApp → components) ──────────────────────────
    # FastAPIApp centre-bottom → each row-2 class top
    app_cx = 9.0   # FastAPIApp centre-x
    app_by = 8.2   # FastAPIApp bottom-y
    targets_r2 = [1.85, 5.3, 8.75, 12.25, 15.95]  # centre-x of row-2 boxes
    for tx in targets_r2:
        ax.annotate("", xy=(tx, 7.15), xytext=(app_cx, app_by),
                    arrowprops=dict(arrowstyle="-|>", color=C_BLUE,
                                   lw=1.0, mutation_scale=10))

    # FastAPIApp → row-3 (classifiers, audit, pdf) via vertical drop
    targets_r3 = [1.95, 5.65, 9.6, 13.35, 16.45]
    for tx in targets_r3:
        # junction at y=4.0 then down
        ax.plot([app_cx, app_cx], [app_by, 3.95], color=C_BORDER, lw=0.7, ls="--")
        ax.annotate("", xy=(tx, 3.75), xytext=(tx, 4.0),
                    arrowprops=dict(arrowstyle="-|>", color=C_BORDER,
                                   lw=0.8, mutation_scale=9))
    ax.plot([targets_r3[0], targets_r3[-1]], [3.95, 3.95],
            color=C_BORDER, lw=0.7, ls="--")

    # ── LEGEND ────────────────────────────────────────────────────────────────
    legend_patches = [
        mpatches.Patch(color=C_BLUE,   label="FastAPI / Pipeline"),
        mpatches.Patch(color=C_GREEN,  label="Classical Classifier"),
        mpatches.Patch(color=C_PURPLE, label="Quantum Classifiers"),
        mpatches.Patch(color=C_ORANGE, label="AuditDatabase"),
        mpatches.Patch(color=C_RED,    label="PDFGenerator"),
    ]
    ax.legend(handles=legend_patches, loc="lower left", fontsize=7.5,
              framealpha=0.9, edgecolor=C_BORDER, ncol=5,
              bbox_to_anchor=(0.0, 0.0))

    out = os.path.join(OUT_DIR, "fig4_4_class_diagram.png")
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved → {out}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4.5 — SEQUENCE DIAGRAM
# ══════════════════════════════════════════════════════════════════════════════

def make_sequence_diagram():
    FW, FH = 22, 18
    fig, ax = plt.subplots(figsize=(FW, FH))
    ax.set_xlim(0, FW)
    ax.set_ylim(0, FH)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    fig.suptitle(
        "Fig. 4.5  UML Sequence Diagram — POST /predict Endpoint (QuCardio)",
        fontsize=13, fontweight="bold", fontfamily=SANS, y=0.99)

    # ── Lifeline x-positions ──────────────────────────────────────────────────
    XR   = 1.2   # React Client
    XF   = 3.6   # FastAPI Backend
    XRED = 5.8   # RedisCache
    XG   = 8.1   # ECGGatekeeper
    XFE  = 10.4  # FeatureExtractor
    XDR  = 12.7  # DimensionalityReducer
    XSVC = 15.2  # QSVCClassifier
    XADB = 18.2  # AuditDB

    lifelines = [
        (XR,   "React",          "Client",          C_BLUE),
        (XF,   "FastAPI",        "Backend",         C_BLUE),
        (XRED, "RedisCache",     "",                C_BLUE),
        (XG,   "ECGGatekeeper",  "",                C_BLUE),
        (XFE,  "FeatureExtractor","",               C_BLUE),
        (XDR,  "Dimensionality", "Reducer",         C_BLUE),
        (XSVC, "QSVCClassifier", "",                C_PURPLE),
        (XADB, "AuditDB",        "",                C_ORANGE),
    ]

    TOP_Y = FH - 0.55
    BOT_Y = 0.45
    HDR_H = 0.60
    HDR_W = 1.80

    for (cx, l1, l2, col) in lifelines:
        # top header box
        ax.add_patch(mpatches.FancyBboxPatch(
            (cx - HDR_W/2, TOP_Y - HDR_H), HDR_W, HDR_H,
            boxstyle="square,pad=0", linewidth=1.1,
            edgecolor=col, facecolor=col))
        dy = 0.07 if l2 else 0
        ax.text(cx, TOP_Y - HDR_H/2 + dy, l1,
                ha="center", va="center", fontsize=8.5,
                color="white", fontfamily=SANS, fontweight="bold")
        if l2:
            ax.text(cx, TOP_Y - HDR_H/2 - 0.14, l2,
                    ha="center", va="center", fontsize=8.5,
                    color="white", fontfamily=SANS, fontweight="bold")
        # dashed lifeline
        ax.plot([cx, cx], [TOP_Y - HDR_H, BOT_Y + HDR_H],
                color=col, lw=0.9, ls="--", alpha=0.45, zorder=0)
        # bottom mirror box
        ax.add_patch(mpatches.FancyBboxPatch(
            (cx - HDR_W/2, BOT_Y), HDR_W, HDR_H,
            boxstyle="square,pad=0", linewidth=1.1,
            edgecolor=col, facecolor=col))
        ax.text(cx, BOT_Y + HDR_H/2 + dy, l1,
                ha="center", va="center", fontsize=8.5,
                color="white", fontfamily=SANS, fontweight="bold")
        if l2:
            ax.text(cx, BOT_Y + HDR_H/2 - 0.14, l2,
                    ha="center", va="center", fontsize=8.5,
                    color="white", fontfamily=SANS, fontweight="bold")

    # FastAPI activation bar (runs for whole pipeline)
    ACT_TOP = TOP_Y - HDR_H - 0.05
    ACT_BOT = BOT_Y + HDR_H + 0.05
    ax.add_patch(mpatches.FancyBboxPatch(
        (XF - 0.10, ACT_BOT), 0.20, ACT_TOP - ACT_BOT,
        boxstyle="square,pad=0", linewidth=0.7,
        edgecolor=C_BLUE, facecolor=C_BLUE_BG, zorder=2))

    # ── Helpers ───────────────────────────────────────────────────────────────
    def msg(y, x1, x2, label, num=None, ret=False,
            col=C_TEXT, act_x=None, act_h=0.40, fs=7.5):
        style = "<-" if ret else "->"
        lw    = 0.9 if ret else 1.1
        ls_conn = "arc3,rad=0.0"
        ax.annotate("", xy=(x2, y), xytext=(x1, y),
                    arrowprops=dict(arrowstyle=style, color=col, lw=lw,
                                   mutation_scale=11,
                                   connectionstyle=ls_conn),
                    zorder=3)
        lx = (x1 + x2) / 2
        txt = (f"{num}. " if num else "    ") + label
        ax.text(lx, y + 0.10, txt, ha="center", va="bottom",
                fontsize=fs, fontfamily=SANS, color=col)
        if act_x is not None and not ret:
            ax.add_patch(mpatches.FancyBboxPatch(
                (act_x - 0.075, y - act_h), 0.15, act_h,
                boxstyle="square,pad=0", linewidth=0.5,
                edgecolor=C_BORDER, facecolor="#CFD8DC", zorder=4))

    def note_box(y, x, w, text, ec="#7E57C2", fc="#EDE7F6", tcol="#4527A0"):
        ax.add_patch(mpatches.FancyBboxPatch(
            (x, y - 0.16), w, 0.34,
            boxstyle="round,pad=0.05", linewidth=0.7,
            edgecolor=ec, facecolor=fc, zorder=5))
        ax.text(x + 0.12, y + 0.01, text,
                va="center", fontsize=7.0, fontfamily=SANS, color=tcol)

    def subtext(y, x, text, col="#546E7A"):
        ax.text(x, y, text, ha="left", va="center",
                fontsize=6.8, fontfamily=SANS, color=col, style="italic")

    # ── Step Y coordinates — evenly spaced from top to bottom ─────────────────
    #  step:  y
    S = {
        1:   15.85,   # POST /predict
        "2s": 15.35,  # cache get →
        "2r": 14.90,  # ← cache miss
        # alt frame spans 14.55 – 15.60
        3:   14.10,   # validate(image) →
        "3a": 13.72,  # sub-step notes
        "3b": 13.44,
        "3c": 13.16,
        "3r": 12.70,  # ← is_ecg=True
        4:   12.10,   # extract(img) →
        "4r": 11.65,  # ← feat[462400]
        5:   11.10,   # transform(feat) →
        "5r": 10.65,  # ← x9d[9]
        "5b": 10.18,  # mahalanobis note
        6:    9.68,   # ×π note
        7:    9.10,   # predict(x9d_pi) →
        "7r":  8.62,  # ← (class_idx, probs)
        8:    8.00,   # log(…) →
        "8r":  7.55,  # ← ack
        9:    6.90,   # cache set →
        "9r":  6.45,  # ← ack
        10:   5.70,   # ← JSON response
    }

    # ── ALT frame (cache miss / cache hit) ────────────────────────────────────
    alt_bot = S["2r"] - 0.35
    alt_top = S["2s"] + 0.55
    ax.add_patch(mpatches.FancyBboxPatch(
        (XF - 0.2, alt_bot), XADB - XF + 0.4, alt_top - alt_bot,
        boxstyle="square,pad=0", linewidth=0.9,
        edgecolor="#F9A825", facecolor="#FFFDE7", alpha=0.55, zorder=1))
    ax.text(XF - 0.05, alt_top - 0.06, "alt",
            ha="left", va="top", fontsize=8, fontfamily=SANS,
            fontweight="bold", color="#F57F17")
    ax.text(XF + 0.42, alt_top - 0.06, "[cache miss]",
            ha="left", va="top", fontsize=7.2, fontfamily=SANS,
            color="#795548", style="italic")
    # divider between cache miss / cache hit
    div_y = (S["2s"] + S["2r"]) / 2 - 0.05
    ax.plot([XF - 0.2, XADB + 0.6], [div_y, div_y],
            color="#F9A825", lw=0.7, ls="--")
    ax.text(XF - 0.05, div_y - 0.07,
            "[cache hit]  →  return cached JSON directly (skip to step 10)",
            ha="left", va="top", fontsize=7.0, fontfamily=SANS,
            color="#795548", style="italic")

    # ── MESSAGES ──────────────────────────────────────────────────────────────

    # 1. POST /predict
    msg(S[1], XR, XF, "POST /predict (image_bytes, model=\"quantum\")", num=1)

    # 2. Redis lookup
    msg(S["2s"], XF, XRED, "get( sha256(image) + \"quantum\" )", num=2,
        act_x=XRED, act_h=0.38)
    msg(S["2r"], XRED, XF, "cache miss", ret=True, col="#78909C")

    # 3. ECG gate
    msg(S[3], XF, XG, "validate( image )", num=3, act_x=XG, act_h=1.35)
    subtext(S["3a"], XG + 0.14, "3a. MobileNetV2 binary classifier  (prob_ecg ≥ 0.5)")
    subtext(S["3b"], XG + 0.14, "3b. periodicity score ≥ 0.14  (on preprocessed 340×340 crop)")
    subtext(S["3c"], XG + 0.14, "3c. axis-angle fraction ≤ 0.42  (on preprocessed 340×340 crop)")
    msg(S["3r"], XG, XF, "is_ecg = True", ret=True, col="#78909C")
    ax.annotate("HTTP 422 if any check fails",
                xy=(XF + 0.18, S[3] - 0.65),
                xytext=(XF + 1.6, S[3] - 0.25),
                fontsize=7.0, color="#E53935", fontfamily=SANS,
                arrowprops=dict(arrowstyle="->", color="#E53935", lw=0.8))

    # 4. Extract
    msg(S[4], XF, XFE, "extract( preprocessed_img )", num=4,
        act_x=XFE, act_h=0.40)
    msg(S["4r"], XFE, XF, "feat[462400]", ret=True, col="#78909C")

    # 5. SVD + MinMax
    msg(S[5], XF, XDR, "transform( feat )", num=5, act_x=XDR, act_h=0.40)
    msg(S["5r"], XDR, XF, "x9d[9]", ret=True, col="#78909C")

    # 5b. Mahalanobis note
    note_box(S["5b"], XF + 0.15, 9.5,
             "5b.  Mahalanobis distance check  →  ood_flag + ood_distance"
             " appended to response  (non-blocking — classification always proceeds)",
             ec="#7E57C2", fc="#EDE7F6", tcol="#4527A0")

    # 6. ×π note
    note_box(S[6], XF + 0.15, 8.8,
             "6.   x9d_pi = x9d × \u03c0   \u2192   [0, \u03c0]"
             "   (quantum model only;  classical SVM uses [0,1] directly)",
             ec=C_BLUE, fc=C_BLUE_BG, tcol=C_SUBTEXT)

    # 7. Classify
    msg(S[7], XF, XSVC, "predict( x9d_pi )", num=7, act_x=XSVC, act_h=0.40)
    msg(S["7r"], XSVC, XF, "( class_idx, probs )", ret=True, col="#78909C")

    # 8. Audit log
    msg(S[8], XF, XADB,
        "log(timestamp, hash, \"quantum\", class, conf)  [sync]",
        num=8, act_x=XADB, act_h=0.40)
    msg(S["8r"], XADB, XF, "ack", ret=True, col="#78909C")

    # 9. Cache set
    msg(S[9], XF, XRED, "set(key, result, TTL=86400s)", num=9,
        act_x=XRED, act_h=0.40)
    msg(S["9r"], XRED, XF, "ack", ret=True, col="#78909C")

    # 10. Final response
    msg(S[10], XF, XR,
        "JSON { prediction, confidence, probabilities, image_b64, timing_ms, ood_flag }",
        num=10, ret=True, col="#1565C0", fs=7.5)

    out = os.path.join(OUT_DIR, "fig4_5_sequence.png")
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved → {out}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Generating UML diagrams...")
    make_class_diagram()
    make_sequence_diagram()
    print("Done.")
