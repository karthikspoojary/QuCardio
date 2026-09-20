import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Polygon

# ------------------------------------------------------------------
# IEEE-style UML Class Diagram  –  QuCardio Core System
# Key fixes vs previous version:
#   1. All connector lines are drawn BEFORE class boxes (low zorder)
#      so they never overlap/cover text.
#   2. Arrow-tip placement corrected: the annotate xy is exactly the
#      top edge of the target box – no pixel gap.
#   3. Aggregation diamonds sit flush on the bottom edge of FastAPIApp.
#   4. Font sizes increased: header=12, body=9.5 (no layout overflow).
# ------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(22, 14), dpi=300)
ax.set_xlim(0, 2200)
ax.set_ylim(0, 1400)
ax.axis('off')
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

# Title (IEEE caption style)
ax.text(1100, 1370,
        "Fig. 4.4  UML Class Diagram — QuCardio Core System",
        ha='center', va='center',
        fontsize=14, fontweight='bold', family='serif', color='#000000')

# ── Constants ──────────────────────────────────────────────────────
HEADER_H   = 42      # height of the class-name banner
LINE_H     = 21      # vertical step per text line
PAD_TOP    = 14      # space between a divider and first text line
FONT_BODY  = 9.5     # monospace body text
FONT_TITLE = 12.0    # class-name text


def draw_uml_class(ax, x, y, w, h,
                   title, fields, methods,
                   header_color='#BEE3F8',
                   border_color='#2B6CB0',
                   zorder_box=10):
    """Draw a UML class box.  All text/box zorders are >= zorder_box.
       Connectors must be drawn BEFORE calling this with lower zorder."""

    # Outer rectangle
    rect = patches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="square,pad=0",
        linewidth=1.8, edgecolor=border_color,
        facecolor='#FFFFFF', zorder=zorder_box)
    ax.add_patch(rect)

    # Header band
    h_rect = patches.Rectangle(
        (x, y + h - HEADER_H), w, HEADER_H,
        linewidth=1.8, edgecolor=border_color,
        facecolor=header_color, zorder=zorder_box + 1)
    ax.add_patch(h_rect)

    # Class title
    ax.text(x + w / 2, y + h - HEADER_H / 2,
            title, ha='center', va='center',
            fontsize=FONT_TITLE, fontweight='bold',
            family='sans-serif', color='#1A365D', zorder=zorder_box + 2)

    # Field section height
    f_height = LINE_H * max(len(fields), 1) + 6
    f_top    = y + h - HEADER_H         # top of the fields area

    # Divider line between header and fields
    ax.plot([x, x + w], [f_top, f_top],
            color=border_color, linewidth=1.2, zorder=zorder_box + 1)

    # Divider line between fields and methods
    div_y = f_top - f_height
    ax.plot([x, x + w], [div_y, div_y],
            color=border_color, linewidth=1.2, zorder=zorder_box + 1)

    # Fields
    curr_y = f_top - PAD_TOP
    for f in fields:
        ax.text(x + 12, curr_y, f,
                ha='left', va='center',
                fontsize=FONT_BODY, family='monospace',
                color='#2D3748', zorder=zorder_box + 2)
        curr_y -= LINE_H

    # Methods
    curr_y = div_y - PAD_TOP
    for m in methods:
        ax.text(x + 12, curr_y, m,
                ha='left', va='center',
                fontsize=FONT_BODY, family='monospace',
                color='#2D3748', zorder=zorder_box + 2)
        curr_y -= LINE_H


# ══════════════════════════════════════════════════════════════════
#  LAYOUT  (coordinates tuned for the larger canvas 2200 × 1400)
#
#   Row 0  –  FastAPIApp banner   y=1080..1330   (h=250)
#   Row 1  –  Pipeline boxes      y= 630..1000   (h=220 approx.)
#   Row 2  –  Classifiers + svc   y= 130.. 520   (h=250 approx.)
# ══════════════════════════════════════════════════════════════════

# ── Row 0: FastAPIApp ──────────────────────────────────────────────
FA_X, FA_Y, FA_W, FA_H = 600, 1060, 1000, 270
fastapi_fields = [
    "- app: FastAPI",
    "- redis_client: Redis  (optional, TTL = 86400s)",
    "- VALID_MODELS: set = { 'classical', 'quantum', 'pegasos' }",
]
fastapi_methods = [
    "+ startup_event(): void",
    "+ predict(file: UploadFile, model: str): PredictResponse",
    "+ predict_batch(files: List[UploadFile], model: str): BatchResponse",
    "+ predict_pdf(file: UploadFile, model: str, ...): FileResponse",
    "+ history(page: int): HistoryResponse",
]

# Aggregation diamond dimensions (drawn BEFORE the box so it sits behind nothing)
DIA_HALF  = 12          # half-width of diamond
DIA_FULL  = DIA_HALF * 2

# Diamond centre: horizontally centred on FastAPIApp, vertically at its bottom edge
dia_cx = FA_X + FA_W / 2   # 1100
dia_cy = FA_Y               # bottom of FastAPIApp box

# ── CONNECTORS (drawn first – zorder ≤ 5 – so boxes cover them) ──

# Vertical trunk: from diamond tip downward to horizontal bus
BUS1_Y = 960    # y of horizontal distribution bus 1 (into pipeline)
BUS2_Y = 500    # y of horizontal distribution bus 2 (into classifiers/services)

# Pipeline drop points (x of each box centre in Row 1)
ROW1_CENTRES = [200, 590, 1000, 1410, 1720]   # approx centre x values below
ROW2_CENTRES = [195, 570, 950, 1370, 1730, 2020]

# Draw horizontal buses FIRST (zorder=1)
bus_left1  = ROW1_CENTRES[0]
bus_right1 = ROW1_CENTRES[-1]
bus_left2  = ROW2_CENTRES[0]
bus_right2 = ROW2_CENTRES[-1]

ax.plot([bus_left1, bus_right1], [BUS1_Y, BUS1_Y],
        color='#2B6CB0', linewidth=1.6, zorder=1)
ax.plot([bus_left2, bus_right2], [BUS2_Y, BUS2_Y],
        color='#2B6CB0', linewidth=1.6, zorder=1)

# Vertical trunk from diamond down through both buses
ax.plot([dia_cx, dia_cx], [dia_cy, BUS2_Y],
        color='#2B6CB0', linewidth=1.6, zorder=1)

# ── Aggregation diamond (open, "has-a", sits at bottom of FastAPIApp)
dia_pts = [
    (dia_cx,            dia_cy),          # bottom tip
    (dia_cx - DIA_HALF, dia_cy + DIA_FULL//2 + 2),  # left
    (dia_cx,            dia_cy + DIA_FULL + 4),       # top tip
    (dia_cx + DIA_HALF, dia_cy + DIA_FULL//2 + 2),   # right
]
diamond = Polygon(dia_pts, closed=True,
                  edgecolor='#2B6CB0', facecolor='white',
                  linewidth=1.8, zorder=6)
ax.add_patch(diamond)

# ── Row 1 pipeline component positions ────────────────────────────
#   ECGGatekeeper   x=40  w=310
#   PreprocessModule x=385 w=370
#   FeatureExtractor x=795 w=370
#   DimensionalityReducer x=1210 w=380
#   GradCAMModel    x=1630 w=300

R1 = [
    dict(x=40,   y=720, w=310, h=220,
         title="ECGGatekeeper",
         fields=["- mobilenet_model: Model", "- threshold: float = 0.5"],
         methods=["+ validate(img: ndarray):", "    bool"],
         hc='#BEE3F8', bc='#2B6CB0'),
    dict(x=390,  y=700, w=380, h=240,
         title="preprocess_for_inference",
         fields=["<<module>>"],
         methods=["+ preprocess_for_inference(",
                  "    img_bgr: ndarray):",
                  "    ndarray[340, 340]"],
         hc='#BEE3F8', bc='#2B6CB0'),
    dict(x=820,  y=720, w=360, h=220,
         title="FeatureExtractor",
         fields=["- resnet_pool1: Model", "    (ResNet50 pool1_pool)"],
         methods=["+ extract(img: ndarray):", "    ndarray[462400]"],
         hc='#BEE3F8', bc='#2B6CB0'),
    dict(x=1230, y=700, w=380, h=240,
         title="DimensionalityReducer",
         fields=["- svd: TruncatedSVD(9)", "- scaler: MinMaxScaler"],
         methods=["+ transform(x: ndarray):", "    ndarray[9]  in [0, 1]"],
         hc='#BEE3F8', bc='#2B6CB0'),
    dict(x=1665, y=720, w=320, h=220,
         title="GradCAMModel",
         fields=["- resnet_gradcam: Model", "    (conv5 + pool1)"],
         methods=["+ compute_gradcam(", "    img: ndarray):", "    ndarray[340, 340]"],
         hc='#BEE3F8', bc='#2B6CB0'),
]

# Vertical drops from bus1 to top of each Row1 box (drawn before boxes)
for info in R1:
    cx   = info['x'] + info['w'] / 2
    top  = info['y'] + info['h']       # top edge of the box
    # vertical from bus down to just above box top
    ax.plot([cx, cx], [BUS1_Y, top],
            color='#2B6CB0', linewidth=1.6, zorder=1)
    # arrowhead pointing DOWN into the box (tip = top of box)
    ax.annotate('', xy=(cx, top), xytext=(cx, top + 18),
                arrowprops=dict(arrowstyle="-|>", color='#2B6CB0',
                                lw=1.5, mutation_scale=14))

# ── Row 2 component positions ─────────────────────────────────────
R2 = [
    dict(x=30,   y=130, w=320, h=260,
         title="ClassicalSVMClassifier",
         fields=["- calibrated_svc:",
                 "    CalibratedClassifierCV",
                 "- kernel: str = 'rbf'",
                 "- C: float = 10.0"],
         methods=["+ predict(x: ndarray[9]):",
                  "    tuple[int, ndarray[4]]",
                  "+ predict_proba(x: ndarray[9]):",
                  "    ndarray[4]"],
         hc='#C6F6D5', bc='#276749'),
    dict(x=395,  y=110, w=360, h=280,
         title="QSVCClassifier",
         fields=["<<Quantum>>",
                 "- zzfeaturemap: ZZFeatureMap",
                 "    (9q, reps=2, circular)",
                 "- sv_train_cache:",
                 "    ndarray[742, 512]"],
         methods=["+ kernel_row(x: ndarray[9]):",
                  "    ndarray[742]",
                  "+ predict(x: ndarray[9]):",
                  "    tuple[int, ndarray[4]]"],
         hc='#E9D8FD', bc='#553C9A'),
    dict(x=810,  y=110, w=390, h=280,
         title="PegasosMulticlassClassifier",
         fields=["<<Quantum>>",
                 "- binary_models:",
                 "    dict[(c1,c2)->PegasosQSVC]",
                 "- CLASS_PAIRS: list (6 pairs)"],
         methods=["+ pegasos_multiclass_predict(",
                  "    k_row: ndarray): int",
                  "+ predict_node(k_row,",
                  "    c1, c2): int"],
         hc='#E9D8FD', bc='#553C9A'),
    dict(x=1260, y=130, w=340, h=260,
         title="AuditDatabase",
         fields=["- db_path: str",
                 "- conn: sqlite3.Connection"],
         methods=["+ init_db(): void",
                  "+ log_prediction(patient_id, hash,",
                  "    model, cls, conf): int",
                  "+ history(page: int): list"],
         hc='#FEEBC8', bc='#9C4221'),
    dict(x=1660, y=130, w=360, h=260,
         title="PDFGenerator",
         fields=["- template_env: Jinja2Env",
                 "- css_styles: str"],
         methods=["+ generate_pdf_report(",
                  "    result: dict,",
                  "    filename: str): bytes"],
         hc='#FED7D7', bc='#9B2C2C'),
]

# Vertical drops from bus2 to top of each Row2 box
for info in R2:
    cx  = info['x'] + info['w'] / 2
    top = info['y'] + info['h']
    ax.plot([cx, cx], [BUS2_Y, top],
            color='#2B6CB0', linewidth=1.6, zorder=1)
    ax.annotate('', xy=(cx, top), xytext=(cx, top + 18),
                arrowprops=dict(arrowstyle="-|>", color='#2B6CB0',
                                lw=1.5, mutation_scale=14))

# ── Now draw all class boxes ON TOP of connectors ─────────────────
draw_uml_class(ax, FA_X, FA_Y, FA_W, FA_H,
               "FastAPIApp",
               fastapi_fields, fastapi_methods,
               header_color='#90CDF4', border_color='#2B6CB0',
               zorder_box=10)

for info in R1:
    draw_uml_class(ax, info['x'], info['y'], info['w'], info['h'],
                   info['title'], info['fields'], info['methods'],
                   header_color=info['hc'], border_color=info['bc'],
                   zorder_box=10)

for info in R2:
    draw_uml_class(ax, info['x'], info['y'], info['w'], info['h'],
                   info['title'], info['fields'], info['methods'],
                   header_color=info['hc'], border_color=info['bc'],
                   zorder_box=10)

# ── Legend ────────────────────────────────────────────────────────
legend_items = [
    ('#90CDF4', '#2B6CB0', "FastAPI / Pipeline"),
    ('#C6F6D5', '#276749', "Classical Classifier"),
    ('#E9D8FD', '#553C9A', "Quantum Classifiers"),
    ('#FEEBC8', '#9C4221', "AuditDatabase"),
    ('#FED7D7', '#9B2C2C', "PDFGenerator"),
]
lx, ly = 60, 55
for fc, ec, label in legend_items:
    rect = patches.Rectangle((lx, ly - 8), 22, 16,
                              linewidth=1.2, edgecolor=ec, facecolor=fc, zorder=15)
    ax.add_patch(rect)
    ax.text(lx + 28, ly, label, ha='left', va='center',
            fontsize=9, family='sans-serif', color='#2D3748', zorder=15)
    lx += 230

plt.tight_layout(pad=0.3)
target_path = 'Doc/report/extracted_images/fig4_4_class_diagram.png'
plt.savefig(target_path, bbox_inches='tight', dpi=300)
print(f"Saved class diagram → {target_path}")
