import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ------------------------------------------------------------------
# IEEE-style UML Sequence Diagram – POST /predict (QuCardio)
# Key fixes vs previous version:
#   1. Arrow tips (xy) point exactly to the lifeline centre-x, not
#      offset by ±10 pixels.  The line is shortened by the dx to
#      avoid the line poking through the arrowhead.
#   2. Message labels are placed ABOVE the arrow line on a white
#      background rectangle so they never obscure other elements.
#   3. All activation bars have high zorder so they sit on top of
#      dashed lifelines.
#   4. Font sizes increased: lifeline header=11, message labels=10
#      with generous figure width so nothing overlaps.
# ------------------------------------------------------------------

LWIDTH  = 200   # lifeline header box width
LHEIGHT = 50    # lifeline header box height
FONT_LL = 11    # lifeline name font size
FONT_MSG = 9.5  # message label font size

fig, ax = plt.subplots(figsize=(24, 16), dpi=300)
ax.set_xlim(0, 2400)
ax.set_ylim(0, 1440)
ax.axis('off')
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

# ── Title ──────────────────────────────────────────────────────────
ax.text(1200, 1410,
        "Fig. 4.5  UML Sequence Diagram — POST /predict Endpoint (QuCardio)",
        ha='center', va='center',
        fontsize=14, fontweight='bold', family='serif', color='#000000')

# ── Lifeline definitions ───────────────────────────────────────────
lifelines = [
    ("React\nClient",           150),
    ("FastAPI\nBackend",        430),
    ("RedisCache",              710),
    ("ECGGatekeeper",           990),
    ("FeatureExtractor",       1270),
    ("Dimensionality\nReducer",1550),
    ("QSVCClassifier",         1830),
    ("AuditDB",                2100),
]

TOP_Y    = 1320   # y of top lifeline boxes
BOTTOM_Y = 80     # y of bottom lifeline boxes

# Draw lifeline top + bottom headers, then dashed lines (lowest zorder)
for name, x in lifelines:
    for cy in [TOP_Y, BOTTOM_Y]:
        rect = patches.FancyBboxPatch(
            (x - LWIDTH/2, cy - LHEIGHT/2), LWIDTH, LHEIGHT,
            boxstyle="round,pad=3",
            linewidth=1.8, edgecolor='#2B6CB0', facecolor='#EBF8FF', zorder=8)
        ax.add_patch(rect)
        ax.text(x, cy, name,
                ha='center', va='center',
                fontsize=FONT_LL, fontweight='bold',
                color='#1A365D', zorder=9)

    # Dashed lifeline (drawn before activation bars)
    ax.plot([x, x], [TOP_Y - LHEIGHT/2, BOTTOM_Y + LHEIGHT/2],
            linestyle='--', color='#A0AEC0', linewidth=1.2, zorder=1)

# ── Activation bar helper ──────────────────────────────────────────
def draw_activation(x, y_top, y_bot, color='#BEE3F8'):
    """Draw a narrow activation rectangle. y_top > y_bot."""
    w = 18
    rect = patches.Rectangle(
        (x - w/2, y_bot), w, y_top - y_bot,
        linewidth=1.2, edgecolor='#2B6CB0', facecolor=color, zorder=5)
    ax.add_patch(rect)

# Long activations for React and FastAPI (spans full message exchange)
draw_activation(150,  1260, 125)
draw_activation(430,  1260, 125)

# Short activations per component
draw_activation(710,  1200, 1155)   # Redis lookup
draw_activation(990,  1090, 1040)   # Gatekeeper
draw_activation(1270,  995,  950)   # FeatureExtractor
draw_activation(1550,  900,  855)   # DimReducer
draw_activation(1830,  725,  660)   # QSVC
draw_activation(2100,  565,  520)   # AuditDB
draw_activation(710,   400,  355)   # Redis set

# ── Message arrow helper ───────────────────────────────────────────
ARROWHEAD_LEN = 12   # pixels to shorten the line so the arrowhead fits

def draw_message(x1, x2, y, text,
                 is_dashed=False,
                 line_color='#1A2030',
                 label_color='#1A2030',
                 label_above=True):
    """Draw a horizontal message arrow with label.

    The line is drawn from x1 to (x2 ± ARROWHEAD_LEN) so the
    arrowhead annotation tip sits exactly at x2 with no gap.
    """
    direction = 1 if x2 >= x1 else -1
    line_end   = x2 - direction * ARROWHEAD_LEN   # stop the line here
    style      = "--" if is_dashed else "-"

    ax.plot([x1, line_end], [y, y],
            linestyle=style, color=line_color, linewidth=1.4, zorder=3)

    # Arrowhead: tip at exactly x2
    ax.annotate('', xy=(x2, y), xytext=(line_end, y),
                arrowprops=dict(
                    arrowstyle="-|>" if not is_dashed else "->",
                    color=line_color, lw=1.4, mutation_scale=14))

    # Label on white background so it never hides other content
    mid_x   = (x1 + x2) / 2
    label_y = y + (9 if label_above else -9)
    ax.text(mid_x, label_y, text,
            ha='center', va='bottom' if label_above else 'top',
            fontsize=FONT_MSG, fontweight='normal',
            family='sans-serif', color=label_color,
            bbox=dict(boxstyle='square,pad=0.25',
                      facecolor='white', edgecolor='none', alpha=0.95),
            zorder=6)

# ── Step labels (left margin) ──────────────────────────────────────
def step_label(y, n):
    ax.text(18, y, str(n) + ".",
            ha='left', va='center',
            fontsize=FONT_MSG, fontweight='bold',
            color='#555555', zorder=7)

# ── Alt/loop frame helper ──────────────────────────────────────────
def draw_alt_frame(x, y_bot, w, h, label):
    rect = patches.FancyBboxPatch(
        (x, y_bot), w, h,
        boxstyle="square,pad=0",
        linewidth=1.4, edgecolor='#718096',
        facecolor='none', zorder=4)
    ax.add_patch(rect)
    # Small tab in top-left corner
    tab = patches.Rectangle(
        (x, y_bot + h - 22), 55, 22,
        linewidth=1.4, edgecolor='#718096',
        facecolor='#EDF2F7', zorder=4)
    ax.add_patch(tab)
    ax.text(x + 6, y_bot + h - 11, label,
            ha='left', va='center',
            fontsize=9, fontweight='bold',
            family='sans-serif', color='#2D3748', zorder=5)

# ══════════════════════════════════════════════════════════════════
# MESSAGES  (y values spaced ~110px apart for clear vertical rhythm)
# ══════════════════════════════════════════════════════════════════

# 1. POST /predict  ─────────────────────────────────────────────
step_label(1240, 1)
draw_message(150, 430, 1240,
             'POST /predict  (image_bytes, model="quantum")',
             label_color='#1A365D')

# 2. Redis cache check  ─────────────────────────────────────────
# alt frame wraps the two Redis arrows
draw_alt_frame(440, 1115, 310, 115, "alt")
ax.text(500, 1220, "[cache miss]",
        ha='left', va='center', fontsize=9, fontstyle='italic',
        color='#276749', zorder=6)
step_label(1185, 2)
draw_message(430, 710, 1185,
             'get( sha256(image) + "quantum" )')
draw_message(710, 430, 1148,
             'cache miss', is_dashed=True,
             line_color='#718096', label_color='#718096')

# [cache hit] dashed divider + note
ax.plot([440, 750], [1135, 1135],
        linestyle=':', color='#A0AEC0', linewidth=1.0, zorder=3)
ax.text(448, 1130, "[cache hit]  →  cached result (skip to step 10)",
        ha='left', va='top', fontsize=8.5, fontstyle='italic',
        color='#718096', zorder=6)

# 3. Gatekeeper validate  ──────────────────────────────────────
step_label(1080, 3)
draw_message(430, 990, 1080, 'validate( image )')
draw_message(990, 430, 1043,
             'is_ecg = True',
             is_dashed=True, line_color='#718096', label_color='#718096')

# 4. Feature extraction  ───────────────────────────────────────
step_label(980, 4)
draw_message(430, 1270, 980, 'extract( preprocessed_img )')
draw_message(1270, 430, 943,
             'feat[462400]',
             is_dashed=True, line_color='#718096', label_color='#718096')

# 5. Dimensionality reduction  ─────────────────────────────────
step_label(880, 5)
draw_message(430, 1550, 880, 'transform( feat )')
draw_message(1550, 430, 843,
             'x9d[9]',
             is_dashed=True, line_color='#718096', label_color='#718096')

# 6. Self-loop: scale to pi  ───────────────────────────────────
step_label(790, 6)
ax.plot([430, 500, 500, 430], [800, 800, 770, 770],
        color='#2B6CB0', linewidth=1.4, zorder=3)
ax.annotate('', xy=(430, 770), xytext=(444, 770),
            arrowprops=dict(arrowstyle="-|>", color='#2B6CB0',
                            lw=1.4, mutation_scale=12))
ax.text(510, 787,
        'x9d_pi = x9d × π   [0, π]',
        ha='left', va='center',
        fontsize=FONT_MSG, fontweight='normal',
        family='sans-serif', color='#2B6CB0',
        bbox=dict(boxstyle='square,pad=0.25',
                  facecolor='white', edgecolor='none', alpha=0.95),
        zorder=6)

# 7. QSVC predict  ─────────────────────────────────────────────
step_label(700, 7)
draw_message(430, 1830, 700, 'predict( x9d_pi )')
draw_message(1830, 430, 660,
             '(class_idx, probs)',
             is_dashed=True, line_color='#718096', label_color='#718096')

# 8. Audit log  ────────────────────────────────────────────────
step_label(555, 8)
draw_message(430, 2100, 555,
             'log(timestamp, hash, "quantum", class, conf)  [sync]')
draw_message(2100, 430, 515,
             'ack',
             is_dashed=True, line_color='#718096', label_color='#718096')

# 9. Redis set  ────────────────────────────────────────────────
step_label(400, 9)
draw_message(430, 710, 400,
             'set(key, result, TTL=86400s)')
draw_message(710, 430, 360,
             'ack',
             is_dashed=True, line_color='#718096', label_color='#718096')

# 10. Return JSON to React  ────────────────────────────────────
step_label(250, 10)
draw_message(430, 150, 250,
             'JSON {prediction, confidence, probabilities, image_b64, timing_ms}',
             is_dashed=True,
             line_color='#276749', label_color='#276749')

plt.tight_layout(pad=0.3)
target_path = 'Doc/report/extracted_images/fig4_5_sequence.png'
plt.savefig(target_path, bbox_inches='tight', dpi=300)
print(f"Saved sequence diagram → {target_path}")
