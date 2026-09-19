import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Canvas: tall enough that actor + 6 use cases never crowd ──────────
fig, ax = plt.subplots(figsize=(16, 14), dpi=200)
W, H = 1600, 1400
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

UC_EDGE  = "#1D4ED8"; UC_FILL = "#DBEAFE"; UC_TXT = "#1E3A5F"
INC_EDGE = "#065F46"; INC_FILL = "#D1FAE5"
ACTOR_C  = "#111827"
ARROW_C  = "#1D4ED8"
SYS_EDGE = "#111827"

# ── FONT SIZES (IEEE large-print level) ─────────────────────────────
FS_TITLE = 22    # system boundary title
FS_UC    = 16    # use-case ellipse labels
FS_INC   = 13    # <<include>> stereotype label
FS_ACTOR = 16    # actor label

# ── System boundary box ───────────────────────────────────────────────
SYS_LEFT  = 280
SYS_BOT   = 25
SYS_W     = 1280
SYS_H     = 1330
ax.add_patch(mpatches.FancyBboxPatch(
    (SYS_LEFT, SYS_BOT), SYS_W, SYS_H,
    boxstyle="square,pad=0", linewidth=2.8,
    edgecolor=SYS_EDGE, facecolor="white", zorder=1))
ax.text(SYS_LEFT + SYS_W/2, SYS_BOT + SYS_H - 22, "QuCardio System",
        ha="center", va="center",
        fontsize=FS_TITLE, fontweight="bold", color=SYS_EDGE, zorder=5)

# ── Actor (stick figure) ─────────────────────────────────────────────
def draw_actor(cx, cy, label):
    # head
    ax.add_patch(plt.Circle((cx, cy + 82), 24, color=ACTOR_C, zorder=5))
    # body
    ax.plot([cx, cx],            [cy + 58, cy],        color="black", lw=2.5, zorder=5)
    # arms
    ax.plot([cx - 38, cx + 38],  [cy + 35, cy + 35],   color="black", lw=2.5, zorder=5)
    # legs
    ax.plot([cx, cx - 34],       [cy, cy - 52],        color="black", lw=2.5, zorder=5)
    ax.plot([cx, cx + 34],       [cy, cy - 52],        color="black", lw=2.5, zorder=5)
    ax.text(cx, cy - 82, label,  ha="center", va="center",
            fontsize=FS_ACTOR, fontweight="bold", color="black", zorder=5)

# Actor centred vertically on the 6 use cases
ACTOR_CX = 145
ACTOR_CY = 680
draw_actor(ACTOR_CX, ACTOR_CY, "Clinician /\nUser")

# ── Use-case helper ───────────────────────────────────────────────────
def draw_uc(cx, cy, w, h, line1, line2="", filled=True):
    ec = UC_EDGE if filled else INC_EDGE
    fc = UC_FILL if filled else INC_FILL
    ax.add_patch(mpatches.Ellipse((cx, cy), w, h, linewidth=2.2,
                 edgecolor=ec, facecolor=fc, zorder=3))
    c = UC_TXT if filled else INC_EDGE
    if line2:
        ax.text(cx, cy + 12, line1, ha="center", va="center",
                fontsize=FS_UC, fontweight="bold", color=c, zorder=4)
        ax.text(cx, cy - 12, line2, ha="center", va="center",
                fontsize=FS_UC, fontweight="bold", color=c, zorder=4)
    else:
        ax.text(cx, cy, line1, ha="center", va="center",
                fontsize=FS_UC, fontweight="bold", color=c, zorder=4)

def draw_include(x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="->", color=ARROW_C,
                        lw=2.0, linestyle="dashed",
                        connectionstyle="arc3,rad=0.0"))
    ax.text((x1+x2)/2, (y1+y2)/2 + 16, "<<include>>",
            ha="center", va="bottom",
            fontsize=FS_INC, style="italic", color=ARROW_C,
            bbox=dict(boxstyle="square,pad=0.12", facecolor="white",
                      edgecolor="none", alpha=0.92), zorder=6)

def draw_assoc(x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="->", color="black", lw=2.0,
                        connectionstyle="arc3,rad=0.0"))

# ── Primary use-case column ───────────────────────────────────────────
UC_CX = 610
UC_W  = 310
UC_H  = 82
# 6 use cases evenly spaced
UC_YS = [1220, 1010, 810, 610, 410, 210]

draw_uc(UC_CX, UC_YS[0], UC_W, UC_H, "Upload ECG Image")
draw_uc(UC_CX, UC_YS[1], UC_W, UC_H, "Select Classifier", "(SVM / QSVC / Pegasos)")
draw_uc(UC_CX, UC_YS[2], UC_W, UC_H, "View Diagnosis", "and Confidence")
draw_uc(UC_CX, UC_YS[3], UC_W, UC_H, "View Probability", "Distribution")
draw_uc(UC_CX, UC_YS[4], UC_W, UC_H, "Download PDF", "Report")
draw_uc(UC_CX, UC_YS[5], UC_W, UC_H, "View Prediction", "History")

# Association lines: actor arm → left edge of each UC ellipse
for yy in UC_YS:
    draw_assoc(ACTOR_CX + 34, ACTOR_CY + 82 - (ACTOR_CY + 82 - yy)*0 ,
               ACTOR_CX, ACTOR_CY)  # placeholder rewritten below

# Redraw associations properly from actor torso to UC left edge
# (clear previous lines are not drawn yet; the loop above was placeholder)
# Correct actor body centre used for arrow origin
A_ORIGIN_X = ACTOR_CX + 2   # chest centre X
A_ORIGIN_Y = ACTOR_CY + 35  # torso mid Y

for yy in UC_YS:
    draw_assoc(A_ORIGIN_X, A_ORIGIN_Y, UC_CX - UC_W//2 - 4, yy)

# ── Included use-case column ─────────────────────────────────────────
INC_CX = 1220
INC_W  = 300
INC_H  = 82

draw_uc(INC_CX, UC_YS[0], INC_W, INC_H, "Validate ECG", "(Gatekeeper)", filled=False)
draw_include(UC_CX + UC_W//2, UC_YS[0], INC_CX - INC_W//2 - 4, UC_YS[0])

draw_uc(INC_CX, UC_YS[2], INC_W, INC_H, "Run Automated", "ECG Diagnosis", filled=False)
draw_include(UC_CX + UC_W//2, UC_YS[2], INC_CX - INC_W//2 - 4, UC_YS[2])

draw_uc(INC_CX, UC_YS[4], INC_W, INC_H, "Generate PDF", "Report", filled=False)
draw_include(UC_CX + UC_W//2, UC_YS[4], INC_CX - INC_W//2 - 4, UC_YS[4])

plt.tight_layout(pad=0.3)
OUT = "/home/karthik/projects/QuCardio/Doc/report/extracted_images/fig4_2_use_case.png"
plt.savefig(OUT, bbox_inches="tight", dpi=200)
print("Saved:", OUT)
