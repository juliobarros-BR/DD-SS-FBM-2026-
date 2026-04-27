import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

# ============================================================
# CONFIG
# ============================================================

MODEL_FILE = "best_cand_MS_history.csv"
REFERENCE_FILE = "reference_mechanosorption_model.csv"
OUTPUT_PLOT = "model_vs_experiment_components.png"

FIGSIZE = (16, 12)

# ============================================================
# STYLE
# ============================================================

mpl.rcParams.update(mpl.rcParamsDefault)


plt.rcParams.update({
    'font.size': 28,
    'axes.labelsize': 28,
    'xtick.labelsize': 23,
    'ytick.labelsize': 23,
    'legend.fontsize': 24,
    'figure.titlesize': 25,
    'lines.linewidth': 3.5,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'Computer Modern Roman'],
    'mathtext.fontset': 'cm',
    'mathtext.default': 'it',
    'text.usetex': True,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.major.width': 1.5,
    'ytick.major.width': 1.5,
    'axes.linewidth': 1,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'legend.framealpha': 1.0,
    'legend.edgecolor': 'black',
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'figure.dpi': 150,
})

# ============================================================
# HELPERS
# ============================================================

def interp_previous(x_new, x, y):
    idx = np.searchsorted(x, x_new, side="right") - 1
    idx = np.clip(idx, 0, len(y)-1)
    return y[idx]

def fix_initial_zero(arr):
    arr = arr.copy()
    mask = arr > 0
    if np.any(mask):
        arr[np.argmax(mask)] = 0.0
    return arr

def get_loading_mask(load, threshold=1e-6):
    return load > threshold

def get_mask_intervals(x, mask):
    intervals = []
    start = None
    for i, m in enumerate(mask):
        if m and start is None:
            start = x[i]
        elif not m and start is not None:
            intervals.append((start, x[i]))
            start = None
    if start is not None:
        intervals.append((start, x[-1]))
    return intervals

def apply_loading_shade(ax, intervals):
    for x0, x1 in intervals:
        ax.axvspan(x0, x1, color="gray", alpha=0.10, zorder=0)

from matplotlib.patches import FancyBboxPatch

# def annotate_right_figure(fig, ax, labels, colors, spacing=0.05):
#     pos = ax.get_position()
#     y_mid = 0.5 * (pos.y0 + pos.y1)

#     n = len(labels)
#     offsets = -(np.arange(n) - (n - 1) / 2)
#     ys = y_mid + offsets * spacing

#     x_text = 0.87

#     # ---- draw box FIRST ----
#     pad_y = spacing * 0.8
#     y_top = max(ys) + pad_y
#     y_bot = min(ys) - pad_y

#     box = FancyBboxPatch(
#         (x_text +0.01 , y_bot+0.047),              # (x, y)
#         0.08,                               # width
#         (y_top - y_bot)*0.55,                      # height
#         boxstyle="round,pad=0.02",
#         transform=fig.transFigure,
#         facecolor="white",
#         edgecolor="black",
#         linewidth=1.5,
#         zorder=2
#     )
#     fig.patches.append(box)

#     # ---- draw text on top ----
#     for y, label, c in zip(ys, labels, colors):
#         fig.text(
#             x_text, y, label,
#             color=c,
#             ha="left", va="center",
#             zorder=3
#         )


def annotate_right_figure(fig, ax, labels, colors, spacing=0.05):
    pos = ax.get_position()
    y_mid = 0.5 * (pos.y0 + pos.y1)

    n = len(labels)
    offsets = -(np.arange(n) - (n - 1) / 2)
    ys = y_mid + offsets * spacing

    x_text = 0.87

    # ---- draw texts first (temporary) ----
    texts = []
    for y, label, c in zip(ys, labels, colors):
        t = fig.text(x_text, y, label,
                     color=c, ha="left", va="center",
                     transform=fig.transFigure)
        texts.append(t)

    # ---- force draw to compute sizes ----
    fig.canvas.draw()

    # ---- get bounding boxes in figure coords ----
    renderer = fig.canvas.get_renderer()
    bboxes = [t.get_window_extent(renderer=renderer) for t in texts]

    # convert from pixels → figure coordinates
    inv = fig.transFigure.inverted()
    bboxes_fig = [inv.transform_bbox(bb) for bb in bboxes]

    # ---- compute overall box ----
    x0 = min(bb.x0 for bb in bboxes_fig)
    x1 = max(bb.x1 for bb in bboxes_fig)
    y0 = min(bb.y0 for bb in bboxes_fig)
    y1 = max(bb.y1 for bb in bboxes_fig)

    pad_x = -0.016
    pad_y = spacing * 0.05

    # ---- draw box ----
    box = FancyBboxPatch(
        (x0 - pad_x, y0 + 0.018),
        (x1 - x0) + 2 * pad_x,
        (y1 - y0) *0.75,
        boxstyle="round,pad=0.02",
        transform=fig.transFigure,
        facecolor="white",
        edgecolor="black",
        linewidth=1.5,
        zorder=2
    )
    fig.patches.append(box)

    # ---- bring text to front ----
    for t in texts:
        t.set_zorder(3)

# ============================================================
# LOAD DATA
# ============================================================

model_df = pd.read_csv(MODEL_FILE)
ref_df = pd.read_csv(REFERENCE_FILE)

t_ref = ref_df["time_fbm"].to_numpy()
t_model = model_df["Time"].to_numpy()

load = interp_previous(t_ref, t_model, model_df["Load"].to_numpy())
phi = interp_previous(t_ref, t_model, model_df["Moisture"].to_numpy())
phi = 0.05 + 0.15 * phi

load[1] = 0
# ============================================================
# INTERPOLATION
# ============================================================

model_interp = dict(
    total      = interp_previous(t_ref, t_model, model_df["Total_strain"]),
    elastic    = interp_previous(t_ref, t_model, model_df["Elastic"]),
    visco      = interp_previous(t_ref, t_model, model_df["Creep"]),
    hygro      = interp_previous(t_ref, t_model, model_df["Hygroexp"]),
    mechano    = interp_previous(t_ref, t_model, model_df["Slip_strain"]),
    total_new  = interp_previous(t_ref, t_model, model_df["Total_model"]),
    creep_new  = interp_previous(t_ref, t_model, model_df["Creep_model"]),
    mech_new   = interp_previous(t_ref, t_model, model_df["Mech_model"]),
    template   = interp_previous(t_ref, t_model, model_df["templ_global"]),
    plastic    = interp_previous(t_ref, t_model, model_df["plastic"]),
)

model_interp["total_new"].iloc[1] = 0
model_interp["total_new"].iloc[1] = 0
model_interp["elastic"].iloc[1] = 0


model_interp["total"] = fix_initial_zero(model_interp["total"])

# ============================================================
# COLORS
# ============================================================

color_exp = "black"
color_model = "#d62728"
color_new = "#1f77b4"

# ============================================================
# FIGURE
# ============================================================

fig = plt.figure(figsize=FIGSIZE)

outer = GridSpec(5, 1, height_ratios=[2.7, 1.6, 1.4, 1.4, 1.0], hspace=0.10)

ax_total   = fig.add_subplot(outer[0])
ax_ms      = fig.add_subplot(outer[1], sharex=ax_total)
ax_creep   = fig.add_subplot(outer[2], sharex=ax_total)
ax_elastic = fig.add_subplot(outer[3], sharex=ax_total)
ax_load    = fig.add_subplot(outer[4], sharex=ax_total)

# ============================================================
# SHRINK AXES (CORRECT WAY)
# ============================================================

# right_margin = 0.15

# for ax in [ax_total, ax_ms, ax_creep, ax_elastic, ax_load]:
#     pos = ax.get_position()
#     ax.set_position([
#         pos.x0,
#         pos.y0,
#         pos.width * (1 - right_margin),
#         pos.height
#     ])

RIGHT_COLUMN = 0.15

plt.subplots_adjust(
    left=0.08,
    right=1 - RIGHT_COLUMN,
    bottom=0.08,
    top=0.90,
    hspace=0.10
)

# ============================================================
# SHADE
# ============================================================

intervals = get_mask_intervals(t_ref, get_loading_mask(load))

# ============================================================
# PLOTS
# ============================================================

# (a)
ax_total.plot(t_ref, ref_df["strain_total"], color=color_exp)
ax_total.plot(t_ref, model_interp["total_new"], '-', color=color_model)
ax_total.plot(t_ref, model_interp["total_new"], '-', color=color_new)
ax_total.set_ylabel(r"$\varepsilon$")
ax_total.text(0.02, 0.9, "(a)", transform=ax_total.transAxes)

# (b)
ax_ms.plot(t_ref, ref_df["strain_mechanosorption"], color=color_exp)
ax_ms.plot(t_ref, model_interp["mechano"], color=color_model)
ax_ms.plot(t_ref, model_interp["mech_new"], color=color_new)
ax_ms.set_ylabel(r"$\varepsilon$")
ax_ms.text(0.02, 0.85, "(b)", transform=ax_ms.transAxes)

# (c)
ax_creep.plot(t_ref, ref_df["strain_viscoelastic"], color=color_exp)
ax_creep.plot(t_ref, model_interp["visco"], color=color_model)
ax_creep.plot(t_ref, model_interp["creep_new"], color=color_new)
ax_creep.set_ylabel(r"$\varepsilon^{VE}$")
ax_creep.text(0.02, 0.85, "(c)", transform=ax_creep.transAxes)

# (d)
ref_he = ref_df["strain_elastic"] + ref_df["strain_hygro"]
ax_elastic.plot(t_ref, ref_he, color=color_exp)
ax_elastic.plot(t_ref, model_interp["elastic"] + model_interp["hygro"], color=color_model)
ax_elastic.plot(t_ref, model_interp["template"] + model_interp["plastic"], color=color_new)
ax_elastic.set_ylabel(r"$\varepsilon^{HE}$")
ax_elastic.text(0.02, 0.85, "(d)", transform=ax_elastic.transAxes)

# (e)
ax_load.plot(t_ref, load, color="black")
ax_phi = ax_load.twinx()
ax_phi.plot(t_ref, phi, color="black", alpha=0.5)
ax_load.set_ylabel(r"$\sigma$ [MPa]", labelpad=36)
ax_phi.set_ylabel(r"$\varphi$")
ax_load.set_xlabel(r"$t/\tau_1$")
ax_load.text(0.02, 0.8, "(e)", transform=ax_load.transAxes)
ax_load.set_ylim(0,3)
ax_load.set_yticks([0,  2])
ax_phi.spines.right.set_visible(True)
# ============================================================
# SHADE
# ============================================================

for ax in [ax_total, ax_ms, ax_creep, ax_elastic, ax_load]:
    apply_loading_shade(ax, intervals)
    ax.set_xlim(t_ref.min(), t_ref.max())

# ============================================================
# RIGHT COLUMN LABELS
# ============================================================



annotate_right_figure(fig, ax_total,
    [r"$\varepsilon_{ref}$", r"$\varepsilon$", r"$\varepsilon$"],
    [color_exp, color_model, color_new])


annotate_right_figure(fig, ax_ms,
    [r"$\varepsilon^{MS}_{ref}$", r"$\varepsilon^S$", r"$\varepsilon^{*MS}$"],
    [color_exp, color_model, color_new])

annotate_right_figure(fig, ax_creep,
    [r"$\varepsilon_{ref}^{VE}$", r"$\varepsilon^{VE}$", r"$\varepsilon^{*VE}$"],
    [color_exp, color_model, color_new])

annotate_right_figure(fig, ax_elastic,
    [r"$\varepsilon_{ref}^E+\varepsilon_{ref}^H$", r"$\varepsilon^E+\varepsilon^H$", r"$\varepsilon^{*HE}$"],
    [color_exp, color_model, color_new])

# ============================================================
# GLOBAL LEGEND (TOP BAR)
# ============================================================

legend_handles = [
    Line2D([0], [0], color=color_exp, lw=3.5),
    Line2D([0], [0], color=color_model, lw=3.5),
    Line2D([0], [0], color=color_new, lw=3.5),
]

legend_labels = [
    r"Reference (Ferrara and Wittel 2025b)",
    r"DD-SS-FBM",
    r"Macroscopic Strain Decomposition"
]

fig.legend(
    legend_handles,
    legend_labels,
    loc="upper center",
    ncol=3,
    bbox_to_anchor=(0.5, 0.98),
    frameon=True
)

plt.subplots_adjust(top=0.90)
ax_total.grid(True, alpha=0.5)
ax_ms.grid(True, alpha=0.5)
ax_creep.grid(True, alpha=0.5)
ax_elastic.grid(True, alpha=0.5)
ax_load.grid(True, alpha=0.5)
# ============================================================
# FINAL
# ============================================================

plt.setp(ax_total.get_xticklabels(), visible=False)
plt.setp(ax_ms.get_xticklabels(), visible=False)
plt.setp(ax_creep.get_xticklabels(), visible=False)
plt.setp(ax_elastic.get_xticklabels(), visible=False)

plt.savefig("Fig12.png", dpi=300)
print("Saved plot:", OUTPUT_PLOT)