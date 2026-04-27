import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # comment out if you want interactive windows
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# STYLE (same as your previous nice plot)
# ============================================================
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
    'savefig.bbox': None,  # do not auto-crop
})


# ============================================================
# PATHS
# ============================================================
OUTDIR = Path("plastic_analysis/unified_plastic_sweeps_LoadDmax_0p700/")
PARTA = OUTDIR / "partA_load_sweep" / "partA_load_sweep.csv"
PARTB = OUTDIR / "partB_moist_sweep" / "partB_moist_sweep.csv"

# output names (new, nicer)
OUT_A = OUTDIR / "plot_partA_load_sweep_nice2.png"
OUT_B = OUTDIR / "plot_partB_moist_sweep_nice2.png"
OUT_C = OUTDIR / "plot_combined_overall_nice2.png"

# ============================================================
# LOAD
# ============================================================
if not PARTA.exists():
    raise FileNotFoundError(f"Missing: {PARTA}")
if not PARTB.exists():
    raise FileNotFoundError(f"Missing: {PARTB}")

dfA = pd.read_csv(PARTA)
dfB = pd.read_csv(PARTB)

# Basic cleaning (keep finite values)
def finite_xy(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    return x[ok], y[ok]

# ============================================================
# PART A: slip vs load_d (dry)
# ============================================================
xA, yA = finite_xy(dfA["load_d"], dfA["slip_final"])

# figA, axA = plt.subplots(figsize=(8, 6), dpi=300)
# axA.plot(xA, yA, marker="o", linestyle="-")

# axA.set_xlabel(r"$\sigma_0$")                 # shorter than "load_d (dry state)"
# axA.set_ylabel(r"$\varepsilon^{P}$")          # no "proxy"
# axA.grid(True, alpha=0.3)

# figA.tight_layout()
# figA.savefig(OUT_A)
# plt.close(figA)

# ============================================================
# PART B: slip vs moisture step
# ============================================================
xB, yB = finite_xy(dfB["moist_step_human"], dfB["slip_final"])

# figB, axB = plt.subplots(figsize=(8, 6), dpi=300)
# axB.plot(xB, yB, marker="o", linestyle="-")

# axB.set_xlabel(r"Moisture step")              # short + clear
# axB.set_ylabel(r"$\varepsilon^{P}$")
# axB.grid(True, alpha=0.3)

# figB.tight_layout()
# figB.savefig(OUT_B)
# plt.close(figB)

# ============================================================
# COMBINED: progress axis (0->1 load, 1->2 moisture)
# ============================================================
# segment A: include x=1
sA = np.linspace(0.0, 1.0, len(dfA), endpoint=True) if len(dfA) > 1 else np.array([1.0])

# segment B: start at x=1 (keep as-is)
sB = np.linspace(1.0, 2.0, len(dfB), endpoint=True) if len(dfB) > 0 else np.array([])


# --- Combined plot that (1) keeps true figsize geometry and (2) never chops y-label ---
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# IMPORTANT:
# - Use constrained_layout (no tight_layout)
# - Do NOT use negative labelpad (it confuses layout + increases clipping risk)
# - Save WITHOUT bbox_inches="tight" to preserve the true 8x6 geometry
FORCE_CRIT = 4.154739120251522  
figC, axC = plt.subplots(figsize=(5.3, 6.5), dpi=300)

# -------------------------
# Plot data
# -------------------------
if len(sA):
    axC.plot(sA, yA/FORCE_CRIT, marker="o", linestyle="-", label=r"$\Delta\sigma$")
if len(sB):
    axC.plot(sB, yB/FORCE_CRIT, marker="o", linestyle="-", label=r"$\Delta\varphi$")

axC.axvline(1.0, ls="--", lw=1)

# -------------------------
# Axes formatting
# -------------------------
axC.set_xlim(-0.2, 2.2)
axC.set_xticks([0.0, 1.0, 2.0])
axC.set_xticklabels([
    r"\shortstack{$\sigma=0$\\$\varphi=\varphi_d$}",
    r"\shortstack{$\sigma=0.4\sigma_c$\\$\varphi=\varphi_d$}",
    r"\shortstack{$\sigma=0.4\sigma_c$\\$\varphi=\varphi_w$}",
])

axC.set_ylabel(r"$\varepsilon^{*P}/\varepsilon_c$")

# Scientific notation
from matplotlib.ticker import ScalarFormatter
formatter = ScalarFormatter(useMathText=True)
formatter.set_scientific(True)
formatter.set_powerlimits((-2, -2))
axC.yaxis.set_major_formatter(formatter)

# -------------------------
# Grid + legend
# -------------------------
axC.grid(True, alpha=0.3)
axC.legend(loc="upper left")

# -------------------------
# Manual positioning (FIXED VERSION)
# -------------------------
pos = axC.get_position()

axC.set_position([
    pos.x0 * 1.45,   # shift right (like your working script)
    pos.y0 * 1.13,   # shift up slightly (THIS was missing before)
    pos.width * 0.95,  # slightly shrink width (instead of expanding!)
    pos.height * 1.07  # slightly shrink height (prevents overflow)
])

# -------------------------
# Save
# -------------------------
figC.savefig("./Fig5a.png", dpi=300, bbox_inches=None)

plt.close(figC)



print("Saved:")
print("./Fig5a.png")
