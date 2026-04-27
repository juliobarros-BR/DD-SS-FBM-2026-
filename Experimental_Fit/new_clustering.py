import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# LOAD
DATA_FILE = "./merged_full.csv"
df = pd.read_csv(DATA_FILE)

# BASIC FILTERS (keep only entries with valid MSE and not broken)
df = df[
    np.isfinite(df.get("mse_plain")) &
    (df.get("broken_kv", 0) == 0) &
    (df.get("broken_ms", 0) == 0)
].copy()

# drop nonpositive J_w to avoid divide-by-zero
df = df[df.get("J_w_kv", 1e-99) > 0].copy()

# SCORES
ms = df["mse_plain"].to_numpy()
jeff = df["J_eff_kv"].to_numpy()

ms_norm = (ms - ms.min()) / (ms.max() - ms.min() + 1e-16)
jeff_norm = (jeff - jeff.min()) / (jeff.max() - jeff.min() + 1e-16)
combined = ms_norm + jeff_norm

df["ms_norm"] = ms_norm
df["jeff_norm"] = jeff_norm
df["combined_score"] = combined

best_idx = np.argsort(combined)[:10]
df_best = df.iloc[best_idx]

# RC PARAMETERS (avoid text.usetex for portability)
mpl.rcParams.update(mpl.rcParamsDefault)
plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 14,
    'lines.linewidth': 1.5,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times'],
    'mathtext.fontset': 'cm',
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.major.width': 1.0,
    'ytick.major.width': 1.0,
    'axes.linewidth': 1.0,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'legend.framealpha': 1.0,
    'legend.edgecolor': 'black',
    'savefig.dpi': 300,
})

# FIGURE (manual layout) - this reproduces combined_surrogate_plot_clean
fig = plt.figure(figsize=(16, 7))
ax1 = fig.add_axes([0.08, 0.17, 0.35, 0.8])
ax2 = fig.add_axes([0.6, 0.17, 0.35, 0.8])
cbar_ax = fig.add_axes([0.48, 0.18, 0.015, 0.7])

ratio = df["lambda_Weibull_kv"] / df["J_w_kv"]
scale = 1e5  # original scaling intent (used only for display in original script)
mse_for_color = ms  # color = raw MSE

# LEFT: ratio vs J_eff
sc = ax1.scatter(
    ratio,
    df["J_eff_kv"],
    c=mse_for_color,
    cmap="viridis_r",
    s=35,
    alpha=0.85,
    edgecolors="none"
)
ax1.scatter(
    df_best["lambda_Weibull_kv"] / df_best["J_w_kv"],
    df_best["J_eff_kv"],
    facecolors="none",
    edgecolors="red",
    s=120,
    linewidths=2.5,
    zorder=5,
    label="Top candidates"
)
ax1.axhline(0.0095, color="black", linestyle="--", linewidth=1.5, label="Reference compliance")
ax1.set_xlabel(r"$\lambda / J_w$")
ax1.set_ylabel(r"$J_{\mathrm{eff}}$")
ax1.grid(alpha=0.25)
ax1.text(0.5, -0.17, "a)", transform=ax1.transAxes, ha="center", va="center")

# RIGHT: J_w vs J_eff
sc2 = ax2.scatter(
    df["J_w_kv"],
    df["J_eff_kv"],
    c=mse_for_color,
    cmap="viridis_r",
    s=35,
    alpha=0.85,
    edgecolors="none"
)
ax2.scatter(
    df_best["J_w_kv"],
    df_best["J_eff_kv"],
    facecolors="none",
    edgecolors="red",
    s=120,
    linewidths=2.5,
    zorder=5
)
ax2.axhline(0.0095, color="black", linestyle="--", linewidth=1.5, label="Reference Wet Compliance")
ax2.set_xlabel(r"$J_w$")
ax2.set_ylabel(r"$J_{\mathrm{eff}}$")
ax2.grid(alpha=0.25)
ax2.text(0.5, -0.17, "b)", transform=ax2.transAxes, ha="center", va="center")
ax1.legend(loc="upper right", framealpha=1.0, edgecolor="black")

# COLORBAR (clean)
cbar = fig.colorbar(sc, cax=cbar_ax)
cbar.set_label(r"$E_{MS}$", size=12)
cbar.ax.yaxis.set_ticks_position("left")
formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_powerlimits((0, 0))
cbar.ax.yaxis.set_major_formatter(formatter)

# ALIGN Y LIMITS
ymin = df["J_eff_kv"].min() * 0.95
ymax = df["J_eff_kv"].max() * 1.05
ax1.set_ylim(ymin, ymax)
ax2.set_ylim(ymin, ymax)

plt.savefig("Fig11.png", dpi=300)
plt.close()
