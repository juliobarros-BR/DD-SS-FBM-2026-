#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 28 15:14:45 2025

@author: jortiz
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# Apply your style parameters
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
# Load previously computed results
# ============================================================
csv_path = Path("./moisture_sweep_results22/LoadD_0p6/moisture_sweep_summary.csv")
df = pd.read_csv(csv_path)

# Extract the REAL data points
m = df["moisture_frac"].to_numpy()
J_target = df["J_target"].to_numpy()
J_eff    = df["J_eff"].to_numpy()

# ============================================================
# Quadratic fit:  J_fit(m) = a m^2 + b m + c
# ============================================================
coef = np.polyfit(m, J_eff, 2)   # [a, b, c]
poly = np.poly1d(coef)

m_fit = np.linspace(m.min(), m.max(), 400)
J_fit = poly(m_fit)

fit_label = r"$Quadratic fit$"

# ============================================================
# Main plot
# ============================================================
fig, ax = plt.subplots(figsize=(5.3, 6.5))

# Applied (input) viscoelastic compliance
ax.plot(
    m, J_target,
    marker="o", linestyle="None",
    label=r"$\sum_{n=1}^{4} J_n$"
)

# EFFECTIVE bundle compliance (REAL data points from CSV)
ax.plot(
    m, J_eff,
    marker="s", linestyle="None",
    label=r"$\sum_{n=1}^{4} J_{\mathrm{eff}(n)}$"
)

# Quadratic fit curve
ax.plot(
    m_fit, J_fit,
    linestyle="-", color="black",
    label=fit_label,
    alpha=0.3
)

ax.set_xlabel(r"$\varphi$")
ax.set_ylabel(r"$J$[-]")

ax.set_xticks([0, 0.5, 1])
ax.set_xticklabels([0.05, 0.125, 0.2])
from matplotlib.ticker import ScalarFormatter

formatter = ScalarFormatter(useMathText=True)
formatter.set_powerlimits((-1, -1))  # force ×10⁻²

ax.yaxis.set_major_formatter(formatter)
ax.ticklabel_format(axis="y", style="sci", scilimits=(-1, -1))

# Optional: move the ×10⁻² a bit
ax.yaxis.get_offset_text().set_x(-0.12)
ax.yaxis.get_offset_text().set_y(1.01)

ax.grid(True, alpha=0.3)
ax.legend()

pos = ax.get_position()
ax.set_position([pos.width*0.17, pos.y0*1.13, pos.width*1.06, pos.height*1.07])

# fig.tight_layout()
fig.savefig("./Fig5b.png",bbox_inches=None,dpi=300)
# plt.show()
