#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 10 14:37:27 2025

@author: jortiz
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from pathlib import Path

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



# ======================================================
# FUNCTIONS
# ======================================================

def tX_multiKV(tau_list, X=0.99, J_list=None):
    """Find t where normalized compliance reaches fraction X."""
    tau_list = np.asarray(tau_list, dtype=float)
    if J_list is None:
        J_list = np.ones_like(tau_list)
    J_list = np.asarray(J_list, dtype=float)

    def f(t):
        # J(t)/J_inf = 1 - Σ exp(-t/τ) weighted by J_list
        return 1 - np.sum(J_list * np.exp(-t / tau_list)) / np.sum(J_list) - X

    return brentq(f, 0, 50 * np.max(tau_list))


def viscoelastic_curve(t, tau_list, J_list):
    """Return normalized creep compliance evolution 1 - Σ exp(-t/τ)."""
    t = np.asarray(t)
    val = 1 - np.sum(J_list * np.exp(-t[:, None] / tau_list[None, :]), axis=1) / np.sum(J_list)
    return val


# ======================================================
# MAIN
# ======================================================
if __name__ == "__main__":

    # -------------------------------
    # USER SETTINGS
    # -------------------------------
    tau = 1.0
    KV_num = 4
    exponential = True
    THRESHOLDS = [0.63, 0.95, 0.99]
    COLORS = ['C2', 'C1', 'C3']
    SAVE_FIG = True
    OUT_FIG = Path("Fig3b.png")

    # Kelvin–Voigt parameters
    tau_list = tau * 10 ** np.arange(KV_num)
    weights = 2 ** np.arange(KV_num) if exponential else np.ones(KV_num)
    J_weights = weights / np.sum(weights)

    # -------------------------------
    # GENERATE EVOLUTION CURVE
    # -------------------------------
    t = np.linspace(0, 50 * np.max(tau_list), 500)
    J_norm = viscoelastic_curve(t, tau_list, J_weights)

    # -------------------------------
    # COMPUTE THRESHOLD TIMES
    # -------------------------------
    results = []
    for th in THRESHOLDS:
        idx = np.argmax(J_norm >= th)
        results.append((th, t[idx], J_norm[idx]))
        print(t[idx])

    # -------------------------------
    # PLOT
    # -------------------------------
    fig, ax = plt.subplots(figsize=((8, 6)),dpi=300)
    ax.plot(t/1000, J_norm, color='C0')

    for (th, t_val, J_val), col in zip(results, COLORS):
        t_val /= 1000
        ax.axhline(th, color=col, ls='--', alpha=0.6)
        ax.axvline(t_val, color=col, ls='--', alpha=0.6)
        ax.plot(t_val, J_val, 'o', color=col, markersize=8, label = rf"$T_{{\tau}}^{{{int(th * 100)}}}$")
        # ax.text(
        #     t_val + 0.2/1000 * np.max(tau_list), J_val - 0.05,
        #     f"$T_{{\tau}}^{{{int(th*100)}}}$",
        #     color=col, ha='center', va='bottom', fontsize=18,
        #     fontweight='bold',
        #     bbox=dict(facecolor='white', edgecolor=col, boxstyle='round,pad=0.15', alpha=1.0)
        # )

    ax.set_xlabel(r"$t / \tau_4$")
    ax.set_ylabel(r"$J(t)/J$")
    # ax.set_title("Viscoelastic Response Evolution")
    ax.set_xlim(0,10000/1000)
    ax.legend()
    ax.grid(alpha=0.3)
    # fig.tight_layout()
    pos = ax.get_position()
    ax.set_position([pos.width*0.15, pos.y0*1.15, pos.width*1.1, pos.height*1.1])
    ax.set_ylim(0,1.05)

    if SAVE_FIG:
        plt.savefig(OUT_FIG,bbox_inches=None,dpi=300)
        print(f"Saved figure → {OUT_FIG}")
        # plt.show()
    # else:
        # plt.show()
