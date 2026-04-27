#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 10 14:20:20 2025

@author: jortiz
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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



def find_avg_equilibrium_times(df, target_moisture, thresholds, phi0=0.05):
    groups = df.groupby(df.columns[0])
    times = np.array(list(groups.groups.keys()))
    avg_moist = np.array([np.nanmean(g["moisture"]) for _, g in groups])

    idx_sort = np.argsort(times)
    times = times[idx_sort]
    avg_moist = avg_moist[idx_sort]

    # normalize to 0..1 using (phi - phi0)/(target - phi0)
    denom = (target_moisture - phi0)
    avg_norm = (avg_moist - phi0) / denom

    results = []
    for th in thresholds:
        hit = np.where(avg_norm >= th)[0]
        if len(hit) == 0:
            results.append((th, np.nan, np.nan))
        else:
            idx = hit[0]
            results.append((th, times[idx], avg_norm[idx]))
    return times, avg_norm, results



if __name__ == "__main__":
    # ===============================
    # USER SETTINGS
    # ===============================
    CSV_PATH = Path("./long_moistening.csv")
    TARGET_MOISTURE = 0.2
    THRESHOLDS = [0.63, 0.95, 0.99]  # diffusion-style criteria
    COLORS = ['C2', 'C1', 'C3']      # for points/lines
    SAVE_FIG = True
    OUT_FIG = Path("Fo_chi.png")

    # ===============================
    # LOAD DATA
    # ===============================
    df = pd.read_csv(CSV_PATH)
    if "time" not in df.columns[0].lower():
        df.columns = ["time", "fiber", "moisture"][:len(df.columns)]

    # ===============================
    # PROCESS
    # ===============================
    times, avg_moist, results = find_avg_equilibrium_times(df, TARGET_MOISTURE, THRESHOLDS, phi0=0.05)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(times/5000-1, avg_moist, color='C0')

    for (th, t_val, m_val), col in zip(results, COLORS):
        t_val /= 5000
        ax.axhline(th, color=col, ls='--', alpha=0.6)     # <-- now threshold is exactly th
        ax.axvline(t_val-1, color=col, ls='--', alpha=0.6)
        ax.plot(t_val-1, m_val, 'o', color=col, markersize=7,
                label=rf"$T_{{\chi}}^{{{int(th * 100)}}}$")

    # for (th, t_val, m_val), col in zip(results, COLORS):
    #     t_val /= 5000
    #     ax.axhline((TARGET_MOISTURE * th-0.05)/0.15, color=col, ls='--', alpha=0.6)
    #     ax.axvline(t_val, color=col, ls='--', alpha=0.6)
    #     ax.plot(t_val, (m_val-0.05)/0.15, 'o', color=col, markersize=7, label=rf"$T_{{\chi}}^{{{int(th * 100)}}}$")
    #     ax.text(
    #         t_val+130/5000, m_val-0.015, rf"$T_d^{{{int(th*100)}}}$",
    #         color=col, ha='center', va='bottom', fontsize=9,
    #         fontweight='bold',
    # bbox=dict(facecolor='white', edgecolor=col, boxstyle='round,pad=0.15', alpha=1.0)
    #     )

    # Formatting
    ax.set_xlabel("Moistening segment [-]")
    ax.set_ylabel(r"$(\langle\varphi\rangle$-0.05)/0.15 [-]")
    # ax.set_title("Moisture Equilibrium Evolution")
    ax.legend()
    ax.grid(alpha=0.3)
    pos = ax.get_position()
    ax.set_position([pos.width*0.15, pos.y0*1.15, pos.width*1.1, pos.height*1.1])
    # ax.set_xlim(0,1)
    ax.set_ylim(0,1.05)
    ax.tick_params(axis="y", pad=7.5)
    ax.tick_params(axis="x", pad=5.5)


    if SAVE_FIG:
        plt.savefig("Fig3a.png",bbox_inches=None, dpi=300)
        print(f"Saved figure → {OUT_FIG}")
        plt.show()
    else:
        plt.show()
