#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  3 13:24:01 2025

@author: jortiz
"""

import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from pathlib import Path
import matplotlib.pyplot as plt
import re
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# ==============================================================
# CONFIGURATION
# ==============================================================
BASE_DIR = Path("./")
SEARCH_PATTERN = "Moisture_Profiles_*"
OVERWRITE = False   # ⬅️  run all again if True, skip completed if False
N_INTERP = 100
PLOT_EVERY = 0
VMIN, VMAX = 0.05, 0.20
# ==============================================================


def interpolate_fiber_group(df):
    """Interpolate one fiber's time series."""
    t = df['time_s'].values
    omega = df['moisture'].values

    if len(t) < 2:
        return pd.DataFrame()

    t_interp = np.linspace(t.min(), t.max(), N_INTERP)
    f_interp = interp1d(t, omega, kind='cubic', fill_value="extrapolate")
    omega_interp = f_interp(t_interp)

    result = pd.DataFrame({
        'fiber_index': df['fiber_index'].iloc[0],
        'x_mm': df['x_mm'].iloc[0],
        'y_mm': df['y_mm'].iloc[0],
        'time_s': t_interp,
        'moisture': omega_interp
    })
    return result

def interpolate_file(file: Path):
    """Interpolate all fibers for one input file."""
    df = pd.read_csv(file)
    df.sort_values(['fiber_index', 'time_s'], inplace=True)
    interpolated = df.groupby('fiber_index', group_keys=False).apply(interpolate_fiber_group)
    out_file = file.with_name(file.stem + "_interpolated.csv")
    interpolated.to_csv(out_file, index=False)
    print(f"✓ Interpolated: {out_file.name}")
    return out_file

def apply_transform(xy, matrix, offset):
    return (np.dot(xy, matrix) + offset).astype(int)

def make_masks(file: Path, data_folder: Path, plot_every=PLOT_EVERY):
    """
    Export only the interpolated triangular/base moisture profile
    without expanding to the symmetric full square bundle.
    """

    df = pd.read_csv(file)
    all_times = sorted(df['time_s'].unique())

    output_rows = []

    for t_idx, t in enumerate(all_times):

        df_t = df[df['time_s'] == t].copy()

        # Convert coordinates to integer grid indices
        df_t['x'] = (df_t['x_mm'] / 1000).astype(int)
        df_t['y'] = (df_t['y_mm'] / 1000).astype(int)

        temp_df = df_t[['fiber_index', 'x', 'y', 'moisture']].copy()
        temp_df['time_s'] = t

        output_rows.append(temp_df)

        # Optional plotting
        if plot_every and t_idx % plot_every == 0:
            plt.figure(figsize=(6, 6))
            sc = plt.scatter(
                temp_df['x'],
                temp_df['y'],
                c=temp_df['moisture'],
                cmap='viridis',
                s=8,
                vmin=VMIN,
                vmax=VMAX
            )

            plt.gca().invert_yaxis()
            plt.title(f"Triangular Moisture Profile (t={t:.2f}s)")
            plt.colorbar(sc, label='Moisture')
            plt.axis('equal')
            plt.tight_layout()

            plot_path = data_folder / f"{file.stem}_frame_{t_idx:03d}.png"
            plt.savefig(plot_path, dpi=300)
            plt.close()

    # Combine all time steps
    final_df = pd.concat(output_rows, ignore_index=True)

    # Rebuild consistent time_index per fiber
    final_df.sort_values(by=['fiber_index', 'time_s'], inplace=True)
    final_df['time_index'] = final_df.groupby('fiber_index').cumcount()

    final_df.drop(columns=['time_s'], inplace=True)

    final_df = final_df[['fiber_index', 'x', 'y', 'moisture', 'time_index']]

    prefix = "mask_moistening" if "moistening" in file.stem else "mask_drying"
    cycle_num = re.search(r"cycle_(\d+)", file.stem).group(1)

    output_path = data_folder / f"{prefix}_{cycle_num}_cycle.csv"

    final_df.to_csv(output_path, index=False)

    print(f"✓ Saved triangular mask: {output_path.name}")

def process_folder(folder: Path):
    # pick only raw moistening/drying files (exclude interpolated/others)
    candidates = [
        f for f in folder.glob("*.csv*")
        if (
            ("moistening" in f.name or "drying" in f.name)
            and ("interpolated" not in f.name)
            and ("mask_" not in f.name)          #  exclude masks
            and ("_mask" not in f.name)          #  just in case of alternate naming
        )
    ]


    def sort_key(f: Path):
        name = f.stem
        phase = 0 if "moistening" in name else 1 if "drying" in name else 2
        m = re.search(r"cycle_(\d+)", name)
        cyc = int(m.group(1)) if m else 10**9
        return (phase, cyc, name)

    files = sorted(candidates, key=sort_key)
    if not files:
        return

    # --- Skip folder if already complete ---
    mask_files = list(folder.glob("mask_*.csv"))
    if not OVERWRITE and len(mask_files) >= len(files):
        print(f"✓ Skipping {folder.name} — all masks already computed ({len(mask_files)}/{len(files)}).")
        return

    print(f"\n=== Processing folder: {folder.name} ===")
    for file in files:
        try:
            # Skip individual files if mask already exists (unless OVERWRITE)
            cycle_num = re.search(r"cycle_(\d+)", file.stem)
            cycle_num = cycle_num.group(1) if cycle_num else "X"
            prefix = "mask_moistening" if "moistening" in file.stem else "mask_drying"
            mask_path = folder / f"{prefix}_{cycle_num}_cycle.csv"

            if not OVERWRITE and mask_path.exists():
                print(f"⏩ Skipping {mask_path.name} (already exists).")
                continue

            interpolated = interpolate_file(file)
            make_masks(interpolated, folder)

        except Exception as e:
            fail_path = folder / "FAILED.txt"
            with open(fail_path, "a") as f:
                f.write(f"File {file.name} failed:\n{repr(e)}\n\n")
            print(f"⚠️  Skipping {file.name} due to error: {e}")
            print(f"    → Logged in {fail_path.name}")
            continue



# ==============================================================
# RUN ALL FOLDERS
# ==============================================================
for folder in sorted(BASE_DIR.glob(SEARCH_PATTERN)):
    if folder.is_dir():
        process_folder(folder)
