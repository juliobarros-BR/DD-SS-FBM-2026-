#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 19 16:28:32 2025

@author: jortiz
"""
from Fiber_Grid import FiberGrid
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os, gc
import pandas as pd
from scipy.interpolate import interp1d
import re

# ==========================================================
# PARAMETERS
# ==========================================================
max_cycles = 50
convergence_threshold = 1e-10
n_substeps = 200

segment_durations = [5000]
ramp_percents = [0.0]

max_steps_per_event = 1_000_000
snapshot_every_steps = 100
n_final_snapshots = 250
plot_frequency = 0
GENERATE_VIDEO = False

# Interpolation
N_INTERP = 100
VMIN, VMAX = 0.05, 0.20

# ==========================================================
# VIDEO EXPORT
# ==========================================================
def save_snapshot_video_opencv(grid, filename, folder):
    os.makedirs(folder, exist_ok=True)
    save_path = os.path.join(folder, filename)

    if len(grid.snapshots) == 0:
        return

    indices = np.linspace(0, len(grid.snapshots)-1, min(20, len(grid.snapshots)), dtype=int)
    frames = []

    for i in indices:
        snapshot = grid.snapshots[i]
        original = grid.moisture.copy()
        grid.moisture = snapshot

        fig = grid.plot_fibers_video(title=f"t={grid.snapshot_times[i]:.2f}s", full_square=True)
        fig.canvas.draw()

        buf = np.asarray(fig.canvas.buffer_rgba())
        img = buf[..., :3].copy()

        frames.append(img)
        plt.close(fig)
        grid.moisture = original

    h, w, _ = frames[0].shape
    out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), 10, (w, h))

    for frame in frames:
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    out.release()
    del frames
    gc.collect()


# ==========================================================
# INTERPOLATION
# ==========================================================
def interpolate_fiber_group(df):
    t = df['time_s'].values
    omega = df['moisture'].values

    if len(t) < 2:
        return pd.DataFrame()

    t_interp = np.linspace(t.min(), t.max(), N_INTERP)
    f_interp = interp1d(t, omega, kind='cubic', fill_value="extrapolate")
    omega_interp = f_interp(t_interp)

    return pd.DataFrame({
        'fiber_index': df['fiber_index'].iloc[0],
        'x_mm': df['x_mm'].iloc[0],
        'y_mm': df['y_mm'].iloc[0],
        'time_s': t_interp,
        'moisture': omega_interp
    })


def interpolate_file(file):
    df = pd.read_csv(file)
    df.sort_values(['fiber_index', 'time_s'], inplace=True)

    interpolated = df.groupby('fiber_index', group_keys=False).apply(interpolate_fiber_group)

    out_file = file.with_name(file.stem + "_interpolated.csv")
    interpolated.to_csv(out_file, index=False)

    return out_file


# ==========================================================
# MASK GENERATION
# ==========================================================
def make_mask(file, folder):
    df = pd.read_csv(file)
    all_times = sorted(df['time_s'].unique())

    output_rows = []

    for t_idx, t in enumerate(all_times):
        df_t = df[df['time_s'] == t].copy()

        df_t['x'] = (df_t['x_mm'] / 1000).astype(int)
        df_t['y'] = (df_t['y_mm'] / 1000).astype(int)

        temp_df = df_t[['fiber_index', 'x', 'y', 'moisture']].copy()
        temp_df['time_s'] = t

        output_rows.append(temp_df)

    final_df = pd.concat(output_rows, ignore_index=True)

    final_df.sort_values(by=['fiber_index', 'time_s'], inplace=True)
    final_df['time_index'] = final_df.groupby('fiber_index').cumcount()

    final_df = final_df[['fiber_index', 'x', 'y', 'moisture', 'time_index']]

    prefix = "mask_moistening" if "moistening" in file.stem else "mask_drying"
    cycle_num = re.search(r"cycle_(\d+)", file.stem).group(1)

    out_path = folder / f"{prefix}_{cycle_num}_cycle.csv"
    final_df.to_csv(out_path, index=False)


# ==========================================================
# SIMULATION STEP
# ==========================================================
def ramp_and_hold_linear(grid, start, target, seg_dur, ramp_dur,
                         prefix, cycle, folder):

    grid.reset_snapshots()

    ramp_dur = min(ramp_dur, seg_dur)

    # --- Ramp ---
    if ramp_dur > 0:
        levels = np.linspace(start, target, n_substeps)
        for level in levels:
            grid.set_boundary_moisture(level)
            grid.evolve_until_time(
                target_duration=ramp_dur / n_substeps,
                max_steps=max_steps_per_event,
                save_snapshots=True,
                snapshot_every_steps=snapshot_every_steps,
            )

    # --- Hold ---
    hold_time = max(0, seg_dur - ramp_dur)
    if hold_time > 0:
        grid.set_boundary_moisture(target)
        grid.evolve_until_time(
            target_duration=hold_time,
            max_steps=max_steps_per_event,
            save_snapshots=True,
            snapshot_every_steps=snapshot_every_steps,
        )

    grid.finalize_snapshots(n_final_snapshots)

    # --- EXPORT CSV ---
    csv_path = folder / f"{prefix}_cycle_{cycle}.csv"
    grid.export_snapshots_to_csv(csv_path)

    # --- INTERPOLATE + MASK ---
    interp_file = interpolate_file(csv_path)
    make_mask(interp_file, folder)

    # --- VIDEO ---
    if GENERATE_VIDEO:
        save_snapshot_video_opencv(grid, f"{prefix}_cycle_{cycle}.mp4", folder)

    avg_end = float(np.mean(grid.snapshots[-1]))

    grid.reset_snapshots()
    return avg_end


# ==========================================================
# MAIN LOOP
# ==========================================================
for seg_dur in segment_durations:
    for ramp_percent in ramp_percents:

        ramp_dur = seg_dur * ramp_percent
        folder = Path(f"Moisture_Profiles_{int(seg_dur)}_ramp{ramp_percent:.2f}")
        folder.mkdir(exist_ok=True)

        grid = FiberGrid(n_rows=100, width_mm=1.0, height_mm=1.0)

        prev_avg = None

        for cycle in range(1, max_cycles + 1):

            print(f"\n🌊 Cycle {cycle} - Moistening")
            avg_m = ramp_and_hold_linear(grid, 0.05, 0.20, seg_dur, ramp_dur,
                                         "moistening", cycle, folder)

            print(f"🔥 Cycle {cycle} - Drying")
            ramp_and_hold_linear(grid, 0.20, 0.05, seg_dur, ramp_dur,
                                 "drying", cycle, folder)

            # cleanup
            grid.history = [grid.history[-1]]
            grid.time = [grid.time[-1]]
            plt.close('all')
            gc.collect()

            # convergence
            if prev_avg is not None:
                if abs(avg_m - prev_avg) < convergence_threshold:
                    print("✅ Converged.")
                    break

            prev_avg = avg_m