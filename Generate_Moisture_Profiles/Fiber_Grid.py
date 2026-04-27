#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 25 14:28:29 2025

@author: jortiz
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from scipy.ndimage import convolve
import pandas as pd
import gc


class FiberGrid:
    def __init__(self, n_rows, width_mm, height_mm, D=1, dt=1):
        self.width = width_mm * 1e-3
        self.height = height_mm * 1e-3
        self.n_rows = n_rows
        self.N = self.n_rows * (self.n_rows + 1) // 2
        self.D0 = D
        self.dt = dt
        self.use_variable_D = True
        self.beta = np.log(10) / (0.20 - 0.05)

        # Geometry
        self.fiber_positions = self._generate_triangular_grid()
        self.kdtree = KDTree(self.fiber_positions)
        self.neighbor_radius = 1
        self.neighbor_list = self._compute_all_neighbors()

        # Initial condition
        self.moisture = np.full(len(self.fiber_positions), 0.05)
        self.history = [0.05]
        self.time = [0.0]
        self.snapshots, self.snapshot_times = [], []

    # ---------------------------------------------------------------------
    # Geometry & diffusion setup
    # ---------------------------------------------------------------------
    def _generate_triangular_grid(self):
        positions = []
        for row in range(self.n_rows):
            n_cols = self.n_rows - row
            for col in range(n_cols):
                x = self.n_rows - n_cols + col
                y = row
                positions.append((x, y))
        return np.array(positions)

    def _compute_all_neighbors(self):
        neighbors = []
        for i in range(len(self.fiber_positions)):
            idx = self.kdtree.query_ball_point(self.fiber_positions[i], r=self.neighbor_radius)
            neighbors.append([j for j in idx if j != i])
        return neighbors

    def get_D(self, omega):
        """Moisture-dependent diffusion coefficient."""
        if not self.use_variable_D:
            return self.D0
        return self.D0 * np.exp(self.beta * (omega - 0.05))

    def set_boundary_moisture(self, value, atol=1e-12):
        bottom_indices = np.where(np.isclose(self.fiber_positions[:, 1], 0.0, atol=atol))[0]
        self.moisture[bottom_indices] = value

    # ---------------------------------------------------------------------
    # Main evolution routine (optimized)
    # ---------------------------------------------------------------------
    def evolve_until_time(self,
                          target_duration,
                          max_steps=1_000_000,
                          save_snapshots=True,
                          snapshot_every_steps=20,
                          n_snapshots=100,
                          plot_frequency=0,
                          cfl_coeff=0.25):
        """
        Evolves moisture for target_duration.
        """
        if not hasattr(self, "raw_snapshots"):
            self.raw_snapshots, self.raw_snapshot_times = [], []

        t_start = self.time[-1]
        t_target = t_start + float(target_duration)
        step = 0

        while step < max_steps and self.time[-1] + 1e-12 < t_target:
            D_all = self.get_D(self.moisture)
            max_D = float(np.max(D_all))
            dt_cfl = cfl_coeff / ((self.neighbor_radius**2) * max_D if max_D > 0 else 1e-9)
            remaining = t_target - self.time[-1]
            self.dt = min(dt_cfl, remaining * 1.001)

            grid_size = self.n_rows
            moisture_grid = np.full((grid_size, grid_size), np.nan)
            D_grid = np.full((grid_size, grid_size), np.nan)
            tri_mask = np.fromfunction(lambda y, x: x >= y, (grid_size, grid_size))
            moisture_grid[tri_mask] = self.moisture
            D_grid[tri_mask] = D_all

            kernel = np.array([[0, 1, 0],
                               [1, 0, 1],
                               [0, 1, 0]])

            valid_mask = np.zeros_like(moisture_grid, dtype=float)
            valid_mask[tri_mask] = 1.0
            neighbor_count = convolve(valid_mask, kernel, mode='constant', cval=0.0)
            moist_filled = np.nan_to_num(moisture_grid, nan=0.0)
            neighbor_sum = convolve(moist_filled, kernel, mode='constant', cval=0.0)
            delta = neighbor_sum - moisture_grid * neighbor_count
            delta = np.where(tri_mask, delta, 0.0)
            updated_grid = moisture_grid + D_grid * delta * self.dt

            fix_mask = np.zeros_like(moisture_grid, dtype=bool)
            fix_mask[0, :] = tri_mask[0, :]  # y = 0 row
            updated_grid[0, fix_mask[0, :]] = moisture_grid[0, fix_mask[0, :]]


            self.moisture = updated_grid[tri_mask]
            self.time.append(self.time[-1] + self.dt)
            self.history.append(float(np.mean(self.moisture)))

            if save_snapshots and snapshot_every_steps > 0 and (step % snapshot_every_steps == 0):
                self.raw_snapshots.append(self.moisture.copy())
                self.raw_snapshot_times.append(self.time[-1])

            if plot_frequency > 0 and (step % plot_frequency == 0):
                self.plot_fibers(title=f"t={self.time[-1]:.2f}s")

            step += 1

        # Ensure final frame saved
        if abs(self.time[-1] - t_target) > 1e-9:
            self.time.append(t_target)
            self.history.append(float(np.mean(self.moisture)))
            if save_snapshots:
                self.raw_snapshots.append(self.moisture.copy())
                self.raw_snapshot_times.append(t_target)

    # ---------------------------------------------------------------------
    # Snapshot handling
    # ---------------------------------------------------------------------
    def finalize_snapshots(self, n_final_snapshots=100):
        if not hasattr(self, 'raw_snapshots') or len(self.raw_snapshots) == 0:
            print("⚠️ No raw snapshots to finalize.")
            return

        if len(self.raw_snapshots) <= n_final_snapshots:
            self.snapshots = np.array(self.raw_snapshots)
            self.snapshot_times = np.array(self.raw_snapshot_times)
        else:
            times = np.array(self.raw_snapshot_times)
            indices = np.linspace(0, len(times) - 1, n_final_snapshots, dtype=int)
            self.snapshots = np.array([self.raw_snapshots[i] for i in indices])
            self.snapshot_times = times[indices]

    def export_snapshots_to_csv(self, filename_prefix, include_positions=True):
        all_data = []
        for step_idx, snapshot in enumerate(self.snapshots):
            t = self.snapshot_times[step_idx]
            for i, omega in enumerate(snapshot):
                row = {"time_s": t, "fiber_index": i, "moisture": omega}
                if include_positions:
                    row["x_mm"] = self.fiber_positions[i, 0] * 1e3
                    row["y_mm"] = self.fiber_positions[i, 1] * 1e3
                all_data.append(row)
        pd.DataFrame(all_data).to_csv(f"{filename_prefix}.csv", index=False)

    # ---------------------------------------------------------------------
    # Visualization
    # ---------------------------------------------------------------------
    def plot_fibers(self, title="", full_square=False):
        plt.figure(figsize=(6, 5))
        plt.scatter(self.fiber_positions[:, 0], self.fiber_positions[:, 1],
                    c=self.moisture, cmap="viridis", s=5, vmin=0.05, vmax=0.20)
        plt.colorbar(label="Moisture (g/g)")
        plt.title(title)
        plt.tight_layout()
        plt.show()

    def plot_fibers_video(self, title="", full_square=True):
        """Lightweight plot for video export (non-interactive)."""
        fig, ax = plt.subplots(figsize=(6, 5))
        sc = ax.scatter(self.fiber_positions[:, 0], self.fiber_positions[:, 1],
                        c=self.moisture, cmap="viridis", s=25, vmin=0.05, vmax=0.20)
        plt.colorbar(sc, ax=ax, label="Moisture (g/g)")
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect('equal')
        plt.tight_layout()
        return fig

    def reset_snapshots(self):
        self.raw_snapshots, self.raw_snapshot_times = [], []
        self.snapshots, self.snapshot_times = [], []
        gc.collect()

