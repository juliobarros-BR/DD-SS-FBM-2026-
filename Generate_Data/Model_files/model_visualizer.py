#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  7 15:41:06 2025

@author: jortiz
"""

import matplotlib.pyplot as plt
import numpy as np
import decimal
import matplotlib.gridspec as gridspec
from matplotlib import cm, colors

class ModelVisualizer:
    """
    Visualization helper for the Fiber Bundle Model.

    Contains all plotting utilities that depend on a Model instance.
    """

    def __init__(self, model):
        """
        Initialize the visualizer with a reference to a Model instance.

        Parameters
        ----------
        model : Model
            The simulation model whose state will be visualized.
        """
        self.model = model


    # ===============================================================
    # 1. Plot distance kernel
    # ===============================================================
    def plot_distance_matrix(self, cmap='viridis'):
        """Visualize the full distance matrix as a 2D color map."""
        dm = self.model.distance_matrix
        grid_size = dm.shape[0]

        plt.figure(figsize=(6, 6))
        im = plt.matshow(dm, cmap=cmap, fignum=0)
        plt.colorbar(im, fraction=0.046, pad=0.04, label='Normalized weight')
        plt.title(f'Distance Kernel (γ = {self.model.gamma_deg:.2f})')
        plt.axis('off')
        plt.show()


    # ===============================================================
    # 2. Plot local forces
    # ===============================================================
    def plot_local_forces(self, cmap='coolwarm', normalize=False, show_sum=True):
        """
        Visualize the current per-fiber local force field as a 2D matrix.
        """
        grid_size = int(np.ceil(np.sqrt(self.model.N)))

        # reshape into square grid
        force_matrix = np.full((grid_size, grid_size), np.nan)
        force_matrix.flat[:self.model.N] = self.model.local_force

        if normalize and self.model.load != 0:
            force_matrix /= self.model.load

        if show_sum:
            total = np.nansum(force_matrix)
            print(f"Σ local_forces = {total:.6e} (model.load = {self.model.load:.6e})")

        plt.figure(figsize=(6, 6))
        im = plt.matshow(force_matrix, cmap=cmap, fignum=0)
        plt.colorbar(im, fraction=0.046, pad=0.04,
                     label='Normalized force' if normalize else 'Local force')
        plt.title(f"Local Force Field ({'normalized' if normalize else 'absolute'})")
        plt.axis('off')
        plt.show()


    # ===============================================================
    # 3. Plot fiber moisture + state fields
    # ===============================================================
    def plot_fiber_moisture(self, cur_time, increment):
        """
        Plot:
          - Moisture field
          - Local forces
          - Slip count
          - Histogram of distance to threshold
        """
        model = self.model
        if model.geometry != 0:
            raise ValueError("plot_fiber_moisture currently supports geometry=0 (square grid) only.")

        grid_size = int(np.ceil(np.sqrt(model.N)))

        # Initialize matrices
        moisture_matrix = np.full((grid_size, grid_size), np.nan)
        force_matrix = np.full((grid_size, grid_size), np.nan)
        slip_matrix = np.full((grid_size, grid_size), np.nan)

        # Fill data
        moisture_matrix.flat[:model.N] = model.fiber_moisture
        force_matrix.flat[:model.N] = model.local_force
        slip_matrix.flat[:model.N] = model.slip_count

        # Compute degradation, threshold, etc.
        decay = model.sys_var.get("decay")
        degradation_mode = model.sys_var.get("degradation_mode", 0)
        if degradation_mode == 0:
            degrad_th = model.threshold * np.exp(-model.slip_count / decay)
        else:
            min_factor = np.exp(-model.max_slip / decay)
            lin_factor = 1 - (1 - min_factor) * (model.slip_count / model.max_slip)
            degrad_th = model.threshold * lin_factor

        idx_reverse = np.ones(model.N)
        idx_reverse[np.where(model.local_slip > model.total_strain)] = -1
        wet_sc = (1 + (model.sys_var.get('wet_scale') - 1)
                  * model.normalized_moisture * model.fiber_moisture)
        reverse_sc = (1 + (model.sys_var.get('reverse_scale') - 1)
                      * (1 - idx_reverse) / 2)

        diff = (np.abs(
            model.total_strain
            - model.local_slip
            - (model.fiber_moisture - model.sys_var.get("start_wet"))
              * model.alpha * model.sys_var.get("hygro_slip")
        ) * model.local_intact
               - degrad_th * wet_sc * reverse_sc)
        diff[np.where(model.local_force < 0)] = -diff[np.where(model.local_force < 0)]

        valid_diff = diff[(model.local_intact != 0) & (model.local_force < 0) & (model.slip_count < 60)]
        valid_diff = valid_diff / model.critical_strain

        # === Plot ===
        fig = plt.figure(figsize=(10, 10))
        gs = gridspec.GridSpec(2, 2, figure=fig)
        ax0, ax1, ax2, ax3 = [fig.add_subplot(gs[i]) for i in [(0, 0), (0, 1), (1, 0), (1, 1)]]

        # Moisture
        im0 = ax0.imshow(moisture_matrix, origin='lower', cmap='viridis', vmin=0, vmax=1)
        ax0.set_title(r"Normalized Moisture Content ($\omega / 0.3$)")
        ax0.axis('off')
        plt.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04)

        # Histogram
        ax1.hist(valid_diff, bins=50, color='darkorange', edgecolor='black')
        ax1.axvline(0, color='red', linestyle='--', label="Threshold Reached")
        ax1.set_title("Fiber Distance to Threshold")
        ax1.set_xlabel(r"Distance to threshold ($(\varepsilon - \varepsilon^S_i - \varepsilon^{th}_i)/\varepsilon_c$)")
        ax1.set_ylabel("Fiber Count")
        ax1.legend()

        # Local Force
        im2 = ax2.imshow(force_matrix / model.critical_load, origin='lower', cmap='viridis')
        ax2.set_title(r"Local Normalized Stress ($\sigma_i / \sigma_c$)")
        ax2.axis('off')
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

        # Slip Count
        im3 = ax3.imshow(slip_matrix, origin='lower', cmap='viridis')
        ax3.set_title("Slip Count [-]")
        ax3.axis('off')
        plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

        increment_str = f"{decimal.Decimal(increment / model.critical_strain):.2e}"
        fig.text(0.5, 0.45, f"Time = {cur_time:.2f}\n Increment = {increment_str}",
                 fontsize=14, ha='center', va='center',
                 bbox=dict(facecolor='white', edgecolor='black', alpha=0.8))

        plt.tight_layout()
        plt.show()


    # ===============================================================
    # 4. Plot geometry / positions
    # ===============================================================
    def plot_positions(self):
        """Plot fiber positions and geometry."""
        model = self.model
        plt.figure(figsize=(8, 8))
        if model.geometry == 0:
            plt.scatter(model.positions[:, 0], model.positions[:, 1],
                        s=1, c='blue', label='Square Geometry')
        elif model.geometry == 1:
            plt.scatter(model.positions[:, 0], model.positions[:, 1],
                        s=1, c='green', label='Circular Geometry')
        else:
            raise ValueError(f"Unsupported geometry type: {model.geometry}")

        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("Fiber Positions Visualization")
        plt.axis('equal')
        plt.legend()
        plt.show()
        
        
    # ===============================================================
    # 5. Plot moisture
    # ===============================================================    
    
    def plot_fiber_moisture_absolute(self, cmap="viridis", show_colorbar=True):
        """
        Plot the *absolute* moisture content of each fiber in matrix form
        (no normalization or clipping).
    
        Parameters
        ----------
        cmap : str
            Matplotlib colormap (default 'viridis').
        show_colorbar : bool
            Whether to display a colorbar.
        """
        model = self.model
        grid_size = int(np.ceil(np.sqrt(model.N)))
    
        # Reshape the 1D moisture array into a 2D matrix
        moisture_matrix = np.full((grid_size, grid_size), np.nan)
        moisture_matrix.flat[:model.N] = model.fiber_moisture
    
        plt.figure(figsize=(6, 6))
        im = plt.matshow(
            moisture_matrix,
            cmap=cmap,
            fignum=0,
            origin="lower"
        )
        if show_colorbar:
            plt.colorbar(im, fraction=0.046, pad=0.04, label="Absolute Moisture Content")
    
        plt.title("Fiber Moisture (Absolute Values)")
        plt.axis("off")
        plt.show()
        
        
    # ===============================================================
    # 5. Plot slips
    # ===============================================================    
    
        
    def plot_slip_distribution(self, cmap="Greys", normalize=False, show_colorbar=True):
        """
        Visualize the current slip distribution across the fiber bundle as a 2D matrix.
    
        Parameters
        ----------
        cmap : str
            Colormap for the visualization (default: 'inferno').
        normalize : bool
            If True, normalizes slip values by the global maximum slip (for relative comparison).
        show_colorbar : bool
            Whether to display a colorbar.
        """
        model = self.model
        grid_size = int(np.ceil(np.sqrt(model.N)))
    
        # Reshape the 1D slip array into a 2D grid
        slip_matrix = np.full((grid_size, grid_size), np.nan)
        slip_matrix.flat[:model.N] = model.slip_count
    
        # Normalize if requested
        if normalize:
            max_slip = np.nanmax(slip_matrix)
            if max_slip > 0:
                slip_matrix /= max_slip
        # for num in slip_matrix.flat:  # iterate over every element in the 2D array
            # if not np.isnan(num) and not float(num).is_integer():
            #     print(num)
        # Plot
        plt.figure(figsize=(6, 6))
        im = plt.matshow(
            slip_matrix,
            cmap=cmap,
            origin="lower",
            fignum=0
        )
    
        if show_colorbar:
            label = "Normalized Slip [-]" if normalize else "Slip Strain [-]"
            plt.colorbar(im, fraction=0.046, pad=0.04, label=label)
    
        plt.title(f"Slip Distribution ({'normalized' if normalize else 'absolute'})")
        plt.axis("off")
        plt.show()
        
        
    def plot_local_strain(self, cmap="inferno", normalize=False, show_colorbar=True):
        """
        Visualize the current slip distribution across the fiber bundle as a 2D matrix.
    
        Parameters
        ----------
        cmap : str
            Colormap for the visualization (default: 'inferno').
        normalize : bool
            If True, normalizes slip values by the global maximum slip (for relative comparison).
        show_colorbar : bool
            Whether to display a colorbar.
        """
        model = self.model
        grid_size = int(np.ceil(np.sqrt(model.N)))
    
        # Create strain matrix
        strain_matrix = np.full((grid_size, grid_size), np.nan)
        strain_matrix.flat[:model.N] = model.local_strain
    
        # Create intact mask (1 = intact, 0 = broken)
        intact_mask = np.full((grid_size, grid_size), np.nan)
        intact_mask.flat[:model.N] = model.local_intact
    
        # Mask broken fibers (set them to NaN so we can color them white)
        strain_matrix[intact_mask == 0] = np.nan
    
        # Normalize if requested
        if normalize:
            max_val = np.nanmax(strain_matrix)
            if max_val > 0:
                strain_matrix /= max_val
    
        # --- Custom colormap with white for NaNs ---
        cmap_obj = plt.get_cmap(cmap).copy()
        cmap_obj.set_bad(color="white")
    
        # Plot
        plt.figure(figsize=(6, 6))
        im = plt.imshow(
            strain_matrix,
            cmap=cmap_obj,
            origin="lower",
            interpolation="nearest"
        )
    
        if show_colorbar:
            label = "Normalized Strain [-]" if normalize else "Local Strain [-]"
            plt.colorbar(im, fraction=0.046, pad=0.04, label=label)
    
        plt.title(f"Local Strain Distribution (Total = {model.total_strain:.4f})")
        plt.axis("off")
        plt.tight_layout()
        plt.show()
        
    def plot_slip_strain_only(
        self,
        outpath="slip_strain_pixels.png",
        cmap="viridis",
        figsize=(8, 6),
        vmin=None,
        vmax=0.915927150509827,   # fixed vmax
        # ===== MANUAL LAYOUT (figure coords [0,1]) =====
        LEFT=0.02,      # main axis left
        BOTTOM=0.12,    # main axis bottom
        SIZE=0.76,      # main axis is SIZE x SIZE (square)
        GAP=0.01,       # gap to colorbar
        CB_W=0.035,     # colorbar width
        # ==============================================
        # ---- average text toggle (easy to comment out by setting False) ----
        SHOW_AVG=True,
    ):
        """
        Plot ONLY slip strain as a per-fiber pixel map (1 fiber = 1 square).
        - No coarsening: each fiber = one square
        - Broken fibers plotted in solid red
        - No title
        - Shows x/y tick values + grid
        - Colorbar label only: \\varepsilon_i^S
        - Manual, publication-safe layout: main axis + separate cbar axis (won't override set_position)
        - Text box: Broken fibers (red) + optional average slip over intact fibers (broken excluded)
        """
        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib as mpl

        # -----------------------------
        # Style (your rcParams)
        # -----------------------------
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
            'savefig.bbox': None,  # keep fixed canvas; don't auto-crop
            'figure.dpi': 150,
        })

        model = self.model
        grid_size = int(np.ceil(np.sqrt(model.N)))

        # -----------------------------
        # Build 2D slip + intact masks
        # -----------------------------
        slip_flat = np.full(grid_size * grid_size, np.nan, dtype=float)
        slip_flat[:model.N] = model.local_slip
        slip_2d = slip_flat.reshape((grid_size, grid_size))

        intact_flat = np.zeros(grid_size * grid_size, dtype=bool)
        intact_flat[:model.N] = (model.local_intact == 1)
        intact_2d = intact_flat.reshape((grid_size, grid_size))

        broken_mask = ~intact_2d

        # intact-only field (broken excluded via NaN)
        slip_intact = np.where(intact_2d, slip_2d, np.nan)

        # vmin auto from intact unless provided
        if vmin is None:
            vmin = np.nanmin(slip_intact)

        # -----------------------------
        # Figure + MANUAL layout (AX + CBAR)
        # IMPORTANT: do NOT use fig.colorbar(..., ax=ax, ...)
        # because it will shrink/move ax and override your manual placement.
        # -----------------------------
        fig = plt.figure(figsize=figsize, dpi=150)

        # main plot axis (square) + separate colorbar axis
        ax  = fig.add_axes([LEFT, BOTTOM, SIZE, SIZE])
        cax = fig.add_axes([LEFT + SIZE + GAP, BOTTOM, CB_W, SIZE])

        # -----------------------------
        # Colormap: NaNs transparent
        # -----------------------------
        cmap_obj = mpl.cm.get_cmap(cmap).copy()
        cmap_obj.set_bad((1, 1, 1, 0))  # transparent

        # Main slip map (intact only)
        im = ax.imshow(
            slip_intact,
            cmap=cmap_obj,
            vmin=vmin, vmax=vmax,
            aspect="equal",
            origin="upper",
            interpolation="nearest",
        )

        # Overlay broken fibers in solid red
        broken_overlay = np.where(broken_mask, 1.0, np.nan)
        red_cmap = mpl.colors.ListedColormap(["red"])
        red_cmap.set_bad((1, 1, 1, 0))
        ax.imshow(
            broken_overlay,
            cmap=red_cmap,
            vmin=0, vmax=1,
            aspect="equal",
            origin="upper",
            interpolation="nearest",
        )

        # -----------------------------
        # Ticks similar to your other plot
        # -----------------------------
        mid = grid_size // 2
        max_i = grid_size - 1

        ax.set_xticks([0, mid, max_i])
        ax.set_yticks([0, mid, max_i])

        ax.set_xticklabels([0, mid, grid_size])
        ax.set_yticklabels([grid_size, mid, 0])

        ax.grid(True, zorder=10)

        # -----------------------------
        # Colorbar (manual axis)
        # -----------------------------
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label(r"$\varepsilon_i^S$")

        # -----------------------------
        # Broken fibers + average slip (intact only)
        # -----------------------------
        intact_sum = int(np.sum(model.local_intact))  # 1/0 array
        broken = int(model.N) - intact_sum

        text_lines = [rf"Broken Fibers: {broken:d}"]

        # ---- Average slip over intact fibers (easy to disable) ----
        if SHOW_AVG:
            avg_slip = np.nanmean(slip_intact)  # broken excluded via NaN
            text_lines.append(rf"$\langle \varepsilon_i^S \rangle = {avg_slip:.4f}$")
        # -----------------------------------------------------------

        # -----------------------------
        # Text: Broken fibers (red)
        # -----------------------------
        # ax.text(
        #     0.02, 0.98,
        #     s=28,
        #     rf"Broken Fibers: {broken:d}",
        #     transform=ax.transAxes,
        #     ha="left", va="top",
        #     color="red",
        #     bbox=dict(
        #         boxstyle="round,pad=0.3",
        #         facecolor="white",
        #         edgecolor="black",
        #         alpha=0.9
        #     )
        # )

        # # -----------------------------
        # # Text: average slip (black)
        # # -----------------------------
        # if SHOW_AVG:
        #     ax.text(
        #         0.02, 0.85,   # slightly lower y
        #         s=28,
        #         rf"$\langle \varepsilon_i^S \rangle = {avg_slip:.4f}$",
        #         transform=ax.transAxes,
        #         ha="left", va="top",
        #         color="black",
        #         bbox=dict(
        #             boxstyle="round,pad=0.3",
        #             facecolor="white",
        #             edgecolor="black",
        #             alpha=0.9
        #         )
        #     )


        # No title
        # ax.set_title("")

        plt.savefig(outpath, dpi=300)
        plt.show()
        plt.close(fig)

    def plot_joint_bundle_state(self, cur_time=None, cmap_force="coolwarm", cmap_slip="Greys"):
        """
        Plot a 2x3 grid summarizing the current bundle state:
        Top: distance matrix, local strain, local force
        Bottom: moisture, slip strain, slip count
        """
        model = self.model
        grid_size = int(np.ceil(np.sqrt(model.N)))
        fig, axes = plt.subplots(2, 3, figsize=(15, 10), dpi=200)
    
        # -----------------------------------
        # --- TOP ROW
        # -----------------------------------
    
        # 1️⃣ Distance matrix (cropped to bundle size)
        dm = getattr(model, "distance_matrix", None)
        if dm is not None:
            dm_size = dm.shape[0]
            center = dm_size // 2
            half_grid = grid_size // 2
            crop_start = max(center - half_grid, 0)
            crop_end = min(center + half_grid + 1, dm_size)
            dm_cropped = dm[crop_start:crop_end, crop_start:crop_end]
    
            im0 = axes[0, 0].imshow(dm_cropped, cmap="viridis", origin="lower")
            axes[0, 0].set_title(f"Distance Kernel (γ={getattr(model, 'gamma_deg', 0):.2f})")
            axes[0, 0].axis("off")
            fig.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04, label="Normalized weight")
        else:
            axes[0, 0].text(0.5, 0.5, "No distance matrix", ha="center", va="center")
            axes[0, 0].axis("off")

    
        # 2️⃣ Local strain
        local_strain = np.full((grid_size, grid_size), np.nan)
        local_strain.flat[:model.N] = model.local_strain
        im1 = axes[0, 1].imshow(local_strain, cmap=cmap_force, origin="lower")
        axes[0, 1].set_title("Local Strain Distribution")
        axes[0, 1].axis("off")
        fig.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04, label="ε_local [-]")
    
        # 3️⃣ Local force
        local_force = np.full((grid_size, grid_size), np.nan)
        local_force.flat[:model.N] = model.local_force
        im2 = axes[0, 2].imshow(local_force, cmap=cmap_force, origin="lower")
        axes[0, 2].set_title("Local Force Distribution")
        axes[0, 2].axis("off")
        fig.colorbar(im2, ax=axes[0, 2], fraction=0.046, pad=0.04, label="F_local [-]")
    
        # -----------------------------------
        # --- BOTTOM ROW
        # -----------------------------------
    
        # 4️⃣ Moisture
        fiber_moisture = np.full((grid_size, grid_size), np.nan)
        fiber_moisture.flat[:model.N] = model.fiber_moisture
        im3 = axes[1, 0].imshow(fiber_moisture, cmap="viridis", origin="lower", vmin=0, vmax=1)
        if cur_time is not None:
            axes[1, 0].set_title(rf"Moisture Profile ($t = {cur_time:.3f}$)")
        else:
            axes[1, 0].set_title(r"Normalized Moisture Content ($\omega / 0.2$)")
        axes[1, 0].axis("off")
        fig.colorbar(im3, ax=axes[1, 0], fraction=0.046, pad=0.04)
    
        # 5️⃣ Slip strain (COARSENED)
        # --------------------------------------------------
        
        # 5️⃣ Slip strain (COARSENED, alpha = intact fraction)
        # --------------------------------------------------
        
        COARSE = 2
        # block size: COARSE x COARSE fibers
        
        # --- Build intact mask (flat)
        intact_flat = np.zeros(grid_size * grid_size, dtype=bool)
        intact_flat[:model.N] = (model.local_intact == 1)
        
        # --- Build slip field (flat)
        local_slip_flat = np.full(grid_size * grid_size, np.nan)
        local_slip_flat[:model.N] = model.local_slip
        
        # Reshape to 2D
        local_slip_2d = local_slip_flat.reshape((grid_size, grid_size))
        intact_2d = intact_flat.reshape((grid_size, grid_size))
        
        # ==================================================
        # COARSENING
        # ==================================================
        ny, nx = local_slip_2d.shape
        nyc, nxc = ny // COARSE, nx // COARSE
        
        # Trim to exact multiple
        slip_trim = local_slip_2d[:nyc*COARSE, :nxc*COARSE]
        intact_trim = intact_2d[:nyc*COARSE, :nxc*COARSE]
        
        # --- Count intact fibers per block
        intact_count = np.sum(
            intact_trim.reshape(nyc, COARSE, nxc, COARSE),
            axis=(1, 3)
        )
        
        block_size = COARSE * COARSE
        intact_fraction = intact_count / block_size  # ∈ [0, 1]
        
        # --- Average slip ONLY over intact fibers
        slip_masked = np.where(intact_trim, slip_trim, np.nan)
        
        slip_coarse = np.nanmean(
            slip_masked.reshape(nyc, COARSE, nxc, COARSE),
            axis=(1, 3)
        )
        
        # ==================================================
        # PLOTTING
        # ==================================================
        
        im4 = axes[1, 1].imshow(
            slip_coarse,
            cmap="viridis",          # perceptually uniform
            origin="lower",
            interpolation="nearest",
            alpha=intact_fraction   # 👈 THE KEY IDEA
        )
        
        axes[1, 1].set_title(
            f"Slip strain (coarse {COARSE}×{COARSE}, α = intact fraction)"
        )
        axes[1, 1].axis("off")
        
        fig.colorbar(
            im4,
            ax=axes[1, 1],
            fraction=0.046,
            pad=0.04,
            label="⟨ε_slip⟩ intact fibers [-]"
        )




    
        # 6️⃣ Slip count
        slip_count = np.full((grid_size, grid_size), np.nan)
        slip_count.flat[:model.N] = model.slip_count
        im5 = axes[1, 2].imshow(slip_count, cmap=cmap_slip, origin="lower")
        axes[1, 2].set_title("Slip Count Distribution")
        axes[1, 2].axis("off")
        fig.colorbar(im5, ax=axes[1, 2], fraction=0.046, pad=0.04, label="Slip count [-]")
    
        # -----------------------------------
        # --- Final layout
        # -----------------------------------
        plt.suptitle("Bundle State Summary", fontsize=14, y=0.98)
        plt.tight_layout()
        plt.savefig( "joint_bundle_state2.png", dpi=300)
        plt.show()
        plt.close(fig)


