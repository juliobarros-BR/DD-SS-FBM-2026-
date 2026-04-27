

import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # keep deterministic for rendering to video/PNG
import matplotlib.pyplot as plt

# ================= USER CONFIG ===================
FOLDERS = [
    Path("./Moisture_Profiles_1000_ramp0.30"),
]

OUT_DIR = Path("visualization_output")
OUT_DIR.mkdir(exist_ok=True)

# MODE: "single" (only FRAME_TO_SAVE) or "video" (export full video)
MODE = "single"           # "single" or "video"
FRAME_TO_SAVE = 110       # used when MODE="single"
MAX_FRAMES_VIDEO = None   # e.g. 1500 to cap runtime; None = no cap

FPS = 10
MAX_CYCLES = 16
TARGET_SECONDS = 20        # <<< NEW: desired total video length (seconds)

# spatial clustering (1 means full resolution in your grid units)
CLUSTER_SIZE = 1

# output video size
W, H = 1200, 600

# moisture plotting range in %
VMIN, VMAX = 5, 20
# =================================================
import colorsys
from matplotlib.colors import LinearSegmentedColormap

def precompute_full_curves(files, outer_ids, center_ids, max_cycles):
    """
    Read all files up to max_cycles and return FULL concatenated curves in %,
    plus consistent x_vals and total_frames.
    """
    y_outer_list, y_center_list, y_avg_list = [], [], []
    last_n_steps = None

    current_cycle = -1
    for cycle_num, phase, file in files:
        if cycle_num != current_cycle:
            current_cycle = cycle_num
            if current_cycle >= max_cycles:
                break

        df = pd.read_csv(file, usecols=["fiber_index", "moisture", "time_index"])

        moist_df = df.pivot(index="fiber_index", columns="time_index", values="moisture").sort_index()
        moist = moist_df.to_numpy()          # (n_fib, n_steps)
        fiber_ids = moist_df.index.to_numpy()
        n_steps = moist.shape[1]
        last_n_steps = n_steps

        id_to_row = {fid: i for i, fid in enumerate(fiber_ids)}
        outer_rows  = [id_to_row[i] for i in outer_ids  if i in id_to_row]
        center_rows = [id_to_row[i] for i in center_ids if i in id_to_row]

        y_avg_list.append(moist.mean(axis=0))
        y_outer_list.append(moist[outer_rows].mean(axis=0) if len(outer_rows) else np.full(n_steps, np.nan))
        y_center_list.append(moist[center_rows].mean(axis=0) if len(center_rows) else np.full(n_steps, np.nan))

    # full curves in %
    y_outer_full  = np.concatenate(y_outer_list)  * 100.0
    y_center_full = np.concatenate(y_center_list) * 100.0
    y_avg_full    = np.concatenate(y_avg_list)    * 100.0

    # your cycle axis convention
    denom = 2.0 * float(last_n_steps if last_n_steps else 1.0)
    x_vals_full = np.linspace(0.0, len(y_avg_full) / denom, len(y_avg_full))

    total_frames = len(y_avg_full)
    return x_vals_full, y_outer_full, y_center_full, y_avg_full, total_frames


def desaturate_rgb(rgb, factor=0.6):
    """
    rgb: (r,g,b) in [0,1]
    factor: 1.0 = original, 0.0 = grayscale
    """
    h, l, s = colorsys.rgb_to_hls(*rgb)
    s *= factor
    return colorsys.hls_to_rgb(h, l, s)



cmyk_points = [
    (40,  0, 100,  0),
    # (20,  0,  60,  0),
    # (20,  0,  20,  0),
    (60,  0,  20,  0),
    # (40,  0,  40,  0),
    (60,  0,  20,  0),
    (100,100, 0,  0),
]

def cmyk_to_rgb(c, m, y, k):
    """
    CMYK in [0,100] -> RGB in [0,1]
    """
    c, m, y, k = [x / 100.0 for x in (c, m, y, k)]
    r = (1 - c) * (1 - k)
    g = (1 - m) * (1 - k)
    b = (1 - y) * (1 - k)
    return (r, g, b)


from matplotlib.colors import LinearSegmentedColormap

rgb_colors = [cmyk_to_rgb(*cmyk) for cmyk in cmyk_points]

rgb_desat = [desaturate_rgb(c, factor=0.55) for c in rgb_colors]

CMAP_CMYK = LinearSegmentedColormap.from_list(
    "cmyk_soft",
    rgb_desat,
    N=256
)



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


# ============================================================
# HELPERS
# ============================================================
def find_mask_files(folder: Path):
    files = list(folder.glob("mask_*_*_cycle.csv"))
    parsed = []
    for f in files:
        if "moistening" in f.name:
            phase = "moistening"
        elif "drying" in f.name:
            phase = "drying"
        else:
            continue
        m = re.search(r"_(\d+)_cycle", f.name)
        if m:
            cycle = int(m.group(1))
            parsed.append((cycle, phase, f))
    return sorted(parsed, key=lambda x: (x[0], 0 if x[1] == "moistening" else 1))


def fig_to_bgr(fig):
    """Backend-agnostic Matplotlib figure -> BGR uint8 for OpenCV."""
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())      # (H, W, 4) RGBA
    rgb = buf[..., :3].copy()                       # RGB
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def get_fiber_groups(first_file: Path):
    df = pd.read_csv(first_file, usecols=["fiber_index", "x", "y"]).drop_duplicates("fiber_index")
    outer_y = df.y.max()
    outer_ids = df[df.y == outer_y].fiber_index.to_numpy()
    cx, cy = df.x.median(), df.y.median()
    center_ids = df[(df.x.sub(cx).abs() <= 1) & (df.y.sub(cy).abs() <= 1)].fiber_index.to_numpy()
    all_ids = df.fiber_index.to_numpy()
    return np.sort(outer_ids), np.sort(center_ids), np.sort(all_ids)


def make_grid_mapper(x, y, cluster_size: int):
    """
    Precompute mapping from each fiber to a grid cell id:
      cell_id[i] in [0, n_cells)
    and grid shape (H, W).
    """
    xq = (np.floor(x / cluster_size)).astype(np.int32)
    yq = (np.floor(y / cluster_size)).astype(np.int32)

    x0 = xq.min()
    y0 = yq.min()
    xq = xq - x0
    yq = yq - y0

    Wg = int(xq.max() + 1)
    Hg = int(yq.max() + 1)

    cell_id = yq * Wg + xq
    n_cells = Wg * Hg
    return cell_id, Hg, Wg, n_cells


def grid_mean(values, cell_id, n_cells):
    """Fast mean per cell using bincount."""
    values = values.astype(np.float64, copy=False)
    sumv = np.bincount(cell_id, weights=values, minlength=n_cells)
    cnt = np.bincount(cell_id, minlength=n_cells)
    out = sumv / np.maximum(cnt, 1)
    out[cnt == 0] = np.nan
    return out


def build_frame_figure(
    heat_percent_2d,
    x_vals,
    y_outer,
    y_center,
    y_avg,
    frame_i,
    folder_name: str,
    two_sided_yaxis: bool = True,
    # --- ramp annotation controls (data coords) ---
    ramp_x0: float = 1.0,
    ramp_x1: float = 1.25,
    ramp_x2: float = 1.5,
    ramp_text: str = r"$T_r=50\%$",
):
    """
    1x3 panel:
      LEFT  = curves
      MID   = vertical label
      RIGHT = heatmap
    - Heatmap has no ticklabels.
    - Curve y-axis ticks shown on BOTH sides; NO y-label on the curve plot.
    - Colorbar is placed on the LEFT of the heatmap (in the middle).
    - Curve x ticks + ticklabels are kept.
    - Optional ramp annotation: two thin horizontal segments + centered text.
    """
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    fig, axs = plt.subplots(
        1, 3, figsize=(16, 6),
        gridspec_kw={"width_ratios": [1, 0.05, 1]}
    )

    fig.subplots_adjust(
        left=0.055,
        right=0.985,
        bottom=0.14,
        top=0.93,
        wspace=0.04   # <<< this is the big one (gap between panels)
    )


    ax_curve, ax_mid, ax_img = axs

    # =====================================================
    # ---- LEFT: curves ----
    # =====================================================
    ax_curve.plot(x_vals, y_outer, color="gray", alpha=0.4, zorder=0)
    ax_curve.plot(x_vals, y_center, color="gray", alpha=0.4, zorder=0)
    ax_curve.plot(x_vals, y_avg,  color="gray", alpha=0.4, zorder=0)

    is_avg = "avg" in folder_name.lower()
    if is_avg:
        segment = 10
        base_orders = [[1, 2, 3], [2, 3, 1], [3, 1, 2]]
        n_segments = max(1, frame_i // segment + 1)

        for s in range(n_segments):
            start = max(0, s * segment - 1)
            end = min((s + 1) * segment, frame_i + 1)
            z = base_orders[s % len(base_orders)]

            ax_curve.plot(
                x_vals[start:end], y_outer[start:end],
                color="red", lw=3, zorder=z[0],
                label=r"$\varphi(\Omega_{\mathrm{out}})$" if s == 0 else ""
            )
            ax_curve.plot(
                x_vals[start:end], y_center[start:end],
                color="green", lw=3, zorder=z[1],
                label=r"$\varphi(\Omega_{\mathrm{cen}})$" if s == 0 else ""
            )
            ax_curve.plot(
                x_vals[start:end], y_avg[start:end],
                color="blue", lw=3, zorder=z[2],
                label=r"$\langle\varphi\rangle$" if s == 0 else ""
            )
    else:
        ax_curve.plot(
            x_vals[:frame_i+1], y_outer[:frame_i+1],
            color="red", lw=3,
            label=r"$\varphi(\Omega_{\mathrm{out}})$"
        )
        ax_curve.plot(
            x_vals[:frame_i+1], y_center[:frame_i+1],
            color="green", lw=3, zorder=10,
            label=r"$\varphi(\Omega_{\mathrm{cen}})$"
        )
        ax_curve.plot(
            x_vals[:frame_i+1], y_avg[:frame_i+1],
            color="blue", lw=3,
            label=r"$\langle\varphi\rangle$"
        )

    idx_vline = min(frame_i, len(x_vals) - 1)
    ax_curve.axvline(x_vals[idx_vline], color="k", linestyle="--", lw=3)

    # -----------------------------------------------------
    # Axes formatting (MAIN axis handles x-label/ticks)
    # -----------------------------------------------------
    ax_curve.set_xlim(left=0)
    ax_curve.set_ylim(VMIN, VMAX)
    ax_curve.set_yticks([5, 10, 15, 20])
    ax_curve.set_ylabel("")  # NO y-label

    # x axis: KEEP label + ticks
    ax_curve.set_xlabel("Cycle", labelpad=-4)
    ax_curve.set_xticks([0, 1, 2, 3, 4])
    ax_curve.set_xticklabels([r"$0$", r"$1$", r"$2$", r"$3$", r"$4$"])
    ax_curve.tick_params(axis="x", which="both", bottom=True, labelbottom=True)

    ax_curve.axvline(1.0, color="k", linestyle=":", lw=2)
    ax_curve.axvline(1.25, color="k", linestyle=":", lw=2)
    ax_curve.axvline(1.50, color="k", linestyle=":", lw=2)

    ax_curve.grid(True, zorder=0)

    # ---- duplicate y ticks on RIGHT side (same subplot) ----
    # Only create if requested; otherwise left-only.
    if two_sided_yaxis:
        ax_curve_r = ax_curve.twinx()
        ax_curve_r.set_ylim(VMIN, VMAX)
        ax_curve_r.set_yticks([5, 10, 15, 20])
        ax_curve_r.set_ylabel("")  # NO y-label
        ax_curve_r.tick_params(axis="y", which="both", right=True, labelright=True)
        
        ax_curve.set_xticks([0, 1, 2, 3, 4])
        ax_curve.set_xticklabels([r"$0$", r"$1$", r"$2$", r"$3$", r"$4$"])
        ax_curve_r.set_xlabel("")
        ax_curve_r.spines["right"].set_visible(True)

    # =====================================================
    # ---- ramp annotation (two thin segments + text) ----
    # =====================================================
    # Place it near the top, but inside the plot. Use data coords.
    y_anno = VMAX  # tweak if you want higher/lower
    lw_anno = 2

    # segment 1: [ramp_x0, ramp_x1]
    ax_curve.hlines(y_anno-0.05, ramp_x0, ramp_x1, colors="k", linewidth=lw_anno, zorder=20, linestyle=":")
    # segment 2: [ramp_x1, ramp_x2]
    ax_curve.hlines(y_anno-0.05, ramp_x1, ramp_x2, colors="k", linewidth=lw_anno, zorder=20, linestyle=":")

    # # text centered over the first segment (2 to 2.25) like you asked
    ax_curve.text(
        0.5 * (ramp_x0 + ramp_x1),
        y_anno ,
        ramp_text,
        ha="center", va="bottom",
        fontsize=25,
        zorder=21
    )

    # legend
    leg = ax_curve.legend(loc="upper right", fontsize=22, frameon=True)
    leg.set_zorder(10)

    # =====================================================
    # ---- MIDDLE: vertical label ----
    # =====================================================
    ax_mid.axis("off")
    ax_mid.text(
        2, 0.5, r"$\varphi$ (\%)",
        ha="center", va="center",
        fontsize=28,
        fontfamily="serif",
        transform=ax_mid.transAxes
    )

    # =====================================================
    # ---- RIGHT: heatmap + colorbar on its LEFT ----
    # =====================================================
    im = ax_img.imshow(
        heat_percent_2d,
        cmap="GnBu",  # or your CMAP_CMYK_SOFT
        vmin=VMIN, vmax=VMAX,
        aspect="equal",
    )
    # --- full frame (box) around the heatmap ---
    for side in ["left", "right", "top", "bottom"]:
        ax_img.spines[side].set_visible(True)
        ax_img.spines[side].set_linewidth(1.0)  # match your paper style
        ax_img.spines[side].set_color("black")

    ax_img.set_xticks([])
    ax_img.set_yticks([])

    divider = make_axes_locatable(ax_img)
    cax = divider.append_axes("left", size="4%", pad=0.15)
    cbar = fig.colorbar(im, cax=cax, ticks=[5, 10, 15, 20])
    cax.yaxis.tick_left()
    cax.yaxis.set_label_position("left")
    

    return fig






# ============================================================
# MAIN PROCESSING
# ============================================================
def process_folder(folder: Path):
    files = find_mask_files(folder)
    if not files:
        print(f"No mask files in {folder}")
        return

    outer_ids, center_ids, _all_ids = get_fiber_groups(files[0][2])

    # ---------- PREPASS: build FULL curves once ----------
    x_vals_full, y_outer_full, y_center_full, y_avg_full, total_frames = precompute_full_curves(
        files, outer_ids, center_ids, MAX_CYCLES
    )

    # Video writer
    writer = None
    joint_path = OUT_DIR / f"{folder.name}_joint.mp4"
    if MODE == "video":
        n_out = int(FPS * TARGET_SECONDS)  # <<< NEW
        keep_idx = set(np.linspace(0, total_frames - 1, n_out).astype(int))  # <<< NEW

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(joint_path), fourcc, FPS, (W, H))
        if not writer.isOpened():
            raise RuntimeError("Could not open VideoWriter.")

    global_frame = 0
    current_cycle = -1

    try:
        for cycle_num, phase, file in files:
            if cycle_num != current_cycle:
                current_cycle = cycle_num
                if current_cycle >= MAX_CYCLES:
                    break

            df = pd.read_csv(file, usecols=["fiber_index", "x", "y", "moisture", "time_index"])
            moist_df = df.pivot(index="fiber_index", columns="time_index", values="moisture").sort_index()
            moist = moist_df.to_numpy()
            fiber_ids = moist_df.index.to_numpy()
            n_steps = moist.shape[1]

            coords = (
                df.drop_duplicates("fiber_index")[["fiber_index", "x", "y"]]
                .sort_values("fiber_index")
                .set_index("fiber_index")
                .reindex(fiber_ids)
            )
            x = coords["x"].to_numpy(dtype=float)
            y = coords["y"].to_numpy(dtype=float)

            cell_id, Hg, Wg, n_cells = make_grid_mapper(x, y, CLUSTER_SIZE)

            for t in range(n_steps):

                if MODE == "video":

                    if global_frame in keep_idx:
                        values = moist[:, t]
                        heat_flat = grid_mean(values, cell_id, n_cells)
                        heat = heat_flat.reshape(Hg, Wg) * 100.0

                        frame_i = min(global_frame, total_frames - 1)

                        fig = build_frame_figure(
                            heat_percent_2d=heat,
                            x_vals=x_vals_full,
                            y_outer=y_outer_full,
                            y_center=y_center_full,
                            y_avg=y_avg_full,
                            frame_i=frame_i,
                            folder_name=folder.name,
                        )

                        frame_bgr = fig_to_bgr(fig)
                        frame_bgr = cv2.resize(frame_bgr, (W, H), interpolation=cv2.INTER_AREA)
                        writer.write(frame_bgr)
                        plt.close(fig)

                elif MODE == "single":
                    if global_frame == FRAME_TO_SAVE:
                        # --- build the same frame as in video mode ---
                        values = moist[:, t]
                        heat_flat = grid_mean(values, cell_id, n_cells)
                        heat = heat_flat.reshape(Hg, Wg) * 100.0

                        frame_i = min(global_frame, total_frames - 1)

                        fig = build_frame_figure(
                            heat_percent_2d=heat,
                            x_vals=x_vals_full,
                            y_outer=y_outer_full,
                            y_center=y_center_full,
                            y_avg=y_avg_full,
                            frame_i=frame_i,
                            folder_name=folder.name,
                        )

                        out_png = OUT_DIR / f"{folder.name}_frame_{FRAME_TO_SAVE:05d}.png"
                        fig.savefig(out_png)  # uses your rcParams dpi etc.
                        plt.close(fig)
                        print(f"Saved image: {out_png}")
                        return  # stop after saving the requested frame

                global_frame += 1
            del df, moist_df, moist

            if MODE == "video" and MAX_FRAMES_VIDEO is not None and global_frame >= MAX_FRAMES_VIDEO:
                break

        if MODE == "video":
            print(f"Saved video: {joint_path}")

    finally:
        if writer is not None:
            writer.release()




if __name__ == "__main__":
    for folder in FOLDERS:
        if folder.exists():
            process_folder(folder)
        else:
            print(f"Missing: {folder}")
