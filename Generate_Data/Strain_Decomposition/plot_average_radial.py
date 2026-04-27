#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np
import joblib
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# ===================== USER CONFIG =====================
FILE_A = Path("./Slip_snapshot_cycle_1.joblib")       # Case A  ->  φ_i
FILE_B = Path("./avg_slip_snapshot_cycle_1_before.joblib")   # Case B  ->  <φ_i>

OUTPATH = Path("./count_radial_mean_slip_rolling_cycle_1.png")

FIGSIZE = (8, 6)

# Rolling window thickness (in grid units = "fiber pixels")
DR_WINDOW = 15.0          # ΔR (full width). Try 4..12 depending on N.

# How many R evaluation points along the curve
N_EVAL = 100

# Require at least this many intact fibers inside each window to accept the point
MIN_COUNT = 50

# Use Gaussian weights inside the window for smoother curves?
USE_GAUSSIAN_WEIGHTS = True
GAUSS_SIGMA = None       # if None: sigma = DR_WINDOW/4.0

# Plot R normalized by Rmax?
NORMALIZE_R = True

# Inset: where to show the annotated example radius R (as fraction of Rmax)
R_ANNOT_FRACTION = 0.55  # e.g. 0.55 means show R at 55% of Rmax

# Inset position (axes fraction) [x0, y0, w, h]
INSET_BBOX = [0.63, 0.65, 0.35, 0.35]

# Grayscale heatmap settings in inset
INSET_CMAP = "gray"
INSET_SHOW_BROKEN_AS_BLACK = True
# =======================================================


def apply_pub_style():
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
        'savefig.bbox': None,
        'figure.dpi': 150,
    })


def load_snapshot(path: Path):
    snap = joblib.load(path)
    for k in ("N", "local_slip", "local_intact"):
        if k not in snap:
            raise KeyError(f"{path} missing key '{k}' (needs N, local_slip, local_intact)")

    N = int(snap["N"])
    slip = np.asarray(snap["local_slip_count"], dtype=float)
    intact = (np.asarray(snap["local_intact"]).astype(int) == 1)
    max_intact = np.max(slip[intact])
    min_intact = np.min(slip[intact])
    mean_intact = np.mean(slip[intact])

    print(f"{path.name}: min={min_intact:.6g}, mean={mean_intact:.6g}, max={max_intact:.6g}")

    if slip.shape[0] != N or intact.shape[0] != N:
        raise ValueError(f"{path}: arrays must have length N={N}")

    return N, slip, intact


def fiber_radii_in_grid(N: int):
    """
    Same embedding as your pixel heatmap:
      grid = ceil(sqrt(N))
      index -> (row, col) row-major
    returns:
      grid, r (length N), (row,col) arrays (length N), center coordinate c0
    """
    grid = int(np.ceil(np.sqrt(N)))
    idx = np.arange(N)
    rr = idx // grid
    cc = idx % grid

    c0 = (grid - 1) / 2.0
    r = np.sqrt((rr - c0) ** 2 + (cc - c0) ** 2)

    return grid, r, rr, cc, c0


# def rolling_radial_mean(r_all, y_all, intact_mask, dr_window, n_eval=120,
#                         min_count=50, use_gauss=True, gauss_sigma=None):
#     """
#     Rolling/sliding radial window:
#       for many radii Rk, average y over intact fibers with |r - Rk| <= dr_window/2.

#     Optionally apply Gaussian weights inside the window.
#     """
#     r_i = r_all[intact_mask]
#     y_i = y_all[intact_mask]

#     if r_i.size == 0:
#         return np.array([]), np.array([]), np.array([])

#     rmin = float(np.min(r_i))
#     rmax = float(np.max(r_i))

#     # Evaluate over the physically relevant range with a bit of margin
#     half = 0.5 * float(dr_window)
#     R_eval = np.linspace(rmin + half, rmax - half, int(n_eval))

#     means = np.full_like(R_eval, np.nan, dtype=float)
#     counts = np.zeros_like(R_eval, dtype=int)

#     if use_gauss:
#         sigma = (dr_window / 4.0) if gauss_sigma is None else float(gauss_sigma)

#     for k, Rk in enumerate(R_eval):
#         d = r_i - Rk
#         m = np.abs(d) <= half
#         cnt = int(np.sum(m))
#         counts[k] = cnt

#         if cnt < min_count:
#             continue

#         if use_gauss:
#             w = np.exp(-0.5 * (d[m] / sigma) ** 2)
#             means[k] = float(np.sum(w * y_i[m]) / np.sum(w))
#         else:
#             means[k] = float(np.mean(y_i[m]))

#     ok = np.isfinite(means)
#     return R_eval[ok], means[ok], counts[ok]


def rolling_square_mean(rr, cc, y_all, intact_mask,
                        dr_window, n_eval=120,
                        min_count=50,
                        use_gauss=True,
                        gauss_sigma=None):
    """
    Rolling mean using square shells expanding from center.

    Distance definition:
        d = max(|x-c|, |y-c|)

    which produces square contours aligned with the grid.
    """

    grid = int(np.ceil(np.sqrt(len(y_all))))
    c0 = (grid - 1) / 2.0

    x = cc[intact_mask]
    y = rr[intact_mask]
    vals = y_all[intact_mask]

    # square distance
    d = np.maximum(np.abs(x - c0), np.abs(y - c0))

    dmin = float(np.min(d))
    dmax = float(np.max(d))

    half = dr_window * 0.5
    S_eval = np.linspace(dmin + half, dmax - half, int(n_eval))

    means = np.full_like(S_eval, np.nan, dtype=float)
    counts = np.zeros_like(S_eval, dtype=int)

    if use_gauss:
        sigma = (dr_window / 4.0) if gauss_sigma is None else float(gauss_sigma)

    for k, Sk in enumerate(S_eval):

        dist = d - Sk
        m = np.abs(dist) <= half

        cnt = int(np.sum(m))
        counts[k] = cnt

        if cnt < min_count:
            continue

        if use_gauss:
            w = np.exp(-0.5 * (dist[m] / sigma) ** 2)
            means[k] = np.sum(w * vals[m]) / np.sum(w)
        else:
            means[k] = np.mean(vals[m])

    ok = np.isfinite(means)
    return S_eval[ok], means[ok], counts[ok]


def build_slip_image(N, slip, intact):
    """
    Build 2D slip image (NaN for padding; NaN also for broken in 'slip_intact'),
    plus broken mask in full grid.
    """
    grid, r, rr, cc, c0 = fiber_radii_in_grid(N)
    total = grid * grid

    slip_flat = np.full(total, np.nan, dtype=float)
    slip_flat[:N] = slip
    slip_2d = slip_flat.reshape((grid, grid))

    intact_flat = np.zeros(total, dtype=bool)
    intact_flat[:N] = intact
    intact_2d = intact_flat.reshape((grid, grid))

    broken_2d = ~intact_2d
    slip_intact = np.where(intact_2d, slip_2d, np.nan)

    # Rmax to corner (diagonal)
    Rmax = c0 * np.sqrt(2.0)

    return grid, c0, Rmax, slip_intact, broken_2d


def draw_inset_with_annotations(ax_parent, N, slip, intact, dr_window, r_fraction=0.55):
    """
    Inset: grayscale heatmap of Case A + annotations for square-shell averaging:
      - line center -> right side of square shell
      - square at S
      - dashed squares at S ± ΔS/2
      - diagonal line to corner for R_max
    """
    axins = ax_parent.inset_axes(INSET_BBOX)

    grid, c0, Rmax, slip_intact, broken_2d = build_slip_image(N, slip, intact)

    cmap = mpl.cm.get_cmap(INSET_CMAP).copy()
    cmap.set_bad((1, 1, 1, 1))

    axins.imshow(
        slip_intact,
        cmap=cmap,
        origin="upper",
        interpolation="nearest",
        aspect="equal",
    )

    if INSET_SHOW_BROKEN_AS_BLACK:
        broken_overlay = np.where(broken_2d, 1.0, np.nan)
        dead_cmap = mpl.colors.ListedColormap(["black"])
        dead_cmap.set_bad((1, 1, 1, 0))
        axins.imshow(
            broken_overlay,
            cmap=dead_cmap,
            vmin=0, vmax=1,
            origin="upper",
            interpolation="nearest",
            aspect="equal",
        )

    axins.set_xticks([])
    axins.set_yticks([])

    # square "radius" S in Chebyshev metric
    Smax = c0
    S = float(np.clip(r_fraction, 0.05, 0.95)) * float(Smax)
    half = 0.5 * float(dr_window)

    cx, cy = c0, c0

    # line from center to square boundary along +x
    xS, yS = cx + S, cy
    axins.plot([cx, xS], [cy, yS], linewidth=2.0, color="white")

    # main square at S
    axins.add_patch(Rectangle(
        (cx - S, cy - S), 2*S, 2*S,
        fill=False, linewidth=2.0, edgecolor="white"
    ))

    # dashed inner/outer square shell boundaries
    s_in = max(0.0, S - half)
    s_out = min(Smax, S + half)

    axins.add_patch(Rectangle(
        (cx - s_in, cy - s_in), 2*s_in, 2*s_in,
        fill=False, linewidth=1.5, linestyle="--", edgecolor="white"
    ))
    axins.add_patch(Rectangle(
        (cx - s_out, cy - s_out), 2*s_out, 2*s_out,
        fill=False, linewidth=1.5, linestyle="--", edgecolor="white"
    ))

    # label S
    axins.text(
        cx + 0.55 * S, cy - 0.06 * grid,
        r"$S$",
        ha="center", va="bottom",
        fontsize=18,
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="black", alpha=0.8)
    )

    # diagonal to corner for R_max
    # Vertical line showing Smax (center -> top edge)
    ySmax = grid - 1
    axins.plot([cx, cx], [cy, ySmax], linewidth=2.0, color="white")

    axins.text(
        cx + 0.08 * grid,
        cy + 0.75 * (ySmax - cy),
        r"$S_{\max}$",
        ha="left",
        va="center",
        fontsize=18,
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="black", alpha=0.8)
    )

    for spine in axins.spines.values():
        spine.set_linewidth(1.2)

    return axins


def main():
    apply_pub_style()

    if not FILE_A.exists():
        raise FileNotFoundError(FILE_A.resolve())
    if not FILE_B.exists():
        raise FileNotFoundError(FILE_B.resolve())

    # Load snapshots
    N_a, slip_a, intact_a = load_snapshot(FILE_A)
    N_b, slip_b, intact_b = load_snapshot(FILE_B)

    # Radii arrays
    grid_a, _, rr_a, cc_a, c0_a = fiber_radii_in_grid(N_a)
    grid_b, _, rr_b, cc_b, c0_b = fiber_radii_in_grid(N_b)

    # Rmax for normalization
    Rmax_a = c0_a
    Rmax_b = c0_b
    Rmax_common = min(Rmax_a, Rmax_b) if (Rmax_a > 0 and Rmax_b > 0) else 1.0

    # Rolling mean curves
    Sa, ya, ca = rolling_square_mean(
        rr_a, cc_a,
        slip_a, intact_a,
        dr_window=DR_WINDOW,
        n_eval=N_EVAL,
        min_count=MIN_COUNT,
        use_gauss=USE_GAUSSIAN_WEIGHTS,
        gauss_sigma=GAUSS_SIGMA
    )

    Sb, yb, cb = rolling_square_mean(
        rr_b, cc_b,
        slip_b, intact_b,
        dr_window=DR_WINDOW,
        n_eval=N_EVAL,
        min_count=MIN_COUNT,
        use_gauss=USE_GAUSSIAN_WEIGHTS,
        gauss_sigma=GAUSS_SIGMA
    )

    if NORMALIZE_R:
        x_a = Sa / Rmax_common
        x_b = Sb / Rmax_common
        xlabel = r"$S/S_{\max}$"
        # Inset uses absolute units internally, but we annotate via fraction anyway
    else:
        x_a = Sa
        x_b = Sb
        xlabel = r"$S$ (grid units)"

    # ---- Figure ----
    fig = plt.figure(figsize=FIGSIZE, dpi=150)
    ax = fig.add_axes([0.15, 0.16, 0.80, 0.80])  # compact manual layout

    # Labels requested
    ax.plot(x_a, ya, label=r"$\langle \varepsilon_i^S \rangle$")
    ax.plot(x_b, yb, label=r"$\langle \overline{{\varepsilon_i^S}} \rangle$")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"[-]")

    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.7, zorder=0)
    ax.set_xlim(left=0)
    # Inset: grayscale heatmap + R/Rmax + window circles (Case A)
    _ = draw_inset_with_annotations(
        ax_parent=ax,
        N=N_a, slip=slip_a, intact=intact_a,
        dr_window=DR_WINDOW,
        r_fraction=R_ANNOT_FRACTION
    )

    fig.savefig(OUTPATH, dpi=300)
    plt.show()
    plt.close(fig)

    print(f"Saved: {OUTPATH.resolve()}")
    print(f"DR_WINDOW={DR_WINDOW}, MIN_COUNT={MIN_COUNT}, N_EVAL={N_EVAL}, GAUSS={USE_GAUSSIAN_WEIGHTS}")


if __name__ == "__main__":
    main()
