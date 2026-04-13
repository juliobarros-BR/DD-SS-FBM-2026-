#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  8 09:22:06 2025

@author: jortiz
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
from matplotlib.colors import TwoSlopeNorm
from pathlib import Path
from matplotlib.cm import ScalarMappable
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.lines import Line2D
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.cm import ScalarMappable
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.cm import ScalarMappable

# =========================================================
# STYLE (unchanged)
# =========================================================
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
})

# =========================================================
# CONFIG
# =========================================================
CSV_FILE = Path("fourier_meta/results_systems.csv")
FIT_DIR  = Path("fourier_globalfit_from_results_Fo99")
OUT_DIR  = Path("Plots_Fo_eff")
OUT_DIR.mkdir(exist_ok=True, parents=True)

# ----------------------------
# "More freedom" switches
# ----------------------------
METRIC_COL = "Mech_max_unload"
SECOND_METRIC_COL = "SlipCount_unload"

FIT_LABEL_PREFIX = "non_avg"   # e.g. "avg" or "non_avg"

TYPE_MODE = "path"  # "path" or "column"
TYPE_COLUMN = "is_avg"  # only used if TYPE_MODE="column"

LOADD_FILTER = [0.7]
LOAD_TOL = 1e-9

USE_LOGISTIC = True
ENFORCE_SECOND_EQ_ONE = True

# Ignore avg/full pairs where this column is zero (in either avg or full)
ZERO_FILTER_COL = "Mech_unload"   # <-- set to the exact column name in results_systems.csv
# If you actually mean your METRIC_COL (e.g. "Mech_max_unload"), just set this to METRIC_COL instead.


# Optional hard limits
X_LIM = None
Y_LIM = None

# ----------------------------
# Outlier-resistant color scaling (so 1 freak point doesn't nuke the colormap)
# ----------------------------
USE_ROBUST_COLOR_NORM = True
ROBUST_Q_LO = 0.02     # 2% quantile
ROBUST_Q_HI = 0.98     # 98% quantile

# ----------------------------
# Debugging outputs
# ----------------------------
PRINT_WORST_N = 15
SAVE_WORST_CSV = True   # saves worst cases per LoadD into OUT_DIR

# =========================================================
# SMALL HELPERS
# =========================================================
def _to_float(x):
    """0.7 / '0.7' / '0p7' -> float, else None"""
    try:
        if isinstance(x, str):
            return float(x.replace("p", "."))
        return float(x)
    except Exception:
        return None

def _float_in(x: float | None, allowed, tol=1e-9) -> bool:
    if x is None:
        return False
    for a in allowed:
        aa = _to_float(a)
        if aa is None:
            continue
        if abs(x - aa) <= tol:
            return True
    return False

def type_from_row(row) -> str:
    if TYPE_MODE == "column":
        return "avg" if bool(row[TYPE_COLUMN]) else "full"
    return "avg" if "_avg" in str(row["CSV_path"]) else "full"

def load_fit_params_for_loadd(load_d: float):
    """Read (a,b,c) from fit_parameters_{FIT_LABEL_PREFIX}_LoadD_{tag}.csv"""
    load_tag = f"{load_d:.3f}".replace(".", "p")
    fit_csv = FIT_DIR / f"fit_parameters_{FIT_LABEL_PREFIX}_LoadD_{load_tag}.csv"
    if not fit_csv.exists():
        print(f"⚠️ Missing fit params: {fit_csv}")
        return None

    df_fit = pd.read_csv(fit_csv)
    try:
        a = float(df_fit.loc[df_fit["parameter"] == "a", "value"].iloc[0])
        b = float(df_fit.loc[df_fit["parameter"] == "b", "value"].iloc[0])
        c = float(df_fit.loc[df_fit["parameter"] == "c", "value"].iloc[0])
        return a, b, c
    except Exception as e:
        print(f"⚠️ Could not parse a,b,c from {fit_csv}: {e}")
        return None

def robust_minmax(arr, qlo=0.02, qhi=0.98):
    """Quantile-based min/max for colormap scaling."""
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan, np.nan
    lo = np.quantile(arr, qlo)
    hi = np.quantile(arr, qhi)
    if not np.isfinite(lo) or not np.isfinite(hi) or abs(hi - lo) < 1e-15:
        lo = np.nanmin(arr)
        hi = np.nanmax(arr)
    return lo, hi

# =========================================================
# FIT FUNCTIONS
# =========================================================
def logistic_4p(x, L, U, k, x0):
    return L + (U - L) / (1.0 + np.exp(-k * (np.log(x) - x0)))

def logistic_fixedU(U):
    def f(x, L, k, x0):
        return L + (U - L) / (1.0 + np.exp(-k * (np.log(x) - x0)))
    return f

def fit_logistic_mech(x, y):
    p0 = [np.nanmin(y), np.nanmax(y), -5.0, 0.0]
    popt, _ = curve_fit(logistic_4p, x, y, p0=p0, maxfev=20000)
    return popt

def fit_logistic_fixed_upper(x, y, U=1.0):
    f = logistic_fixedU(U)
    p0 = [np.nanmin(y), 5.0, 0.0]
    bounds = ([0.0, 0.0, -10.0], [U, 50.0, 10.0])
    popt, _ = curve_fit(f, x, y, p0=p0, bounds=bounds, maxfev=20000)
    L, k, x0 = popt
    return (L, U, k, x0)

def fit_linear(x, y):
    m = LinearRegression()
    m.fit(np.log(x).reshape(-1,1), y)
    return m

# =========================================================
# MAIN
# =========================================================
df = pd.read_csv(CSV_FILE)

if "LoadD" not in df.columns:
    raise KeyError("results_systems.csv must contain a numeric 'LoadD' column.")

df["LoadD"] = df["LoadD"].astype(float).round(3)

if LOADD_FILTER:
    df = df[df["LoadD"].apply(lambda x: _float_in(_to_float(x), LOADD_FILTER, tol=LOAD_TOL))].copy()
    print(f"▶ LoadD filter active: {LOADD_FILTER}  (remaining rows: {len(df)})")
    if df.empty:
        raise ValueError(f"No rows left after LOADD_FILTER={LOADD_FILTER}")

needed_cols = [METRIC_COL, "Fo_diff", "Fo_visc", "Ramp", "Diff", "Tau", "CSV_path"]
if SECOND_METRIC_COL is not None:
    needed_cols.append(SECOND_METRIC_COL)
for c in needed_cols:
    if c not in df.columns:
        raise KeyError(f"Missing column in results_systems.csv: '{c}'")

load_levels = sorted(df["LoadD"].dropna().unique())
if len(load_levels) == 0:
    raise ValueError("No valid LoadD values found.")

for load_d in load_levels:
    print(f"\n=== LoadD = {load_d:.3f} ===")

    abc = load_fit_params_for_loadd(load_d)
    if abc is None:
        print("→ Skipping this load (no fit parameters file).")
        continue
    a, b, c = abc

    # --- subset per load ---
    dfL = df[df["LoadD"] == load_d].copy()
    dfL["Type"] = dfL.apply(type_from_row, axis=1)
    # dfL["Fo_diff"] = dfL["Fo_diff"]/2
    # dfL["Fo_visc"] = dfL["Fo_visc"]/2

    # --- compute Fo_eff (per-load a,b,c) ---
    dfL["Fo_eff"] = (
        (dfL["Fo_diff"].astype(float) ** a) *
        (dfL["Fo_visc"].astype(float) ** b) *
        np.exp(c * dfL["Ramp"].astype(float))
    )

    df_full = dfL[dfL["Type"] == "full"].copy()
    df_avg  = dfL[dfL["Type"] == "avg"].copy()

    on_cols = ["Diff", "Ramp", "Tau"]
    merged = pd.merge(df_full, df_avg, on=on_cols, suffixes=("_full", "_avg"))
        # ---------------------------------------------------------
    # FILTER: drop pairs where Mech_unload is 0 (avg or full)
    # ---------------------------------------------------------
    if ZERO_FILTER_COL is not None:
        col_full = f"{ZERO_FILTER_COL}_full"
        col_avg  = f"{ZERO_FILTER_COL}_avg"

        if col_full not in merged.columns or col_avg not in merged.columns:
            raise KeyError(
                f"ZERO_FILTER_COL='{ZERO_FILTER_COL}' but '{col_full}' or '{col_avg}' not in merged. "
                f"Check the exact column name in results_systems.csv."
            )

        before = len(merged)
        merged = merged[
            (merged[col_full].astype(float) != 0.0) &
            (merged[col_avg].astype(float)  != 0.0)
        ].copy()
        after = len(merged)
        if after < before:
            print(f"→ Dropped {before-after} pairs because {ZERO_FILTER_COL} was 0 (avg or full).")

        if merged.empty:
            print("→ All pairs removed by zero filter; skipping.")
            continue


    if merged.empty:
        print("→ No avg/full pairs after merge; skipping.")
        continue

    # Ratios
    y_main = merged[f"{METRIC_COL}_avg"] / merged[f"{METRIC_COL}_full"]
    y_second = None if SECOND_METRIC_COL is None else (merged[f"{SECOND_METRIC_COL}_avg"] / merged[f"{SECOND_METRIC_COL}_full"])

    # Clean mask
    mask = np.isfinite(merged["Fo_eff_full"]) & (merged["Fo_eff_full"] > 0) & np.isfinite(y_main)
    if y_second is not None:
        mask &= np.isfinite(y_second)

    data = merged.loc[mask].copy()
    if data.empty:
        print("→ No finite data after cleaning; skipping.")
        continue

    # Numpy arrays
    x = data["Fo_eff_full"].to_numpy(float)
    y_main = (data[f"{METRIC_COL}_avg"] / data[f"{METRIC_COL}_full"]).to_numpy(float)
    if y_second is not None:
        y_second = (data[f"{SECOND_METRIC_COL}_avg"] / data[f"{SECOND_METRIC_COL}_full"]).to_numpy(float)

    # =========================================================
    # DEBUG: identify outliers that wreck the colormap / limits
    # =========================================================
    data["Fo_eff"] = x
    data["y_main"] = y_main
    if y_second is not None:
        data["y_second"] = y_second

    # Which rows are the worst?
    id_cols = ["CSV_path_full", "CSV_path_avg", "Diff", "Ramp", "Tau", "LoadD_full"]
    raw_cols = [f"{METRIC_COL}_full", f"{METRIC_COL}_avg"]
    if y_second is not None:
        raw_cols += [f"{SECOND_METRIC_COL}_full", f"{SECOND_METRIC_COL}_avg"]

    show_cols = id_cols + ["Fo_eff", "y_main"] + (["y_second"] if y_second is not None else []) + raw_cols

    # Print + save worst (low ratios)
    print(f"    y_main:   min={np.nanmin(y_main):.6g}, max={np.nanmax(y_main):.6g}")
    if y_second is not None:
        print(f"    y_second: min={np.nanmin(y_second):.6g}, max={np.nanmax(y_second):.6g}")

    worst_main = data.sort_values("y_main", ascending=True).loc[:, show_cols].head(PRINT_WORST_N)
    print("\n=== LOWEST y_main (avg/full) rows ===")
    with pd.option_context("display.max_colwidth", None, "display.width", 220):
        print(worst_main.to_string(index=False))

    if y_second is not None:
        worst_second = data.sort_values("y_second", ascending=True).loc[:, show_cols].head(PRINT_WORST_N)
        print("\n=== LOWEST y_second (avg/full) rows ===")
        with pd.option_context("display.max_colwidth", None, "display.width", 220):
            print(worst_second.to_string(index=False))

        # Flag tiny denominators (often the true reason ratios explode)
        denom = data[f"{SECOND_METRIC_COL}_full"].astype(float).to_numpy()
        denom_abs = np.abs(denom[np.isfinite(denom)])
        if denom_abs.size > 0:
            tiny_thr = np.quantile(denom_abs, 0.01)
            suspicious = data[np.abs(data[f"{SECOND_METRIC_COL}_full"].astype(float)) <= tiny_thr] \
                           .sort_values("y_second", ascending=True) \
                           .loc[:, show_cols] \
                           .head(PRINT_WORST_N)
            print("\n=== Suspicious y_second (tiny denominator in FULL) ===")
            with pd.option_context("display.max_colwidth", None, "display.width", 220):
                print(suspicious.to_string(index=False))

    if SAVE_WORST_CSV:
        load_tag = f"{load_d:.3f}".replace(".", "p")
        worst_main.to_csv(OUT_DIR / f"worst_y_main_LoadD_{load_tag}.csv", index=False)
        if y_second is not None:
            worst_second.to_csv(OUT_DIR / f"worst_y_second_LoadD_{load_tag}.csv", index=False)

    # =========================================================
    # FITS
    # =========================================================
    x_smooth = np.logspace(np.log10(x.min()), np.log10(x.max()), 500)

    if USE_LOGISTIC:
        Lm, Um, km, x0m = fit_logistic_mech(x, y_main)
        y_main_fit = logistic_4p(x_smooth, Lm, Um, km, x0m)

        if y_second is not None:
            if ENFORCE_SECOND_EQ_ONE:
                Ls, Us, ks, x0s = fit_logistic_fixed_upper(x, y_second, U=1.0)
            else:
                Ls, Us, ks, x0s = fit_logistic_mech(x, y_second)
            y_second_fit = logistic_4p(x_smooth, Ls, Us, ks, x0s)
        else:
            y_second_fit = None
    else:
        lin_main = fit_linear(x, y_main)
        y_main_fit = lin_main.predict(np.log(x_smooth).reshape(-1, 1))

        if y_second is not None:
            lin_second = fit_linear(x, y_second)
            y_second_fit = lin_second.predict(np.log(x_smooth).reshape(-1, 1))
        else:
            y_second_fit = None

    # =========================================================
    # COLORS (center at 1) — robust against outliers if enabled
    # =========================================================
    cmap = plt.cm.coolwarm.copy()

    y_all = y_main if y_second is None else np.concatenate([y_main, y_second])

    if USE_ROBUST_COLOR_NORM:
        rmin, rmax = robust_minmax(y_all, qlo=ROBUST_Q_LO, qhi=ROBUST_Q_HI)
        # still ensure the true min/max are printed for debugging
        print(f"    color-norm (robust): vmin~{rmin:.6g}  vmax~{rmax:.6g}  (center=1)")
    else:
        rmin = np.nanmin(y_all)
        rmax = np.nanmax(y_all)
        print(f"    color-norm (full):   vmin={rmin:.6g}  vmax={rmax:.6g}  (center=1)")

    # avoid degenerate vmin=vmax
    if not np.isfinite(rmin) or not np.isfinite(rmax):
        rmin, rmax = 0.999, 1.001
    if abs(rmax - rmin) < 1e-12:
        rmin -= 1e-3
        rmax += 1e-3

    norm = TwoSlopeNorm(vmin=rmin, vcenter=1.0, vmax=rmax)
    norm.clip = True


    # =========================================================
    # PLOT
    # =========================================================
    fig, ax = plt.subplots(figsize=(8.0, 6.0))

    # -------------------------
    # SCATTER POINTS
    # -------------------------
    ax.scatter(
        x, y_main, s=100, marker="o",
        c=cmap(norm(y_main)), edgecolor="k", label="_nolegend_", zorder=10
    )

    if y_second is not None:
        ax.scatter(
            x, y_second, s=100, marker="s",
            c=cmap(norm(y_second)), edgecolor="k", label="_nolegend_", zorder=10
        )

    # -------------------------
    # FIT LINES
    # -------------------------
    ax.plot(x_smooth, y_main_fit, color="red", lw=3, label=f"{METRIC_COL} (fit)")
    if y_second_fit is not None:
        ax.plot(x_smooth, y_second_fit, color="blue", lw=3, label=f"{SECOND_METRIC_COL} (fit)")

    # -------------------------
    # REFERENCE + AXES
    # -------------------------
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=2)

    ax.set_xscale("log")
    ax.set_xlabel(r"$Fo_{\mathrm{eff}}$")
    ax.set_ylabel(r"Prop. Ratio $\langle \varphi_i \rangle / \varphi_i [-]$")

    ax.grid(alpha=0.3)
    ax.tick_params(axis="x", which="major", pad=6)

    # Optional: set your preferred limits if you want the exact old look
    # ax.set_xlim(0.0035, 4.5)
    # ax.set_ylim(0.89, np.nanmax(y_main) + 0.0034)

    # -------------------------
    # MARKER LEGEND (your latex labels)
    # -------------------------
    legend_elements = [
        Line2D([0], [0], marker='o', markersize=12, color='w',
            markeredgecolor='k', label=r"$\varepsilon_\infty^{MS}/\varepsilon_\infty$"),
    ]
    if y_second is not None:
        legend_elements.append(
            Line2D([0], [0], marker='s', markersize=12, color='w',
                markeredgecolor='k', label=r"$N_\infty$")
        )

    ax.legend(
        handles=legend_elements,
        loc="center",
        bbox_to_anchor=(0.8, 0.9),
        handletextpad=0.1,
        framealpha=1.0
    )

    # -------------------------
    # COLORBAR ON THE LEFT (proper axis; doesn't break limits)
    # -------------------------
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    divider = make_axes_locatable(ax)
    # pad is in inches-ish relative units; increase if you still see collisions
    cax = inset_axes(
        ax,
        width="2.8%",      # bar width
        height="100%",      # bar height
        loc="lower left",
        bbox_to_anchor=(0.0, 0.0, 1, 1),  # <--- THIS IS THE PLACING LINE
        bbox_transform=ax.transAxes,
        borderpad=0
    )

    cb = fig.colorbar(sm, cax=cax, orientation="vertical", extend="neither")


    # Kill ticks and labels
    cb.set_ticks([])
    cb.ax.set_yticklabels([])
    cb.ax.tick_params(left=False, right=False)

    # Outline (same look as before)
    cb.outline.set_edgecolor("black")
    cb.outline.set_linewidth(1.2)
    # -------------------------
    # SAVE
    # -------------------------
    out_path = OUT_DIR / f"Combined_Ratio_{METRIC_COL}_LoadD_{load_tag}_abc_{FIT_LABEL_PREFIX}.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

    print("Saved:", out_path)

print("\n✅ Done!")
