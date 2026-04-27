

import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from matplotlib.patches import Rectangle, ConnectionPatch
from compute_mech2 import analyze_creep, ViscoelasticConfig, PlasticConfig, make_plastic_config_from_dir, choose_plastic_sweep_dir
from matplotlib.widgets import Slider

# ===============================
# STYLE (your publication params)
# ===============================
plt.close("all")
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
    # 'savefig.bbox': 'tight',
    # 'savefig.pad_inches': 0.05,
    'figure.dpi': 300,
    'savefig.bbox': None,  # do not auto-crop
})


# ===============================
# USER CONFIG
# ===============================
INPUT_CSV = Path("./all_new_paper_01_04_26.csv")  # <- file you want for paper
SUMMARY_CSV = Path("./moisture_sweep_results22/LoadD_0p7/moisture_sweep_summary.csv")

# ---- MANUAL CONDITION LABELS (for MS fits lookup only) ----
DIFF_LABEL = "Diff_1000"
RAMP_LABEL = "Ramp_50%"
TAU_LABEL  = "Tau_0.001"

SUMMARY_MS_FITS = Path("./Analysis_folders/mechsorption_analysis/summary_ms_fits.csv")

TAU_VALUE = 0.001
N_CYCLES_TO_PLOT = 10

SUBTRACT_ELASTIC0 = False
USE_MS_FITS = True   # set True only if summary_ms_fits.csv exists

# If you want to hard-force crit for plotting (overrides analyze_creep output):
FORCE_CRIT = 4.154739120251522  # e.g. 4.154739120251522


# ===============================
# HELPERS
# ===============================
def sat_exp(n, A, n0, C):
    return A * (1.0 - np.exp(-n / max(n0, 1e-12))) + C


def load_ms_fit_params(summary_ms_fits: Path, diff_label: str, ramp_label: str, tau_label: str):
    """
    Read summary_ms_fits.csv and return A,n0,C for matching diff/ramp/tau.
    tau in file is numeric, tau_label looks like 'Tau_0.001'.
    """
    tau_value = float(tau_label.replace("Tau_", ""))

    df = pd.read_csv(summary_ms_fits)
    m = (
        (df["diff"] == diff_label) &
        (df["ramp"] == ramp_label) &
        (np.isclose(df["tau"].astype(float), tau_value, rtol=1e-6, atol=1e-12))
    )
    if not np.any(m):
        print(f"⚠️ No MS fit params found for {diff_label}/{ramp_label}/{tau_label} in {summary_ms_fits}")
        return None

    row = df.loc[m].iloc[0]
    return float(row["A"]), float(row["n0"]), float(row["C"])


def shade(c, f):
    rgb = np.array(mcolors.to_rgb(c))
    if f >= 1:
        return tuple(1 - (1 - rgb) / f)
    else:
        return tuple(rgb * f)


def infer_load_signal_from_csv(input_csv: Path, t_plot: np.ndarray, tau4: float):
    """
    Try to infer a load-like signal from INPUT_CSV. If found and lengths mismatch,
    interpolate onto t_plot if time column exists; otherwise index-resample.
    Returns (load_sig, load_label) or (None, None).
    """
    try:
        df_in = pd.read_csv(input_csv)
    except Exception as e:
        print(f"⚠️ Could not read INPUT_CSV for load extraction: {e}")
        return None, None

    candidates = [
        "Load", "load", "Force", "force", "F", "stress", "Stress",
        "Sigma", "sigma", "Load_N", "Force_N"
    ]
    col = next((c for c in candidates if c in df_in.columns), None)
    if col is None:
        return None, None

    load_sig = np.asarray(df_in[col], float)

    if len(load_sig) == len(t_plot):
        return load_sig, str(col)

    # Try interpolation using a time column (assumed same units as `time`)
    tcol = next((c for c in ["time", "Time", "t", "T"] if c in df_in.columns), None)
    if tcol is not None:
        t_in = np.asarray(df_in[tcol], float)
        t_in_plot = t_in / tau4
        # guard against non-monotonic time
        order = np.argsort(t_in_plot)
        t_in_plot = t_in_plot[order]
        load_sig_sorted = load_sig[order]
        # unique time points
        uniq = np.concatenate([[True], np.diff(t_in_plot) > 0])
        t_in_plot = t_in_plot[uniq]
        load_sig_sorted = load_sig_sorted[uniq]
        if len(t_in_plot) >= 2:
            load_sig = np.interp(t_plot, t_in_plot, load_sig_sorted)
            return load_sig, str(col)

    # Last resort: index-resample
    x_old = np.linspace(0, 1, len(load_sig))
    x_new = np.linspace(0, 1, len(t_plot))
    load_sig = np.interp(x_new, x_old, load_sig)
    return load_sig, str(col)


def plot_last_cycle_template(results, ax=None):

    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    # ----------------------------
    # arrays
    # ----------------------------
    time      = np.asarray(results["time"], float)
    moisture  = np.asarray(results["moisture"], float)
    reduced   = np.asarray(results["reduced"], float)
    templ_all = np.asarray(results["templ_global"], float)

    pre_cycles  = results["pre_unload_cycles"]
    post_cycles = results["post_unload_cycles"]

    alpha_L = float(results["a_loaded"])   / FORCE_CRIT
    alpha_UL = float(results["a_unloaded"]) / FORCE_CRIT

    # ----------------------------
    # colors
    # ----------------------------
    COL_REDUCED = "#2CA02C"
    COL_TEMPLATE = "#9467BD"
    
    COL_MOIST   = "0.65"
    COL_RED = "red"

    # ----------------------------
    # axis
    # ----------------------------
    if ax is None:
        fig, ax = plt.subplots(figsize=(6,6))
    else:
        fig = ax.figure

    ax.set_axis_off()

    # ----------------------------
    # layout parameters (0–1 inside inset)
    # reduce vertical gap so top and bottom plots are closer
    # ----------------------------
    left = 0.0
    bottom = 0.0

    h_gap = 0.15
    v_gap = 0.04  # reduced from 0.12

    width  = (1 - h_gap) / 2
    height = (1 - v_gap) / 2

    # ----------------------------
    # create axes INSIDE inset
    # ----------------------------
    ax_L_top = ax.inset_axes([0, 1 - height, width, height])
    ax_R_top = ax.inset_axes([width + h_gap, 1 - height, width, height])

    ax_L_bot = ax.inset_axes([0, 0, width, height])
    ax_R_bot = ax.inset_axes([width + h_gap, 0, width, height])
    # =========================================================
    # TOP LEFT — LOADED REDUCED
    # =========================================================
    i1, i2 = pre_cycles[-1]
    s = reduced[i1:i2+1] / FORCE_CRIT
    x_top = np.linspace(0,1,len(s))

    ax_L_top.plot(x_top, s, color=COL_REDUCED, lw=3)

    ax_L_top.set_xlim(0,1)
    ax_L_top.set_xticks([0,1])
    # remove tick labels for top plots
    ax_L_top.set_xticklabels([])
    # ax_L_top.set_yticks([])

    ax_L_top.set_ylabel(r"$\varepsilon/\varepsilon_c$")
    ax_L_top.set_title("Loaded", fontsize=22)

    # =========================================================
    # TOP RIGHT — UNLOADED REDUCED
    # =========================================================
    i1, i2 = post_cycles[-1]
    s = reduced[i1:i2+1] / FORCE_CRIT
    x_top = np.linspace(0,1,len(s))

    ax_R_top.plot(x_top, s, color=COL_REDUCED, lw=3)

    ax_R_top.set_xlim(0,1)
    ax_R_top.set_xticks([0,1])
    ax_R_top.set_xticklabels([])
    ax_R_top.set_yticks([0.08,0.12])

    ax_R_top.set_title("Unloaded", fontsize=22)

    # =========================================================
    # BOTTOM LEFT — LOADING TEMPLATE
    # =========================================================
    i1, i2 = pre_cycles[0]

    t = time[i1:i2+1]
    phi = moisture[i1:i2+1]
    theta = templ_all[i1:i2+1] / FORCE_CRIT

    phi0 = phi[0]
    theta0 = theta[0]

    def map_L(p):
        return theta0 + alpha_L*(p - phi0)

    def inv_L(e):
        return phi0 + (e - theta0)/alpha_L

    # use normalized cycle x axis from 0 to 1 for bottom plots
    x_bot = np.linspace(0, 1, len(theta))

    ax_L_bot.plot(x_bot, theta, color=COL_TEMPLATE, lw=3)
    ax_L_bot.plot(x_bot, map_L(phi), color=COL_MOIST, lw=2)

    # slope line
    ax_L_bot.plot([0, 1], [theta[0], theta[-1]], color=COL_RED, lw=2)

    # place cycle label and ticks
    ax_L_bot.set_xlim(0,1)
    ax_L_bot.set_xticks([0,1])
    ax_L_bot.set_xticklabels(["0","1"])
    ax_L_bot.set_xlabel("Cycle", fontsize=22)

    # place explanatory text with coordinates appropriate for normalized axis
    tx = 0.12
    ty = theta[0] + 0.7*(theta[-1]-theta[0])
    ax_L_bot.text(
        0.2,
        0.34,
        r"$\Delta \varepsilon = \alpha_E \Delta \varphi$",
        color="red",
        fontsize=20
    )

    ax_L_bot.set_ylabel(r"$\varepsilon/\varepsilon_c$")

    # =========================================================
    # BOTTOM RIGHT — UNLOADING TEMPLATE
    # =========================================================
    i1, i2 = post_cycles[-1]

    t = time[i1:i2+1]
    phi = moisture[i1:i2+1]
    theta = templ_all[i1:i2+1] / FORCE_CRIT

    phi0 = phi[0]
    theta0 = theta[0]

    def map_UL(p):
        return theta0 + alpha_UL*(p - phi0)

    def inv_UL(e):
        return phi0 + (e - theta0)/alpha_UL

    x_bot = np.linspace(0, 1, len(theta))

    ax_R_bot.plot(x_bot, theta, color=COL_TEMPLATE, lw=3)
    ax_R_bot.plot(x_bot, map_UL(phi), color=COL_MOIST, lw=2)

    # horizontal line
    ax_R_bot.plot([0, 1], [theta0, theta0], color=COL_RED, lw=2)

    # vertical measurement at mid-cycle
    x_mid = x_bot[len(x_bot)//2]
    ax_R_bot.plot([x_mid, x_mid], [0, theta0], color=COL_RED, lw=2)

    ax_R_bot.text(
        0.2,
        0.03,
        r"$\Delta \varepsilon = \alpha_E \varphi_f$",
        color="red",
        fontsize=20
    )

    ax_R_bot.set_xlim(0,1)
    ax_R_bot.set_xticks([0,1])
    ax_R_bot.set_xticklabels(["0","1"])
    ax_R_bot.set_xlabel("Cycle", fontsize=22)

    ax_R_bot2 = ax_R_bot.twinx()
    y1,y2 = ax_R_bot.get_ylim()
    ax_R_bot2.set_ylim(inv_UL(y1), inv_UL(y2))
    ax_R_bot2.set_ylabel(r"$\varphi/0.2$", color=COL_MOIST)
    ax_R_bot2.tick_params(axis="y", colors=COL_MOIST)
    ax_R_bot2.spines["right"].set_visible(True)
    ax_R_bot2.spines["right"].set_color(COL_MOIST)
    ax_R_bot2.set_zorder(ax_R_bot.get_zorder() + 1)
    ax_R_bot2.patch.set_visible(False)

    # =========================================================
    # CLEANUP
    # =========================================================
    for a in [ax_L_top, ax_R_top, ax_L_bot, ax_R_bot]:
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)
        a.tick_params(direction="in")

    # add panel numbers at the top-right of each inset
    _panel_labels = ["1", "2", "3", "4"]
    _panel_axes = [ax_L_top, ax_R_top, ax_L_bot, ax_R_bot]
    for _ax, _lbl in zip(_panel_axes, _panel_labels):
        _ax.text(
            0.95, 0.98, _lbl,
            transform=_ax.transAxes,
            ha="right", va="top",
            fontsize=22, fontweight="bold",
            bbox=dict(
                facecolor="white",
                edgecolor="black",
                boxstyle="square,pad=0.25",
                linewidth=1.0,
                alpha=0.95
            ),
            zorder=20
        )

    return fig, ax
# ===============================
# MAIN
if __name__ == "__main__":

    # --- run decomposition using your canonical implementation ---
    # ---- viscoelastic config ----
    ve_cfg = ViscoelasticConfig(
        summary_csv=str(SUMMARY_CSV),
        tau=TAU_VALUE,
        degree=2,
        kv_num=4,
    )

    # ---- plastic config ----
    plastic_dir = choose_plastic_sweep_dir(
        input_csv=INPUT_CSV,
        sweeps_root="./plastic_analysis/",
        default_dirname="unified_plastic_sweeps_LoadDmax_0p700",
    )

    plastic_cfg = make_plastic_config_from_dir(
        plastic_dir,
        enabled=True,
        strict=True,
    )

    # ---- run ----
    results = analyze_creep(
        input_csv=str(INPUT_CSV),
        ve_cfg=ve_cfg,
        plastic_cfg=plastic_cfg,
        manual_valleys=[],
        make_plots=False,
        exclude_plastic=False,
    )


    # ----------------------------
    # Required arrays
    # ----------------------------
    time     = np.asarray(results["time"], float)
    moisture = np.asarray(results["moisture"], float)
    mech     = np.asarray(results["mech"], float)
    slip     = np.asarray(results["slip"], float)

    total_after_plastic = np.asarray(results.get("total", np.nan), float)
    total_raw           = np.asarray(results.get("total_raw", total_after_plastic), float)
    reduced             = np.asarray(results.get("reduced", 0.0), float)
    creep               = np.asarray(results.get("creep", 0.0), float)
    creep_sim           = np.asarray(results.get("creep_sim", 0.0), float)

    templ_global = np.asarray(results.get("templ_global", np.nan), float)
    elastic      = np.asarray(results.get("elastic", np.zeros_like(time)), float)

    # optional model-term: hygroexpansion (if analyze_creep provides it)
    hygro = results.get("hygro", None)
    if hygro is None:
        hygro = np.zeros_like(time, dtype=float)
    else:
        hygro = np.asarray(hygro, float)

    crit = float(results.get("critical_strain", 1.0))
    if FORCE_CRIT is not None:
        crit = float(FORCE_CRIT)

    plastic_scler = results.get("plastic", None)
    cycle_bounds  = results.get("cycle_bounds", None)
    t_unload      = results.get("t_unload", None)

    if plastic_scler is None:
        plastic_scler = np.zeros_like(time, dtype=float)
        print("⚠️ results['plastic'] not found. Plotting plastic as zeros.")
    else:
        plastic_scler = np.asarray(plastic_scler, float)

    if cycle_bounds is None or len(cycle_bounds) == 0:
        raise RuntimeError("results['cycle_bounds'] is missing or empty.")

    if t_unload is None:
        raise RuntimeError("results['t_unload'] is missing; needed to separate loaded/unloaded cycles.")
    t_unload = float(t_unload)

    # ----------------------------
    # Split cycles into loaded vs unloaded (for last-cycle overlays)
    # ----------------------------
    pre_unload_cycles  = [(int(i1), int(i2)) for (i1, i2) in cycle_bounds if float(time[int(i2)]) <= t_unload]
    post_unload_cycles = [(int(i1), int(i2)) for (i1, i2) in cycle_bounds if float(time[int(i1)]) >= t_unload]

    if not pre_unload_cycles:
        raise ValueError("No cycles before unload. Check t_unload and cycle_bounds.")

    last_loaded_i1, last_loaded_i2 = pre_unload_cycles[-1]
    if post_unload_cycles:
        last_unloaded_i1, last_unloaded_i2 = post_unload_cycles[-1]
    else:
        last_unloaded_i1, last_unloaded_i2 = last_loaded_i1, last_loaded_i2

    i_unload = int(np.argmin(np.abs(time - t_unload)))

    # ----------------------------
    # Normalized time axis
    # ----------------------------
    KV_num = 4
    tau4 = TAU_VALUE * (10.0 ** (KV_num - 1))   # tau * 1000 for KV=4
    t_plot = time / tau4
    t_unload_plot = t_unload / tau4

    # ----------------------------
    # Elastic offset subtraction based on FIRST cycle start
    # ----------------------------
    i0_cycle = int(cycle_bounds[0][0])
    elastic0_norm = float(elastic[i0_cycle]) / max(crit, 1e-30)

    def shift(arr):
        if not SUBTRACT_ELASTIC0:
            return arr
        return np.asarray(arr, float) - elastic0_norm

    total_raw_p   = shift(total_raw)
    total_after_p = shift(total_after_plastic)
    templ_p       = shift(templ_global)
    reduced_p     = shift(reduced)
    mech_p        = shift(mech)
    hygro_p       = shift(hygro)
    creep_p       = shift(creep)

    plastic_p = plastic_scler
    slip_p    = np.asarray(slip, float)

    # ----------------------------
    # Load MS fit params (unrecoverable part)
    # ----------------------------
    params = None
    if USE_MS_FITS:
        if SUMMARY_MS_FITS.exists():
            params = load_ms_fit_params(SUMMARY_MS_FITS, DIFF_LABEL, RAMP_LABEL, TAU_LABEL)
        else:
            print(f"⚠️ MS fits file not found: {SUMMARY_MS_FITS}")
            print("   Continuing without unrecoverable MS overlay.")

    # ----------------------------
    # Cycle ends from cycle_bounds (NO moisture valley detection)
    # ----------------------------
    n_plot = min(N_CYCLES_TO_PLOT, len(cycle_bounds))
    cycle_end_indices = np.array([cycle_bounds[k][1] for k in range(n_plot)], dtype=int)
    cycle_end_indices = cycle_end_indices[(cycle_end_indices >= 0) & (cycle_end_indices < len(time))]
    n_plot = len(cycle_end_indices)

    cycle_numbers_plot = np.arange(1, n_plot + 1, dtype=float)
    cycle_numbers_fit  = cycle_numbers_plot - 1.0

    t_cycle_end = t_plot[cycle_end_indices]
    mech_end    = mech_p[cycle_end_indices]

    if params is None:
        ms_unrec_pred = np.full_like(cycle_numbers_plot, np.nan, dtype=float)
        ms_rec_pred   = np.full_like(cycle_numbers_plot, np.nan, dtype=float)
    else:
        A_fit, n0_fit, C_fit = params
        ms_unrec_pred = sat_exp(cycle_numbers_fit, A_fit, n0_fit, C_fit)
        ms_unrec_pred = np.maximum(ms_unrec_pred, 0.0)
        ms_rec_pred   = mech_end - ms_unrec_pred

    # ----------------------------
    # Try to infer a load signal for the bottom panel
    # ----------------------------
    load_sig, load_label = infer_load_signal_from_csv(INPUT_CSV, t_plot, tau4)


    # plt.figure(figsize=(6,6))
    # # plt.plot(t_plot, total_raw_p/crit, label="total_raw_p")
    # # plt.plot(t_plot, (plastic_p+creep_p+templ_p+mech_p)/crit, label="sum of terms")
    # plt.plot(t_plot, (elastic+hygro_p+creep_sim)/crit, label="el+hygro")
    # plt.plot(t_plot, (templ_p+creep_p)/crit, label="total_raw_p")
    # plt.show()
    # plt.close()

    # ----------------------------
    # Plot (3 panels)
    # ----------------------------
    # top/mid/bottom height split: 8,8,2
    fig = plt.figure(figsize=(16, 16), dpi=300)
    gs = GridSpec(3, 1, figure=fig, height_ratios=[8, 8, 2], hspace=0.10)
    # fig = plt.figure(figsize=(12, 14), dpi=300)

    # ---- manual layout ----
    left   = 0.08
    width  = 0.83

    ax_top = fig.add_axes([left, 0.63, width, 0.35])
    ax_mid = fig.add_axes([left, 0.25, width, 0.35], sharex=ax_top)
    ax_bot = fig.add_axes([left, 0.08, width, 0.15], sharex=ax_top)


    # ---- palette (consistent + chill) ----
    COL_TOTAL   = "k"
    COL_ELASTIC = "0.25"
    COL_VE      = "#1F77B4"
    COL_HYGRO   = "#17BECF"
    COL_SLIP    = "0.45"

    COL_PLASTIC = "#F0C300"
    COL_MS      = "#D62728"
    COL_REDUCED = "#2CA02C"
    COL_TEMPL   = "#9467BD"
    COL_MOIST   = "0.65"
    COL_LOAD    = "0.35"

    i_start = int(cycle_bounds[0][0])

    # ======= LOADED region shading (same logic you had) =======
    eps_for_loaded = total_raw_p
    nz = np.where(np.isfinite(eps_for_loaded) & (np.abs(eps_for_loaded) > 1e-12))[0]
    i_loaded_start = int(nz[0]) if len(nz) > 0 else i_start

    ax_top.axvspan(t_plot[i_loaded_start], t_unload_plot, color="0.92", zorder=0, label="_nolegend_")
    ax_mid.axvspan(t_plot[i_loaded_start], t_unload_plot, color="0.92", zorder=0, label="_nolegend_")
    ax_bot.axvspan(t_plot[i_loaded_start], t_unload_plot, color="0.92", zorder=0, label="_nolegend_")

    # ===========================
    # TOP: MODEL DECOMPOSITION
    # ===========================
    ax_top.plot(t_plot, total_raw_p/crit, color=COL_TOTAL,   alpha=0.75, label=r"$\varepsilon$")
    ax_top.plot(t_plot, elastic/crit,     color=COL_ELASTIC, alpha=0.75, label=r"$\varepsilon^E$")
    ax_top.plot(t_plot, creep_sim/crit,     color=COL_VE,      alpha=0.75, label=r"$\varepsilon^{VE}$")
    ax_top.plot(t_plot, hygro_p/crit,     color=COL_HYGRO,   alpha=0.75, label=r"$\varepsilon^H$")
    ax_top.plot(t_plot, slip_p/crit,      color=COL_SLIP,    alpha=0.75, label=r"$\varepsilon^S$")

    ax_top.set_ylabel(r"FBM Decomposition: $\varepsilon/\varepsilon_c$")
    ax_top.grid(True, alpha=0.7,zorder=0)
    ax_top.set_ylim(bottom=0.0)
    ax_top.tick_params(labelbottom=False)

    # ===========================
    # MIDDLE: YOUR FULL DECOMPOSITION
    # ===========================
    templ_loaded_color   = shade(COL_TEMPL, 0.85)
    templ_unloaded_color = shade(COL_TEMPL, 1.50)

    ax_mid.plot(t_plot, total_raw_p/crit, color=COL_TOTAL,   alpha=0.75, label=r"$\varepsilon$")
    ax_mid.plot(t_plot, plastic_p/crit,   color=COL_PLASTIC, alpha=0.75, label=r"$\varepsilon^{*P}$")
    ax_mid.plot(t_plot, creep_p/crit,     color=COL_VE,      alpha=0.75, label=r"$\varepsilon^{*VE}$")
    ax_mid.plot(t_plot, reduced_p/crit,   color=COL_REDUCED, alpha=0.75, label=r"$\varepsilon^{*R}$")

    # template split before/after unloading
    ax_mid.plot(t_plot[i_start:i_unload+1], templ_p[i_start:i_unload+1]/crit,
                color=templ_loaded_color, alpha=0.70, label=r"$\varepsilon^{*HE}$")
    ax_mid.plot(t_plot[i_unload:], templ_p[i_unload:]/crit,
                color=templ_loaded_color, alpha=0.70)

    # reduced overlays for template base cycles (dashed, no extra legend)
    # ax_mid.plot(t_plot[last_loaded_i1:last_loaded_i2+1], reduced_p[last_loaded_i1:last_loaded_i2+1]/crit,
    #             color=templ_loaded_color, linestyle="--", alpha=0.70, label="_nolegend_")
    # ax_mid.plot(t_plot[last_unloaded_i1:last_unloaded_i2+1], reduced_p[last_unloaded_i1:last_unloaded_i2+1]/crit,
    #             color=templ_unloaded_color, linestyle="--", alpha=0.70, label="_nolegend_")

    # mechsorption curve
    ax_mid.plot(t_plot[i_start:], mech_p[i_start:]/crit, color=COL_MS, label=r"$\varepsilon^{*MS}$", zorder=10)

    # MS recoverable / non-recoverable points
    ms_unrec_color = shade(COL_MS, 0.80)
    ms_rec_color   = shade(COL_MS, 1.60)
    ax_mid.scatter(t_cycle_end, ms_unrec_pred/crit, s=55, marker="s",
                   color=ms_unrec_color, edgecolor="none", label=r"$\varepsilon^{*MSnr}$", zorder=11)
    ax_mid.scatter(t_cycle_end, ms_rec_pred/crit,   s=55, marker="^",
                   color=ms_rec_color, edgecolor="none", label=r"$\varepsilon^{*MSr}$", zorder=11)

    ax_mid.set_ylabel(r"Macro. Strain Decomp. $\varepsilon/\varepsilon_c$")
    ax_mid.grid(True, alpha=0.7, zorder=0)
    ax_mid.set_ylim(bottom=0.0)
    ax_mid.tick_params(labelbottom=False)

    # =====================================
    # INSET: last-cycle template behaviour
    # =====================================

    # =====================================
    # INSET: last-cycle template behaviour
    # =====================================

    in_ax = inset_axes(
        ax_mid,
        width="35%",
        height="40%",
        bbox_to_anchor=(0.6, 0.3, 1.1, 1.8),
        bbox_transform=ax_mid.transAxes,
        loc="lower left",
        borderpad=0
    )

    plot_last_cycle_template(results, ax=in_ax)

    x1, x2 = 89, 109
    y1, y2 = 0.43, 0.54

    # zoom rectangle
    rect = Rectangle(
        (x1, y1),
        x2-x1,
        y2-y1,
        linewidth=1.5,
        edgecolor="black",
        facecolor="none",
        zorder=5
    )
    # ensure this rectangle is actually added to the axis
    ax_mid.add_patch(rect)

    dx = 0.02 * (x2 - x1)
    dy = 0.05 * (y2 - y1)
    ax_mid.text(
        x2 - dx,
        y2 - dy,
        "1",
        transform=ax_mid.transData,
        ha="right",
        va="top",
        fontsize=20,
        fontweight="bold",
        bbox=dict(facecolor='none', edgecolor='none', pad=0.2)
    )

    x1, x2 = 9, 29
    y1, y2 = 0.3, 0.41

    # zoom rectangle
    rect = Rectangle(
        (x1, y1),
        x2-x1,
        y2-y1,
        linewidth=1.5,
        edgecolor="black",
        facecolor="none",
        zorder=5
    )

    # place a "1" at the top-right of the rectangle (slightly inset)
    dx = 0.02 * (x2 - x1)
    dy = 0.05 * (y2 - y1)
    ax_mid.text(
        x2 - dx,
        y2 - dy,
        "3",
        transform=ax_mid.transData,
        ha="right",
        va="top",
        fontsize=20,
        fontweight="bold",
        bbox=dict(facecolor='none', edgecolor='none', pad=0.2)
    )

    ax_mid.add_patch(rect)


    x1, x2 = 189, 210
    y1, y2 = 0.01, 0.13

    # zoom rectangle
    rect = Rectangle(
        (x1, y1),
        x2-x1,
        y2-y1,
        linewidth=1.5,
        edgecolor="black",
        facecolor="none",
        zorder=5
    )

    # place a "1" at the top-right of the rectangle (slightly inset)
    dx = 0.02 * (x2 - x1)
    dy = 0.05 * (y2 - y1)
    ax_mid.text(
        x2 - dx,
        y2 - dy,
        "2,4",
        transform=ax_mid.transData,
        ha="right",
        va="top",
        fontsize=20,
        fontweight="bold",
        bbox=dict(facecolor='none', edgecolor='none', pad=0.2)
    )

    ax_mid.add_patch(rect)

    # connector
    # con = ConnectionPatch(
    #     xyA=(x2, y1),
    #     coordsA=ax_mid.transData,
    #     xyB=(0,1),
    #     coordsB=in_ax.transAxes,
    #     color="black",
    #     lw=1
    # )

    # fig.add_artist(con)

    # ===========================
    # BOTTOM: LOAD + MOISTURE
    # ===========================
    if load_sig is not None:
        # normalize load by its max (proxy for sigma_c if you don't have it explicitly)
        sig_c = float(np.nanmax(np.abs(load_sig))) if np.any(np.isfinite(load_sig)) else 7
        sig_c = max(sig_c, 1e-30)
        # print("AM I GETTING HERE?")
        sig_c=1.8817433196624958
        ax_bot.plot(t_plot, load_sig / sig_c, color="k", alpha=1.0)   # full black, no label
        ax_bot.set_ylabel(r"$\sigma/\sigma_c$")
    else:
        ax_bot.set_ylabel(r"$\sigma/\sigma_c$")

    ax_bot.grid(True, alpha=0.7, zorder=0)
    ax_bot.set_xlabel(r"Normalized time ($t/\tau_1$)")

    ax_bot2 = ax_bot.twinx()
    ax_bot2.plot(t_plot, moisture, color=COL_MOIST)  # no label -> no legend
    ax_bot2.set_ylabel(r"$\langle\varphi\rangle/0.2$", color=COL_MOIST)
    ax_bot2.tick_params(axis="y", colors=COL_MOIST)
    ax_bot2.spines["right"].set_visible(True)
    ax_bot2.spines["right"].set_color(COL_MOIST)
    ax_bot2.set_zorder(ax_bot.get_zorder() + 1)
    ax_bot2.patch.set_visible(False)


    # ===========================
    # SEPARATE LEGENDS (top + mid)
    # ===========================
    # ===========================
    # TWO-COLUMN GROUPED LEGEND
    # ===========================

    # --- handles ---
    h_top, l_top = ax_top.get_legend_handles_labels()
    h_mid, l_mid = ax_mid.get_legend_handles_labels()

    # --- LEFT: model (top plot) ---
    leg1 = ax_top.legend(
        h_top,
        l_top,
        loc="upper center",
        bbox_to_anchor=(0.6, 1.05),
        frameon=True,
        title="DD-SS-FBM",
        borderaxespad=0.0
    )

    # --- RIGHT: experimental (mid plot) ---
    leg2 = ax_top.legend(
        h_mid,
        l_mid,
        loc="upper center",
        bbox_to_anchor=(0.85, 1.05),
        frameon=True,
        title="Macro. Strain Decomp.",
        borderaxespad=0.0
    )

    # keep both legends
    ax_top.add_artist(leg1)


    seen = set()
    handles, labels = [], []
    # for h, l in list(zip(h_top, l_top)) + list(zip(h_mid, l_mid)) + list(zip(h_bot, l_bot)):
    #     if l == "_nolegend_":
    #         continue
    #     if l not in seen:
    #         handles.append(h)
    #         labels.append(l)
    #         seen.add(l)

    # # stable margin for legend
    # fig.subplots_adjust(right=0.78)
    # fig.legend(
    #     handles, labels,
    #     loc="center left",
    #     bbox_to_anchor=(0.80, 0.50),
    #     frameon=True
    # )
    # ===========================
    # AXIS BASELINES AT ZERO
    # ===========================
    # x starts at 0
    ax_top.set_xlim(left=0.0)

    # y starts at 0 for each left axis
    ax_top.set_ylim(bottom=0.0)
    ax_mid.set_ylim(bottom=0.0)
    ax_bot.set_ylim(bottom=0.0)

    # moisture axis also from 0 (right axis)
    ax_bot2.set_ylim(bottom=0.0)

    # ---- save ----
    out = Path("./Fig6.png")
    # plt.show()
    fig.savefig(out, dpi=300)
    plt.close(fig)

    print(f"✅ Saved: {out.resolve()}")
    if load_sig is None:
        print("ℹ️ Bottom panel: no load-like column found in INPUT_CSV; plotted moisture only.")
    if np.allclose(hygro, 0.0):
        print("ℹ️ Top panel: hygroexp not provided by analyze_creep; plotted as zeros.")
