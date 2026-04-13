from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from matplotlib.patches import Rectangle

# ============================================================
# STYLE (reuse your publication params)
# ============================================================
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

# ============================================================
# PATHS
# ============================================================
META_DIR = Path("fourier_meta")
IN_CSV = META_DIR / "results_systems.csv"

FIT_DIR = Path("fourier_globalfit_from_results")  # your existing fit outputs
OUT_DIR = Path("fourier_load_effect_from_fixed_Fo_window")
OUT_DIR.mkdir(exist_ok=True)

# ============================================================
# CONFIG
# ============================================================
MECH_COLUMN = "Mech_max_ratio"      # or "Mech_max_unload"
LOADD_MAX = 0.7

# Which LoadD fit to use as "reference regime" (you said 0.600)
LOADD_REF_TAG = "0p600"  # must match file name

# Width of the Fo window around x0 (in log space)
# delta=0.35 means Fo within exp(x0±0.35) ~ factor 1.42
DELTA_LOG_FO = 35

# Linear fit window in LoadD
LIN_LO = 0.3
LIN_HI = 0.5
LIN_TOL = 1e-9
REQUIRED_LOADS = (0.3, 0.4, 0.5)  # enforce these three exist

# Linear overlay behavior
PLOT_PER_SYSTEM_LINES = True     # faint gray line per fitted system
PLOT_MEAN_LINEAR_LINE = False     # one bold "mean fit" line (mean m and b)
PLOT_MEAN_BAND = False           # optional: +/- 1 std band of mean m,b (rough)
MEAN_LINE_COLOR = "k"
MEAN_LINE_STYLE = "-"
MEAN_LINE_LW = 3.0

# Exponential model in LoadD (your existing)
def mech_vs_load_exp(loadD, A, d):
    return A * np.exp(d * loadD)

def read_fit_params(param_csv: Path):
    p = pd.read_csv(param_csv)
    dct = dict(zip(p["parameter"].astype(str), p["value"]))
    needed = ["a", "b", "c", "x0"]
    missing = [k for k in needed if k not in dct]
    if missing:
        raise ValueError(f"Missing parameters {missing} in {param_csv}")
    return float(dct["a"]), float(dct["b"]), float(dct["c"]), float(dct["x0"])

def fo_eff(Fo_diff, Fo_visc, Ramp, a, b, c):
    return (Fo_diff ** a) * (Fo_visc ** b) * np.exp(c * Ramp)

def r2_score(y_true, y_pred):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot <= 0:
        return np.nan
    return 1.0 - ss_res / ss_tot

def _has_required_loads(loads, required=REQUIRED_LOADS, tol=LIN_TOL):
    loads = np.asarray(loads, float)
    for r in required:
        if not np.any(np.isclose(loads, r, atol=tol, rtol=0.0)):
            return False
    return True

def per_system_linear_fits(dfW, group_cols, xcol="LoadD", ycol=MECH_COLUMN):
    """
    For each system (group), fit y = m*LoadD + b in [LIN_LO, LIN_HI]
    only if it has 0.3, 0.4, 0.5 present.
    Returns dataframe with m,b,R2,n per system.
    """
    rows = []

    dfi = dfW[(dfW[xcol] >= LIN_LO - LIN_TOL) & (dfW[xcol] <= LIN_HI + LIN_TOL)].copy()
    if dfi.empty:
        return pd.DataFrame(columns=group_cols + ["n", "m", "b", "R2"])

    for key, g in dfi.groupby(group_cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)

        x = g[xcol].to_numpy(float)
        y = g[ycol].to_numpy(float)

        if len(x) < 3:
            continue
        if not _has_required_loads(x):
            continue

        # linear regression
        m, b = np.polyfit(x, y, 1)
        yhat = m * x + b
        R2 = r2_score(y, yhat)

        row = {col: val for col, val in zip(group_cols, key)}
        row.update({"n": len(x), "m": m, "b": b, "R2": R2})
        rows.append(row)

    return pd.DataFrame(rows)

def run_one(is_avg: bool, label: str):
    # -------------------------
    # load fit params at LoadD=0.6 (reference for Fo window)
    # -------------------------
    fit_csv = FIT_DIR / f"fit_parameters_{label}_LoadD_{LOADD_REF_TAG}.csv"
    a, b, c, x0 = read_fit_params(fit_csv)

    logFo_min = x0 - DELTA_LOG_FO
    logFo_max = x0 + DELTA_LOG_FO

    # -------------------------
    # Load + clean
    # -------------------------
    df = pd.read_csv(IN_CSV).copy()
    df = df[df["is_avg"] == is_avg].copy()

    for col in ["Fo_diff", "Fo_visc", "Ramp", "LoadD", MECH_COLUMN, "Diff", "Tau"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["Fo_diff", "Fo_visc", "Ramp", "LoadD", MECH_COLUMN, "Diff", "Tau"])
    df = df[df["LoadD"] <= LOADD_MAX].copy()
    df = df[df[MECH_COLUMN] > 0].copy()

    # -------------------------
    # Compute Fo_eff_ref using reference (a,b,c)
    # -------------------------
    df["Fo_eff_ref"] = fo_eff(df["Fo_diff"].values, df["Fo_visc"].values, df["Ramp"].values, a, b, c)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["Fo_eff_ref"]).copy()
    df = df[df["Fo_eff_ref"] > 0].copy()
    df["logFo_eff_ref"] = np.log(df["Fo_eff_ref"])

    # -------------------------
    # Apply fixed Fo window
    # -------------------------
    dfW = df[(df["logFo_eff_ref"] >= logFo_min) & (df["logFo_eff_ref"] <= logFo_max)].copy()
    if dfW.empty:
        print(f"[{label}] No points in Fo window. Increase DELTA_LOG_FO.")
        return

    # -------------------------
    # Exponential fit (your existing)
    # -------------------------
    x_all = dfW["LoadD"].values
    y_all = dfW[MECH_COLUMN].values

    p0 = [np.median(y_all), 1.0]
    popt, pcov = curve_fit(mech_vs_load_exp, x_all, y_all, p0=p0, maxfev=20000)
    A_fit, d_fit = popt
    d_err = float(np.sqrt(np.diag(pcov))[1]) if pcov.shape == (2, 2) else np.nan

    # save exp fit
    pd.DataFrame({
        "parameter": ["A", "d", "d_std", "a_used", "b_used", "c_used", "x0_used",
                      "logFo_min", "logFo_max", "DELTA_LOG_FO", "LoadD_max"],
        "value":     [A_fit, d_fit, d_err, a, b, c, x0,
                      logFo_min, logFo_max, DELTA_LOG_FO, LOADD_MAX],
    }).to_csv(OUT_DIR / f"load_fit_{label}_FoWindowAround_x0.csv", index=False)

    # -------------------------
    # per-system linear fits in [0.3,0.5]
    # define system = Diff, Ramp, Tau
    # -------------------------
    group_cols = ["Diff", "Ramp", "Tau"]
    lin_tbl = per_system_linear_fits(dfW, group_cols=group_cols, xcol="LoadD", ycol=MECH_COLUMN)
    lin_csv = OUT_DIR / f"linear_fits_{label}_LoadD_{LIN_LO:.1f}_to_{LIN_HI:.1f}.csv"
    lin_tbl.to_csv(lin_csv, index=False)

    # stats for plot annotation (mean ± std across systems)
    if lin_tbl.empty:
        mean_m = mean_b = np.nan
        std_m = std_b = np.nan
        mean_r2 = std_r2 = np.nan
        n_ok = 0
    else:
        mean_m = float(np.nanmean(lin_tbl["m"]))
        mean_b = float(np.nanmean(lin_tbl["b"]))

        # ddof=1 (sample std) if >1 system, else 0
        n_m = int(lin_tbl["m"].notna().sum())
        n_b = int(lin_tbl["b"].notna().sum())
        n_r = int(lin_tbl["R2"].notna().sum())

        std_m = float(lin_tbl["m"].std(ddof=1)) if n_m > 1 else 0.0
        std_b = float(lin_tbl["b"].std(ddof=1)) if n_b > 1 else 0.0

        mean_r2 = float(np.nanmean(lin_tbl["R2"]))
        std_r2  = float(lin_tbl["R2"].std(ddof=1)) if n_r > 1 else 0.0

        n_ok = int(lin_tbl.shape[0])

    print(f"[{label}] Linear fits in [{LIN_LO:.1f},{LIN_HI:.1f}] requiring {REQUIRED_LOADS}:")
    print(f"  qualified systems: {n_ok}")
    if n_ok > 0:
        print(f"  mean R²: {mean_r2:.4f} ± {std_r2:.4f}")
        print(f"  mean slope m: {mean_m:.6g} ± {std_m:.3g}")
        print(f"  mean intercept b: {mean_b:.6g} ± {std_b:.3g}")
        print(f"  saved per-system fits: {lin_csv}")
    else:
        print(f"  no qualified systems (saved empty table): {lin_csv}")

        # -------------------------
    # Plot: SAME SCATTER + EXP FIT, plus linear overlays
    # but larger + legend panel (colors/markers/alpha) like your collapse plot
    # -------------------------
    Fo_diffs_unique = sorted(dfW["Fo_diff"].round(6).unique())
    Fo_visc_unique  = sorted(dfW["Fo_visc"].round(6).unique())
    ramps_sorted    = sorted(dfW["Ramp"].dropna().unique())

    colors = plt.cm.viridis(np.linspace(0, 1, len(Fo_diffs_unique)))
    color_map = {fd: colors[i] for i, fd in enumerate(Fo_diffs_unique)}

    markers = ["o", "s", "^", "D", "p", "X", "v", "<", ">"]
    marker_map = {fv: markers[i % len(markers)] for i, fv in enumerate(Fo_visc_unique)}

    alpha_map = {0.0: 1.0, 10.0: 0.7, 30.0: 0.4, 50.0: 0.15}

    # --- figure + gridspec (main plot + legend panel) ---
    fig = plt.figure(figsize=(16, 6), dpi=200)
    gs = fig.add_gridspec(nrows=1, ncols=2, width_ratios=[1.0, 0.55])

    ax = fig.add_subplot(gs[0, 0])

    # --- scatter points (same logic) ---
    for _, r in dfW.iterrows():
        fd = round(float(r["Fo_diff"]), 6)
        fv = round(float(r["Fo_visc"]), 6)
        ax.scatter(
            r["LoadD"], r[MECH_COLUMN],
            s=120,
            facecolor=color_map[fd],
            edgecolor="black",
            linewidth=1.0,
            marker=marker_map[fv],
            alpha=alpha_map.get(float(r["Ramp"]), 0.5),
            zorder=5
        )

    # --- exponential fit (same dashed) ---
    xx = np.linspace(dfW["LoadD"].min(), dfW["LoadD"].max(), 200)
    ax.plot(xx, mech_vs_load_exp(xx, A_fit, d_fit), "--k", zorder=10)

    # --- linear fit overlays only in [0.3,0.5] ---
    x_lin = np.array([LIN_LO, LIN_HI], dtype=float)

    # A) per-system lines (faint)
    if PLOT_PER_SYSTEM_LINES and (not lin_tbl.empty):
        for _, row in lin_tbl.iterrows():
            m = float(row["m"])
            b0 = float(row["b"])
            ax.plot(
                x_lin, m * x_lin + b0,
                color="0.5", alpha=0.25, lw=2.0, zorder=7
            )

    # B) mean line from averaged coefficients
    if PLOT_MEAN_LINEAR_LINE and np.isfinite(mean_m) and np.isfinite(mean_b):
        ax.plot(
            x_lin, mean_m * x_lin + mean_b,
            color=MEAN_LINE_COLOR, ls=MEAN_LINE_STYLE, lw=MEAN_LINE_LW, zorder=12
        )

        if PLOT_MEAN_BAND and np.isfinite(std_m) and np.isfinite(std_b):
            y_lo = (mean_m - std_m) * x_lin + (mean_b - std_b)
            y_hi = (mean_m + std_m) * x_lin + (mean_b + std_b)
            ax.fill_between(x_lin, y_lo, y_hi, color="0.2", alpha=0.12, zorder=6)

    ax.set_xlabel(r"$\sigma/\sigma_c$")
    ax.set_ylabel(r"$\varepsilon^{MS}_\infty/\varepsilon_\infty$")

    ax.axvspan(LIN_LO, LIN_HI, color="0.8", alpha=0.12, zorder=0)
    ax.grid(True, ls="--", alpha=0.35)

    # Annotation: include std of R² on the plot
    if n_ok > 0:
        ax.text(
            0.02, 0.98,
            rf"$\langle R^2_{{0.3-0.5}}\rangle = {mean_r2:.2f} \pm {std_r2:.2f}$",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=20,
            bbox=dict(facecolor="white", edgecolor="black", alpha=1.0, boxstyle="round,pad=0.2")
        )

    # ------------------------------------------------------
    # LEGEND PANEL (same structure as your collapse plot)
    # ------------------------------------------------------
    legend_ax = fig.add_subplot(gs[0, 1])
    legend_ax.axis("off")

    handles, labels = [], []

    # 1) Colors = Fo_diff
    for fd in Fo_diffs_unique:
        handles.append(Rectangle((0, 0), 1, 1, facecolor=color_map[fd], edgecolor="black"))
        labels.append(f"{fd:.3g}")

    # 2) Markers = Fo_visc
    for fv in Fo_visc_unique:
        handles.append(plt.Line2D([], [], marker=marker_map[fv], color="black", lw=0, markersize=12))
        labels.append(f"{fv:.3g}")

    # 3) Alpha = Ramp
    for rr in ramps_sorted:
        handles.append(Rectangle((0, 0), 1, 1, facecolor="black", edgecolor="black",
                                 alpha=alpha_map.get(float(rr), 0.6)))
        labels.append(f"{int(rr)}%")

    title_text = r"$Fo_{\chi}\hspace{2.5cm}Fo_{\tau}\hspace{2.5cm}T_r$"

    big_legend = legend_ax.legend(
        handles, labels,
        title=title_text,
        ncol=3,
        fontsize=16,
        title_fontsize=20,
        handlelength=1.4,
        columnspacing=2.0,
        frameon=True,
        loc="center",
        bbox_to_anchor=(0.78, 0.76),
        bbox_transform=fig.transFigure,
    )
    big_legend.get_frame().set_facecolor("white")
    big_legend.get_frame().set_edgecolor("black")
    big_legend.get_frame().set_linewidth(1.5)
    big_legend.get_frame().set_alpha(1.0)

    fig.tight_layout()

    out_png = OUT_DIR / f"{MECH_COLUMN}_vs_LoadD_{label}_fixedFoWindow_withLinear.png"
    fig.savefig(out_png, dpi=300)
    plt.close(fig)

    print(f"✔ Saved: {out_png}")
    print(f"  Exp Fit: {MECH_COLUMN} ≈ A exp(d LoadD),  d = {d_fit:.4f} ± {d_err:.4f}")



def main():
    run_one(is_avg=False, label="non_avg")
    run_one(is_avg=True,  label="avg")

if __name__ == "__main__":
    main()
