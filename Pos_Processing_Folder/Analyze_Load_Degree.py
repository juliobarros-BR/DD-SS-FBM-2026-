from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from matplotlib.patches import Rectangle

# ============================================================
# STYLE (UNCHANGED)
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

FIT_DIR = Path("fourier_globalfit_from_results_Fo99")
OUT_DIR = Path("fourier_load_effect_from_fixed_Fo_window")
OUT_DIR.mkdir(exist_ok=True)

# ============================================================
# CONFIG
# ============================================================
MECH_COLUMN = "Mech_max_ratio"
LOADD_MAX = 0.7
LOADD_REF_TAG = "0p700"
DELTA_LOG_FO = 35

# ============================================================
# MODELS
# ============================================================
def mech_vs_load_exp(loadD, A, d):
    return A * np.exp(d * loadD)

def read_fit_params(param_csv: Path):
    p = pd.read_csv(param_csv)
    dct = dict(zip(p["parameter"].astype(str), p["value"]))
    return float(dct["a"]), float(dct["b"]), float(dct["c"]), float(dct["x0"])

def fo_eff(Fo_diff, Fo_visc, Ramp, a, b, c):
    return (Fo_diff ** a) * (Fo_visc ** b) * np.exp(c * Ramp)

# ============================================================
# NEW HELPERS (NON-INTRUSIVE)
# ============================================================
def build_envelope(df, xcol, ycol):
    g = df.groupby(xcol)[ycol]
    x = np.array(sorted(g.mean().index))
    y_min = g.min().reindex(x).values
    y_max = g.max().reindex(x).values
    return x, y_min, y_max

def fit_exp(x, y):
    mask = np.isfinite(x) & np.isfinite(y) & (y > 0)
    x = x[mask]
    y = y[mask]
    if len(x) < 5:
        return None
    p0 = [np.median(y), 1.0]
    popt, _ = curve_fit(mech_vs_load_exp, x, y, p0=p0, maxfev=20000)
    return popt

# ============================================================
# MAIN
# ============================================================
def r2_score(y_true, y_pred):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot <= 0:
        return np.nan
    return 1.0 - ss_res / ss_tot

def _has_required_loads(loads, required=(0.3, 0.4, 0.5), tol=1e-9):
    loads = np.asarray(loads, float)
    for r in required:
        if not np.any(np.isclose(loads, r, atol=tol, rtol=0.0)):
            return False
    return True

def per_system_linear_fits(dfW, group_cols, xcol="LoadD", ycol="MS_total"):
    rows = []

    dfi = dfW[(dfW[xcol] >= 0.3 - 1e-9) & (dfW[xcol] <= 0.5 + 1e-9)].copy()
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

        m, b = np.polyfit(x, y, 1)
        yhat = m * x + b
        R2 = r2_score(y, yhat)

        row = {col: val for col, val in zip(group_cols, key)}
        row.update({"n": len(x), "m": m, "b": b, "R2": R2})
        rows.append(row)

    return pd.DataFrame(rows)

def run_one(is_avg: bool, label: str):

    # -------------------------
    # Load fit params
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

    for col in ["Fo_diff", "Fo_visc", "Ramp", "LoadD", MECH_COLUMN, "Diff", "Tau", "ms_nr_ratio"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["Fo_diff", "Fo_visc", "Ramp", "LoadD", MECH_COLUMN, "Diff", "Tau", "ms_nr_ratio"])
    df = df[df["LoadD"] <= LOADD_MAX]
    df = df[df[MECH_COLUMN] > 0]

    # -------------------------
    # Fo collapse
    # -------------------------
    df["Fo_eff_ref"] = fo_eff(df["Fo_diff"], df["Fo_visc"], df["Ramp"], a, b, c)
    df = df[df["Fo_eff_ref"] > 0]
    df["logFo_eff_ref"] = np.log(df["Fo_eff_ref"])

    dfW = df[(df["logFo_eff_ref"] >= logFo_min) &
             (df["logFo_eff_ref"] <= logFo_max)].copy()

    if dfW.empty:
        print("No data in Fo window")
        return

    # ============================================================
    # MS DECOMPOSITION (SAFE ADDITION)
    # ============================================================
    dfW["MS_total"] = dfW[MECH_COLUMN]
    dfW["MS_nr"] = dfW["ms_nr_ratio"]
    dfW["MS_rec"] = dfW["MS_total"] - dfW["MS_nr"]
    dfW.loc[dfW["MS_rec"] < 0, "MS_rec"] = np.nan
    # ============================================================
    # LINEAR FITS (RESTORED)
    # ============================================================
    group_cols = ["Fo_diff", "Ramp", "Fo_visc"]
    lin_tbl = per_system_linear_fits(dfW, group_cols=group_cols,
                                    xcol="LoadD", ycol="MS_total")
    if lin_tbl.empty:
        mean_r2 = std_r2 = np.nan
    else:
        mean_r2 = float(np.nanmean(lin_tbl["R2"]))
        n_r = int(lin_tbl["R2"].notna().sum())
        std_r2 = float(lin_tbl["R2"].std(ddof=1)) if n_r > 1 else 0.0
    
    # ============================================================
    # SELECT EXTREME SYSTEMS (NEW)
    # ============================================================
    # define extremes from data automatically
    Fo_diff_max = dfW["Fo_diff"].max()
    Fo_diff_min = dfW["Fo_diff"].min()

    Fo_visc_max = dfW["Fo_visc"].max()
    Fo_visc_min = dfW["Fo_visc"].min()

    Ramp_max = dfW["Ramp"].max()
    Ramp_min = dfW["Ramp"].min()

    # helper to select system
    def select_extreme(df, fd, fv, rr):
        return df[
            (np.isclose(df["Fo_diff"], fd)) &
            (np.isclose(df["Fo_visc"], fv)) &
            (np.isclose(df["Ramp"], rr))
        ]

    # select rows in linear table
    lin_high = lin_tbl[
        (np.isclose(lin_tbl["Fo_diff"], Fo_diff_max)) &
        (np.isclose(lin_tbl["Fo_visc"], Fo_visc_max)) &
        (np.isclose(lin_tbl["Ramp"], Ramp_min))
    ]

    lin_low = lin_tbl[
        (np.isclose(lin_tbl["Fo_diff"], Fo_diff_min)) &
        (np.isclose(lin_tbl["Fo_visc"], Fo_visc_min)) &
        (np.isclose(lin_tbl["Ramp"], Ramp_max))
    ]


    # ============================================================
    # FITS (SAFE ADDITION)
    # ============================================================
    x_all = dfW["LoadD"].values
    y_all = dfW["MS_total"].values
    p0 = [np.median(y_all), 1.0]
    A_fit, d_fit = curve_fit(mech_vs_load_exp, x_all, y_all, p0=p0, maxfev=20000)[0]

    fit_rec = fit_exp(dfW["LoadD"].values, dfW["MS_rec"].values)
    fit_nr  = fit_exp(dfW["LoadD"].values, dfW["MS_nr"].values)

    # envelopes
    x_env, rec_min, rec_max = build_envelope(dfW, "LoadD", "MS_rec")
    _, nr_min, nr_max = build_envelope(dfW, "LoadD", "MS_nr")

    # ============================================================
    # ORIGINAL PLOT STRUCTURE (UNTOUCHED)
    # ============================================================
    Fo_diffs_unique = sorted(dfW["Fo_diff"].round(6).unique())
    Fo_visc_unique  = sorted(dfW["Fo_visc"].round(6).unique())
    ramps_sorted    = sorted(dfW["Ramp"].dropna().unique())

    colors = plt.cm.viridis(np.linspace(0, 1, len(Fo_diffs_unique)))
    color_map = {fd: colors[i] for i, fd in enumerate(Fo_diffs_unique)}

    markers = ["o", "s", "^", "D", "p", "X", "v", "<", ">"]
    marker_map = {fv: markers[i % len(markers)] for i, fv in enumerate(Fo_visc_unique)}

    alpha_map = {0.0: 1.0, 10.0: 0.7, 30.0: 0.4, 50.0: 0.15}

    fig = plt.figure(figsize=(16, 6), dpi=200)
    gs = fig.add_gridspec(nrows=1, ncols=2, width_ratios=[1.0, 0.55])

    ax = fig.add_subplot(gs[0, 0])

    # --- ORIGINAL scatter ---
    for _, r in dfW.iterrows():
        fd = round(float(r["Fo_diff"]), 6)
        fv = round(float(r["Fo_visc"]), 6)
        ax.scatter(
            r["LoadD"], r["MS_total"],
            s=120,
            facecolor=color_map[fd],
            edgecolor="black",
            linewidth=1.0,
            marker=marker_map[fv],
            alpha=alpha_map.get(float(r["Ramp"]), 0.5),
            zorder=5
        )

    # --- NEW (non-intrusive visual layers) ---
    ax.fill_between(x_env, rec_min, rec_max, color="green", alpha=0.18, zorder=1)
    ax.fill_between(x_env, nr_min, nr_max, color="red", alpha=0.18, zorder=2)

    xx = np.linspace(dfW["LoadD"].min(), dfW["LoadD"].max(), 200)

    ax.plot(xx, mech_vs_load_exp(xx, A_fit, d_fit), "--k", zorder=10)

    if fit_rec is not None:
        ax.plot(xx, mech_vs_load_exp(xx, *fit_rec),"--", color="green", lw=3, zorder=11)

    if fit_nr is not None:
        ax.plot(xx, mech_vs_load_exp(xx, *fit_nr), "--", color="red", lw=3, zorder=12)

    ax.set_xlabel(r"$\sigma/\sigma_c$")
    ax.set_ylabel(r"$\varepsilon^{*MS}_\infty/\varepsilon_\infty$")
    ax.grid(True, ls="--", alpha=0.7, zorder=0)
    # --- linear fits (same style as before) ---
    # --- linear fits (ONLY EXTREMES) ---
    x_lin = np.array([0.3, 0.5], dtype=float)

    # HIGH system (strong mechano response)
    if not lin_high.empty:
        row = lin_high.iloc[0]
        m = float(row["m"])
        b0 = float(row["b"])

        ax.plot(
            x_lin, m * x_lin + b0,
            color="black",    # same color as lowest Fo_diff
            alpha=0.3,
            lw=4.0,
            zorder=20
        )

    # LOW system (weak mechano response)
    if not lin_low.empty:
        row = lin_low.iloc[0]
        m = float(row["m"])
        b0 = float(row["b"])

        ax.plot(
            x_lin, m * x_lin + b0,
            color="black",    # same color as lowest Fo_diff
            alpha=0.3,
            lw=4.0,
            zorder=20
        )
    # ============================================================
    # LEGEND PANEL (100% ORIGINAL — NOT TOUCHED)
    # ============================================================
    legend_ax = fig.add_subplot(gs[0, 1])
    legend_ax.axis("off")

    handles, labels = [], []

    for fd in Fo_diffs_unique:
        handles.append(Rectangle((0, 0), 1, 1, facecolor=color_map[fd], edgecolor="black"))
        labels.append(f"{fd:.2f}")

    for fv in Fo_visc_unique:
        handles.append(plt.Line2D([], [], marker=marker_map[fv], color="black", lw=0, markersize=12))
        labels.append(f"{fv:.2f}")

    for rr in ramps_sorted:
        handles.append(Rectangle((0, 0), 1, 1, facecolor="black",
                                 edgecolor="black",
                                 alpha=alpha_map.get(float(rr), 0.6)))
        labels.append(f"{int(rr)}%")

    title_text = r"$Fo_{\chi}^{99}\hspace{2.5cm}Fo_{\tau}^{99}\hspace{2.5cm}T_r$"

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
    legend_ax.add_artist(big_legend)

    # --- SMALL LEGEND (extended but same structure) ---
    fit_handles = [
        plt.Line2D([], [], ls="--", color="black", lw=3,
                label=r"Global $\varepsilon^{*MS}_\infty/\varepsilon_\infty$ fit (exponential)"),

        plt.Line2D([], [], ls="--", color="green", lw=3,
                label=r"Global $\varepsilon^{*MSr}_\infty/\varepsilon_\infty$ fit (exponential)"),

        plt.Line2D([], [], ls="--", color="red", lw=3,
                label=r"Global $\varepsilon^{*MSnr}_\infty/\varepsilon_\infty$ fit (exponential)"),

        Rectangle((0,0),1,1,color="green",alpha=0.18,
                label=r"$\varepsilon^{*MSr}_\infty/\varepsilon_\infty$ range"),

        Rectangle((0,0),1,1,color="red",alpha=0.18,
                label=r"$\varepsilon^{*MSnr}_\infty/\varepsilon_\infty$ range"),

        plt.Line2D([], [], color="black", lw=4.0, alpha=0.3,
            label="Linear fit (extreme systems)"),
    ]

    if not np.isnan(mean_r2):
        ax.text(
            0.02, 0.98,
            rf"$\langle R^2_{{0.3-0.5}}\rangle = {mean_r2:.3f} \pm {std_r2:.3f}$",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=20,
            bbox=dict(
                facecolor="white",
                edgecolor="black",
                alpha=1.0,
                boxstyle="round,pad=0.2"
            )
        )

    fit_legend = legend_ax.legend(
        handles=fit_handles,
        fontsize=16,
        frameon=True,
        loc="center",
        bbox_to_anchor=(0.78, 0.40),
        bbox_transform=fig.transFigure
    )

    fit_legend.get_frame().set_facecolor("white")
    fit_legend.get_frame().set_edgecolor("black")
    fit_legend.get_frame().set_linewidth(1.5)
    fit_legend.get_frame().set_alpha(1.0)

    fig.tight_layout()

    out_png = "Figures/Fig9.png"
    fig.savefig(out_png, dpi=300)
    plt.close(fig)

    print(f"✔ Saved: {out_png}")

# ============================================================
# RUN
# ============================================================
def main():
    # run_one(False, "non_avg")
    run_one(True, "avg")

if __name__ == "__main__":
    main()