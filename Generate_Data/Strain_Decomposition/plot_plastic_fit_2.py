import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # comment this if you want interactive windows
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import curve_fit

# ============================================================
# STYLE (same as your plastic ratio plot)
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
    'savefig.bbox': None,  # do not auto-crop
})

# ============================================================
# INPUTS
# ============================================================
OUT_ROOT = Path("./Analysis_folders/mechsorption_analysis/")
CSV_ALL  = OUT_ROOT / "ms_unrec_all_points.csv"

# Optional filter (set to None to plot ALL groups)
ONLY_DIFF = "Diff_1000"       # e.g. "Diff_300"
ONLY_RAMP = "Ramp_50%"       # e.g. "Ramp_00%"
ONLY_TAU  = 0.001       # e.g. 0.001 (float)

# ============================================================
# Fit model: y = A*(1-exp(-n/n0)) + C
# ============================================================
def sat_exp(n, A, n0, C):
    return A * (1.0 - np.exp(-n / np.maximum(n0, 1e-12))) + C


def fit_sat_exp(n, y):
    """Return popt, R2, n_fit, y_fit. Raises if fit fails."""
    n = np.asarray(n, float)
    y = np.asarray(y, float)

    ok = np.isfinite(n) & np.isfinite(y)
    n = n[ok]
    y = y[ok]
    if len(n) < 4:
        raise RuntimeError("Need >=4 finite points to fit.")

    # robust-ish initial guesses
    A0  = float(np.nanmax(y) - np.nanmin(y))
    n00 = float(0.2 * np.nanmax(n)) if np.nanmax(n) > 0 else 1.0
    C0  = float(np.nanmin(y))
    p0 = [max(A0, 1e-12), max(n00, 1e-6), C0]
    bounds = ([0, 1e-6, -np.inf], [np.inf, np.inf, np.inf])

    popt, _ = curve_fit(sat_exp, n, y, p0=p0, bounds=bounds)

    n_fit = np.linspace(float(np.min(n)), float(np.max(n)), 400)
    y_fit = sat_exp(n_fit, *popt)

    y_hat = sat_exp(n, *popt)
    r2 = 1 - np.sum((y - y_hat) ** 2) / (np.sum((y - y.mean()) ** 2) + 1e-12)

    return popt, float(r2), n_fit, y_fit


def make_one_plot(df_grp, out_path, title_str):
    df_grp = df_grp.sort_values("cycles_loaded")
    n = df_grp["cycles_loaded"].to_numpy(float)
    y = df_grp["min_mech_after_unload"].to_numpy(float)
    FORCE_CRIT = 4.154739120251522  
    fig, ax = plt.subplots(figsize=(5.3, 6.5), dpi=300)

    # data points
    ax.plot(n, y/FORCE_CRIT, marker="o", linestyle="None")

    # fit
    fit_label = None
    try:
        (A_fit, n0_fit, C_fit), r2, n_fit, y_fit = fit_sat_exp(n, y)
        ax.plot(
            n_fit, y_fit/FORCE_CRIT, linestyle="-", color="black",
            alpha=0.3,
            label=rf"$A(1-e^{{-C_L/C_0}})+B$"
        )
        fit_label = (A_fit, n0_fit, C_fit, r2)
    except Exception:
        # no fit; keep scatter only
        pass

    ax.set_xlabel(r"$C_L$")
    ax.set_ylabel(r"$\varepsilon^{*MSnr}/\varepsilon_c$")  # rename if you prefer
    ax.grid(True, alpha=0.3)
    # ax.set_title(title_str)
    from matplotlib.ticker import ScalarFormatter

    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-2, -2))  # force ×10⁻²

    ax.yaxis.set_major_formatter(formatter)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, -2))

    # Optional: move the ×10⁻² a bit
    ax.yaxis.get_offset_text().set_x(-0.12)
    ax.yaxis.get_offset_text().set_y(1.01)

    ax.legend(
        loc="lower right",
        bbox_to_anchor=(1.05, 0.02),
    )


    pos = ax.get_position()
    ax.set_position([pos.width*0.23, pos.y0*1.13, pos.width*1.0, pos.height*1.07])

    # out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path,bbox_inches=None,dpi=300)
    plt.close(fig)

    return fit_label


def main():
    if not CSV_ALL.exists():
        raise FileNotFoundError(f"Could not find {CSV_ALL}")

    df = pd.read_csv(CSV_ALL)

    # apply optional filters
    if ONLY_DIFF is not None:
        df = df[df["diff"] == ONLY_DIFF]
    if ONLY_RAMP is not None:
        df = df[df["ramp"] == ONLY_RAMP]
    if ONLY_TAU is not None:
        # robust float compare
        df = df[np.isclose(df["tau"].astype(float), float(ONLY_TAU), rtol=0, atol=1e-12)]

    if df.empty:
        raise RuntimeError("No rows left after filtering.")

    # loop groups
    for (diff_label, ramp_label, tau_value), grp in df.groupby(["diff", "ramp", "tau"]):
        tau_value_f = float(tau_value)

        out_dir = OUT_ROOT / diff_label / ramp_label / f"Tau_{tau_value_f}"
        out_png = "./Fig5c.png"

        title = rf"{diff_label} / {ramp_label} / $\tau={tau_value_f}$"
        make_one_plot(grp, out_png, title)

        print(f"Saved: {out_png}")

if __name__ == "__main__":
    main()
