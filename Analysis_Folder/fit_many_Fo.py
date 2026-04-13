import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import matplotlib.patches as mpatches
from matplotlib.transforms import Bbox

# ============================================================
# PUBLICATION STYLE (MATCHES YOUR "PREVIOUS" SCRIPT)
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
OUT_DIR = Path("fourier_globalfit_from_results")
OUT_DIR.mkdir(exist_ok=True)

# ============================================================
# SELECT COLUMN TO FIT
# ============================================================
MECH_COLUMN = "Mech_max_ratio"   # change freely

# ============================================================
# LoadD filter (empty => ALL)
# ============================================================
LOADD_FILTER = [0.7]
LOAD_TOL = 1e-9

# ============================================================
# IMPORTANT: NO "/max" NORMALIZATION HERE
# We fit raw y by adding scale+offset: y = y0 + A * sigmoid(...)
# ============================================================

def _to_float(x):
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

# ============================================================
# SIGMOID MODELS (RAW-Y READY)
# ============================================================
def fo_eff_from_params(X, a, b, c):
    Fo_diff = X[:, 0]
    Fo_visc = X[:, 1]
    Ramp = X[:, 2]
    return (Fo_diff ** a) * (Fo_visc ** b) * np.exp(c * Ramp)

def logistic_model_asym(logFo_eff, k, x0, v):
    # outputs in (0, 1]
    return 1.0 / (1.0 + np.exp(-k * (logFo_eff - x0))) ** v

def master_sigmoid_unit(X, a, b, c, k, x0, v):
    Fo_eff = fo_eff_from_params(X, a, b, c)
    return logistic_model_asym(np.log(Fo_eff), k, x0, v)

def master_sigmoid_scaled(X, a, b, c, k, x0, v, A, y0):
    # raw fit: allow amplitude + offset
    return y0 + A * master_sigmoid_unit(X, a, b, c, k, x0, v)

def global_sigmoid_fit_raw(df):
    """
    Fits RAW Mech without dividing by max:
        y = y0 + A * sigmoid(Fo_eff)

    Also uses a better x0 initial guess based on a proxy Fo_eff.
    Works robustly for tags 63 / 95 / 99.
    """
    X_data = df[["Fo_diff", "Fo_visc", "Ramp"]].values
    y_raw = df["Mech"].values

    # x0 guess from a proxy Fo_eff0
    Fo_eff0 = fo_eff_from_params(X_data, 0.3, 0.3, -0.01)
    x0_0 = float(np.median(np.log(Fo_eff0)))

    # initial A,y0 from data range
    y_min = float(np.min(y_raw))
    y_max = float(np.max(y_raw))
    A0 = max(1e-12, y_max - y_min)
    y00 = y_min

    p0 = [0.3, 0.3, -0.01, 5.0, x0_0, 1.5, A0, y00]

    # bounds (mild) to keep fits sane across tags
    lower = [-3.0, -3.0, -2.0,  0.1, x0_0 - 10.0, 0.2, 0.0, -np.inf]  # A >= 0
    upper = [ 3.0,  3.0,  2.0, 50.0, x0_0 + 10.0, 5.0, np.inf, np.inf]

    popt, pcov = curve_fit(
        master_sigmoid_scaled,
        X_data, y_raw,
        p0=p0,
        bounds=(lower, upper),
        maxfev=120000
    )
    return popt, pcov

# ============================================================
# Fourier meta loading + mapping
# ============================================================
def load_fourier_meta(tag: str) -> dict:
    p = META_DIR / f"fourier_lookup_{tag}.json"
    if not p.exists():
        raise FileNotFoundError(f"Missing lookup: {p}")

    with p.open("r") as f:
        meta = json.load(f)

    Fo_diff_map = {float(k): float(v) for k, v in meta["Fo_diff"].items()}
    Fo_visc_map = {float(k): float(v) for k, v in meta["Fo_visc"].items()}
    return {
        "tag": tag,
        "Fo_diff_map": Fo_diff_map,
        "Fo_visc_map": Fo_visc_map,
    }

def pick_col(df: pd.DataFrame, candidates, required=True):
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise KeyError(
            f"Could not find required column. Tried: {candidates}. "
            f"Available columns: {list(df.columns)}"
        )
    return None

def map_to_Fo(df: pd.DataFrame, meta: dict) -> pd.DataFrame:
    diff_col = pick_col(df, ["Diff", "D", "diff", "D_diff", "Diffusion", "Diff_value"])
    tau_col  = pick_col(df, ["Tau", "tau", "Tau_base", "tau_base", "Tau_value"])

    df2 = df.copy()
    df2["_D_"] = df2[diff_col].apply(_to_float)
    df2["_tau_"] = df2[tau_col].apply(_to_float)

    df2["Fo_diff"] = df2["_D_"].map(meta["Fo_diff_map"])
    df2["Fo_visc"] = df2["_tau_"].map(meta["Fo_visc_map"])

    if df2["Fo_diff"].isna().any() or df2["Fo_visc"].isna().any():
        missing_d = sorted(df2.loc[df2["Fo_diff"].isna(), "_D_"].dropna().unique())
        missing_t = sorted(df2.loc[df2["Fo_visc"].isna(), "_tau_"].dropna().unique())
        raise ValueError(
            "Some Diff/Tau values were not found in the Fourier lookup maps.\n"
            f"Missing D values (examples): {missing_d[:10]}\n"
            f"Missing tau values (examples): {missing_t[:10]}\n"
            "Fix by ensuring the values in results_systems.csv match exactly the keys used to build fourier_lookup_*.json."
        )

    return df2.drop(columns=["_D_", "_tau_"])

# ============================================================
# Plot style mapping (consistent)
# ============================================================
LINESTYLE_BY_TAG = {"99": "-", "95": "--", "63": ":"}
MARKER_BY_TAG    = {"99": "o", "95": "s",  "63": "^"}
COLOR_BY_TAG     = {"99": "C0", "95": "C1", "63": "C2"}  # keep matplotlib defaults

# ============================================================
# CORE ROUTINE (AVG / NON-AVG)
# ============================================================
def run_one_system_class(df_all, is_avg, label, metas):
    print(f"\n▶ Running Fo-compare collapse for: {label}")

    df = df_all[df_all["is_avg"] == is_avg].copy()

    if "LoadD" not in df.columns:
        df["LoadD"] = np.nan

    if LOADD_FILTER:
        df = df[df["LoadD"].apply(lambda x: _float_in(_to_float(x), LOADD_FILTER, tol=LOAD_TOL))].copy()
        print(f"  ▶ LoadD filter active: {LOADD_FILTER}  (remaining N={len(df)})")

    if df.empty:
        print(f"  - No data left after LoadD filter for {label}")
        return

    load_levels = sorted([x for x in df["LoadD"].dropna().unique()])
    if len(load_levels) == 0:
        load_levels = [None]

    for load_d in load_levels:
        if load_d is None:
            dfL0 = df.copy()
            load_str = "ALL"
            load_text = "LoadD: (not in CSV)"
        else:
            dfL0 = df[df["LoadD"] == load_d].copy()
            load_str = f"{load_d:.3f}".replace(".", "p")
            load_text = rf"LoadD = {load_d:.3f}"

        dfL0 = dfL0.replace([np.inf, -np.inf], np.nan)
        dfL0 = dfL0.dropna(subset=["Ramp", "Mech"])
        dfL0 = dfL0[dfL0["Mech"] > 0]

        if dfL0.empty:
            print(f"  - No valid data for {label} at LoadD={load_d}")
            continue

        print(f"  ▶ Load group: {load_text}  (N={len(dfL0)})")

        # ------------------------------------------------------
        # FIT EACH Fo-DEFINITION (99/95/63) ON RAW Mech
        # ------------------------------------------------------
        fit_results = []

        for meta in metas:
            tag = meta["tag"]
            dfL = map_to_Fo(dfL0, meta)

            dfL = dfL.replace([np.inf, -np.inf], np.nan)
            dfL = dfL.dropna(subset=["Fo_diff", "Fo_visc", "Ramp", "Mech"])
            dfL = dfL[(dfL["Fo_diff"] > 0) & (dfL["Fo_visc"] > 0) & (dfL["Mech"] > 0)]
            dfL["Fo_diff"] = dfL["Fo_diff"]/2
            dfL["Fo_visc"] = dfL["Fo_visc"]/2

            if dfL.empty:
                print(f"    - No valid mapped data for Fo^{tag} at LoadD={load_d}")
                continue

            popt, pcov = global_sigmoid_fit_raw(dfL)
            a, b, c, k, x0, v, A, y0 = popt

            X = dfL[["Fo_diff", "Fo_visc", "Ramp"]].values
            dfL["Fo_eff"] = fo_eff_from_params(X, a, b, c)

            fit_results.append({
                "tag": tag,
                "params": (a, b, c, k, x0, v, A, y0),
                "df": dfL,
            })

        if len(fit_results) == 0:
            print(f"  - No fits produced for {label} at LoadD={load_d}")
            continue

        # ------------------------------------------------------
        # SAVE PARAMETER TABLE (one row per approach)
        # ------------------------------------------------------
        fit_label = f"{label}_LoadD_{load_str}_FoCompare"
        rows = []
        for fr in fit_results:
            tag = fr["tag"]
            a, b, c, k, x0, v, A, y0 = fr["params"]
            ab = a / b if abs(b) > 1e-15 else np.nan
            bc = b / c if abs(c) > 1e-15 else np.nan
            rows.append({
                "Fo_tag": tag,
                "a": a, "b": b, "c": c, "k": k, "x0": x0, "v": v,
                "A": A, "y0": y0,
                "a_over_b": ab,
                "b_over_c": bc,
            })
        pd.DataFrame(rows).to_csv(OUT_DIR / f"fit_parameters_{fit_label}.csv", index=False)

        # ------------------------------------------------------
        # FIGURE: left = data+3 curves ; right = big 3-col param box
        # (same figure size and styling as your other script)
        # ------------------------------------------------------
        fig = plt.figure(figsize=(16, 6))
        gs = fig.add_gridspec(nrows=1, ncols=2, width_ratios=[1.0, 0.62])

        ax = fig.add_subplot(gs[0, 0])
        legax = fig.add_subplot(gs[0, 1])
        legax.axis("off")

        # --- data points: same color family as curve, lighter (alpha) ---
        point_alpha = 0.22
        point_size = 120
        edge_lw = 0.8

        for fr in fit_results:
            tag = fr["tag"]
            dfP = fr["df"]
            col = COLOR_BY_TAG.get(tag, "k")

            ax.scatter(
                dfP["Fo_eff"].values,
                dfP["Mech"].values,
                marker=MARKER_BY_TAG.get(tag, "o"),
                s=point_size,
                facecolor=col,
                edgecolor=col,
                linewidth=edge_lw,
                alpha=point_alpha,
                zorder=3
            )

        # --- plot each fitted curve over its own x-range ---
        xmin_all, xmax_all = np.inf, -np.inf

        for fr in fit_results:
            tag = fr["tag"]
            a, b, c, k, x0, v, A, y0 = fr["params"]
            dfP = fr["df"]
            col = COLOR_BY_TAG.get(tag, "k")

            Fo_min_i = float(dfP["Fo_eff"].min())
            Fo_max_i = float(dfP["Fo_eff"].max())
            xmin_all = min(xmin_all, Fo_min_i)
            xmax_all = max(xmax_all, Fo_max_i)

            Fo_line_i = np.logspace(np.log10(Fo_min_i), np.log10(Fo_max_i), 400)
            y_line_i = y0 + A * logistic_model_asym(np.log(Fo_line_i), k, x0, v)

            ax.plot(
                Fo_line_i, y_line_i,
                color=col,
                linestyle=LINESTYLE_BY_TAG.get(tag, "-"),
                lw=3.5,   # matches your plot style
                zorder=10
            )

        ax.set_xscale("log")
        ax.set_xlim(xmin_all * 0.8, xmax_all * 1.2)
        ax.set_xlabel(r"$Fo_{\mathrm{eff}}$")

        # RAW y label (no "Normalized" wording)
        if MECH_COLUMN == "Mech_Ratio":
            ax.set_ylabel(r"$\varepsilon^{MS}_{\infty} / \varepsilon_{\infty}$")
        else:
            ax.set_ylabel(r"$\varepsilon^{MS}_{\infty} / \varepsilon_{\infty}$")

        ax.grid(True, ls="--", alpha=0.4)

        # ------------------------------------------------------
        # RIGHT: big legend-like parameter box (3 columns)
        # ------------------------------------------------------
        fig.tight_layout()
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()

        order = {"99": 0, "95": 1, "63": 2}
        fit_results_sorted = sorted(fit_results, key=lambda fr: order.get(fr["tag"], 99))

        SHIFT_LEFT = -0.06
        COL_GAP    = 0.38
        HEADER_Y   = 0.9
        LINE_LEN   = 0.18
        MARKER_DX  = 0.205

        ytop = 0.85
        dy   = 0.070

        PAD_X = 0.040
        PAD_Y = 0.075
        TOP_PAD_EXTRA = 0.030

        x0 = 0.05 + SHIFT_LEFT
        xcols = [x0 + i * COL_GAP for i in range(3)]

        text_artists = []
        header_artists = []

        for i, fr in enumerate(fit_results_sorted[:3]):
            tag = fr["tag"]
            a, b, c, k, x0p, v, A, y0 = fr["params"]
            ab = a / b if abs(b) > 1e-15 else np.nan
            bc = b / c if abs(c) > 1e-15 else np.nan

            col = COLOR_BY_TAG.get(tag, "k")
            ls  = LINESTYLE_BY_TAG.get(tag, "-")
            mk  = MARKER_BY_TAG.get(tag, "o")

            (line_artist,) = legax.plot(
                [xcols[i], xcols[i] + LINE_LEN], [HEADER_Y, HEADER_Y],
                transform=legax.transAxes,
                color=col, linestyle=ls, lw=4,
                solid_capstyle="round",
                zorder=8, clip_on=False
            )
            (marker_artist,) = legax.plot(
                [xcols[i] + MARKER_DX], [HEADER_Y],
                transform=legax.transAxes,
                marker=mk, color=col, markersize=10, lw=0,
                zorder=9, clip_on=False
            )
            header_artists.extend([line_artist, marker_artist])

            lines = [
                (rf"$Fo^{{{tag}}}$", "black"),
                (rf"$a={a:.2f}$", "black"),
                (rf"$b={b:.2f}$", "black"),
                (rf"$c={c:.3f}$", "black"),
                (rf"$k={k:.2f}$", "black"),
                (rf"$x_0={x0p:.2f}$", "black"),
                (rf"$v={v:.2f}$", "black"),
                (rf"$A={A:.3f}$", "black"),
                (rf"$y_0={y0:.3f}$", "black"),
                (rf"$a/b={ab:.2f}$", "red"),
                (rf"$b/c={bc:.2f}$", "red"),
            ]

            for j, (s, color) in enumerate(lines):
                t = legax.text(
                    xcols[i], ytop - j * dy, s,
                    transform=legax.transAxes,
                    ha="left", va="top",
                    fontsize=22,
                    color=color,
                    zorder=10
                )
                text_artists.append(t)

        # --- bbox from ALL artists (text + header) ---
        # --- bbox from ALL artists (text + header) ---
        fig.tight_layout()
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()

        all_artists = text_artists + header_artists
        bb_disp = Bbox.union([a.get_window_extent(renderer=renderer) for a in all_artists])
        bb_axes = bb_disp.transformed(legax.transAxes.inverted())

        # padding
        x0b = bb_axes.x0 - PAD_X
        y0b = bb_axes.y0 - PAD_Y
        x1b = bb_axes.x1 + PAD_X
        y1b = bb_axes.y1 + PAD_Y + TOP_PAD_EXTRA

        # independent SHIFT (this is what you want)
        SHIFT_X = -0.00   # left (try -0.02 first)
        SHIFT_Y =  0.00   # up/down
        x0b += SHIFT_X; x1b += SHIFT_X
        y0b += SHIFT_Y; y1b += SHIFT_Y

        patch = mpatches.FancyBboxPatch(
            (x0b, y0b),
            x1b - x0b, y1b - y0b,
            transform=legax.transAxes,
            boxstyle="round,pad=0.02",
            facecolor="none",        # or "white"
            edgecolor="black",
            linewidth=1.8,
            zorder=0,
            clip_on=False            # IMPORTANT if you want it to go outside legax
        )
        legax.add_patch(patch)


        # put artists above patch
        for a in all_artists:
            a.set_zorder(5)


        fig.savefig(OUT_DIR / f"global_collapse_{fit_label}.png", dpi=300)
        plt.close(fig)

        print(f"✔ Saved Fo-compare collapse for {fit_label}")


# ============================================================
# MAIN
# ============================================================
def main():
    # prepared for all criteria: 63, 95, 99
    metas = [load_fourier_meta("99"), load_fourier_meta("95"), load_fourier_meta("63")]

    df = pd.read_csv(IN_CSV)
    if MECH_COLUMN not in df.columns:
        raise KeyError(f"MECH_COLUMN='{MECH_COLUMN}' not found. Available: {list(df.columns)}")
    df = df.rename(columns={MECH_COLUMN: "Mech"})

    run_one_system_class(df, is_avg=False, label="non_avg", metas=metas)
    run_one_system_class(df, is_avg=True,  label="avg", metas=metas)

if __name__ == "__main__":
    main()
