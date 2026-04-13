import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.legend_handler import HandlerPatch
from scipy.optimize import curve_fit

# ============================================================
# PUBLICATION STYLE (UNCHANGED)
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

# ============================================================
# SELECT COLUMN TO FIT
# ============================================================
MECH_COLUMN = "Mech_max_ratio"   # change freely

# ============================================================
# WHICH Fo-DEFINITION TO USE HERE (easy switch: "99" <-> "63")
# ============================================================
FO_TAG = "99"  # <- set to "63" for 63%, "99" for 99%

# ============================================================
# OUTPUT
# ============================================================
OUT_DIR = Path(f"fourier_globalfit_from_results_Fo{FO_TAG}")
OUT_DIR.mkdir(exist_ok=True)

# ============================================================
# LoadD filter (empty => ALL)
# ============================================================
LOADD_FILTER = [0.7]
LOAD_TOL = 1e-9

# ============================================================
# FLAGS
# ============================================================
# 1) If True: fit on y normalized to max=1 (per load group)
NORMALIZE_MECH_TO_1 = False

# 2) If False normalization: allow fitting y = y0 + A * sigmoid(...)
#    (recommended if your raw y is e.g. 0.05..0.35 etc.)
FIT_SCALE_OFFSET_WHEN_RAW = True


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def _to_float(x):
    """0.7 / '0.7' / '0p7' -> float, else None"""
    try:
        if isinstance(x, str):
            return float(x.replace("p", "."))
        return float(x)
    except Exception:
        return None

def _float_in(x: float | None, allowed, tol=1e-9) -> bool:
    """True if x matches any value in allowed within tol"""
    if x is None:
        return False
    for a in allowed:
        aa = _to_float(a)
        if aa is None:
            continue
        if abs(x - aa) <= tol:
            return True
    return False

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

# ------------------------------------------------------------
# Fourier meta loading + mapping
# ------------------------------------------------------------
def load_fourier_meta(tag: str) -> dict:
    p = META_DIR / f"fourier_lookup_{tag}.json"
    if not p.exists():
        raise FileNotFoundError(f"Could not find Fourier lookup file: {p}")

    with p.open("r") as f:
        meta = json.load(f)

    if "Fo_diff" not in meta or "Fo_visc" not in meta:
        raise KeyError(
            f"{p} must contain keys 'Fo_diff' and 'Fo_visc'. "
            f"Found keys: {list(meta.keys())}"
        )

    Fo_diff_map = {float(k): float(v) for k, v in meta["Fo_diff"].items()}
    Fo_visc_map = {float(k): float(v) for k, v in meta["Fo_visc"].items()}

    return {
        "tag": tag,
        "Fo_diff_map": Fo_diff_map,
        "Fo_visc_map": Fo_visc_map,
    }

def map_to_Fo(df: pd.DataFrame, meta: dict) -> pd.DataFrame:
    """
    Takes raw Diff + Tau columns and maps them to Fo_diff + Fo_visc.
    Ignores any pre-existing Fo_* columns in the CSV.
    """
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
# SIGMOID MODELS
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
    """unit sigmoid in [0,1]"""
    Fo_eff = fo_eff_from_params(X, a, b, c)
    return logistic_model_asym(np.log(Fo_eff), k, x0, v)

def master_sigmoid_scaled(X, a, b, c, k, x0, v, A, y0):
    """scaled sigmoid: y0 + A * unit_sigmoid"""
    return y0 + A * master_sigmoid_unit(X, a, b, c, k, x0, v)

def global_sigmoid_fit(df):
    """
    - If NORMALIZE_MECH_TO_1: fit unit sigmoid to y/max(y).
    - Else: if FIT_SCALE_OFFSET_WHEN_RAW: fit y0 and A too.
            otherwise: fit unit sigmoid directly to raw y (usually bad).
    """
    X_data = df[["Fo_diff", "Fo_visc", "Ramp"]].values
    y_raw = df["Mech"].values

    # x0 initial guess based on a proxy Fo_eff scale:
    Fo_eff0 = fo_eff_from_params(X_data, 0.3, 0.3, -0.01)
    x0_0 = float(np.median(np.log(Fo_eff0)))

    # shared initial guesses for (a,b,c,k,x0,v)
    p0_base = [0.3, 0.3, -0.01, 5.0, x0_0, 1.5]
    lower_base = [-3.0, -3.0, -2.0,  0.1, x0_0 - 10.0, 0.2]
    upper_base = [ 3.0,  3.0,  2.0, 50.0, x0_0 + 10.0, 5.0]

    if NORMALIZE_MECH_TO_1:
        mech_max = np.nanmax(y_raw)
        if not np.isfinite(mech_max) or mech_max <= 0:
            raise ValueError(f"Invalid mech_max: {mech_max}")
        y_data = y_raw / mech_max
        print("MAXIMUM MS (used for normalization): ", mech_max)

        popt, pcov = curve_fit(
            master_sigmoid_unit,
            X_data, y_data,
            p0=p0_base,
            bounds=(lower_base, upper_base),
            maxfev=60000
        )
        return popt, pcov, "unit"

    # --- raw fit ---
    print("MAXIMUM MS (not used): ", np.nanmax(y_raw))

    if FIT_SCALE_OFFSET_WHEN_RAW:
        # Fit A and y0 so the model can match raw scale
        y_min = float(np.nanmin(y_raw))
        y_max = float(np.nanmax(y_raw))
        A0 = max(1e-12, y_max - y_min)
        y00 = y_min

        p0 = p0_base + [A0, y00]
        lower = lower_base + [0.0, -np.inf]     # A >= 0
        upper = upper_base + [np.inf, np.inf]

        popt, pcov = curve_fit(
            master_sigmoid_scaled,
            X_data, y_raw,
            p0=p0,
            bounds=(lower, upper),
            maxfev=120000
        )
        return popt, pcov, "scaled"

    # If you insist on raw fit with unit sigmoid (usually poor)
    popt, pcov = curve_fit(
        master_sigmoid_unit,
        X_data, y_raw,
        p0=p0_base,
        bounds=(lower_base, upper_base),
        maxfev=60000
    )
    return popt, pcov, "unit_raw_direct"

# ============================================================
# LEGEND RECT HANDLER
# ============================================================
class FixedSizeRectHandler(HandlerPatch):
    def __init__(self, width_pt, height_pt, **kwargs):
        super().__init__(**kwargs)
        self.width_pt = width_pt
        self.height_pt = height_pt

    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        fig = legend.get_figure()
        dpi = fig.dpi
        w = self.width_pt / dpi
        h = self.height_pt / dpi
        x = xdescent + (width - w) / 2
        y = ydescent + (height - h) / 2

        return [Rectangle((x, y), w, h,
                          facecolor=orig_handle.get_facecolor(),
                          edgecolor=orig_handle.get_edgecolor(),
                          lw=orig_handle.get_linewidth(),
                          transform=trans)]

# ============================================================
# CORE ROUTINE (AVG / NON-AVG)
# ============================================================
def run_one_system_class(df_all, is_avg, label, meta):
    print(f"\n▶ Running collapse for: {label}  (Fo-tag={meta['tag']})")

    df = df_all[df_all["is_avg"] == is_avg].copy()

    # Ensure LoadD exists
    if "LoadD" not in df.columns:
        df["LoadD"] = np.nan

    # Apply LoadD filter
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

        # --- MAP Diff/Tau -> Fo_* ---
        dfL = map_to_Fo(dfL0, meta)

        # ---------------- CLEAN ----------------
        dfL = dfL.replace([np.inf, -np.inf], np.nan)
        dfL = dfL.dropna(subset=["Fo_diff", "Fo_visc", "Ramp", "Mech"])
        dfL = dfL[(dfL["Mech"] > 0) & (dfL["Fo_diff"] > 0) & (dfL["Fo_visc"] > 0)]
        dfL["Fo_diff"] = dfL["Fo_diff"]/2
        dfL["Fo_visc"] = dfL["Fo_visc"]/2
        if dfL.empty:
            print(f"  - No valid data for {label} at LoadD={load_d}")
            continue

        print(f"  ▶ Load group: {load_text}  (N={len(dfL)})")

        # ---------------- FIT ----------------
        popt, pcov, fit_mode = global_sigmoid_fit(dfL)

        if fit_mode == "scaled":
            a, b, c, k, x0, v, A, y0 = popt
        else:
            a, b, c, k, x0, v = popt
            A, y0 = None, None

        X = dfL[["Fo_diff", "Fo_visc", "Ramp"]].values
        dfL["Fo_eff"] = fo_eff_from_params(X, a, b, c)

        # y used for plotting
        if NORMALIZE_MECH_TO_1:
            dfL["y_plot"] = dfL["Mech"] / dfL["Mech"].max()
            ylab = r"Normalized $\varepsilon^{MS}_{\infty} / \varepsilon_{\infty}$"
        else:
            dfL["y_plot"] = dfL["Mech"]
            ylab = r"$\varepsilon^{MS}_{\infty} / \varepsilon_{\infty}$"

        # ---------------- SAVE FIT ----------------
        fit_label = f"{label}_LoadD_{load_str}"

        if fit_mode == "scaled":
            params_out = ["a", "b", "c", "k", "x0", "v", "A", "y0"]
            values_out = [a, b, c, k, x0, v, A, y0]
        else:
            params_out = ["a", "b", "c", "k", "x0", "v"]
            values_out = [a, b, c, k, x0, v]

        pd.DataFrame({
            "parameter": params_out,
            "value": values_out,
        }).to_csv(OUT_DIR / f"fit_parameters_{fit_label}.csv", index=False)

        # ---------------- COLORS / MARKERS ----------------
        Fo_diffs_unique = sorted(dfL["Fo_diff"].round(3).unique())
        Fo_viscs_unique = sorted(dfL["Fo_visc"].round(3).unique())
        ramps_sorted = sorted(dfL["Ramp"].unique())

        colors = plt.cm.viridis(np.linspace(0, 1, len(Fo_diffs_unique)))
        color_map = {d: colors[i] for i, d in enumerate(Fo_diffs_unique)}

        markers = ["o", "s", "^", "D", "p", "X", "v", "<", ">"]
        marker_map = {vv: markers[i % len(markers)] for i, vv in enumerate(Fo_viscs_unique)}

        alpha_map = {0.0: 1.0, 10.0: 0.7, 30.0: 0.4, 50.0: 0.1}

        # ======================================================
        # FIGURE + GRID
        # ======================================================
        fig = plt.figure(figsize=(16, 6))

        gs = fig.add_gridspec(
            nrows=2, ncols=2,
            width_ratios=[1.0, 0.55],
            height_ratios=[1.0, 0.25],
        )

        ax = fig.add_subplot(gs[:, 0])

        for _, row in dfL.iterrows():
            ax.scatter(
                row["Fo_eff"],
                row["y_plot"],
                facecolor=color_map[round(row["Fo_diff"], 3)],
                edgecolor="black",
                marker=marker_map[round(row["Fo_visc"], 3)],
                s=120,
                alpha=alpha_map.get(row["Ramp"], 0.6),
                linewidth=1.2,
                zorder=9
            )

        Fo_line = np.logspace(
            np.log10(dfL["Fo_eff"].min()),
            np.log10(dfL["Fo_eff"].max()),
            400
        )

        # predicted curve
        if fit_mode == "scaled":
            y_line = master_sigmoid_scaled(
                np.column_stack([Fo_line, Fo_line*0 + np.median(dfL["Fo_visc"]), Fo_line*0 + np.median(dfL["Ramp"])]),
                a, b, c, k, x0, v, A, y0
            )
            # NOTE: above is just to keep a callable shape; we want the real line:
            # build X_line properly using representative Ramp, but varying Fo_eff needs both Fo_diff and Fo_visc.
            # Instead: directly evaluate the unit sigmoid on log(Fo_line) (since line is in Fo_eff space):
            # We plot f(logFo_eff) in Fo_eff-space:
            y_line = y0 + A * logistic_model_asym(np.log(Fo_line), k, x0, v)
        else:
            y_line = logistic_model_asym(np.log(Fo_line), k, x0, v)
            if NORMALIZE_MECH_TO_1:
                pass
            else:
                # this is the "unit to raw" mismatch case; still plot what was fitted
                pass

        ax.plot(Fo_line, y_line, "--k", lw=3, zorder=10)

        ax.set_xscale("log")
        ax.set_xlabel(r"$Fo_{\mathrm{eff}}$")
        ax.set_ylabel(ylab)
        ax.grid(True, ls="--", alpha=0.4)

        # ------------------------------------------------------
        # LEGEND AXIS
        # ------------------------------------------------------
        legend_ax = fig.add_subplot(gs[0, 1])
        legend_ax.axis("off")

        handles, labels = [], []

        for d in Fo_diffs_unique:
            handles.append(Rectangle((0, 0), 1, 1,
                                     facecolor=color_map[d],
                                     edgecolor="black"))
            labels.append(f"{d:.2f}")

        for fv in Fo_viscs_unique:
            handles.append(
                plt.Line2D([], [], marker=marker_map[fv],
                           color="black", lw=0, markersize=12)
            )
            labels.append(f"{fv:.2f}")

        for r in ramps_sorted:
            handles.append(
                Rectangle((0, 0), 1, 1,
                          facecolor="black",
                          alpha=alpha_map.get(r, 0.6))
            )
            labels.append(f"{int(r)}%")

        title_text = (
            rf"$Fo_{{\chi}}^{{{meta['tag']}}}\hspace{{2.5cm}}Fo_{{\tau}}^{{{meta['tag']}}}\hspace{{2.5cm}}T_r$"
        )

        big_legend = legend_ax.legend(
            handles, labels,
            title=title_text,
            ncol=3,
            fontsize=16,
            title_fontsize=20,
            handlelength=1.4,
            columnspacing=2,
            frameon=True,
            loc="center",
            bbox_to_anchor=(0.78, 0.76),
            bbox_transform=fig.transFigure,
        )

        big_legend.get_frame().set_facecolor("white")
        big_legend.get_frame().set_edgecolor("black")
        big_legend.get_frame().set_linewidth(1.5)
        big_legend.get_frame().set_alpha(1.0)

        import matplotlib.patches as mpatches
        from matplotlib.transforms import Bbox

        eq_ax = fig.add_subplot(gs[1, 1])
        eq_ax.axis("off")

        text_Feff = (
            r"$Fo_{\mathrm{eff}} = "
            rf"(Fo_{{\chi}}^{{{meta['tag']}}})^{{a}}\,"
            rf"(Fo_{{\tau}}^{{{meta['tag']}}})^{{b}}\,"
            r"e^{(c\,T_r)}$"
        )

        if fit_mode == "scaled":
            text_sigmoid = (
                r"$f(Fo_{\mathrm{eff}}) = y_0 + A \left[1 + \exp\!\left(-k\,[\ln(Fo_{\mathrm{eff}})-x_0]\right)\right]^{-v}$"
            )
        else:
            text_sigmoid = (
                r"$f(Fo_{\mathrm{eff}}) = "
                r"\left[1 + \exp\!\left(-k\,[\ln(Fo_{\mathrm{eff}})-x_0]\right)\right]^{-v}$"
            )

        if fit_mode == "scaled":
            param_labels = ["$a$", "$b$", "$c$", "$k$", "$x_0$", "$v$", "$A$", "$y_0$"]
            param_values = [a, b, c, k, x0, v, A, y0]
        else:
            param_labels = ["$a$", "$b$", "$c$", "$k$", "$x_0$", "$v$"]
            param_values = [a, b, c, k, x0, v]

        # format parameters into two columns nicely
        half = int(np.ceil(len(param_labels)/2))
        left_pairs  = list(zip(param_labels[:half], param_values[:half]))
        right_pairs = list(zip(param_labels[half:], param_values[half:]))

        left_text  = "\n".join([rf"{n} = {vv:.3f}" for n, vv in left_pairs])
        right_text = "\n".join([rf"{n} = {vv:.3f}" for n, vv in right_pairs])

        x_center = 0.78
        y_top    = 0.55

        t1 = eq_ax.text(x_center, y_top, text_Feff,
                        transform=fig.transFigure, ha="center", va="top", fontsize=20)
        t2 = eq_ax.text(x_center, y_top - 0.075, text_sigmoid,
                        transform=fig.transFigure, ha="center", va="top", fontsize=18)

        y_params = y_top - 0.20
        tL = eq_ax.text(x_center - 0.13, y_params, left_text,
                        transform=fig.transFigure, ha="left", va="top", fontsize=20, linespacing=1.5)
        tR = eq_ax.text(x_center + 0.04, y_params, right_text,
                        transform=fig.transFigure, ha="left", va="top", fontsize=20, linespacing=1.5)

        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        bboxes = [t.get_window_extent(renderer=renderer) for t in (t1, t2, tL, tR)]
        bb = Bbox.union(bboxes)

        pad_px = -10
        bb_padded = Bbox.from_extents(bb.x0 - pad_px, bb.y0 - pad_px, bb.x1 + pad_px, bb.y1 + pad_px)
        bb_fig = bb_padded.transformed(fig.transFigure.inverted())

        patch = mpatches.FancyBboxPatch(
            (bb_fig.x0, bb_fig.y0),
            bb_fig.width, bb_fig.height,
            transform=fig.transFigure,
            boxstyle="round,pad=0.02",
            facecolor="None",
            edgecolor="black",
            alpha=1.0,
            linewidth=1.5,
            zorder=min(t.get_zorder() for t in (t1, t2, tL, tR)) - 1,
        )
        fig.add_artist(patch)


        # ======================================================
        # TOP-LEFT REGIME ANNOTATION (axes coords, independent of data)
        # ======================================================
        # ======================================================
        # TOP-LEFT TIME-SCALE REGIME ANNOTATION (exact wording)
        # ======================================================

        x_text = 0.28   # left margin (axes coords)
        x_arrow_l = 0.28
        x_arrow_r = 0.44

        y_top = 0.8
        dy = 0.055

        # --- First line ---
        ax.text(
            x_text, y_top,
            r"$T_c \gg T_{\chi},\,T_{\tau}$",
            transform=ax.transAxes,
            ha="left", va="center",
            fontsize=20,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=2.0),
            zorder=20,
        )

        # arrow to the right
        ax.annotate(
            "",
            xy=(x_arrow_r, y_top - 1.5*dy),
            xytext=(x_text, y_top - 1.5*dy),
            xycoords=ax.transAxes,
            arrowprops=dict(arrowstyle="->", lw=2.5, color="black"),
            zorder=19,
        )

        # --- Second line ---
        ax.text(
            x_text, y_top - 4.5*dy,
            r"$T_c \ll T_{\chi},\,T_{\tau}$",
            transform=ax.transAxes,
            ha="left", va="center",
            fontsize=20,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=2.0),
            zorder=20,
        )

        # arrow to the left
        ax.annotate(
            "",
            xy=(x_arrow_l, y_top - 3*dy),
            xytext=(x_arrow_r, y_top - 3*dy),
            xycoords=ax.transAxes,
            arrowprops=dict(arrowstyle="->", lw=2.5, color="black"),
            zorder=19,
        )



        ax.tick_params(axis="x", which="major", pad=6)
        fig.tight_layout()

        fig.savefig(OUT_DIR / f"global_collapse_{fit_label}.png", dpi=300)
        plt.close(fig)

        print(f"✔ Saved collapse for {fit_label}")

# ============================================================
# MAIN
# ============================================================
def main():
    # Switching 99<->63 is just changing FO_TAG above,
    # as long as fourier_lookup_<tag>.json exists in META_DIR.
    meta = load_fourier_meta(FO_TAG)

    df = pd.read_csv(IN_CSV)
    # df = pd.read_csv(RESULTS_CSV)

    # ---- robust numeric load column ----
    df["LoadD_num"] = df["LoadD"].apply(_to_float)

    # If you want to see what's in there (debug)
    # print("Unique LoadD raw:", sorted(df["LoadD"].astype(str).unique())[:20])
    # print("Unique LoadD_num:", sorted(df["LoadD_num"].dropna().unique()))

    if LOADD_FILTER:
        allowed = np.array([_to_float(v) for v in LOADD_FILTER], dtype=float)
        df = df[df["LoadD_num"].apply(lambda x: np.any(np.isclose(x, allowed, atol=LOAD_TOL, rtol=0.0)) if np.isfinite(x) else False)].copy()
        print(f"▶ LoadD filter active: {LOADD_FILTER}  (remaining N={len(df)})")

    if MECH_COLUMN not in df.columns:
        raise KeyError(f"MECH_COLUMN='{MECH_COLUMN}' not found in CSV. Available: {list(df.columns)}")
    df = df.rename(columns={MECH_COLUMN: "Mech"})

    run_one_system_class(df, is_avg=False, label="non_avg", meta=meta)
    run_one_system_class(df, is_avg=True,  label="avg", meta=meta)

if __name__ == "__main__":
    main()
