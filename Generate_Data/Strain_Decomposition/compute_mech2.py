import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit



# =========================================================
# CONFIG DATACLASSES
# =========================================================
@dataclass
class PlasticConfig:
    """
    External plastic-reference configuration.

    If enabled=False, plastic strain is set to zero.
    If enabled=True, partA_csv and partB_csv should be provided.
    """
    enabled: bool = True
    partA_csv: str | Path | None = None
    partB_csv: str | Path | None = None
    strict: bool = True
    eps_load: float = 1e-12
    load_col: str = "load_d"
    moist_col: str = "moist_max"
    plastic_col: str = "slip_final"
    use_abs_load: bool = False


@dataclass
class ViscoelasticConfig:
    """
    External viscoelastic-reference configuration.
    """
    summary_csv: str | Path
    tau: float = 0.001
    degree: int = 2
    kv_num: int = 4


# =========================================================
# GENERIC UTILITIES
# =========================================================
def seg_bc(arr, i1, i2):
    """Baseline-corrected array slice arr[i1:i2+1], starts at 0."""
    s = np.asarray(arr[i1:i2 + 1], dtype=float).copy()
    return s - s[0] if len(s) else s


def resize_to(arr, new_len):
    """Resize array arr to length new_len using linear interpolation."""
    arr = np.asarray(arr, dtype=float)
    if len(arr) == 0 or new_len <= 0:
        return np.zeros(new_len, dtype=float)
    xp = np.linspace(0, 1, len(arr))
    return np.interp(np.linspace(0, 1, new_len), xp, arr)


def remove_end_bias(arr):
    """Subtract linear trend so arr[0]==0 and arr[-1]==0."""
    arr = np.asarray(arr, dtype=float)
    if len(arr) < 2:
        return arr.copy()
    trend = np.linspace(0.0, arr[-1], len(arr))
    return arr - trend


def close_cycle_trend(arr):
    """
    Remove linear drift from one cycle so that start and end match.
    Keeps the first point unchanged and forces last-first drift to zero.
    """
    arr = np.asarray(arr, dtype=float)
    if len(arr) < 2:
        return arr.copy()

    drift = arr[-1] - arr[0]
    trend = np.linspace(0.0, drift, len(arr))
    return arr - trend


def stabilize_cycle(strain_slice):
    """
    Make cycle start at original first value and end at the same value,
    after drift correction.
    """
    strain_slice = np.asarray(strain_slice, dtype=float)
    s0 = strain_slice - strain_slice[0]
    s_stable = remove_end_bias(s0)
    return s_stable + strain_slice[0]


def _interp_clamped(x, xp, fp):
    """
    1D linear interpolation with clamping outside [min(xp), max(xp)].
    xp must be sorted ascending.
    """
    x = np.asarray(x, dtype=float)
    xp = np.asarray(xp, dtype=float)
    fp = np.asarray(fp, dtype=float)

    if len(xp) < 2:
        return np.full_like(x, float(fp[0]) if len(fp) else 0.0)

    x_clamped = np.clip(x, xp[0], xp[-1])
    return np.interp(x_clamped, xp, fp)


# =========================================================
# OPTIONAL EXTERNAL HELPERS FOR AUTO-RESOLVING FILES
# (used OUTSIDE analyze_creep if desired)
# =========================================================
def parse_loadD_tag_from_path(input_csv: str | Path) -> str | None:
    """
    From a path like .../LoadD_0p7/... or .../LoadD_0.7/... return tag '0p700'.
    Returns None if not found.
    """
    s = str(Path(input_csv))

    m = re.search(r"LoadD_(\d+(?:[p\.]\d+)?)", s)
    if not m:
        return None

    raw = m.group(1).replace(".", "p")

    try:
        val = float(raw.replace("p", "."))
    except Exception:
        return None

    return f"{val:.3f}".replace(".", "p")


def choose_plastic_sweep_dir(
    input_csv: str | Path,
    sweeps_root: str | Path = ".",
    default_dirname: str = "unified_plastic_sweeps_LoadDmax_0p700",
) -> Path:
    """
    Picks unified_plastic_sweeps_LoadDmax_<tag> if possible,
    else falls back to default_dirname.

    This helper is intentionally OUTSIDE analyze_creep.
    """
    sweeps_root = Path(sweeps_root)
    tag = parse_loadD_tag_from_path(input_csv)

    if tag is not None:
        candidate = sweeps_root / f"unified_plastic_sweeps_LoadDmax_{tag}"
        partA = candidate / "partA_load_sweep" / "partA_load_sweep.csv"
        partB = candidate / "partB_moist_sweep" / "partB_moist_sweep.csv"
        if partA.exists() and partB.exists():
            return candidate

    return sweeps_root / default_dirname


def make_plastic_config_from_dir(
    plastic_dir: str | Path,
    *,
    enabled: bool = True,
    strict: bool = True,
    load_col: str = "load_d",
    moist_col: str = "moist_max",
    plastic_col: str = "slip_final",
    use_abs_load: bool = False,
) -> PlasticConfig:
    """
    Convenience helper to build PlasticConfig from a directory layout:
      plastic_dir/
        partA_load_sweep/partA_load_sweep.csv
        partB_moist_sweep/partB_moist_sweep.csv
    """
    plastic_dir = Path(plastic_dir)
    return PlasticConfig(
        enabled=enabled,
        partA_csv=plastic_dir / "partA_load_sweep" / "partA_load_sweep.csv",
        partB_csv=plastic_dir / "partB_moist_sweep" / "partB_moist_sweep.csv",
        strict=strict,
        load_col=load_col,
        moist_col=moist_col,
        plastic_col=plastic_col,
        use_abs_load=use_abs_load,
    )


# =========================================================
# MASTER CYCLE BUILDERS
# =========================================================
def build_master_cycle(reduced, cycles, N=2, npts=200, detrend=True, debug_plot=False):
    """
    Build average reduced-strain master cycle from the last N cycles.
    """
    if len(cycles) == 0:
        raise ValueError("No cycles provided to build_master_cycle().")

    selected = cycles[-min(N, len(cycles)):]
    resampled = []
    raw_cycles = []

    for i1, i2 in selected:
        s = np.asarray(reduced[i1:i2 + 1], dtype=float)
        s = s - s[0]

        if detrend:
            s = remove_end_bias(s)

        raw_cycles.append(s.copy())
        resampled.append(resize_to(s, npts))

    resampled = np.array(resampled)
    master = np.mean(resampled, axis=0)

    if debug_plot:
        plt.figure(figsize=(7, 5))

        for s in raw_cycles:
            x = np.linspace(0, 1, len(s))
            plt.plot(x, s, alpha=0.6)

        x_master = np.linspace(0, 1, npts)
        for s in resampled:
            plt.plot(x_master, s, linestyle="--", alpha=0.6)

        plt.plot(x_master, master, color="black", linewidth=3, label="MASTER")
        plt.title("Master Cycle Construction")
        plt.xlabel("Normalized cycle position")
        plt.ylabel("Reduced strain")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.show()

    return master


def build_master_moisture(moisture, cycles, N=3, npts=200):
    """
    Build average moisture master cycle from the last N cycles.
    """
    if len(cycles) == 0:
        raise ValueError("No cycles provided to build_master_moisture().")

    selected = cycles[-min(N, len(cycles)):]
    resampled = []

    for i1, i2 in selected:
        m = np.asarray(moisture[i1:i2 + 1], dtype=float)
        xp = np.linspace(0, 1, len(m))
        m_interp = np.interp(np.linspace(0, 1, npts), xp, m)
        resampled.append(m_interp)

    return np.mean(resampled, axis=0)


# =========================================================
# VISCOELASTIC FITTERS
# =========================================================
def fit_Jeff(csv_summary, degree=2):
    """Fit J_eff vs moisture_frac from summary file."""
    csv_path = Path(csv_summary)
    df = pd.read_csv(csv_path)

    x = df["moisture_frac"].to_numpy(dtype=float)
    y = df["J_eff"].to_numpy(dtype=float)

    if degree == 2:
        def quad(xx, a, b, c):
            return a * xx**2 + b * xx + c

        popt, _ = curve_fit(
            quad, x, y,
            bounds=([0, -np.inf, -np.inf], [np.inf, np.inf, np.inf])
        )
        a, b, c = popt
        poly = lambda xx: a * xx**2 + b * xx + c
        coeffs = popt
    else:
        coeffs = np.polyfit(x, y, degree)
        p = np.poly1d(coeffs)
        poly = lambda xx: p(xx)

    return poly, coeffs


def ferrara_branch_compliances(moisture):
    """
    Ferrara KV spectrum vs moisture.
    Returns J_raw with shape (N, 4).
    """
    phi_pts = np.array([0.0, (0.12 - 0.07) / 0.13, 1.0])

    J_table = np.array([
        [2.1e-4, 0.87e-4, 1.8e-4, 2.8e-4],
        [4.2e-4, 12.0e-4, 9.6e-4, 7.6e-4],
        [2.2e-3, 3.4e-3, 2.5e-3, 1.4e-3],
    ])

    J_poly = np.array([
        np.polyfit(phi_pts, J_table[:, i], 2)
        for i in range(J_table.shape[1])
    ])

    moisture = np.asarray(moisture)
    J_raw = np.zeros((len(moisture), 4))

    for i in range(4):
        p = J_poly[i]
        J_raw[:, i] = p[0] * moisture**2 + p[1] * moisture + p[2]

    return J_raw


def ferrara_weights(moisture):
    """
    Normalize Ferrara branch compliances into branch weights.
    """
    J_raw = ferrara_branch_compliances(moisture)
    J_sum = np.sum(J_raw, axis=1, keepdims=True)
    J_sum[J_sum == 0] = 1.0
    return J_raw / J_sum


def fit_Ji(csv_summary, degree=2, clip_nonnegative=True):
    """
    Fit J1..J4 vs moisture_frac from summary file.
    Returns:
      polys: list of 4 callables
      coeffs: list of coefficient arrays
    """
    csv_path = Path(csv_summary)
    df = pd.read_csv(csv_path)
    x = df["moisture_frac"].to_numpy(dtype=float)

    polys = []
    coeffs_out = []

    def make_poly(y):
        if degree == 2:
            def quad(xx, a, b, c):
                return a * xx**2 + b * xx + c

            popt, _ = curve_fit(
                quad, x, y,
                bounds=([0, -np.inf, -np.inf], [np.inf, np.inf, np.inf])
            )
            a, b, c = popt
            poly = lambda xx: a * xx**2 + b * xx + c
            coeffs = popt
        else:
            coeffs = np.polyfit(x, y, degree)
            p = np.poly1d(coeffs)
            poly = lambda xx: p(xx)
        return poly, coeffs

    for col in ["J1", "J2", "J3", "J4"]:
        y = df[col].to_numpy(dtype=float)
        poly, coeffs = make_poly(y)

        if clip_nonnegative:
            poly_raw = poly
            poly = lambda xx, pr=poly_raw: np.maximum(pr(xx), 0.0)

        polys.append(poly)
        coeffs_out.append(coeffs)

    return polys, coeffs_out


def eval_Ji_with_sum_constraint(moisture, poly_Jeff, polys_Ji, eps=1e-15):
    """
    Evaluate Ji(m) from component polys and enforce sum(Ji)=Jeff at each m.
    """
    Ji_raw = np.column_stack([p(moisture) for p in polys_Ji])
    Ji_raw = np.maximum(Ji_raw, 0.0)

    Jeff = np.maximum(poly_Jeff(moisture), 0.0)

    S = Ji_raw.sum(axis=1)
    r = Jeff / np.maximum(S, eps)
    Ji = Ji_raw * r[:, None]
    return Ji, Jeff


def eval_Ji_from_Jeff_weights(moisture, poly_Jeff, weights, clip_nonnegative=True):
    """
    Build Ji(m) by splitting Jeff(m) using fixed weights.
    """
    moisture = np.asarray(moisture, dtype=float)
    Jeff = np.asarray(poly_Jeff(moisture), dtype=float)

    if clip_nonnegative:
        Jeff = np.maximum(Jeff, 0.0)

    w = np.asarray(weights, dtype=float)
    if w.ndim != 1:
        raise ValueError("weights must be a 1D array")
    if np.any(w < 0):
        raise ValueError("weights must be nonnegative")
    if np.sum(w) <= 0:
        raise ValueError("weights must not sum to zero")

    w = w / np.sum(w)
    Ji = Jeff[:, None] * w[None, :]
    return Ji, Jeff


# =========================================================
# PLASTIC SIGNAL
# =========================================================
def build_plastic_signal(
    load,
    moisture,
    partA_csv,
    partB_csv,
    eps_load=1e-12,
    load_col="load_d",
    moist_col="moist_max",
    plastic_col="slip_final",
    use_abs_load=False,
):
    """
    Build eps_plastic(t) using:
      - Part A: plastic(load) during loading ramp
      - Part B: plastic(max moisture reached) after load ramp

    Rules:
      - moisture is non-monotone, but plastic cannot decrease
      - do NOT add partA + partB
      - switch from A to B at end of loading ramp
      - enforce cumulative non-decreasing plasticity
    """
    load = np.asarray(load, dtype=float)
    moisture = np.asarray(moisture, dtype=float)

    if len(load) != len(moisture):
        raise ValueError("load and moisture must have the same length")

    A = pd.read_csv(partA_csv).sort_values(load_col)
    B = pd.read_csv(partB_csv).sort_values(moist_col)

    load_grid = A[load_col].to_numpy(dtype=float)
    plastA_grid = A[plastic_col].to_numpy(dtype=float)

    moist_grid = B[moist_col].to_numpy(dtype=float)
    plastB_grid = B[plastic_col].to_numpy(dtype=float)

    ld = np.abs(load) if use_abs_load else load
    i_max_load = int(np.argmax(ld))

    epsA = _interp_clamped(ld, load_grid, plastA_grid)

    mmax = np.maximum.accumulate(moisture)
    epsB = _interp_clamped(mmax, moist_grid, plastB_grid)

    eps_plastic = np.zeros(len(load), dtype=float)
    eps_plastic[:i_max_load + 1] = epsA[:i_max_load + 1]
    eps_plastic[i_max_load + 1:] = epsB[i_max_load + 1:]

    first_loaded = np.argmax(ld > eps_load)
    if ld[first_loaded] <= eps_load:
        first_loaded = len(load)

    if first_loaded > 0:
        eps_plastic[:first_loaded] = 0.0

    eps_plastic = np.maximum.accumulate(eps_plastic)
    return eps_plastic, i_max_load, mmax


# =========================================================
# MAIN ANALYSIS
# =========================================================
def analyze_creep(
    input_csv,
    ve_cfg: ViscoelasticConfig,
    plastic_cfg: PlasticConfig | None = None,
    manual_valleys=None,  # kept for compatibility
    make_plots=False,
    exclude_plastic=False,
    critical_strain=1.0,
):
    """
    Clean analysis function.

    Important design principle:
    - NO hardcoded external folders/files inside this function
    - all external references are passed through ve_cfg / plastic_cfg
    """

    del manual_valleys  # currently unused, kept for API compatibility

    # -------------------------------------------------
    # helper: fit strain = a*moisture + b on one cycle
    # -------------------------------------------------
    def fit_strain_moisture_slope(moist, strain):
        moist = np.asarray(moist, dtype=float)
        strain = np.asarray(strain, dtype=float)

        mask = np.isfinite(moist) & np.isfinite(strain)
        moist = moist[mask]
        strain = strain[mask]

        if len(moist) < 2:
            return 0.0, 0.0

        a, b = np.polyfit(moist, strain, 1)
        return a, b

    # -------------------------------------------------
    # Load input data
    # -------------------------------------------------
    input_csv = Path(input_csv)
    df = pd.read_csv(input_csv)

    moisture = df["Moisture"].to_numpy(dtype=float)
    time = df["Time"].to_numpy(dtype=float)
    load = df["Load"].to_numpy(dtype=float)
    total = df["Total_strain"].to_numpy(dtype=float)
    slip = df["Slip_strain"].to_numpy(dtype=float)
    creep_sim = df["Creep"].to_numpy(dtype=float)
    elastic = df["Elastic"].to_numpy(dtype=float)
    hygro_exp = df["Hygroexp"].to_numpy(dtype=float)

    total_raw = total.copy()

    # -------------------------------------------------
    # Plastic subtraction
    # -------------------------------------------------
    partA_csv = None
    partB_csv = None
    plastic_source = None

    if plastic_cfg is None or not plastic_cfg.enabled:
        eps_plastic = np.zeros_like(total, dtype=float)
        i_max_load = int(np.argmax(np.abs(load)))
        mmax = np.maximum.accumulate(moisture)

    else:
        partA_csv = Path(plastic_cfg.partA_csv) if plastic_cfg.partA_csv is not None else None
        partB_csv = Path(plastic_cfg.partB_csv) if plastic_cfg.partB_csv is not None else None

        if plastic_cfg.strict:
            if partA_csv is None or partB_csv is None:
                raise ValueError(
                    "PlasticConfig requires both partA_csv and partB_csv when strict=True."
                )
            if not partA_csv.exists() or not partB_csv.exists():
                raise FileNotFoundError(
                    "Plastic sweep files not found.\n"
                    f"partA_csv = {partA_csv} (exists={partA_csv.exists()})\n"
                    f"partB_csv = {partB_csv} (exists={partB_csv.exists()})"
                )

        if (
            partA_csv is not None and partB_csv is not None
            and partA_csv.exists() and partB_csv.exists()
        ):
            eps_plastic, i_max_load, mmax = build_plastic_signal(
                load=load,
                moisture=moisture,
                partA_csv=partA_csv,
                partB_csv=partB_csv,
                eps_load=plastic_cfg.eps_load,
                load_col=plastic_cfg.load_col,
                moist_col=plastic_cfg.moist_col,
                plastic_col=plastic_cfg.plastic_col,
                use_abs_load=plastic_cfg.use_abs_load,
            )
            plastic_source = "external_reference"
        else:
            eps_plastic = np.zeros_like(total, dtype=float)
            i_max_load = int(np.argmax(np.abs(load)))
            mmax = np.maximum.accumulate(moisture)
            plastic_source = "disabled_or_missing"

    if not exclude_plastic:
        total = total - eps_plastic

    # -------------------------------------------------
    # Viscoelastic model
    # -------------------------------------------------
    if ve_cfg is None:
        raise ValueError("ve_cfg is required.")
    if ve_cfg.summary_csv is None:
        raise ValueError("ve_cfg.summary_csv is required.")

    summary_csv = Path(ve_cfg.summary_csv)
    tau = float(ve_cfg.tau)
    kv_num = int(ve_cfg.kv_num)

    poly_Jeff, _ = fit_Jeff(summary_csv, degree=2)
    polys_Ji, _  = fit_Ji(summary_csv, degree=2, clip_nonnegative=True)

    tau_list = tau * 10.0 ** np.arange(kv_num)

    # J_fiber = np.maximum(poly_Jeff(moisture), 0.0)
    # weights = ferrara_weights(moisture)
    # J_i_fiber = J_fiber[:, None] * weights

    J_i_fiber, J_fiber = eval_Ji_with_sum_constraint(moisture, poly_Jeff, polys_Ji)

    time_shifted = time - time[0]
    dt = np.diff(time_shifted, prepend=time_shifted[0])
    dt[dt == 0] = 1e-12

    x = np.zeros((len(time), kv_num), dtype=float)
    for i in range(1, len(time)):
        dti = dt[i]
        sigma_prev = load[i - 1]
        for j, tau_j in enumerate(tau_list):
            alpha = np.exp(-dti / tau_j)
            x[i, j] = alpha * x[i - 1, j] + J_i_fiber[i, j] * sigma_prev * (1.0 - alpha)

    creep_model = np.sum(x, axis=1)
    creep = creep_model / critical_strain

    total_s = total / critical_strain
    reduced = total_s - creep
    reduced2 = total_s - (creep_sim / critical_strain)

    # ============================================================
    # 1) DETECT UNLOAD STEP
    # ============================================================
    eps_unload = 1e-6
    step_to_zero = np.where((load[:-1] > eps_unload) & (load[1:] <= eps_unload))[0]
    unload_idx = int(step_to_zero[0] + 1) if len(step_to_zero) else len(load) - 1
    t_unload = float(time[unload_idx])

    # ============================================================
    # 2) DEFINE CYCLES
    # ============================================================
    cycle_start = 9.0
    cycle_length = 20.0
    time_max = float(np.nanmax(time))

    time_cycles = []
    t1 = cycle_start
    while t1 + cycle_length <= time_max:
        time_cycles.append((t1, t1 + cycle_length))
        t1 += cycle_length

    if not time_cycles:
        raise ValueError("No valid time-based cycles detected.")

    cycle_bounds = []
    for t1, t2 in time_cycles:
        i1 = int(np.argmin(np.abs(time - t1)))
        i2 = int(np.argmin(np.abs(time - t2)))
        if i2 > i1:
            cycle_bounds.append((i1, i2))

    if not cycle_bounds:
        raise ValueError("No valid cycle_bounds detected after indexing.")

    # ============================================================
    # 3) CLASSIFY CYCLES
    # ============================================================
    pre_unload_cycles = [(i1, i2) for (i1, i2) in cycle_bounds if time[i2] <= t_unload]
    post_unload_cycles = [(i1, i2) for (i1, i2) in cycle_bounds if time[i1] >= t_unload]

    skip_unload_points = 3
    if len(post_unload_cycles) > 0:
        i1, i2 = post_unload_cycles[0]
        i1 = min(i1 + skip_unload_points, i2)
        post_unload_cycles[0] = (i1, i2)

    if not pre_unload_cycles:
        raise ValueError("No cycles before unload.")

    # ============================================================
    # 4) BUILD MASTER HYGROELASTIC TEMPLATE
    # ============================================================
    N_template_cycles = 2
    n_template_pts = 200

    master_loaded = build_master_cycle(
        reduced,
        pre_unload_cycles,
        N=N_template_cycles,
        npts=n_template_pts,
        detrend=True,
        debug_plot=False,
    )
    master_moist_loaded = build_master_moisture(
        moisture,
        pre_unload_cycles,
        N=N_template_cycles,
        npts=n_template_pts,
    )

    if len(post_unload_cycles) > 0:
        master_unloaded = build_master_cycle(
            reduced,
            post_unload_cycles,
            N=min(3, len(post_unload_cycles)),
            npts=n_template_pts,
            detrend=True,
            debug_plot=False,
        )
        master_moist_unloaded = build_master_moisture(
            moisture,
            post_unload_cycles,
            N=N_template_cycles,
            npts=n_template_pts,
        )
    else:
        master_unloaded = master_loaded.copy()
        master_moist_unloaded = master_moist_loaded.copy()

    # ============================================================
    # 5) FIT REFERENCE STRAIN-MOISTURE RELATION
    # ============================================================
    ref_loaded_i1, ref_loaded_i2 = pre_unload_cycles[-1]
    a_loaded, b_loaded = fit_strain_moisture_slope(
        moisture[ref_loaded_i1:ref_loaded_i2 + 1],
        reduced[ref_loaded_i1:ref_loaded_i2 + 1],
    )

    if len(post_unload_cycles) > 0:
        ref_unloaded_i1, ref_unloaded_i2 = post_unload_cycles[-1]
        a_unloaded, b_unloaded = fit_strain_moisture_slope(
            moisture[ref_unloaded_i1:ref_unloaded_i2 + 1],
            reduced[ref_unloaded_i1:ref_unloaded_i2 + 1],
        )
    else:
        a_unloaded, b_unloaded = a_loaded, b_loaded

    # ============================================================
    # 6) BUILD CYCLE-WISE TEMPLATE
    # ============================================================
    templ_global = np.full_like(reduced, np.nan)

    for idx, (i1, i2) in enumerate(cycle_bounds):
        m = moisture[i1:i2 + 1].astype(float)

        x_cycle = np.linspace(0.0, 1.0, len(m))
        x_master = np.linspace(0.0, 1.0, n_template_pts)

        is_loaded = time[i2] <= t_unload

        if is_loaded:
            master_strain = master_loaded.copy()
            master_moist = master_moist_loaded.copy()
            a_ref = a_loaded
        else:
            master_strain = master_unloaded.copy()
            master_moist = master_moist_unloaded.copy()
            a_ref = a_unloaded

        templ = np.interp(x_cycle, x_master, master_strain)
        moist_ref = np.interp(x_cycle, x_master, master_moist)

        m_corr = close_cycle_trend(m)
        moist_ref_corr = close_cycle_trend(moist_ref)

        amp_ref = np.max(moist_ref_corr) - np.min(moist_ref_corr)
        amp_cur = np.max(m_corr) - np.min(m_corr)
        scale = (amp_cur / amp_ref) if amp_ref > 1e-12 else 1.0
        templ = templ * scale

        if is_loaded:
            loaded_indices = [k for k, (j1, j2) in enumerate(cycle_bounds) if time[j2] <= t_unload]
            first_loaded_idx = loaded_indices[0]

            if idx == first_loaded_idx:
                offset = reduced[i1] - templ[0]
            else:
                prev_valid = np.where(np.isfinite(templ_global[:i1]))[0]
                if len(prev_valid) > 0:
                    prev_end = templ_global[prev_valid[-1]]
                    offset = prev_end - templ[0]
                else:
                    offset = reduced[i1] - templ[0]

            templ = templ + offset

        else:
            moist_unload = moisture[unload_idx:]
            omega_min = np.min(moist_unload)
            eps_min = a_unloaded * omega_min
            templ_min = np.min(templ)
            offset = eps_min - templ_min
            templ = templ + offset

        delta_m = m[-1] - m[0]
        delta_eps = a_ref * delta_m
        ramp = np.linspace(0.0, 1.0, len(templ))
        templ = templ + delta_eps * ramp

        templ_global[i1:i2 + 1] = templ

    # ============================================================
    # 7) MECHANOSORPTION SIGNAL
    # ============================================================
    mech = reduced - templ_global

    # ============================================================
    # 8) SMOOTH MECH TRANSITION AT UNLOAD
    # ============================================================
    n_left = 7
    n_right = 2
    iL = unload_idx

    i_ref = max(0, iL - n_left)
    val_left = mech[i_ref]
    mech[i_ref:iL + n_right] = val_left

    # ============================================================
    # Optional debug plots
    # ============================================================
    if make_plots:
        plt.figure(figsize=(10, 6))
        plt.plot(load[max(iL - 20, 0):min(iL + 20, len(load))], label="load")
        plt.plot(mech[max(iL - 20, 0):min(iL + 20, len(mech))], label="mechanosorption")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

    # ============================================================
    # RETURN
    # ============================================================
    return dict(
        time=time,
        load=load,
        moisture=moisture,
        total_raw=total_raw / critical_strain,
        plastic=eps_plastic / critical_strain,
        total=total_s,
        total_s=total_s,
        creep=creep,
        creep_sim=creep_sim / critical_strain,
        slip=slip / critical_strain,
        elastic=elastic / critical_strain,
        hygro=hygro_exp / critical_strain,
        reduced=reduced,
        reduced2=reduced2,
        mech=mech,
        templ_global=templ_global,
        t_unload=t_unload,
        cycle_bounds=cycle_bounds,
        pre_unload_cycles=pre_unload_cycles,
        post_unload_cycles=post_unload_cycles,
        time_cycles=time_cycles,
        critical_strain=critical_strain,
        unload_idx=unload_idx,
        plastic_switch_idx=i_max_load,
        moisture_running_max=mmax,
        plastic_source=plastic_source,
        plastic_partA=str(partA_csv) if partA_csv is not None else None,
        plastic_partB=str(partB_csv) if partB_csv is not None else None,
        visco_summary_csv=str(summary_csv),
        tau=tau,
        tau_list=tau_list,
        a_loaded=a_loaded,
        b_loaded=b_loaded,
        a_unloaded=a_unloaded,
        b_unloaded=b_unloaded,
    )


# =========================================================
# EXAMPLE USAGE
# =========================================================
if __name__ == "__main__":
    # --------------------------------------
    # User inputs
    # --------------------------------------
    input_csv = "./all_mech_paper.csv"

    # ---- viscoelastic config (required) ----
    ve_cfg = ViscoelasticConfig(
        summary_csv="./moisture_sweep_results22/LoadD_0p7/moisture_sweep_summary.csv",
        tau=0.001,
        degree=2,
        kv_num=4,
    )

    # ---- plastic config (fully externalized) ----
    # Option 1: explicit files
    # plastic_cfg = PlasticConfig(
    #     enabled=True,
    #     partA_csv="./plastic_analysis/unified_plastic_sweeps_LoadDmax_0p700/partA_load_sweep/partA_load_sweep.csv",
    #     partB_csv="./plastic_analysis/unified_plastic_sweeps_LoadDmax_0p700/partB_moist_sweep/partB_moist_sweep.csv",
    #     strict=True,
    # )

    # Option 2: build from folder helper
    plastic_dir = choose_plastic_sweep_dir(
        input_csv=input_csv,
        sweeps_root="./plastic_analysis/",
        default_dirname="unified_plastic_sweeps_LoadDmax_0p700",
    )
    plastic_cfg = make_plastic_config_from_dir(
        plastic_dir,
        enabled=True,
        strict=True,
    )

    # Option 3: disable plastic entirely
    # plastic_cfg = PlasticConfig(enabled=False)

    # --------------------------------------
    # Run analysis
    # --------------------------------------
    results = analyze_creep(
        input_csv=input_csv,
        ve_cfg=ve_cfg,
        plastic_cfg=plastic_cfg,
        manual_valleys=[],
        make_plots=True,
        exclude_plastic=False,
    )

    # --------------------------------------
    # Save all_mech.csv with mechanosorption
    # --------------------------------------
    df_in = pd.read_csv(input_csv)
    df_in["Mechano_sorption"] = results["mech"]

    out_file = Path(input_csv).with_name("all_mech.csv")
    df_in.to_csv(out_file, index=False)
    print(f"Saved mechanosorption file to: {out_file}")

    # --------------------------------------
    # Quick check plot
    # --------------------------------------
    t = results["time"] / 100.0
    mech = results["mech"]

    plt.figure(figsize=(10, 7))

    t = results["time"] / 100.0

    plt.plot(t, results["total_raw"], label="total_raw (before plastic)")
    plt.plot(t, results["plastic"],   label="plastic (schleronomic)")
    plt.plot(t, results["total"],     label="total (after plastic)")

    plt.plot(t, results["templ_global"], label="templ_global")
    plt.plot(t, results["reduced"],      label="reduced (total - creep)")
    plt.plot(t, results["mech"],         label="mech")
    plt.plot(t, results["slip"],         label="slip")

    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    # --------------------------------------
    # Expose variables in interactive session
    # --------------------------------------
    globals().update(results)
    print("Analysis complete. Variables available in the workspace:")
    for k in results.keys():
        print("  ", k)