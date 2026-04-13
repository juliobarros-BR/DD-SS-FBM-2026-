#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 29 09:03:54 2025

@author: jortiz
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from pathlib import Path
import re

# ---------------------------
# Utilities
# ---------------------------
def seg_bc(arr, i1, i2):
    """Baseline-corrected array slice arr[i1:i2+1], starts at 0."""
    s = arr[i1:i2+1].astype(float).copy()
    return s - s[0] if len(s) else s

def resize_to(arr, new_len):
    """Resize array arr to length new_len using linear interpolation."""
    if len(arr) == 0 or new_len <= 0:
        return np.zeros(new_len, dtype=float)
    xp = np.linspace(0, 1, len(arr))
    return np.interp(np.linspace(0, 1, new_len), xp, arr)



def parse_loadD_tag_from_path(input_csv: str | Path) -> str | None:
    """
    From a path like .../LoadD_0p7/... or .../LoadD_0.7/... return tag '0p700'.
    Returns None if not found.
    """
    p = Path(input_csv)
    s = str(p)

    m = re.search(r"LoadD_(\d+(?:[p\.]\d+)?)", s)
    if not m:
        return None

    raw = m.group(1)  # e.g. '0p7' or '0.7' or '0p700'
    raw = raw.replace(".", "p")

    try:
        val = float(raw.replace("p", "."))
    except Exception:
        return None

    # Make 3 decimals like 0.700 -> '0p700'
    tag = f"{val:.3f}".replace(".", "p")
    return tag


def choose_plastic_sweep_dir(input_csv: str | Path,
                            sweeps_root: str | Path = ".",
                            default_dirname: str = "unified_plastic_sweeps_new3") -> Path:
    """
    Picks unified_plastic_sweeps_LoadDmax_<tag> if possible, else falls back to default_dirname.
    """
    sweeps_root = Path(sweeps_root)
    tag = parse_loadD_tag_from_path(input_csv)
    print(tag)
    if tag is not None:
        candidate = sweeps_root / f"unified_plastic_sweeps_LoadDmax_{tag}"
        partA = candidate / "partA_load_sweep" / "partA_load_sweep.csv"
        partB = candidate / "partB_moist_sweep" / "partB_moist_sweep.csv"
        if partA.exists() and partB.exists():
            return candidate

    # fallback
    return sweeps_root / default_dirname


# ---------------------------
# Fitters for compliances
# ---------------------------

def fit_Jeff(csv_summary, degree=2):
    """Fit J_eff vs moisture_frac from summary file."""
    csv_path = Path(csv_summary)
    df = pd.read_csv(csv_path)

    x = df["moisture_frac"].to_numpy(dtype=float)
    y = df["J_eff"].to_numpy(dtype=float)

    if degree == 2:
        def quad(xx, a, b, c):
            return a*xx**2 + b*xx + c

        popt, _ = curve_fit(
            quad, x, y,
            bounds=([0, -np.inf, -np.inf], [np.inf, np.inf, np.inf])
        )
        a, b, c = popt
        poly = lambda xx: a*xx**2 + b*xx + c
        coeffs = popt
    else:
        coeffs = np.polyfit(x, y, degree)
        p = np.poly1d(coeffs)
        poly = lambda xx: p(xx)

    return poly, coeffs


def fit_Ji(csv_summary, degree=2, clip_nonnegative=True):
    """
    Fit J1..J4 vs moisture_frac from summary file.
    Returns:
      polys: list of 4 callables poly_j(moisture_array) -> Ji_array
      coeffs: list of coefficient arrays (one per Ji)
    """
    csv_path = Path(csv_summary)
    df = pd.read_csv(csv_path)

    x = df["moisture_frac"].to_numpy(dtype=float)

    polys = []
    coeffs_out = []

    def make_poly(y):
        if degree == 2:
            def quad(xx, a, b, c):
                return a*xx**2 + b*xx + c

            popt, _ = curve_fit(
                quad, x, y,
                bounds=([0, -np.inf, -np.inf], [np.inf, np.inf, np.inf])
            )
            a, b, c = popt
            poly = lambda xx: a*xx**2 + b*xx + c
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
    Returns:
      J_i_fiber: (N,4)
      Jeff:      (N,)
    """
    Ji_raw = np.column_stack([p(moisture) for p in polys_Ji])  # (N,4)
    Ji_raw = np.maximum(Ji_raw, 0.0)

    Jeff = poly_Jeff(moisture)
    Jeff = np.maximum(Jeff, 0.0)

    S = Ji_raw.sum(axis=1)
    r = Jeff / np.maximum(S, eps)

    Ji = Ji_raw * r[:, None]
    return Ji, Jeff


def eval_Ji_from_Jeff_weights(moisture, poly_Jeff, weights, clip_nonnegative=True):
    """
    Build Ji(m) by splitting Jeff(m) using fixed weights.
    Returns:
      Ji:   (N, KV_num)
      Jeff: (N,)
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

    w = w / np.sum(w)                  # normalize to sum=1
    Ji = Jeff[:, None] * w[None, :]    # broadcast to (N,KV_num)
    return Ji, Jeff


def remove_end_bias(arr):
    """Subtract linear trend so arr[0]==0 and arr[-1]==0."""
    if len(arr) < 2:
        return arr
    trend = np.linspace(0.0, arr[-1], len(arr))
    return arr - trend
import numpy as np
import pandas as pd
from pathlib import Path

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


def build_plastic_signal(load, moisture,
                         partA_csv,
                         partB_csv,
                         eps_load=1e-12,
                         load_col="load_d",
                         moist_col="moist_max",
                         plastic_col="slip_final",
                         use_abs_load=False):
    """
    Build eps_plastic(t) using:
      - Part A: plastic(load) during loading ramp
      - Part B: plastic(max moisture reached) after load ramp

    Rules implemented:
      - Moisture is non-monotone, but plastic can't decrease:
          use mmax[i] = max(moisture[:i+1])
      - Do NOT add partA + partB.
          Switch from A to B at end of loading ramp.
      - Ensure eps_plastic is non-decreasing over time by cumulative max.

    Assumptions:
      - load input is already "load_d" (same units as partA load_col).
        If not, normalize before calling this function.
      - partB curve is already anchored such that
        partB at its minimum moist equals partA at target load.
    """
    load = np.asarray(load, dtype=float)
    moisture = np.asarray(moisture, dtype=float)
    N = len(load)
    if len(moisture) != N:
        raise ValueError("load and moisture must have the same length")

    # Read sweep tables
    A = pd.read_csv(partA_csv)
    B = pd.read_csv(partB_csv)

    # Sort for interp
    A = A.sort_values(load_col)
    B = B.sort_values(moist_col)

    load_grid = A[load_col].to_numpy(dtype=float)
    plastA_grid = A[plastic_col].to_numpy(dtype=float)

    moist_grid = B[moist_col].to_numpy(dtype=float)
    plastB_grid = B[plastic_col].to_numpy(dtype=float)

    # Choose load for A mapping
    ld = np.abs(load) if use_abs_load else load

    # Identify "loading ramp end" = first index of maximum load (plateau start)
    # (This is the clean A->B switch point)
    i_max_load = int(np.argmax(ld))

    # Part A candidate: plastic as function of load_d (during ramp)
    epsA = _interp_clamped(ld, load_grid, plastA_grid)

    # Part B candidate: plastic as function of running max moisture
    mmax = np.maximum.accumulate(moisture)
    epsB = _interp_clamped(mmax, moist_grid, plastB_grid)

    # Assemble final plastic signal (switch, don't sum)
    eps_plastic = np.zeros(N, dtype=float)

    # Before/including max load index: use Part A
    eps_plastic[:i_max_load+1] = epsA[:i_max_load+1]

    # After max load: use Part B (already anchored to match end of A)
    eps_plastic[i_max_load+1:] = epsB[i_max_load+1:]

    # If load is essentially zero at the very beginning, keep plastic at 0 until load starts
    # (but once plastic appears, it persists)
    first_loaded = np.argmax(ld > eps_load)  # 0 if already loaded
    if ld[first_loaded] <= eps_load:
        first_loaded = N  # never loaded
    if first_loaded > 0:
        eps_plastic[:first_loaded] = 0.0

    # Enforce non-decreasing plasticity
    eps_plastic = np.maximum.accumulate(eps_plastic)

    return eps_plastic, i_max_load, mmax


# ---------------------------
# Main analysis
# ---------------------------
def analyze_creep(
    input_csv,
    summary_csv,
    tau=1.0,
    manual_valleys=None,
    make_plots=False,
    exclude_plastic=False,
    plastic_dir_override=None,
    plastic_sweeps_root="./plastic_analysis/",
    plastic_default_dirname="unified_plastic_sweeps_new3",
    strict_plastic_files=True,
):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from pathlib import Path
    from scipy.signal import find_peaks

    # -------------------------------------------------
    # Load CSV
    # -------------------------------------------------
    df = pd.read_csv(input_csv)

    moisture   = df["Moisture"].to_numpy(dtype=float)
    time       = df["Time"].to_numpy(dtype=float)
    load       = df["Load"].to_numpy(dtype=float)
    total      = df["Total_strain"].to_numpy(dtype=float)
    slip       = df["Slip_strain"].to_numpy(dtype=float)
    creep_sim  = df["Creep"].to_numpy(dtype=float)
    elastic    = df["Elastic"].to_numpy(dtype=float)
    hygro_exp  = df["Hygroexp"].to_numpy(dtype=float)

    total_raw = total.copy()

    # -------------------------------------------------
    # Schleronomic plastic strain subtraction
    #   - choose plastic sweep directory:
    #       * override if provided
    #       * else infer from input_csv path (LoadD_*)
    #       * else fallback to default dir name
    # -------------------------------------------------
    if plastic_dir_override is not None:
        plastic_dir = Path(plastic_dir_override)
    else:
        plastic_dir = choose_plastic_sweep_dir(
            input_csv=input_csv,
            sweeps_root=plastic_sweeps_root,
            default_dirname=plastic_default_dirname,
        )

    partA_csv = plastic_dir / "partA_load_sweep" / "partA_load_sweep.csv"
    partB_csv = plastic_dir / "partB_moist_sweep" / "partB_moist_sweep.csv"

    if strict_plastic_files and (not partA_csv.exists() or not partB_csv.exists()):
        raise FileNotFoundError(
            "Plastic sweep files not found.\n"
            f"plastic_dir = {plastic_dir}\n"
            f"partA_csv   = {partA_csv} (exists={partA_csv.exists()})\n"
            f"partB_csv   = {partB_csv} (exists={partB_csv.exists()})\n\n"
            "Fix options:\n"
            "  1) Pass plastic_dir_override=Path('.../unified_plastic_sweeps_LoadDmax_0p500')\n"
            "  2) Put INPUT_CSV inside a path containing LoadD_... so auto-detect works\n"
            "  3) Set strict_plastic_files=False to continue with plastic=0 (NOT recommended for paper plots)\n"
        )

    if partA_csv.exists() and partB_csv.exists():
        eps_plastic, i_max_load, mmax = build_plastic_signal(
            load=load,
            moisture=moisture,
            partA_csv=partA_csv,
            partB_csv=partB_csv,
            plastic_col="slip_final",   # adjust if your plastic column name differs
            load_col="load_d",
            moist_col="moist_max",
        )
    else:
        # graceful fallback if strict_plastic_files=False
        eps_plastic = np.zeros_like(total, dtype=float)
        i_max_load = int(np.argmax(np.abs(load)))
        mmax = np.maximum.accumulate(moisture)

    # subtract plastic from total strain immediately (before creep subtraction etc.)
    if not exclude_plastic:
        total = total - eps_plastic

    # -------------------------------------------------
    # Critical strain
    # -------------------------------------------------
    critical_strain = 1.0  # keep as you currently do

    # -------------------------------------------------
    # Kelvin–Voigt creep model (use fitted J1..J4, constrained to sum to J_eff)
    # -------------------------------------------------
    KV_num   = 4
    base_tau = float(tau)
    tau_list = base_tau * 10.0 ** np.arange(KV_num)

    poly_Jeff, _ = fit_Jeff(summary_csv, degree=2)
    polys_Ji, _  = fit_Ji(summary_csv, degree=2, clip_nonnegative=True)

    J_i_fiber, J_fiber = eval_Ji_with_sum_constraint(moisture, poly_Jeff, polys_Ji)

    time_shifted = time - time[0]
    dt = np.diff(time_shifted, prepend=time_shifted[0])
    dt[dt == 0] = 1e-12

    x = np.zeros((len(time), KV_num), dtype=float)
    for i in range(1, len(time)):
        Δt = dt[i]
        σ_prev = load[i - 1]
        for j, τ in enumerate(tau_list):
            α = np.exp(-Δt / τ)
            x[i, j] = α * x[i - 1, j] + J_i_fiber[i, j] * σ_prev * (1.0 - α)

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
    # 2) DEFINE CYCLES (time-based, as you had)
    # ============================================================
    cycle_start  = 9.0
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
    for (t1, t2) in time_cycles:
        i1 = int(np.argmin(np.abs(time - t1)))
        i2 = int(np.argmin(np.abs(time - t2)))
        if i2 > i1:
            cycle_bounds.append((i1, i2))

    if not cycle_bounds:
        raise ValueError("No valid cycle_bounds detected after indexing.")

    # ============================================================
    # 3) CLASSIFY CYCLES
    # ============================================================
    pre_unload_cycles  = [(i1, i2) for (i1, i2) in cycle_bounds if time[i2] <= t_unload]
    post_unload_cycles = [(i1, i2) for (i1, i2) in cycle_bounds if time[i1] >= t_unload]

    if not pre_unload_cycles:
        raise ValueError("No cycles before unload.")

    last_loaded_i1, last_loaded_i2 = pre_unload_cycles[-1]
    if post_unload_cycles:
        last_unloaded_i1, last_unloaded_i2 = post_unload_cycles[-1]
    else:
        last_unloaded_i1, last_unloaded_i2 = last_loaded_i1, last_loaded_i2

    # ============================================================
    # 4) FIT STRAIN–MOISTURE RELATIONS
    # ============================================================
    def fit_slope(i1, i2):
        moist = moisture[i1:i2 + 1]
        strain = reduced[i1:i2 + 1]
        a, b = np.polyfit(moist, strain, 1)
        return a

    a_loaded   = fit_slope(last_loaded_i1, last_loaded_i2)
    a_unloaded = fit_slope(last_unloaded_i1, last_unloaded_i2)

    # ============================================================
    # 5) BUILD TEMPLATE FROM MOISTURE SIGNAL
    # ============================================================
    templ_global = np.where(
        np.arange(len(moisture)) < unload_idx,
        moisture * a_loaded + reduced[cycle_bounds[0][0]] / critical_strain,
        moisture * a_unloaded
    ).astype(float)

    # ============================================================
    # 6) FIX DISCONTINUITY AT UNLOAD (NaN window)
    # ============================================================
    skip = 0
    iL = unload_idx
    left = max(0, iL - skip)
    right = min(len(templ_global) - 1, iL + skip)
    templ_global[left:right + 1] = np.nan

    # ============================================================
    # 7) MECHANOSORPTION SIGNAL
    # ============================================================
    mech = reduced - templ_global
    mech[:cycle_bounds[0][0]] = 0.0

    # ============================================================
    # RETURN CLEAN DICT
    # ============================================================
    return dict(
        # --- raw signals ---
        time=time,
        load=load,
        moisture=moisture,

        total_raw=total_raw / critical_strain,     # original total (normalized)
        plastic=eps_plastic / critical_strain,     # plastic you subtracted (normalized)

        # --- decomposition inputs/outputs ---
        total=total_s,                              # plastic-corrected total_s
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

        a_loaded=a_loaded,
        a_unloaded=a_unloaded,

        t_unload=t_unload,
        cycle_bounds=cycle_bounds,
        pre_unload_cycles=pre_unload_cycles,
        post_unload_cycles=post_unload_cycles,
        time_cycles=time_cycles,

        critical_strain=critical_strain,
        unload_idx=unload_idx,

        # optional debug helpers
        plastic_switch_idx=i_max_load,
        moisture_running_max=mmax,

        # for sanity/debug
        plastic_dir=str(plastic_dir),
        plastic_partA=str(partA_csv),
        plastic_partB=str(partB_csv),
    )





if __name__ == "__main__":
    # input_csv = "./Analysis_folders/Structured_Data_final/Diff_600_avg/Ramp_30%/Tau_0.0001/all.csv"
    input_csv = "./Analysis_folders/Plastic_Data_new_high/Diff_1000/Ramp_00%/Tau_0.1/history_1.csv"
    # input_csv = "./Analysis_folders/Diffusion_analysis/5000/all_cap_5000.csv"
    # summary_csv = "./Analysis_folders/moisture_sweep_results/moisture_sweep_summary.csv"
    
    summary_csv = "./moisture_sweep_results/moisture_sweep_summary.csv"

    results = analyze_creep(
        input_csv=input_csv,
        summary_csv=summary_csv,
        tau=0.0001,
        manual_valleys=[],   # now peaks instead of valleys
        make_plots=True
    )
    
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

    
    globals().update(results)
    print("Analysis complete. Variables available in the workspace:")
    for k in results.keys():
        print("  ", k)

