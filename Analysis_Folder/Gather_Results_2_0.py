#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 12 16:52:31 2025

@author: jortiz
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd

# =============================================================
# CONFIG
# =============================================================

BASE_DIR = Path("../Analysis_folders/Structured_Data_Redo")

META_DIR = Path("fourier_meta")
META_DIR.mkdir(exist_ok=True)

OUT_CSV = META_DIR / "results_systems.csv"

# ---------------------------
# LoadD filter (empty => ALL)
# ---------------------------
# Examples:
#   []            -> all LoadD_*
#   [0.7]         -> only ~0.7
#   ["0p7","0p8"] -> only those
LOADD_FILTER = []  # type: list
LOAD_TOL = 1e-9

# -------------------------------------------------------------
# Load Fourier lookup (AUTHORITATIVE FILTER)
# -------------------------------------------------------------
FOURIER_JSON = META_DIR / "fourier_lookup.json"
with FOURIER_JSON.open() as f:
    fo_data = json.load(f)

Fo_diff_map = {float(k): float(v) for k, v in fo_data["Fo_diff"].items()}
Fo_visc_map = {float(k): float(v) for k, v in fo_data["Fo_visc"].items()}

VALID_DIFFS = set(Fo_diff_map.keys())
VALID_TAUS  = set(Fo_visc_map.keys())

# =============================================================
# HELPERS
# =============================================================

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

def parse_loadd(name: str):
    """
    LoadD_0p3 -> (0.3, '0p3')
    LoadD_0.3 -> (0.3, '0.3')
    """
    tag = name.split("LoadD_")[-1]
    val = _to_float(tag)
    return val, tag

def find_plateau_end(load):
    y = np.asarray(load, dtype=float)
    if len(y) < 3:
        return len(y) - 1
    max_val = np.nanmax(y)
    for i in range(len(y) - 2, 0, -1):
        if y[i] >= max_val and (y[i + 1] < y[i]):
            return i - 2
    return int(np.nanargmax(y))

def parse_diff(name):
    tag = name.split("Diff_")[-1]
    is_avg = tag.endswith("_avg")
    if is_avg:
        tag = tag[:-4]
    try:
        return float(tag), is_avg
    except ValueError:
        return None, None

def parse_ramp(name):
    return float(name.split("Ramp_")[-1].rstrip("%"))

def parse_tau(name):
    return float(name.split("Tau_")[-1])

def only_csv_in(folder: Path):
    csvs = sorted(folder.glob("*.csv"))
    if not csvs:
        return None
    if len(csvs) == 1:
        return csvs[0]
    for c in csvs:
        if c.name.lower().startswith("all"):
            return c
    return csvs[0]

# =============================================================
# MAIN
# =============================================================

def main():
    rows = []

    for diff_dir in sorted(BASE_DIR.glob("Diff_*")):
        if not diff_dir.is_dir():
            continue

        D, is_avg = parse_diff(diff_dir.name)
        if D is None or D not in VALID_DIFFS:
            continue

        Fo_diff = Fo_diff_map[D]

        for ramp_dir in sorted(diff_dir.glob("Ramp_*")):
            if not ramp_dir.is_dir():
                continue
            r = parse_ramp(ramp_dir.name)

            for load_dir in sorted(ramp_dir.glob("LoadD_*")):
                if not load_dir.is_dir():
                    continue

                load_d, load_tag = parse_loadd(load_dir.name)
                if load_d is None:
                    continue

                # -----------------------------
                # NEW: Load filter (empty => all)
                # -----------------------------
                if LOADD_FILTER and (not _float_in(load_d, LOADD_FILTER, tol=LOAD_TOL)):
                    continue

                for tau_dir in sorted(load_dir.glob("Tau_*")):
                    if not tau_dir.is_dir():
                        continue

                    t = parse_tau(tau_dir.name)
                    if t not in VALID_TAUS:
                        continue

                    Fo_visc = Fo_visc_map[t]

                    csv_path = only_csv_in(tau_dir)
                    if csv_path is None:
                        continue

                    df = pd.read_csv(csv_path)
                    if not {"Load", "Total_strain"}.issubset(df.columns):
                        continue

                    # -------------------------------------------------
                    # Unloading plateau
                    # -------------------------------------------------
                    idx = find_plateau_end(df["Load"].to_numpy())

                    total_unload = float(df["Total_strain"].iloc[idx])
                    total_final  = float(df["Total_strain"].iloc[-1])
                    

                    # -------------------------------------------------
                    # UNLOADING metrics
                    # -------------------------------------------------
                    Slip_unload = (
                        float(df["Slip_strain"].iloc[idx])
                        if "Slip_strain" in df.columns else np.nan
                    )

                    Slip_Ratio = (
                        Slip_unload / total_unload
                        if Slip_unload == Slip_unload and total_unload > 0 else np.nan
                    )

                    Mech_unload = (
                        float(df["Mech_strain"].iloc[idx])
                        if "Mech_strain" in df.columns else np.nan
                    )

                    Mech_Ratio = (
                        Mech_unload / total_unload
                        if Mech_unload == Mech_unload and total_unload > 0 else np.nan
                    )

                    Mech_max_r = (
                        float(np.max(df["Mech_strain"]))
                        if "Mech_strain" in df.columns else np.nan
                    )

                    Mech_max_rat = (
                        float(np.max(df["Mech_strain"]))/ total_unload
                        if "Mech_strain" in df.columns else np.nan
                    )

                    SlipCount_unload = (
                        float(df["Slip Count"].iloc[idx])
                        if "Slip Count" in df.columns else np.nan
                    )

                    Plastic_unload = (
                        total_final / total_unload
                        if total_unload > 0 else np.nan
                    )

                    Mech_no_Plastic = (
                        Mech_Ratio - Plastic_unload
                        if Mech_Ratio == Mech_Ratio else np.nan
                    )

                    # -------------------------------------------------
                    # DAMAGE / SLIP COUNTS (END + UNLOAD)
                    # -------------------------------------------------
                    slip_end = (
                        float(df["Slip Count"].iloc[-1])
                        if "Slip Count" in df.columns else np.nan
                    )

                    broken_unload = (
                        float(df["Number_of_fibers"].iloc[idx])
                        if "Number_of_fibers" in df.columns else np.nan
                    )

                    broken_end = (
                        float(df["Number_of_fibers"].iloc[-1])
                        if "Number_of_fibers" in df.columns else np.nan
                    )

                    # -------------------------------------------------
                    # Build row (NOW includes LoadD)
                    # -------------------------------------------------
                    rows.append({
                        # Identifiers
                        "Diff": D,
                        "Tau": t,
                        "Ramp": r,
                        "LoadD": load_d,        # NEW (numeric)
                        "LoadD_tag": load_tag,  # NEW (folder tag)
                        "is_avg": is_avg,

                        # Fourier
                        "Fo_diff": Fo_diff/2,
                        "Fo_visc": Fo_visc/2,

                        # Original unloading metrics
                        "Slip_unload": Slip_unload,
                        "Slip_Ratio": Slip_Ratio,
                        "Mech_unload": Mech_unload,
                        "Total_unload": total_unload,
                        "Mech_Ratio": Mech_Ratio,
                        "SlipCount_unload": SlipCount_unload,
                        "Plastic_unload": Plastic_unload,
                        "Mech_no_Plastic": Mech_no_Plastic,
                        "Mech_max_unload": Mech_max_r,
                        "Mech_max_ratio": Mech_max_rat,

                        # New damage / efficiency metrics
                        "Slip_end": slip_end,
                        "BrokenFibers_unload": broken_unload,
                        "BrokenFibers_end": broken_end,
                        "Slip_per_strain_unload": (
                            SlipCount_unload / total_unload
                            if SlipCount_unload == SlipCount_unload and total_unload > 0 else np.nan
                        ),
                        "Slip_per_strain_end": (
                            slip_end / total_final
                            if slip_end == slip_end and total_final > 0 else np.nan
                        ),

                        "CSV_path": str(csv_path),
                    })

    if not rows:
        print("No systems found.")
        return

    df_out = pd.DataFrame(rows)
    df_out.to_csv(OUT_CSV, index=False)

    print("✔ results_systems.csv written")
    print("✔ Rows:", len(df_out))
    print("✔ LoadD filter:", LOADD_FILTER if LOADD_FILTER else "(ALL)")
    print("✔ LoadD included in CSV (LoadD, LoadD_tag)")

# =============================================================
# RUN
# =============================================================
if __name__ == "__main__":
    main()
