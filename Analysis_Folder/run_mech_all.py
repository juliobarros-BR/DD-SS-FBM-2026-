#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 12 23:44:57 2025

@author: jortiz
"""

import re
import numpy as np
import pandas as pd  # type: ignore
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.lines import Line2D
import gc, psutil

from compute_mechsorption import analyze_creep

# =====================================================
# USER CONFIG
# =====================================================
BASE_DIR = Path("./Analysis_folders/Structured_Data_Redo/")

SUMMARY_CSV = Path("./moisture_sweep_results22/LoadD_0p5/moisture_sweep_summary.csv")
RESULTS_DIR = Path("Results_folders/Final_time_strain/")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TAU_DEFAULT = 0.0001
CRITICAL_STRAIN = 1.0

DO_PLOTS = True
OVERWRITE_CSV = True

# ---------------------------
# FILTERS (empty => ALL)
# ---------------------------
LOADD_FILTER = [0.25,0.35,0.45,0.55,0.65]   # e.g. [] or [0.7] or ["0p7", "0p8"]
TAU_FILTER   = []   # e.g. [] or [0.0001, 0.1]
DIFF_FILTER  = []   # e.g. [] or [600, 1000] or ["Diff_600"]
RAMP_FILTER  = []   # e.g. [] or ["10%", "50%"] or ["Ramp_10%"]

AVG_ONLY = False     # None (both) | True (only *_avg) | False (no *_avg)

# Numeric compare tolerances
TOL_LOAD = 1e-9
TOL_TAU  = 1e-12


# =====================================================
# SMALL HELPERS
# =====================================================
def _norm_prefix(s: str, prefix: str) -> str:
    return s[len(prefix):] if s.startswith(prefix) else s

def _to_float(x):
    if x is None:
        return None
    try:
        if isinstance(x, str):
            return float(x.replace("p", "."))
        return float(x)
    except Exception:
        return None

def _float_in(x: float | None, allowed, tol: float) -> bool:
    if x is None:
        return False
    for a in allowed:
        aa = _to_float(a)
        if aa is None:
            continue
        if abs(x - aa) <= tol:
            return True
    return False

def _parse_load(load_name: str) -> float | None:
    # LoadD_0p7 / LoadD_0.7 / LoadD_0p700
    return _to_float(_norm_prefix(load_name, "LoadD_"))

def _parse_tau(tau_name: str) -> float | None:
    return _to_float(_norm_prefix(tau_name, "Tau_"))

def _parse_diff(diff_name: str) -> tuple[int | None, bool]:
    is_avg = diff_name.endswith("_avg")
    base = diff_name[:-4] if is_avg else diff_name  # strip "_avg"
    base = _norm_prefix(base, "Diff_")
    try:
        return int(base), is_avg
    except Exception:
        return None, is_avg

def _parse_ramp(ramp_name: str) -> str:
    # Ramp_10% -> 10%
    return _norm_prefix(ramp_name, "Ramp_")


def passes_filters(diff_name: str, ramp_name: str, load_name: str, tau_name: str) -> bool:
    diff_val, is_avg = _parse_diff(diff_name)
    ramp_val = _parse_ramp(ramp_name)
    load_val = _parse_load(load_name)
    tau_val  = _parse_tau(tau_name)

    # avg filter
    if AVG_ONLY is True and not is_avg:
        return False
    if AVG_ONLY is False and is_avg:
        return False

    # diff filter (accept numeric or "Diff_600")
    if DIFF_FILTER:
        ok = False
        for d in DIFF_FILTER:
            if isinstance(d, str) and d.startswith("Diff_"):
                if diff_name == d:
                    ok = True
            else:
                try:
                    if diff_val is not None and diff_val == int(d):
                        ok = True
                except Exception:
                    pass
        if not ok:
            return False

    # ramp filter (accept "10%" or "Ramp_10%")
    if RAMP_FILTER:
        allowed = {_parse_ramp(str(r)) for r in RAMP_FILTER}
        if ramp_val not in allowed:
            return False

    # load filter (accept 0.7 or "0p7")
    if LOADD_FILTER:
        if not _float_in(load_val, LOADD_FILTER, tol=TOL_LOAD):
            return False

    # tau filter (accept 0.0001 or "0.0001")
    if TAU_FILTER:
        if not _float_in(tau_val, TAU_FILTER, tol=TOL_TAU):
            return False

    return True


# =====================================================
# YOUR EXISTING HELPERS
# =====================================================
def get_tau_value(tau_dir: Path) -> float:
    try:
        return float(tau_dir.name.replace("Tau_", ""))
    except Exception:
        print(f"⚠️ Could not parse tau value from {tau_dir.name}. Using default {TAU_DEFAULT}.")
        return TAU_DEFAULT


def add_mechanosorptive_strain(df: pd.DataFrame, csv_file: Path, tau_value: float) -> pd.DataFrame:
    try:
        print(tau_value)
        results = analyze_creep(
            input_csv=str(csv_file),
            summary_csv=str(SUMMARY_CSV),
            tau=tau_value,
            manual_valleys=[],
            make_plots=False,
            exclude_plastic=True,
        )

        df["Mech_strain"] = np.asarray(results.get("mech", np.nan), np.float32)

        if "plastic" in results:
            df["Plastic_scler"] = np.asarray(results["plastic"], np.float32)
        if "total_raw" in results:
            df["Total_raw_norm"] = np.asarray(results["total_raw"], np.float32)
        if "total" in results:
            df["Total_corr_norm"] = np.asarray(results["total"], np.float32)

        del results
        return df

    except Exception as e:
        print(f"❌ Mechanosorptive analysis failed for {csv_file}: {e}")
        df["Mech_strain"] = np.nan
        df["Plastic_scler"] = np.nan
        df["Total_raw_norm"] = np.nan
        df["Total_corr_norm"] = np.nan
        return df


def make_final_plot(df: pd.DataFrame, diff_name: str, ramp_name: str, load_name: str, tau_name: str,
                    out_path: Path, plastic: np.ndarray | None = None) -> None:
    try:
        for col in ["Time", "Total_strain", "Hygroexp", "Elastic", "Creep", "Slip_strain", "Mech_strain", "Moisture"]:
            if col not in df.columns:
                raise KeyError(f"Missing column '{col}' for plotting")

        t = df["Time"].to_numpy(np.float32)
        total = df["Total_strain"].to_numpy(np.float32) / CRITICAL_STRAIN
        hygro_elastic = (df["Hygroexp"] + df["Elastic"]).to_numpy(np.float32) / CRITICAL_STRAIN
        creep = df["Creep"].to_numpy(np.float32) / CRITICAL_STRAIN
        slip = df["Slip_strain"].to_numpy(np.float32) / CRITICAL_STRAIN
        mech = df["Mech_strain"].to_numpy(np.float32) / CRITICAL_STRAIN
        moist = df["Moisture"].to_numpy(np.float32)

        fig, ax1 = plt.subplots(dpi=300, figsize=(8, 5))
        ax1.plot(t, total, label=r"$\langle \varepsilon \rangle$")
        ax1.plot(t, hygro_elastic, label=r"$\langle \varepsilon_H + \varepsilon_E \rangle$")
        ax1.plot(t, creep, label=r"$\langle \varepsilon_C \rangle$")
        ax1.plot(t, slip, label=r"$\langle \varepsilon_S \rangle$")
        ax1.plot(t, mech, "--", label=r"$\varepsilon_{MS}$")
        if plastic is not None:
            ax1.plot(t, plastic / CRITICAL_STRAIN, "--", label=r"$\varepsilon_P$")

        ax1.set_xlim(0, float(np.max(t)) if len(t) else 1.0)
        ax1.set_ylim(0, 1.3)
        ax1.grid(True)
        ax1.set_xlabel(r"Normalized Time ($t/\tau_4$)")
        ax1.set_ylabel(r"Normalized Strain ($\varepsilon/\varepsilon_c$)")

        info_lines = [Line2D([0], [0], color="none")] * 4
        info_labels = [
            f"D: {diff_name.replace('Diff_', '').replace('_avg','')}",
            f"Ramp: {ramp_name.replace('Ramp_', '')}",
            f"LoadD: {load_name.replace('LoadD_', '').replace('p', '.')}",
            f"τ: {tau_name.replace('Tau_', '')}",
        ]
        legend2 = ax1.legend(
            info_lines, info_labels, loc="upper left",
            framealpha=1, facecolor="white", edgecolor="black"
        )
        ax1.add_artist(legend2)

        ax2 = ax1.twinx()
        ax2.plot(t, moist, alpha=0.3, label=r"$\langle \omega \rangle$")
        ax2.set_ylabel(r"Normalized Moisture ($\omega/0.3$)")
        ax2.set_ylim(0, 1.0)

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2,
                   loc="upper left", bbox_to_anchor=(1.15, 1),
                   framealpha=1, facecolor="white", edgecolor="black")

        fig.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        plt.close("all")

    except Exception as e:
        print(f"⚠️ Plot failed for {out_path.name}: {e}")


# =====================================================
# MAIN LOOP
# =====================================================
process = psutil.Process()
pattern = "Diff_*/Ramp_*/LoadD_*/Tau_*/all.csv"

for csv_file in BASE_DIR.glob(pattern):
    try:
        tau_dir = csv_file.parent
        load_dir = tau_dir.parent
        ramp_dir = load_dir.parent
        diff_dir = ramp_dir.parent

        tau_name  = tau_dir.name
        load_name = load_dir.name
        ramp_name = ramp_dir.name
        diff_name = diff_dir.name

        if not passes_filters(diff_name, ramp_name, load_name, tau_name):
            continue

        print(f"\n=== Processing: {csv_file} ===")

        df = pd.read_csv(csv_file, dtype_backend="numpy_nullable")
        if "Moisture" not in df.columns:
            print("⚠️ Missing Moisture column — skipping.")
            del df
            gc.collect()
            continue

        tau_value = get_tau_value(tau_dir)
        df = add_mechanosorptive_strain(df, csv_file, tau_value)

        if OVERWRITE_CSV:
            df.to_csv(csv_file, index=False)

        if DO_PLOTS:
            out_file = RESULTS_DIR / f"final_strain_vs_time_{diff_name}_{ramp_name}_{load_name}_{tau_name}.png"
            plastic = df["Plastic_scler"].to_numpy(np.float32) if "Plastic_scler" in df.columns else None
            make_final_plot(df, diff_name, ramp_name, load_name, tau_name, out_file, plastic)
            del plastic

        del df
        plt.close("all")
        gc.collect()

        mem = process.memory_info().rss / 1e6
        print(f"→ Memory after cleanup: {mem:.1f} MB")

    except Exception as e:
        print(f"❌ Error in {csv_file}: {e}")
        gc.collect()

print("\n✅ Done!")
