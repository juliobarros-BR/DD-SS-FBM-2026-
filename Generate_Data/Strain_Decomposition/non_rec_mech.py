#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  8 10:37:11 2025

@author: jortiz
"""

"""
Incremental plastic sweep runner (densify schedule without rerunning existing points)

What changed vs your original:
- We DO NOT skip based on plastic_vs_cycles.csv or fit images anymore.
- We skip a tau-folder only if all cycles in the *current* schedule already have history_<n>.csv.
- We run only missing cycles from the schedule.
- If an early-stop (done=true) exists for the tau-folder, we cap the schedule at that cycle
  (i.e., we do not try to run cycles beyond the first done-cycle).

Outputs are the same:
- history_<n>.csv per simulated point
- cycle_<n>_meta.json per simulated point
- plastic_vs_cycles.csv and plastic_vs_cycles_fit.png regenerated via finalize_tau_folder
- summary_fits.csv updated at end (append/replace behavior: append; you can dedupe later if needed)
"""

import os, re, gc, sys, copy, traceback, json
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit
from tqdm import tqdm
import multiprocessing as mp

from Model_files.Model_class_copy_moist_grad_control_new import Model
from Model_files.Sim_class_moist_grad import Simulate


# =============================================================
# CONFIGURATION
# =============================================================

DEST_ROOT = Path("./Analysis_folders/Plastic_Data_new_high/")
DEST_ROOT.mkdir(parents=True, exist_ok=True)

SUMMARY_ROOT = Path("./Analysis_folders/plastic_analysis/")
SUMMARY_ROOT.mkdir(parents=True, exist_ok=True)

MOISTURE_ROOT = Path(
    "/local/home/jortiz/Julio_PhD/Fiber Bundle Model/FBM_Stick_Slip/Revisiting/"
    "Clean_copy_for_paper/Moisture_gradient/Generate_Moisture_Profiles/"
)

BASE_INPUT = Path("input_unitless_plastic.txt")

CYCLES_PRE_LOAD   = 0
CYCLES_UNLOAD     = 1000
MAX_LOADED_CYCLES = 5
FIXED_SEED        = 39

DEFAULT_TAUS = [0.0001, 0.001, 0.01, 0.1]

READONLY_MASKS = False


# =============================================================
# MANUAL FILTERS
# =============================================================

RUN_DIFFS = [1000]   # empty → all
RUN_AVG_MOISTURE = False              # None → both avg + nonavg ; "avg" → only avg ; False → only nonavg
RUN_RAMPS = [50]                        # empty → all
RUN_TAUS  = [0.001]                        # empty → all DEFAULT_TAUS


# =============================================================
# τ-sampling rule (EDIT THIS to densify)
# =============================================================
def cycle_schedule(tau):
    """
    Return an iterable of cycles_loaded values to simulate.

    You can densify by decreasing the stride.
    Example: for tau<=0.0003 change stride 2 -> 1, etc.
    """
    if tau <= 0.0003:
        return range(1, MAX_LOADED_CYCLES + 1, 1)   # was 2
    elif tau <= 0.003:
        return range(1, MAX_LOADED_CYCLES + 1, 1)   # was 4
    elif tau <= 0.03:
        return range(1, MAX_LOADED_CYCLES + 1, 1)   # was 8
    elif tau <= 0.3:
        return range(1, MAX_LOADED_CYCLES + 1, 8)   # was 15
    else:
        return range(1, MAX_LOADED_CYCLES + 1, 15)  # was 25


# =============================================================
# Helpers
# =============================================================
def safe_write_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    with open(path, "rb") as f:
        os.fsync(f.fileno())


def safe_write_csv(path: Path, df: pd.DataFrame):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(df.to_csv(index=False))
    with open(tmp, "rb") as f:
        os.fsync(f.fileno())
    tmp.replace(path)


def safe_savefig(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()
    with open(path, "rb") as f:
        os.fsync(f.fileno())


def build_input_text(base_text: str, folder_moist: str, tau_value: float) -> str:
    """
    Build base input: only folder_moist and tau_value are set.
    Plastic params will be set later inside child_run_one_cycle.
    """
    txt = re.sub(
        r"^(?P<k>\s*folder_moist\s*=\s*)(?P<v>.+?)\s*$",
        rf"\g<k>{folder_moist}",
        base_text,
        flags=re.MULTILINE,
    )
    txt = re.sub(
        r"^(?P<k>\s*tau\s*=\s*)(?P<v>[-+eE0-9.]+)",
        rf"\g<k>{tau_value}",
        txt,
        flags=re.MULTILINE,
    )
    return txt


def build_input_text_plastic(base_text, folder_moist, tau_value,
                             cycles_pre, cycles_loaded, cycles_unload, seed):
    txt = build_input_text(base_text, folder_moist, tau_value)
    txt += f"\ncycles_pre_load = {cycles_pre}"
    txt += f"\ncycles_loaded = {cycles_loaded}"
    txt += f"\ncycles_unload = {cycles_unload}"
    txt += f"\nseed = {seed}\n"
    return txt


def parse_new_folder_name(name: str):
    """
    Parse NEW_300_ramp0.10_avg → (300, 0.10, True)
          NEW_300_ramp0.00     → (300, 0.00, False)
    """
    m = re.match(r"NEW_(\d+)_ramp([0-9.]+)(.*)", name)
    if not m:
        return None
    diff_val = int(m.group(1))
    ramp_val = float(m.group(2))
    is_avg = ("_avg" in m.group(3))
    return diff_val, ramp_val, is_avg


def parse_cycle_from_history(p: Path):
    m = re.match(r"history_(\d+)\.csv$", p.name)
    return int(m.group(1)) if m else None


def parse_cycle_from_meta(p: Path):
    m = re.match(r"cycle_(\d+)_meta\.json$", p.name)
    return int(m.group(1)) if m else None


def find_existing_history_cycles(tau_dir: Path) -> set[int]:
    """
    Existing cycles based on history_<n>.csv presence.
    (This is your requested 'ground truth' for skip.)
    """
    out = set()
    for f in tau_dir.glob("history_*.csv"):
        n = parse_cycle_from_history(f)
        if n is not None:
            out.add(n)
    return out


def find_first_done_cycle(tau_dir: Path) -> int | None:
    """
    If any cycle_<n>_meta.json has done=true, return the smallest n with done=true.
    This caps further runs (avoid useless post-failure cycles).
    """
    done_cycles = []
    for mf in tau_dir.glob("cycle_*_meta.json"):
        n = parse_cycle_from_meta(mf)
        if n is None:
            continue
        try:
            data = json.loads(mf.read_text())
            if bool(data.get("done", False)):
                done_cycles.append(n)
        except Exception:
            continue
    return min(done_cycles) if done_cycles else None


# =============================================================
# Finalization helper (CSV+fit)
# =============================================================
def finalize_tau_folder(tau_dir: Path, diff_label, ramp_label, tau_value, all_fits):
    metas = list(tau_dir.glob("cycle_*_meta.json"))
    if not metas:
        return

    cycles = []
    for mf in metas:
        m = re.match(r"cycle_(\d+)_meta\.json", mf.name)
        if m:
            cyc = int(m.group(1))
            cycles.append((cyc, mf))

    cycles.sort(key=lambda t: t[0])

    df = pd.DataFrame([
        (json.loads(mf.read_text()) | {"cycles_loaded": cyc})
        for cyc, mf in cycles
    ])
    df = df.sort_values("cycles_loaded")
    safe_write_csv(tau_dir / "plastic_vs_cycles.csv", df)

    # Fit if enough points
    if len(df) >= 4:
        n = df["cycles_loaded"].to_numpy(float)
        y = df["final_plastic"].to_numpy(float)

        def sat_exp(n, A, n0, C):
            return A * (1 - np.exp(-n / max(n0, 1e-12))) + C

        A0 = float(np.nanmax(y) - np.nanmin(y))
        n00 = 0.2 * MAX_LOADED_CYCLES
        C0 = float(np.nanmin(y))
        p0 = [A0, n00, C0]
        bounds = ([0, 1e-6, -np.inf], [np.inf, np.inf, np.inf])

        try:
            popt, _ = curve_fit(sat_exp, n, y, p0=p0, bounds=bounds)
            xx = np.linspace(float(np.min(n)), float(np.max(n)), 300)

            fig, ax = plt.subplots(figsize=(6.5, 4.2))
            ax.plot(n, y, "o")
            ax.plot(xx, sat_exp(xx, *popt), "--")
            ax.grid(True)
            ax.set_xlabel("Cycles loaded")
            ax.set_ylabel("Plastic strain")
            ax.set_title(f"{diff_label} / {ramp_label} / Tau={tau_value}")
            safe_savefig(tau_dir / "plastic_vs_cycles_fit.png")

            A_fit, n0_fit, C_fit = popt
            y_hat = sat_exp(n, *popt)
            r2 = 1 - np.sum((y - y_hat) ** 2) / (np.sum((y - y.mean()) ** 2) + 1e-12)

            all_fits.append({
                "diff": diff_label,
                "ramp": ramp_label,
                "tau": float(tau_value),
                "A": float(A_fit),
                "n0": float(n0_fit),
                "C": float(C_fit),
                "R2": float(r2),
                "key": f"{diff_label}/{ramp_label}/Tau_{tau_value}",
            })
        except Exception:
            # Still write a "not enough / fit failed" plot
            fig, ax = plt.subplots(figsize=(6.5, 4.2))
            ax.plot(n, y, "o")
            ax.grid(True)
            ax.set_xlabel("Cycles loaded")
            ax.set_ylabel("Plastic strain")
            ax.set_title("Fit failed (will need more points)")
            safe_savefig(tau_dir / "plastic_vs_cycles_fit.png")
    else:
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        ax.set_title("Not enough points to fit.")
        ax.grid(True)
        safe_savefig(tau_dir / "plastic_vs_cycles_fit.png")
        plt.close(fig)


# =============================================================
# CHILD PROCESS
# =============================================================
RAW_MODEL = None

def child_run_one_cycle(tau_value, nload, tau_dir, base_text, mask_dir):
    """
    Executes a single simulation at cycles_loaded=nload, writes history_<n>.csv and cycle_<n>_meta.json.
    """
    try:
        tau_dir.mkdir(parents=True, exist_ok=True)

        txt = build_input_text_plastic(
            base_text, str(mask_dir), tau_value,
            CYCLES_PRE_LOAD, nload, CYCLES_UNLOAD, FIXED_SEED
        )
        safe_write_text(tau_dir / f"input_cycle_{nload}.txt", txt)

        model = copy.copy(RAW_MODEL)
        model.sys_var["tau"] = float(tau_value)
        model.sys_var["cycles_pre_load"] = CYCLES_PRE_LOAD
        model.sys_var["cycles_loaded"] = nload
        model.sys_var["cycles_unload"] = CYCLES_UNLOAD
        model.sys_var["seed"] = FIXED_SEED

        if hasattr(model, "History"):
            model.History.clear()
        if hasattr(model, "snapshots"):
            model.snapshots.clear()

        if READONLY_MASKS:
            for key, v in model.__dict__.items():
                if isinstance(v, np.ndarray):
                    v.flags.writeable = False

        sim = Simulate(model)
        if hasattr(sim, "n_final_snapshots"): sim.n_final_snapshots = 0
        if hasattr(sim, "snapshot_every_steps"): sim.snapshot_every_steps = 0
        if hasattr(sim, "plot_frequency"): sim.plot_frequency = 0

        # τ-based comparison window
        if tau_value <= 0.001:
            compareN = 10
        elif tau_value < 0.1:
            compareN = 50
        else:
            compareN = 100

        sim.run(compare_n_cycles_ago=compareN)

        if "Time" in sim.History and len(sim.History["Time"]) > 0:
            sim.History["Time"][0] = 0.0

        hist = pd.DataFrame(sim.History)
        safe_write_csv(tau_dir / f"history_{nload}.csv", hist)

        crit = float(getattr(sim.model, "critical_strain", 1.0))
        final_plastic = float(hist["Total_strain"].iloc[-1]) / crit
        max_tot = float(hist["Total_strain"].max())
        done_flag = bool(getattr(sim, "loaded_ignore_start_cycle", False))

        meta = {
            "final_plastic": final_plastic,
            "done": done_flag,
            "max_total_strain": max_tot,
        }
        (tau_dir / f"cycle_{nload}_meta.json").write_text(json.dumps(meta))

        del sim, model, hist
        gc.collect()

    except Exception:
        (tau_dir / "run.log").write_text(
            "ERROR:\n" + "".join(traceback.format_exception(*sys.exc_info()))
        )
        raise


# =============================================================
# MAIN LOOP
# =============================================================
def main():
    mp.set_start_method("fork", force=True)

    base_text = BASE_INPUT.read_text()
    all_fits = []

    all_masks = sorted(
        p for p in MOISTURE_ROOT.iterdir()
        if p.is_dir() and p.name.startswith("NEW_")
    )

    for mask_dir in all_masks:

        parsed = parse_new_folder_name(mask_dir.name)
        if parsed is None:
            continue

        diff_val, ramp_float, is_avg = parsed

        # --- FILTERS ---
        if RUN_DIFFS and diff_val not in RUN_DIFFS:
            continue
        ramp_int = int(round(ramp_float * 100))
        if RUN_RAMPS and ramp_int not in RUN_RAMPS:
            continue
        if RUN_AVG_MOISTURE == "avg" and not is_avg:
            continue
        if RUN_AVG_MOISTURE is False and is_avg:
            continue
        # RUN_AVG_MOISTURE=None → allow both

        diff_label = f"Diff_{diff_val}_avg" if is_avg else f"Diff_{diff_val}"
        ramp_label = f"Ramp_{ramp_int:02d}%"
        ramp_dir = DEST_ROOT / diff_label / ramp_label
        ramp_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n### MASK: {mask_dir.name} → {diff_label}/{ramp_label}")

        # Select list of τ to run
        tau_values = [t for t in DEFAULT_TAUS if (not RUN_TAUS or t in RUN_TAUS)]
        if not tau_values:
            print("   (no tau values selected; skipping)")
            continue

        # ============================================================
        # Build RAW_MODEL ONCE for this mask_dir
        # ============================================================
        print("→ Loading RAW_MODEL (once for this mask)...")
        dummy_tau = tau_values[0]
        template_text = build_input_text(base_text, str(mask_dir), dummy_tau)
        base_input_file = ramp_dir / f"{BASE_INPUT.stem}__BASE.txt"
        safe_write_text(base_input_file, template_text)

        global RAW_MODEL
        RAW_MODEL = Model(str(base_input_file))

        # ============================================================
        # PROCESS EACH τ VALUE
        # ============================================================
        for tau_value in tau_values:

            tau_dir = ramp_dir / f"Tau_{tau_value}"
            tau_dir.mkdir(parents=True, exist_ok=True)

            # cap by early-stop if present
            first_done = find_first_done_cycle(tau_dir)
            schedule = list(cycle_schedule(float(tau_value)))
            if first_done is not None:
                schedule = [c for c in schedule if c <= first_done]

            existing_hist = find_existing_history_cycles(tau_dir)
            to_run = [c for c in schedule if c not in existing_hist]

            if not to_run:
                # folder is "complete" for THIS schedule
                print(f"   ✓ τ={tau_value}: all scheduled cycles already present ({len(schedule)} pts).")
                # still re-finalize in case you changed schedule and want csv/fit refreshed
                finalize_tau_folder(tau_dir, diff_label, ramp_label, float(tau_value), all_fits)
                continue

            print(f"   → τ={tau_value}: running {len(to_run)} missing cycles (of {len(schedule)} scheduled).")

            pbar = tqdm(total=len(to_run), desc=f"τ={tau_value}", ncols=90)
            try:
                for nload in to_run:
                    proc = mp.Process(
                        target=child_run_one_cycle,
                        args=(float(tau_value), int(nload), tau_dir, base_text, mask_dir),
                    )
                    proc.start()
                    proc.join()

                    if proc.exitcode != 0:
                        print(f"   ✗ Error in cycle {nload} for τ={tau_value} (see {tau_dir/'run.log'})")
                        break

                    # check early-stop after each run
                    meta_file = tau_dir / f"cycle_{nload}_meta.json"
                    try:
                        meta = json.loads(meta_file.read_text())
                        if meta.get("done", False):
                            print("   → Early stop triggered (done=true).")
                            break
                    except Exception:
                        pass

                    pbar.update(1)

            finally:
                pbar.close()
                gc.collect()

            # finalize after adding points
            finalize_tau_folder(tau_dir, diff_label, ramp_label, float(tau_value), all_fits)

        RAW_MODEL = None
        gc.collect()

    # =============================================================
    # SAVE SUMMARY
    # =============================================================
    if all_fits:
        summary_df = pd.DataFrame(all_fits)
        safe_write_csv(SUMMARY_ROOT / "summary_fits.csv", summary_df)
        print("\n✔ Plastic simulations complete. Saved summary_fits.csv")
    else:
        # still write empty so pipeline doesn't crash
        safe_write_csv(SUMMARY_ROOT / "summary_fits.csv", pd.DataFrame())
        print("\n✔ No fits produced (maybe no groups had >=4 points). Wrote empty summary_fits.csv.")


if __name__ == "__main__":
    main()
