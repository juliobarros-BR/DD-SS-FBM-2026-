#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  3 14:00:27 2025

@author: jortiz
"""

import os, re, gc, copy, sys, traceback
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shutil

from Model_files.Model_class_copy_moist_grad_control_new import Model
from Model_files.Sim_class_moist_grad import Simulate

# =============================================================
# CONFIGURATION
# =============================================================
CWD = Path.cwd()
BASE_TREE_ROOT = (CWD / "Analysis_folders" / "Structured_Data").resolve()
MOISTURE_ROOT = (CWD / ".." / "Generate_Moisture_Profiles").resolve()
print(MOISTURE_ROOT)
BASE_INPUT = (CWD / "input_many_run.txt")

DEFAULT_TAUS = [0.1]


# Put whatever you want here (ints or floats)
# LOAD_DS = np.arange(0.05, 0.75, 0.05).round(3).tolist()  # 0.0 to 0.6 by 0.05
LOAD_DS = [0.7]
READONLY_MASKS = False

# =============================================================
# MANUAL SELECTION
# =============================================================
RUN_DIFFS = [100,300,600,1000,2000]            # only these diffusion values
RUN_AVG_MOISTURE = False       # True: only avg, False: only non-avg, None: both
RUN_RAMPS = []                # empty = all ramp fractions (integers like 10 for 10%)
RUN_TAUS  = []                # empty = all default taus
RUN_LOAD_DS = []              # empty = all in LOAD_DS
OVERWRITE = False             # overwrite existing results
# =============================================================


# ---------------- Helper functions ----------------

def parse_new_folder_name(name: str):
    """Extract diffusion (e.g. 300), ramp fraction (e.g. 0.10), and avg flag."""
    m = re.match(r"Moisture_Profiles_(\d+)_ramp([0-9.]+)(.*)", name)
    print(m)
    if not m:
        return None
    diff_val = int(m.group(1))
    ramp_val = float(m.group(2))
    suffix = m.group(3)
    is_avg = "_avg" in suffix
    return diff_val, ramp_val, is_avg


def _format_load_d_for_dir(load_d) -> str:
    """
    Directory-safe label for load_d.
    Examples:
      1.0 -> '1'
      0.25 -> '0p25'
      2 -> '2'
      1e-3 -> '0p001' (best-effort)
    """
    try:
        x = float(load_d)
    except Exception:
        return str(load_d).replace(".", "p")

    # Use fixed-ish formatting, then trim
    s = f"{x:.12g}"
    # avoid scientific notation in folder names if possible
    if "e" in s or "E" in s:
        s = f"{x:.12f}".rstrip("0").rstrip(".")
        if s == "":
            s = "0"
    s = s.replace("-", "m").replace(".", "p")
    return s


def build_input_text(base_text: str, folder_moist: str, tau_value: float, load_d_value: float) -> str:
    def _replace_number_keep_comment(txt: str, key: str, value: float) -> str:
        """
        Replace a line like:
          key = 0.6   # comment
        with:
          key = <value>   # comment

        Works even if there are tabs/spaces and optional inline comments.
        """
        pattern = rf"^(?P<pre>\s*{re.escape(key)}\s*=\s*)(?P<num>[-+eE0-9.]+)(?P<post>\s*(?:#.*)?)$"
        repl = rf"\g<pre>{value}\g<post>"
        if re.search(pattern, txt, flags=re.MULTILINE):
            return re.sub(pattern, repl, txt, flags=re.MULTILINE)
        else:
            # If the key doesn't exist at all, append it (keeps behavior robust)
            return txt.rstrip() + f"\n{key} = {value}\n"

    txt = base_text

    # folder_moist might not be a number; keep your original logic
    txt = re.sub(
        r"^(?P<k>\s*folder_moist\s*=\s*)(?P<v>.+?)\s*$",
        rf"\g<k>{folder_moist}",
        txt,
        flags=re.MULTILINE
    )

    # Replace tau and load_d robustly (comments allowed)
    txt = _replace_number_keep_comment(txt, "tau", float(tau_value))
    txt = _replace_number_keep_comment(txt, "load_d", float(load_d_value))

    return txt



def safe_write_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    with open(path, "rb") as f:
        os.fsync(f.fileno())


def safe_write_csv(path: Path, df: pd.DataFrame):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(df.to_csv(index=False))
    with open(tmp, "rb") as f:
        os.fsync(f.fileno())
    tmp.replace(path)


def safe_savefig(path: Path):
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()
    with open(path, "rb") as f:
        os.fsync(f.fileno())


def mark_ndarrays_readonly(obj):
    d = getattr(obj, "__dict__", None)
    if not d:
        return
    for v in d.values():
        if isinstance(v, np.ndarray):
            try:
                v.flags.writeable = False
            except Exception:
                pass
        elif isinstance(v, (list, tuple)):
            for x in v:
                if isinstance(x, np.ndarray):
                    try:
                        x.flags.writeable = False
                    except Exception:
                        pass
        elif isinstance(v, dict):
            for x in v.values():
                if isinstance(x, np.ndarray):
                    try:
                        x.flags.writeable = False
                    except Exception:
                        pass


def required_outputs_present(tau_dir: Path) -> bool:
    return (
        (tau_dir / "all.csv").exists()
        and (tau_dir / "results_total_strain.png").exists()
        and (tau_dir / "results_slip_strain.png").exists()
    )


# =============================================================
#  CHILD PROCESS LOGIC
# =============================================================
RAW_MODEL = None

def child_run_tau(tau_value: float, out_dir: Path, base_text: str, mask_dir: Path, load_d_value: float):
    try:
        out_dir.mkdir(parents=True, exist_ok=True)

        input_text = build_input_text(base_text, str(mask_dir), tau_value, load_d_value)
        input_name = f"{BASE_INPUT.stem}_load_d_{load_d_value}_tau_{tau_value}.txt"
        safe_write_text(out_dir / input_name, input_text)

        model = copy.copy(RAW_MODEL)
        if READONLY_MASKS:
            mark_ndarrays_readonly(model)

        if hasattr(model, "History"):   model.History.clear()
        if hasattr(model, "snapshots"): model.snapshots.clear()

        model.sys_var["tau"] = float(tau_value)
        model.sys_var["load_d"] = float(load_d_value)

        sim = Simulate(model)

        for attr, val in (
            ("n_final_snapshots", 0),
            ("snapshot_every_steps", 0),
            ("plot_frequency", 0),
        ):
            if hasattr(sim, attr):
                setattr(sim, attr, val)

        if tau_value <= 0.001:
            cycles_to_compare1 = 10
        elif 0.001 < tau_value < 0.1:
            cycles_to_compare1 = 50
        else:
            cycles_to_compare1 = 100

        sim.run(compare_n_cycles_ago=cycles_to_compare1)

        hist = sim.History
        if len(hist.get("Time", [])) > 0:
            hist["Time"][0] = 0.0

        df = pd.DataFrame(hist)
        crit = float(getattr(sim.model, "critical_strain", 1.0)) or 1.0

        df["Total_strain_norm"] = np.asarray(df.get("Total_strain", 0), dtype=np.float32) / np.float32(crit)
        df["Slip_strain_norm"]  = np.asarray(df.get("Slip_strain", 0), dtype=np.float32) / np.float32(crit)

        for c in df.columns:
            if df[c].dtype == np.float64:
                df[c] = df[c].astype(np.float32)

        safe_write_csv(out_dir / "all.csv", df)

        t = np.asarray(df["Time"], dtype=np.float32) if "Time" in df else np.arange(len(df), dtype=np.float32)

        plt.figure()
        plt.plot(t, df["Total_strain_norm"])
        plt.xlabel("Normalized Time (t/τ)")
        plt.ylabel("Normalized Total Strain (ε/εc)")
        plt.grid(True)
        safe_savefig(out_dir / "results_total_strain.png")

        plt.figure()
        plt.plot(t, df["Slip_strain_norm"])
        plt.xlabel("Normalized Time (t/τ)")
        plt.ylabel("Slip Strain (εs/εc)")
        plt.grid(True)
        safe_savefig(out_dir / "results_slip_strain.png")

        del df, t, sim, model, hist
        gc.collect()
        return

    except Exception:
        try:
            (out_dir / "run.log").write_text("ERROR:\n" + "".join(traceback.format_exception(*sys.exc_info())))
        except Exception:
            pass
        raise


# =============================================================
# MAIN LOOP (NOW WITH load_d LEVEL)
# =============================================================

def main():
    import multiprocessing as mp
    mp.set_start_method("fork", force=True)

    base_text = BASE_INPUT.read_text()

    all_new_folders = sorted([p for p in MOISTURE_ROOT.iterdir() if p.is_dir() and p.name.startswith("Moisture_Profiles_")])
    print(all)
    # Select load_d values to run
    load_d_values = [ld for ld in LOAD_DS if (not RUN_LOAD_DS or ld in RUN_LOAD_DS)]
    if len(load_d_values) == 0:
        print("⚠️ No load_d selected. Check LOAD_DS / RUN_LOAD_DS.")
        return
    print("at least one load_d selected:", load_d_values)
    for mask_dir in all_new_folders:
        print(mask_dir)
        parsed = parse_new_folder_name(mask_dir.name)
        if parsed is None:
            print(f"Skipping unrecognized folder: {mask_dir.name}")
            continue

        diff_val, ramp_val, is_avg = parsed

        # --- Filters ---
        if RUN_DIFFS and diff_val not in RUN_DIFFS:
            continue
        if RUN_RAMPS and int(ramp_val * 100) not in RUN_RAMPS:
            continue

        # --- Avg moisture logic ---
        if RUN_AVG_MOISTURE is True and not is_avg:
            continue
        if RUN_AVG_MOISTURE is False and is_avg:
            continue

        diff_label = f"Diff_{diff_val}_avg" if is_avg else f"Diff_{diff_val}"
        ramp_label = f"Ramp_{int(ramp_val * 100):02d}%"
        ramp_dir = BASE_TREE_ROOT / diff_label / ramp_label
        ramp_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n→ Preparing {mask_dir.name}")
        print(f"   → {ramp_dir.relative_to(BASE_TREE_ROOT)}")

        # --- Select tau values to run ---
        tau_values = [t for t in DEFAULT_TAUS if (not RUN_TAUS or t in RUN_TAUS)]
        if len(tau_values) == 0:
            print("   ⚠️ No tau selected. Check DEFAULT_TAUS / RUN_TAUS.")
            continue

        # --- Optional PRE-CHECK: skip entire mask_dir if all load_d and all tau done ---
        if not OVERWRITE:
            all_done = True
            for ld in load_d_values:
                load_label = f"LoadD_{_format_load_d_for_dir(ld)}"
                load_dir = ramp_dir / load_label
                for tau_value in tau_values:
                    tau_dir = load_dir / f"Tau_{tau_value}"
                    if not required_outputs_present(tau_dir):
                        all_done = False
                        break
                if not all_done:
                    break

            if all_done:
                print(f"✓ Skipping {mask_dir.name} — all load_d × tau simulations already complete.")
                continue

        # ----------------------------
        # Loop over load_d (NEW LEVEL)
        # ----------------------------
        for ld in load_d_values:
            load_label = f"LoadD_{_format_load_d_for_dir(ld)}"
            load_dir = ramp_dir / load_label
            load_dir.mkdir(parents=True, exist_ok=True)

            # --- PRE-CHECK: skip this load_d if all tau done ---
            if not OVERWRITE:
                all_tau_done = True
                for tau_value in tau_values:
                    tau_dir = load_dir / f"Tau_{tau_value}"
                    if not required_outputs_present(tau_dir):
                        all_tau_done = False
                        break
                if all_tau_done:
                    print(f"   ✓ Skipping {load_label} — all tau simulations already complete.")
                    continue

            print(f"   → load_d = {ld}  ({load_label})")

            # --- Generate input template for THIS load_d (so Model init matches) ---
            template_text = build_input_text(base_text, str(mask_dir), 0.0001, float(ld))
            tmp_input = load_dir / f"{BASE_INPUT.stem}__TEMPLATE__{load_label}.txt"
            safe_write_text(tmp_input, template_text)

            global RAW_MODEL
            RAW_MODEL = Model(str(tmp_input))

            # --- Run all tau folders under this load_d ---
            for tau_value in tau_values:
                tau_dir = load_dir / f"Tau_{tau_value}"
                tau_dir.mkdir(parents=True, exist_ok=True)

                if not OVERWRITE and required_outputs_present(tau_dir):
                    print(f"     ✓ Skipping Tau={tau_value} — outputs already present.")
                    continue

                if OVERWRITE:
                    for f in tau_dir.iterdir():
                        try:
                            if f.is_file() or f.is_symlink():
                                f.unlink()
                            elif f.is_dir():
                                shutil.rmtree(f)
                        except Exception as e:
                            print(f"     ⚠️ Could not remove {f}: {e}")

                print(f"     - Running Tau={tau_value}")
                p = mp.Process(
                    target=child_run_tau,
                    args=(tau_value, tau_dir, base_text, mask_dir, float(ld)),
                    daemon=False,
                )
                p.start()
                p.join()

                if p.exitcode != 0:
                    print(f"     ✗ Failed: {tau_dir.name}")
                    logf = tau_dir / "run.log"
                    if logf.exists():
                        print(f"       See log: {logf}")
                elif not required_outputs_present(tau_dir):
                    print(f"     ✗ Outputs missing: {tau_dir.name}")
                else:
                    print(f"     ✓ Completed: {tau_dir.name}")

            RAW_MODEL = None
            gc.collect()

        gc.collect()

    print("\nAll selected systems processed successfully.")


if __name__ == "__main__":
    main()
