import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from Model_files.Model_class_copy_moist_grad_control_new import Model
from Model_files.Sim_class_moist_grad import Simulate

# =========================================================
# CONFIG
# =========================================================
TEMPLATE_INPUT = "input_unitless_plastic.txt"

N_FIBERS = 40000

# Sweep over these maxima (each one also becomes the fixed load for Part B)
LOAD_D_MAX_LIST = [0.7]

# Keep failure_limit fixed (single value, not swept)
FAILURE_LIMIT = 0.3

# Model params (keep consistent across both sweeps)
PLASTIC_SLIPS = 0

# ---------------------------
# Part A: LOAD sweep at DRY
# ---------------------------
DRY_STEP_HUMAN = 1  # 1..n_steps (dry typically 1)
LOAD_D_START = 0.00
LOAD_D_STEP  = 0.1

# ---------------------------
# Part B: MOIST sweep at FIXED LOAD (= current load_d_max)
# ---------------------------
STEP_STRIDE  = 10  # moisture subsampling (1 => all)

SAVE_PLOTS = True

# =========================================================
# UTILITIES
# =========================================================
def edit_input(template, outpath, updates):
    with open(template, "r") as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        for key, val in updates.items():
            pat = rf'^(\s*{key}\s*=\s*)([-+eE0-9\.]+)(.*)$'
            m = re.match(pat, line)
            if m:
                prefix, _, suffix = m.groups()
                line = f"{prefix}{val}{suffix}\n"
                break
        new_lines.append(line)

    with open(outpath, "w") as f:
        f.writelines(new_lines)

def read_folder_moist_from_template(template_path):
    with open(template_path, "r") as f:
        txt = f.read()
    m = re.search(r"folder_moist\s*=\s*(.+)", txt)
    if not m:
        raise RuntimeError("folder_moist not found in input file")
    return m.group(1).strip()

def get_n_steps_first_cycle(folder_moist):
    files = sorted(Path(folder_moist).glob("mask_moistening_1_cycle.csv"))
    if not files:
        raise RuntimeError("mask_moistening_1_cycle.csv not found")
    df = pd.read_csv(files[0])
    return df["time_index"].nunique()

def run_one_sim(input_path):
    model = Model(input_path)
    sim = Simulate(model)
    sim.run()

    df = pd.DataFrame(sim.History)

    df.to_csv(str(input_path.parent / f"all_{input_path.stem}.csv"))

    slip_final = sim.History["Plastic_like"][-1]
    slip_max   = float(np.max(sim.History["Slip_strain"]))
    tot_mech   = float(np.max(sim.History["Total_strain"]) - np.max(sim.History["Hygroexp"]))
    elastic_max = float(np.max(sim.History["Elastic"]))
    n_fib = int(sim.History["Number_of_fibers"][-1])
    broken = int(bool(sim.History["Broken"][-1]))
    damage = 1.0 - n_fib / N_FIBERS
    moist_mean = float(np.max(sim.History["Moisture"]))

    return {
        "slip_final": slip_final,
        "slip_max": slip_max,
        "tot_mech_max": tot_mech,
        "ratio_slip_over_tot": slip_final / tot_mech if tot_mech != 0 else np.nan,
        "elastic_max": elastic_max,
        "n_fib": n_fib,
        "damage": damage,
        "broken": broken,
        "moist_max": moist_mean,
    }

# =========================================================
# ONE RUN (for a given load_d_max)
# =========================================================
def run_unified_sweep(load_d_max: float, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)

    folder_moist = read_folder_moist_from_template(TEMPLATE_INPUT)
    n_steps = get_n_steps_first_cycle(folder_moist)
    print(f"\n=== load_d_max={load_d_max:.3f} | failure_limit={FAILURE_LIMIT} ===")
    print(f"Detected {n_steps} steps in first moistening cycle")

    dry_step0 = DRY_STEP_HUMAN - 1
    if dry_step0 < 0 or dry_step0 >= n_steps:
        raise ValueError(f"DRY_STEP_HUMAN={DRY_STEP_HUMAN} out of range 1..{n_steps}")

    # Load list for Part A goes up to load_d_max
    load_list = np.arange(LOAD_D_START, load_d_max + 1e-12, LOAD_D_STEP)

    # Moisture step subsampling list for Part B
    step_list = list(range(0, n_steps, STEP_STRIDE))
    if (n_steps - 1) not in step_list:
        step_list.append(n_steps - 1)

    # =====================================================
    # PART A: load sweep at dry
    # =====================================================
    partA_dir = outdir / "partA_load_sweep"
    partA_dir.mkdir(exist_ok=True)

    rowsA = []
    for i, load_d in enumerate(load_list):
        tmp_input = partA_dir / f"input_A_{i:03d}.txt"
        edit_input(
            TEMPLATE_INPUT, tmp_input,
            {
                "failure_limit": float(FAILURE_LIMIT),
                "plastic_slips": int(PLASTIC_SLIPS),
                "frozen_moisture_step": int(dry_step0),
                "load_d": float(load_d),
            }
        )

        out = run_one_sim(tmp_input)
        out.update({
            "failure_limit": float(FAILURE_LIMIT),
            "load_d": float(load_d),
            "load_d_max": float(load_d_max),
            "frozen_moisture_step": int(dry_step0),
            "moist_step_human": int(DRY_STEP_HUMAN),
        })
        rowsA.append(out)

        print(f"[A] load_d={load_d:.3f} slip={out['slip_final']:.4e} broken={out['broken']}")
        if out["broken"]:
            print(f"   💥 broke at load_d={load_d:.3f} -> stopping Part A")
            break

    dfA = pd.DataFrame(rowsA)
    dfA.to_csv(partA_dir / "partA_load_sweep.csv", index=False)

    # =====================================================
    # PART B: moisture sweep at fixed load = load_d_max
    # =====================================================
    partB_dir = outdir / "partB_moist_sweep"
    partB_dir.mkdir(exist_ok=True)

    rowsB = []
    for j, step0 in enumerate(step_list):
        tmp_input = partB_dir / f"input_B_{step0:03d}.txt"
        edit_input(
            TEMPLATE_INPUT, tmp_input,
            {
                "failure_limit": float(FAILURE_LIMIT),
                "plastic_slips": int(PLASTIC_SLIPS),
                "frozen_moisture_step": int(step0),
                "load_d": float(load_d_max),  # <-- fixed load equals the current max
            }
        )

        out = run_one_sim(tmp_input)
        out.update({
            "failure_limit": float(FAILURE_LIMIT),
            "load_d": float(load_d_max),
            "load_d_max": float(load_d_max),
            "frozen_moisture_step": int(step0),
            "moist_step_human": int(step0 + 1),
        })
        rowsB.append(out)

        print(f"[B] step={step0:03d} load_d={load_d_max:.3f} slip={out['slip_final']:.4e} broken={out['broken']}")
        if out["broken"]:
            print(f"   💥 broke at moisture step {step0} -> stopping Part B")
            break

    dfB = pd.DataFrame(rowsB)
    dfB.to_csv(partB_dir / "partB_moist_sweep.csv", index=False)

    # =====================================================
    # PLOTTING
    # =====================================================
    if SAVE_PLOTS:
        figA, axA = plt.subplots(figsize=(6, 4), dpi=300)
        axA.plot(dfA["load_d"], dfA["slip_final"], marker="o", lw=2)
        axA.set_xlabel("load_d (dry state)")
        axA.set_ylabel(r"plastic strain proxy $\varepsilon_P$ (Slip_strain final)")
        axA.set_title(f"failure_limit={FAILURE_LIMIT}, load_d_max={load_d_max:.3f}")
        axA.grid(alpha=0.3)
        figA.tight_layout()
        figA.savefig(outdir / "plot_partA_load_sweep.png", dpi=300)
        plt.close(figA)

        figB, axB = plt.subplots(figsize=(6, 4), dpi=300)
        axB.plot(dfB["moist_step_human"], dfB["slip_final"], marker="o", lw=2)
        axB.set_xlabel("moisture step (cycle 1)")
        axB.set_ylabel(r"plastic strain proxy $\varepsilon_P$ (Slip_strain final)")
        axB.set_title(f"failure_limit={FAILURE_LIMIT}, load_d_fixed={load_d_max:.3f}")
        axB.grid(alpha=0.3)
        figB.tight_layout()
        figB.savefig(outdir / "plot_partB_moist_sweep.png", dpi=300)
        plt.close(figB)

        # Combined "progress" plot
        sA = np.linspace(0.0, 1.0, len(dfA), endpoint=False) if len(dfA) > 0 else np.array([])
        sB = np.linspace(1.0, 2.0, len(dfB), endpoint=True)  if len(dfB) > 0 else np.array([])

        figC, axC = plt.subplots(figsize=(7, 4), dpi=300)
        if len(dfA) > 0:
            axC.plot(sA, dfA["slip_final"], marker="o", lw=2, label="Part A: increase load (dry)")
        if len(dfB) > 0:
            axC.plot(sB, dfB["slip_final"], marker="o", lw=2, label="Part B: increase moisture (fixed load)")

        axC.axvline(1.0, ls="--", lw=1)
        axC.set_xlabel("experiment progress (0→1: load sweep, 1→2: moisture sweep)")
        axC.set_ylabel(r"plastic strain proxy $\varepsilon_P$")
        axC.set_title(f"failure_limit={FAILURE_LIMIT}, load_d_max={load_d_max:.3f}")
        axC.grid(alpha=0.3)
        axC.legend()
        figC.tight_layout()
        figC.savefig(outdir / "plot_combined_overall.png", dpi=300)
        plt.close(figC)

    print(f"Done: {outdir.resolve()}")

# =========================================================
# DRIVER
# =========================================================
def main():
    for load_d_max in LOAD_D_MAX_LIST:
        tag = f"LoadDmax_{load_d_max:.3f}".replace(".", "p")
        outdir = Path(f"unified_plastic_sweeps3_{tag}")
        run_unified_sweep(load_d_max, outdir)

if __name__ == "__main__":
    main()
