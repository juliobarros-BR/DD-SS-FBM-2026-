"""
Random-search calibration of DD-SS-FBM parameters against mechanosorption
and constant-moisture creep targets.

Workflow
--------
1. Sample candidate parameter sets around a manually selected mean/std using
   Latin Hypercube Sampling.
2. Run the mechanosorption simulation for each candidate.
3. Reject candidates that lead to broken systems.
4. Run constant-moisture KV/creep simulations.
5. Estimate effective compliance J_eff.
6. Rank candidates by combined mechanosorption and KV scores.

Author: J. Amando de Barros
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

# Limit BLAS/OpenMP oversubscription during joblib parallelization.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.optimize import curve_fit
from scipy.stats import norm, qmc

from Generate_Data.Model_files.Model_class_copy_moist_grad_control_new import Model
from Generate_Data.Model_files.Sim_class_moist_grad import Simulate


# =============================================================================
# USER SETTINGS
# =============================================================================

REFERENCE_CSV = Path("reference_mechanosorption_model.csv")

BASELINE_INPUT_MS = Path("input_MS.txt")
BASELINE_INPUT_KV = Path("input_KV.txt")

WORKDIR = Path("MS_KV_mean_std")

N_CANDIDATES = 500
N_JOBS = 6       ## Decrease it if too heavy for the system, or set to -1 to use all cores. Parallelization is only applied to candidate evaluation, not to the KV load scan.
RANDOM_SEED = 42

SAVE_PLOTS = True
SAVE_PARAM_JSON = True
SAVE_HISTORY = True

# =============================================================================
# PARAMETER SPACE
# =============================================================================

PARAM_NAMES = [
    "J_w",
    "m_Weibull",
    "lambda_Weibull",
    "decay",
    "failure_limit",
    "wet_scale",
    "reverse_scale",
]

PARAM_MEAN = {
    "J_w": 0.006017,
    "m_Weibull": 1.148311,
    "lambda_Weibull": 0.049582,
    "decay": 26.368723,
    "failure_limit": 0.426439,
    "wet_scale": 0.509833,
    "reverse_scale": 0.783811,
}

PARAM_STD = {
    "J_w": 0.0002,
    "m_Weibull": 0.036560,
    "lambda_Weibull": 0.0015,
    "decay": 11.69321,
    "failure_limit": 0.22,
    "wet_scale": 0.04,
    "reverse_scale": 0.21,
}

PARAM_BOUNDS = {
    "J_w": (0.000757, 0.01),
    "m_Weibull": (1.1, 2.0),
    "lambda_Weibull": (0.005, 0.5),
    "decay": (1.0, 80.0),
    "failure_limit": (0.05, 0.8),
    "wet_scale": (0.40, 1.00),
    "reverse_scale": (0.40, 1.00),
}

# =============================================================================
# KV FIT SETTINGS
# =============================================================================

KV_NUM = 4
TAU_START = 5.0
PLATEAU_FRAC = 0.99

J_EFF_TARGET = 0.0095
LOAD_SCAN = [1.0]


# =============================================================================
# OUTPUT PATHS
# =============================================================================

HISTORY_DIR = WORKDIR / "history"
INPUT_DIR = WORKDIR / "inputs"
PLOT_DIR = WORKDIR / "plots"
PARAM_DIR = WORKDIR / "params"

RESULTS_CSV = WORKDIR / "candidate_dual_fit_results.csv"
BEST_CSV = WORKDIR / "candidate_dual_fit_best.csv"
SAMPLED_CANDIDATES_CSV = WORKDIR / "sampled_candidates.csv"


# =============================================================================
# FILE AND INPUT UTILITIES
# =============================================================================

def prepare_directories() -> None:
    """Create all required output directories."""
    for folder in [WORKDIR, HISTORY_DIR, INPUT_DIR, PLOT_DIR, PARAM_DIR]:
        folder.mkdir(parents=True, exist_ok=True)


def replace_param(text: str, name: str, value: float) -> str:
    """
    Replace a parameter assignment in an input file.

    Preserves inline comments when possible.
    """
    lines = text.splitlines()
    output_lines = []
    found = False

    for line in lines:
        stripped = line.strip()

        if stripped.startswith(name) and "=" in stripped:
            lhs, *comment_parts = line.split("#", 1)
            comment = f"#{comment_parts[0]}" if comment_parts else ""
            output_lines.append(f"{name} = {value} {comment}".rstrip())
            found = True
        else:
            output_lines.append(line)

    if not found:
        output_lines.append(f"{name} = {value}")

    return "\n".join(output_lines)


def write_input_file(base_input: Path, output_path: Path, params: dict[str, float]) -> None:
    """Write a model input file by replacing selected parameters."""
    text = base_input.read_text()

    for name, value in params.items():
        text = replace_param(text, name, value)

    output_path.write_text(text)


def save_json(data: dict[str, Any], output_path: Path) -> None:
    """Save a dictionary as JSON."""
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def save_history_csv(sim: Simulate, output_path: Path) -> None:
    """Save simulation history to CSV."""
    if not SAVE_HISTORY:
        return

    try:
        pd.DataFrame(sim.History).to_csv(output_path, index=False)
    except Exception as exc:
        print(f"[warning] Could not save history to {output_path}: {exc}")


# =============================================================================
# MODEL EXECUTION
# =============================================================================

def run_simulation(input_file: Path) -> Simulate:
    """Run one DD-SS-FBM simulation."""
    model = Model(str(input_file))
    sim = Simulate(model)
    sim.run()

    if "Time" in sim.History and len(sim.History["Time"]) > 0:
        sim.History["Time"][0] = 0.0

    return sim


def is_broken(sim: Simulate) -> bool:
    """Return True if the simulation reached a broken state."""
    history = sim.History

    if "Broken" not in history:
        return False

    broken = np.asarray(history["Broken"], dtype=float)
    return bool(np.any(broken >= 1.0))


# =============================================================================
# PARAMETER SAMPLING
# =============================================================================

def apply_bounds(value: float, bounds: tuple[float | None, float | None]) -> float:
    """Apply lower and upper bounds to a scalar value."""
    lower, upper = bounds

    if lower is not None:
        value = max(value, lower)

    if upper is not None:
        value = min(value, upper)

    return value


def generate_candidates_lhs(n_candidates: int, seed: int) -> pd.DataFrame:
    """
    Generate candidates using Latin Hypercube Sampling.

    A uniform LHS sample u in (0,1) is transformed into a normal sample using:
        x = norm.ppf(u, loc=mean, scale=std)

    Hard parameter bounds are then applied.
    """
    sampler = qmc.LatinHypercube(d=len(PARAM_NAMES), seed=seed)
    samples = sampler.random(n=n_candidates)

    rows = []

    for candidate_index in range(n_candidates):
        row = {"candidate_index": candidate_index}

        for param_index, name in enumerate(PARAM_NAMES):
            mean = float(PARAM_MEAN[name])
            std = float(PARAM_STD[name])
            bounds = PARAM_BOUNDS.get(name, (None, None))

            u = np.clip(samples[candidate_index, param_index], 1e-12, 1.0 - 1e-12)
            value = norm.ppf(u, loc=mean, scale=std)
            value = apply_bounds(float(value), bounds)

            row[name] = value

        rows.append(row)

    return pd.DataFrame(rows)


# =============================================================================
# MECHANOSORPTION SCORING
# =============================================================================

def extract_strain_components(sim: Simulate) -> dict[str, np.ndarray]:
    """Extract relevant strain components from a simulation history."""
    history = sim.History

    return {
        "time": np.asarray(history["Time"], dtype=float),
        "total": np.asarray(history["Total_strain"], dtype=float),
        "elastic": np.asarray(history["Elastic"], dtype=float),
        "visco": np.asarray(history["Creep"], dtype=float),
        "hygro": np.asarray(history["Hygroexp"], dtype=float),
        "mechano": np.asarray(history["Slip_strain"], dtype=float),
    }


def interpolate_simulation_to_reference(
    sim_curve: dict[str, np.ndarray],
    reference_time: np.ndarray,
) -> dict[str, np.ndarray]:
    """Interpolate all simulation components to the reference time vector."""
    return {
        "total": np.interp(reference_time, sim_curve["time"], sim_curve["total"]),
        "elastic": np.interp(reference_time, sim_curve["time"], sim_curve["elastic"]),
        "visco": np.interp(reference_time, sim_curve["time"], sim_curve["visco"]),
        "hygro": np.interp(reference_time, sim_curve["time"], sim_curve["hygro"]),
        "mechano": np.interp(reference_time, sim_curve["time"], sim_curve["mechano"]),
    }


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute MSE while ignoring non-finite values."""
    valid = np.isfinite(y_true) & np.isfinite(y_pred)

    if not np.any(valid):
        return np.inf

    return float(np.mean((y_true[valid] - y_pred[valid]) ** 2))


def score_ms_fit(reference_total: np.ndarray, simulation_total: np.ndarray) -> float:
    """Score mechanosorption fit using total strain MSE."""
    return mean_squared_error(reference_total, simulation_total)


def plot_ms_fit(
    run_id: str,
    reference_df: pd.DataFrame,
    simulation_interp: dict[str, np.ndarray],
    score: float,
    output_path: Path,
) -> None:
    """Plot reference and simulated strain components."""
    time = reference_df["time_fbm"].to_numpy()

    components = [
        ("total", "strain_total", "Total strain"),
        ("elastic", "strain_elastic", "Elastic"),
        ("visco", "strain_viscoelastic", "Viscoelastic"),
        ("hygro", "strain_hygro", "Hygroexpansion"),
        ("mechano", "strain_mechanosorption", "Mechanosorption"),
    ]

    fig, axes = plt.subplots(len(components), 1, figsize=(7, 10), sharex=True)

    for ax, (sim_key, ref_key, label) in zip(axes, components):
        ax.plot(time, reference_df[ref_key].to_numpy(), label="Reference", lw=2)
        ax.plot(time, simulation_interp[sim_key], "--", label="DD-SS-FBM", lw=2)
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)

    axes[0].set_title(f"{run_id} | MS score = {score:.6e}")
    axes[0].legend()
    axes[-1].set_xlabel("Time")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


# =============================================================================
# KV / CREEP SCORING
# =============================================================================

def extract_jeff_plateau(sim: Simulate) -> dict[str, Any]:
    """
    Estimate effective creep compliance J_eff from the plateau creep segment.

    The creep segment starts when the load reaches PLATEAU_FRAC of its maximum.
    A generalized Kelvin-Voigt model is fitted to the creep strain.
    """
    history = sim.History

    time = np.asarray(history["Time"], dtype=float)
    strain = np.asarray(history["Total_strain"], dtype=float)
    load = np.asarray(history["Load"], dtype=float)

    if len(time) < 10:
        raise RuntimeError("Simulation too short for KV fitting.")

    max_load = float(np.max(load))
    plateau_mask = load > PLATEAU_FRAC * max_load

    if not np.any(plateau_mask):
        raise RuntimeError("No load plateau detected.")

    plateau_start = int(np.where(plateau_mask)[0][0])

    creep_time = time[plateau_start:] - time[plateau_start]
    creep_strain = strain[plateau_start:] - strain[plateau_start]

    if len(creep_time) < 6:
        raise RuntimeError("Creep segment too short for KV fitting.")

    sigma0 = float(sim.model.critical_load * sim.model.sys_var["load_d"])

    taus = np.logspace(
        np.log10(TAU_START),
        np.log10(TAU_START) + KV_NUM - 1,
        KV_NUM,
    )

    def kv_model(t: np.ndarray, *compliances: float) -> np.ndarray:
        response = np.zeros_like(t, dtype=float)

        for i in range(KV_NUM):
            response += compliances[i] * (1.0 - np.exp(-t / taus[i]))

        return sigma0 * response

    initial_guess = np.ones(KV_NUM) * 1e-4

    branch_compliances, _ = curve_fit(
        kv_model,
        creep_time,
        creep_strain,
        p0=initial_guess,
        bounds=(0.0, np.inf),
        maxfev=20_000,
    )

    fitted_strain = kv_model(creep_time, *branch_compliances)
    residual_sum_squares = float(np.sum((creep_strain - fitted_strain) ** 2))
    j_eff = float(np.sum(branch_compliances))

    return {
        "J_eff": j_eff,
        "branch_compliances": branch_compliances,
        "rss": residual_sum_squares,
        "t_creep": creep_time,
        "eps_creep": creep_strain,
        "y_fit": fitted_strain,
        "plateau_start": plateau_start,
        "t": time,
        "eps": strain,
    }


def score_kv_fit(j_eff: float) -> float:
    """Score KV fit by squared error to the target effective compliance."""
    return float((j_eff - J_EFF_TARGET) ** 2)


def plot_kv_fit(run_id: str, kv_result: dict[str, Any], output_path: Path) -> None:
    """Plot total strain and fitted KV creep segment."""
    fig, axes = plt.subplots(2, 1, figsize=(7, 8), dpi=150)

    axes[0].plot(kv_result["t"], kv_result["eps"], label="Total strain")
    axes[0].axvline(
        kv_result["t"][kv_result["plateau_start"]],
        color="r",
        linestyle="--",
        label="Plateau start",
    )
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(kv_result["t_creep"], kv_result["eps_creep"], "+", label="Creep data")
    axes[1].plot(kv_result["t_creep"], kv_result["y_fit"], "--", label="KV fit")
    axes[1].set_title(f"J_eff = {kv_result['J_eff']:.6g}")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


# =============================================================================
# CANDIDATE EVALUATION
# =============================================================================

def initialize_result(candidate_index: int, run_id: str, params: dict[str, float]) -> dict[str, Any]:
    """Create a default result row for one candidate."""
    return {
        "candidate_index": candidate_index,
        "run_id": run_id,
        **params,
        "broken_ms": np.nan,
        "broken_kv": np.nan,
        "ms_score": np.inf,
        "kv_score": np.inf,
        "combined_score": np.inf,
        "best_load_d": np.nan,
        "J_eff_kv": np.nan,
        "kv_rss": np.nan,
        "error": "",
    }


def evaluate_kv_scan(run_id: str, params: dict[str, float]) -> pd.DataFrame:
    """Run all KV simulations for one candidate and return their scores."""
    rows = []

    for load_d in LOAD_SCAN:
        kv_params = {
            "J_d": params["J_w"],
            "m_Weibull": params["m_Weibull"],
            "lambda_Weibull": params["lambda_Weibull"] * params["wet_scale"],
            "decay": params["decay"],
            "failure_limit": params["failure_limit"],
            "load_d": load_d,
        }

        kv_input = INPUT_DIR / f"{run_id}_KV_load_{load_d:.2f}.txt"
        write_input_file(BASELINE_INPUT_KV, kv_input, kv_params)

        try:
            sim_kv = run_simulation(kv_input)
            save_history_csv(sim_kv, HISTORY_DIR / f"{run_id}_KV_load_{load_d:.2f}_history.csv")

            if is_broken(sim_kv):
                rows.append({
                    "load_d": load_d,
                    "J_eff": np.nan,
                    "kv_score": np.inf,
                    "rss": np.nan,
                    "broken": True,
                })
                continue

            kv_result = extract_jeff_plateau(sim_kv)
            kv_score = score_kv_fit(kv_result["J_eff"])

            rows.append({
                "load_d": load_d,
                "J_eff": kv_result["J_eff"],
                "kv_score": kv_score,
                "rss": kv_result["rss"],
                "broken": False,
            })

            if SAVE_PLOTS:
                plot_kv_fit(
                    f"{run_id}_load_{load_d:.2f}",
                    kv_result,
                    PLOT_DIR / f"{run_id}_KV_load_{load_d:.2f}.png",
                )

        except Exception as exc:
            rows.append({
                "load_d": load_d,
                "J_eff": np.nan,
                "kv_score": np.inf,
                "rss": np.nan,
                "broken": True,
                "error": f"{type(exc).__name__}: {exc}",
            })

    kv_df = pd.DataFrame(rows)
    kv_df.to_csv(WORKDIR / f"{run_id}_kv_load_scan.csv", index=False)

    return kv_df


def evaluate_candidate(candidate_row: pd.Series, reference_df: pd.DataFrame) -> dict[str, Any]:
    """Evaluate one candidate parameter set."""
    candidate_index = int(candidate_row["candidate_index"])
    run_id = f"cand_{candidate_index:04d}"

    params = {name: float(candidate_row[name]) for name in PARAM_NAMES}
    result = initialize_result(candidate_index, run_id, params)

    if SAVE_PARAM_JSON:
        save_json(params, PARAM_DIR / f"{run_id}_params.json")

    try:
        # ---------------------------------------------------------------------
        # Mechanosorption run
        # ---------------------------------------------------------------------
        ms_input = INPUT_DIR / f"{run_id}_MS.txt"
        write_input_file(BASELINE_INPUT_MS, ms_input, params)

        sim_ms = run_simulation(ms_input)
        save_history_csv(sim_ms, HISTORY_DIR / f"{run_id}_MS_history.csv")

        if is_broken(sim_ms):
            result.update({
                "broken_ms": 1,
                "error": "MS simulation reached broken state.",
            })
            return result

        sim_curve = extract_strain_components(sim_ms)

        reference_time = reference_df["time_fbm"].to_numpy(dtype=float)
        simulation_interp = interpolate_simulation_to_reference(sim_curve, reference_time)

        reference_total = reference_df["strain_total"].to_numpy(dtype=float)
        ms_score = score_ms_fit(reference_total, simulation_interp["total"])

        if SAVE_PLOTS:
            plot_ms_fit(
                run_id,
                reference_df,
                simulation_interp,
                ms_score,
                PLOT_DIR / f"{run_id}_MS.png",
            )

        # ---------------------------------------------------------------------
        # KV scan
        # ---------------------------------------------------------------------
        kv_df = evaluate_kv_scan(run_id, params)
        valid_kv_df = kv_df[np.isfinite(kv_df["kv_score"])].copy()

        if valid_kv_df.empty:
            result.update({
                "broken_ms": 0,
                "broken_kv": 1,
                "ms_score": ms_score,
                "error": "All KV simulations failed.",
            })
            return result

        best_kv = valid_kv_df.sort_values("kv_score").iloc[0]
        combined_score = float(ms_score + best_kv["kv_score"])

        result.update({
            "broken_ms": 0,
            "broken_kv": 0,
            "ms_score": ms_score,
            "kv_score": float(best_kv["kv_score"]),
            "combined_score": combined_score,
            "best_load_d": float(best_kv["load_d"]),
            "J_eff_kv": float(best_kv["J_eff"]),
            "kv_rss": float(best_kv["rss"]),
        })

        return result

    except Exception as exc:
        result.update({
            "error": f"{type(exc).__name__}: {exc}",
        })
        return result


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    """Run the full parameter-search workflow."""
    prepare_directories()

    print("Loading reference data...")
    reference_df = pd.read_csv(REFERENCE_CSV)

    print("Generating candidate parameter sets...")
    candidate_df = generate_candidates_lhs(N_CANDIDATES, RANDOM_SEED)
    candidate_df.to_csv(SAMPLED_CANDIDATES_CSV, index=False)

    print(f"Saved sampled candidates to: {SAMPLED_CANDIDATES_CSV}")
    print(f"Evaluating {len(candidate_df)} candidates with {N_JOBS} workers...")

    rows = Parallel(n_jobs=N_JOBS)(
        delayed(evaluate_candidate)(row, reference_df)
        for _, row in candidate_df.iterrows()
    )

    results_df = pd.DataFrame(rows)
    results_df = results_df.sort_values("combined_score", ascending=True)

    results_df.to_csv(RESULTS_CSV, index=False)
    results_df.head(30).to_csv(BEST_CSV, index=False)

    print("\nSaved full results to:", RESULTS_CSV)
    print("Saved top candidates to:", BEST_CSV)

    summary_columns = [
        "candidate_index",
        "combined_score",
        "ms_score",
        "kv_score",
        "J_eff_kv",
        "best_load_d",
        "broken_ms",
        "broken_kv",
        "error",
    ]

    existing_columns = [col for col in summary_columns if col in results_df.columns]

    print("\nTop candidates:")
    print(results_df[existing_columns].head(10).to_string(index=False))

    print("\nDone.")


if __name__ == "__main__":
    main()