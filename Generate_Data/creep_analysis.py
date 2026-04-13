#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 26 13:59:16 2025

@author: jortiz
"""

# -*- coding: utf-8 -*-
"""
Moisture sweep for bundle creep:
- Sweep D(m), J(m), and lambda_Weibull(m)
- Run creep simulations
- Fit KV (1..KV_MAX) via AIC-like score
- Save per-simulation KV-fit plots (data vs. best fit + annotation)
- Export CSV summary and overview plot

No composite compliance C is computed or plotted.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import curve_fit

from Model_files.Model_class_copy_moist_grad_control_new import Model
from Model_files.Sim_class_moist_grad import Simulate


# ================== USER CONFIG ================== #
BASELINE_INPUT = "input_unitless_KV.txt"

# Run for multiple loading degrees:
LOAD_D_LIST = [0.5]   # <-- set what you want

NUM_STEPS = 15
MOISTURE_MIN, MOISTURE_MAX = 0.0, 1.0

D_MIN, D_MAX = 1.00, 1.30
J_MIN, J_MAX = 0.20, 0.4

LAMBDA_DRY  = 7.0
WET_SCALE   = 0.7

INTERP_SHAPE_D       = "linear"
INTERP_SHAPE_J       = "linear"
INTERP_SHAPE_LAMBDA  = "linear"

KV_FIXED = 4
TAU_START = 1e-1
PLOT_TAU_LOGSPACE = True

OUTPUT_ROOT = Path("moisture_sweep_results22")  # <-- root folder only now
SAVE_INDIVIDUAL_FIT_PLOTS = True
RANDOM_SEED = 39
# ================================================ #


# ---------------- utilities ---------------- #
def interp_curve(x, x0=0.0, x1=1.0, shape="linear"):
    if x1 == x0:
        return 0.0
    u = np.clip((x - x0) / (x1 - x0), 0.0, 1.0)
    if shape == "linear":
        return u
    if shape == "quadratic":
        return u**2
    if shape == "cubic":
        return u**3
    raise ValueError(f"Unknown shape '{shape}'")

def blend(v0, v1, s):
    return (1 - s) * v0 + s * v1

def replace_param(text, name, new_value):
    lines, found = text.splitlines(), False
    out = []
    tgt = name + " "
    for L in lines:
        if L.strip().startswith(tgt):
            lhs, *rhs = L.split("#", 1)
            comment = ("#" + rhs[0]) if rhs else ""
            out.append(f"{name} = {new_value} {comment}".rstrip())
            found = True
        else:
            out.append(L)
    if not found:
        out.append(f"{name} = {new_value}")
    return "\n".join(out)

def load_tag(load_d: float) -> str:
    # 0.35 -> "0p35"
    s = f"{load_d:.4f}".rstrip("0").rstrip(".")
    return s.replace(".", "p")
# -------------------------------------------- #


def run_single_creep_fit(
    baseline_model,
    base_file_path,
    D_val,
    J_val,
    lambda_val,
    load_d,
    output_dir: Path,
    run_tag=None,
):
    """
    Run one creep simulation overriding D_d, J_d, lambda_Weibull AND load_d.
    Saves everything inside output_dir.
    """
    txt = Path(base_file_path).read_text()
    txt = replace_param(txt, "D_d", D_val)
    txt = replace_param(txt, "J_d", J_val)
    txt = replace_param(txt, "lambda_Weibull", lambda_val)
    txt = replace_param(txt, "load_d", load_d)  # <-- NEW

    output_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{run_tag}" if run_tag is not None else "run"

    tmp_file = output_dir / f"tmp_input_{tag}_Ld{load_d:.4f}_D{D_val:.4f}_J{J_val:.4f}_lam{lambda_val:.4f}.txt"
    tmp_file.write_text(txt)

    model = Model(str(tmp_file), baseline_model.moisture_profiles)
    sim = Simulate(model)
    sim.run()

    if "Time" in sim.History and len(sim.History["Time"]) > 0:
        sim.History["Time"][0] = 0.0

    out_csv = output_dir / f"creep_{tag}_Ld{load_d:.4f}_D{D_val:.4f}_J{J_val:.4f}_lam{lambda_val:.4f}.csv"
    pd.DataFrame(sim.History).to_csv(out_csv, index=False)
    df = pd.read_csv(out_csv)

    # ---- critical_strain estimation (keep your logic) ----
    total = df["Total_strain"].to_numpy(dtype=float)
    eps_small = 1e-12

    if "Total_strain_norm" in df.columns:
        total_norm = df["Total_strain_norm"].to_numpy(dtype=float)
        ratio = np.divide(total, np.where(np.abs(total_norm) > eps_small, total_norm, np.nan))
        critical_strain = float(np.nanmedian(ratio))
        if not np.isfinite(critical_strain) or critical_strain == 0.0:
            critical_strain = 1.0
    else:
        critical_strain = 1.0

    # ---- Extract arrays and cut up to peak load ----
    load_all = np.asarray(sim.History["Load"], dtype=float)
    t_all    = np.asarray(sim.History["Time"], dtype=float)
    eps_all  = np.asarray(sim.History["Total_strain"], dtype=float)

    load_limit = np.where(load_all == np.max(load_all))[-1][-1]

    load_raw = load_all[:load_limit]
    t_raw    = t_all[:load_limit]
    eps_raw  = eps_all[:load_limit]

    # ---- User rule: elastic strain is at index 3 ----
    ELASTIC_IDX = 3
    if len(eps_raw) <= ELASTIC_IDX or len(t_raw) <= ELASTIC_IDX:
        raise RuntimeError(
            f"History too short for ELASTIC_IDX={ELASTIC_IDX}. "
            f"len(eps_raw)={len(eps_raw)}, len(t_raw)={len(t_raw)}"
        )

    eps_elastic_measured = float(eps_raw[ELASTIC_IDX])

    t = t_raw[ELASTIC_IDX:] - t_raw[ELASTIC_IDX]
    eps = eps_raw[ELASTIC_IDX:]

    y = (eps - eps_elastic_measured) / critical_strain
    t_ref = 0.0

    # stress amplitude (constant during hold)
    sigma0 = model.critical_load * model.sys_var["load_d"]

    # ---- FIXED-KV fit ----
    KV_num = int(KV_FIXED)
    if KV_num <= 0:
        raise ValueError("KV_FIXED must be >= 1")

    if PLOT_TAU_LOGSPACE:
        taus = np.logspace(np.log10(TAU_START), np.log10(TAU_START) + (KV_num - 1), KV_num)
    else:
        taus = np.linspace(TAU_START, TAU_START * KV_num, KV_num)

    def kv_model(tt, *a):
        dt = np.clip(tt - t_ref, 0.0, None)
        return (sigma0 / critical_strain) * np.sum(
            [a[i] * (1.0 - np.exp(-dt / taus[i])) for i in range(KV_num)],
            axis=0
        )

    p0 = np.ones(KV_num) * max(J_val / KV_num, 1e-12)

    popt, _ = curve_fit(kv_model, t, y, p0=p0, bounds=(0.0, np.inf), method="trf")
    y_fit = kv_model(t, *popt)
    resid = y - y_fit
    rss = float(np.sum(resid**2))
    J_eff = float(np.sum(popt))

    if SAVE_INDIVIDUAL_FIT_PLOTS:
        lines = [f"J_{i+1} = {J:.4g}" for i, J in enumerate(popt)]
        lines.append(f"ΣJ = {np.sum(popt):.4g}")
        lines.append(f"J_input = {J_val:.4g}")
        lines.append(f"load_d = {load_d:.4g}")
        lines.append(f"KV = {KV_num}")
        lines.append(f"RSS = {rss:.3g}")
        text_str = "\n".join(lines)

        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(t, y, "+", label="Total Strain - Elastic Strain (data)")
        ax.plot(t, y_fit, "--", label=f"KV Fit ({KV_num} elements)")
        ax.legend()
        ax.text(
            0.70, 0.35, text_str,
            transform=ax.transAxes,
            fontsize=12, va="top",
            bbox=dict(facecolor="white", edgecolor="black", alpha=0.7)
        )
        ax.set_ylim(bottom=0)
        ax.set_xlabel(r"Normalized Time ($t/\tau_1$)")
        ax.set_ylabel(r"Normalized Creep Strain ($\varepsilon_{ve}/\varepsilon_c$)")
        ax.grid(True)
        ax.set_title("Fixed Kelvin-Voigt Fit (KV=4)")
        fig.tight_layout()
        fig.savefig(
            output_dir / f"kvfit_{tag}_Ld{load_d:.4f}_D{D_val:.4f}_J{J_val:.4f}_lam{lambda_val:.4f}.png",
            dpi=200
        )
        plt.close(fig)

    a_params = np.array(popt, dtype=float)
    taus_out = np.array(taus, dtype=float)

    return dict(
        t=t, y=y, y_fit=y_fit,
        taus=taus_out, KV=KV_num,
        a_params=a_params,
        J_eff=J_eff,
        rss=rss,

        load_d=float(load_d),
        J_target=float(J_val),
        D_imposed=float(D_val),
        lambda_weibull=float(lambda_val),
        sigma0=float(sigma0),

        elastic_idx=int(ELASTIC_IDX),
        eps_elastic_measured=float(eps_elastic_measured),
        critical_strain=float(critical_strain),
    )


def run_moisture_sweep_for_load(load_d: float):
    """
    Runs your entire moisture sweep for one load_d into its own folder.
    """
    output_dir = OUTPUT_ROOT / f"LoadD_{load_tag(load_d)}"
    output_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(RANDOM_SEED)
    baseline_model = Model(BASELINE_INPUT)  # for moisture_profiles

    m_vals = np.linspace(MOISTURE_MIN, MOISTURE_MAX, NUM_STEPS)
    lambda_wet = LAMBDA_DRY * WET_SCALE

    summary_rows = []
    kv_rows = []
    previews = []

    for m in m_vals:
        sD = interp_curve(m, MOISTURE_MIN, MOISTURE_MAX, INTERP_SHAPE_D)
        sJ = interp_curve(m, MOISTURE_MIN, MOISTURE_MAX, INTERP_SHAPE_J)
        sL = interp_curve(m, MOISTURE_MIN, MOISTURE_MAX, INTERP_SHAPE_LAMBDA)

        D_m = blend(D_MIN, D_MAX, sD)
        J_m = blend(J_MIN, J_MAX, sJ)
        lambda_m = blend(LAMBDA_DRY, lambda_wet, sL)

        run_tag = f"m{m:.3f}"
        res = run_single_creep_fit(
            baseline_model=baseline_model,
            base_file_path=BASELINE_INPUT,
            D_val=D_m,
            J_val=J_m,
            lambda_val=lambda_m,
            load_d=load_d,
            output_dir=output_dir,
            run_tag=run_tag
        )

        summary_rows.append({
            "load_d": load_d,
            "moisture_frac": m,
            "D_imposed": res["D_imposed"],
            "J_target": res["J_target"],
            "J_eff": res["J_eff"],
            "KV_fixed": res["KV"],
            "lambda_weibull": res["lambda_weibull"],
            "rss": res["rss"],
            "J1": float(res["a_params"][0]),
            "J2": float(res["a_params"][1]),
            "J3": float(res["a_params"][2]),
            "J4": float(res["a_params"][3]),
            "tau1": float(res["taus"][0]),
            "tau2": float(res["taus"][1]),
            "tau3": float(res["taus"][2]),
            "tau4": float(res["taus"][3]),
        })

        kv_rows.append({
            "load_d": load_d,
            "moisture_frac": m,
            "run_tag": run_tag,
            "D_imposed": res["D_imposed"],
            "J_target": res["J_target"],
            "lambda_weibull": res["lambda_weibull"],
            "KV": res["KV"],
            "rss": res["rss"],
            "J_eff": res["J_eff"],
            "J1": float(res["a_params"][0]),
            "J2": float(res["a_params"][1]),
            "J3": float(res["a_params"][2]),
            "J4": float(res["a_params"][3]),
            "tau1": float(res["taus"][0]),
            "tau2": float(res["taus"][1]),
            "tau3": float(res["taus"][2]),
            "tau4": float(res["taus"][3]),
        })

        if len(previews) < 3 or m in (m_vals[0], m_vals[len(m_vals)//2], m_vals[-1]):
            previews.append((m, res["t"], res["y"], res["y_fit"], res["KV"]))

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(output_dir / "moisture_sweep_summary.csv", index=False)

    df_kv = pd.DataFrame(kv_rows)
    df_kv.to_csv(output_dir / "kv_components_allruns.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(df_summary["moisture_frac"], df_summary["D_imposed"], marker="o", label="Fiber elastic compliance")
    ax.plot(df_summary["moisture_frac"], df_summary["J_target"], marker="s", label="Fiber viscoelastic compliance (input)")
    ax.plot(df_summary["moisture_frac"], df_summary["J_eff"],    marker="s", linestyle="--", label="Bundle viscoelastic compliance (fit)")
    ax.set_xlabel("Moisture content (%)")
    ax.set_xticks([0, 0.5, 1])
    ax.set_xticklabels([0.05, 0.175, 0.3])
    ax.set_ylabel("Compliance [-]")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2)
    ax.set_title(f"Compliance vs Moisture (KV fixed to 4) | load_d={load_d}")
    fig.tight_layout()
    fig.savefig(output_dir / "compliance_vs_moisture.png", dpi=200)
    plt.close(fig)

    if previews:
        fig2, ax2 = plt.subplots(figsize=(7, 4.5))
        for (m, t, y, y_fit, KV) in previews:
            ax2.plot(t, y, "+", label=f"m={m:.2f} data")
            ax2.plot(t, y_fit, "--", label=f"m={m:.2f} KV({KV})")
        ax2.set_xlabel(r"Normalized Time ($t/\tau_1$)")
        ax2.set_ylabel(r"$\varepsilon_{ve}$ (after subtracting elastic part)")
        ax2.grid(True, alpha=0.3)
        ax2.legend(ncol=2, fontsize=8)
        ax2.set_title(f"Fixed KV fits (samples) | load_d={load_d}")
        fig2.tight_layout()
        fig2.savefig(output_dir / "kv_fit_previews.png", dpi=200)
        plt.close(fig2)

    print("Saved:", output_dir / "moisture_sweep_summary.csv")
    print("Saved:", output_dir / "kv_components_allruns.csv")
    print("Saved:", output_dir / "compliance_vs_moisture.png")
    if previews:
        print("Saved:", output_dir / "kv_fit_previews.png")


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    for ld in LOAD_D_LIST:
        run_moisture_sweep_for_load(float(ld))


if __name__ == "__main__":
    main()
