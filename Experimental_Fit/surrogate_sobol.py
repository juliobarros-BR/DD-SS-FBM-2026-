"""
Surrogate modeling, global sensitivity analysis (Sobol), and candidate generation
for DD-SS-FBM parameter calibration.

Workflow
--------
1. Load simulation dataset (previous runs).
2. Train surrogate models for:
   - mechanosorption score (ms_score)
   - effective compliance (J_eff_kv)
3. Define a combined objective function.
4. Perform global sensitivity analysis (Sobol indices).
5. Sample new candidate parameters using the surrogate.
6. Rank, cluster, and export promising candidates.

Author: Your Name
"""

# ============================================================
# IMPORTS
# ============================================================
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from SALib.sample import saltelli
from SALib.analyze import sobol

import matplotlib.pyplot as plt

# ============================================================
# CONFIGURATION
# ============================================================
DATA_FILE = Path("MS_KV_mean_std/candidate_dual_fit_results.csv")
OUTPUT_DIR = Path("surrogate_analysis")
OUTPUT_DIR.mkdir(exist_ok=True)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

TARGET_J = 0.01045

N_NEW_SAMPLES = 20000
N_TOP = 200
N_CLUSTERS = 3

N_SOBOL = 2000

# ============================================================
# PARAMETERS
# ============================================================
PARAM_NAMES = [
    "J_w_kv",
    "m_Weibull_kv",
    "lambda_Weibull_kv",
    "decay_kv",
    "failure_limit_kv",
    "wet_scale_kv",
    "reverse_scale_kv",
]

# ============================================================
# LOAD DATA
# ============================================================
print("Loading dataset...")
df = pd.read_csv(DATA_FILE)

# Keep valid simulations only
df = df[np.isfinite(df["ms_score"]) & np.isfinite(df["J_eff_kv"])].copy()

X = df[PARAM_NAMES].values
y_ms = df["ms_score"].values
y_j  = df["J_eff_kv"].values

print(f"Dataset size: {len(df)} samples")

# ============================================================
# TRAIN SURROGATES
# ============================================================
print("Training surrogate models...")

rf_ms = RandomForestRegressor(
    n_estimators=300,
    max_depth=None,
    random_state=RANDOM_SEED,
    n_jobs=-1,
)

rf_j = RandomForestRegressor(
    n_estimators=300,
    max_depth=None,
    random_state=RANDOM_SEED,
    n_jobs=-1,
)

rf_ms.fit(X, y_ms)
rf_j.fit(X, y_j)

print("Surrogates trained.")

# ============================================================
# OBJECTIVE FUNCTION
# ============================================================
def objective(ms, j):
    """
    Combined objective:
    - minimize mechanosorption error
    - penalize exceeding target J_eff
    """
    penalty = np.maximum(0, j - TARGET_J)
    return ms + 50.0 * penalty


# ============================================================
# PARAMETER BOUNDS
# ============================================================
bounds_min = df[PARAM_NAMES].min().values
bounds_max = df[PARAM_NAMES].max().values

# ============================================================
# SOBOL SENSITIVITY ANALYSIS
# ============================================================
print("Running Sobol sensitivity analysis...")

problem = {
    "num_vars": len(PARAM_NAMES),
    "names": PARAM_NAMES,
    "bounds": list(zip(bounds_min, bounds_max)),
}

param_values = saltelli.sample(
    problem,
    N_SOBOL,
    calc_second_order=False,
)

# Evaluate surrogate
ms_sobol = rf_ms.predict(param_values)
j_sobol  = rf_j.predict(param_values)
obj_sobol = objective(ms_sobol, j_sobol)

Si = sobol.analyze(problem, obj_sobol, calc_second_order=False)

sobol_df = pd.DataFrame({
    "parameter": PARAM_NAMES,
    "S1": Si["S1"],
    "ST": Si["ST"],
})

sobol_df.sort_values("ST", ascending=False, inplace=True)
sobol_df.to_csv(OUTPUT_DIR / "sobol_indices.csv", index=False)

print("\nTop influential parameters:")
print(sobol_df)

# Plot Sobol
plt.figure(figsize=(8, 5))
plt.barh(sobol_df["parameter"], sobol_df["ST"])
plt.xlabel("Total Sobol Index (ST)")
plt.title("Global Sensitivity (Objective)")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "sobol_ST.png", dpi=150)
plt.close()

# ============================================================
# GENERATE NEW CANDIDATES
# ============================================================
print("Sampling new candidate parameters...")

X_new = np.random.uniform(
    bounds_min,
    bounds_max,
    size=(N_NEW_SAMPLES, len(PARAM_NAMES)),
)

ms_pred = rf_ms.predict(X_new)
j_pred  = rf_j.predict(X_new)
obj_pred = objective(ms_pred, j_pred)

# Select best candidates
idx_best = np.argsort(obj_pred)[:N_TOP]

X_best = X_new[idx_best]
ms_best = ms_pred[idx_best]
j_best = j_pred[idx_best]

best_df = pd.DataFrame(X_best, columns=PARAM_NAMES)
best_df["ms_pred"] = ms_best
best_df["J_pred"] = j_best
best_df["objective"] = obj_pred[idx_best]

best_df.to_csv(OUTPUT_DIR / "best_candidates.csv", index=False)

print(f"Saved top {N_TOP} candidates.")

# ============================================================
# CLUSTERING (DIVERSITY)
# ============================================================
print("Clustering best candidates...")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_best)

kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_SEED)
labels = kmeans.fit_predict(X_scaled)

best_df["cluster"] = labels
best_df.to_csv(OUTPUT_DIR / "best_candidates_clustered.csv", index=False)

# ============================================================
# VISUALIZATION
# ============================================================
plt.figure(figsize=(6, 5))

for c in range(N_CLUSTERS):
    mask = labels == c
    plt.scatter(
        ms_best[mask],
        j_best[mask],
        alpha=0.7,
        label=f"cluster {c}",
    )

plt.axhline(TARGET_J, linestyle="--")
plt.xlabel("MS score (predicted)")
plt.ylabel("J_eff (predicted)")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "clusters.png", dpi=150)
plt.close()

print("Clustering plot saved.")

# ============================================================
# DONE
# ============================================================
print("\nPipeline completed successfully.")