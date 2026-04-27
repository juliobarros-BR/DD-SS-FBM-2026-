################################################################
README — Simulation Framework
“The Role of Moisture Gradients and Time Scales in Wood Mechanosorption”
J. Amando de Barros, F. Wittel
Wood Science and Technology (2026)
################################################################


## Overview

This repository contains the full simulation and analysis framework used to study mechanosorption in wood using a moisture-dependent stick–slip Fiber Bundle Model (FBM) coupled to moisture transport simulations.

The workflow is organized into three main stages:

1. Moisture Transport Simulations  
2. Mechanical Simulations (FBM)  
3. Post-processing & Analysis  

Each stage is modular and stored in dedicated folders.


------------------------------------------------------------------------------------------

## Repository Structure

Main folders:

- Generate_Moisture_Profiles/       → Moisture transport simulations  
- Generate_Data/                    → FBM simulations  
- Pos_Processing_Folder/            → Analysis & results figures  
- Characteristic_time_definition/   → Characteristic time definition and interpretation  
- Experimental_Fit/                 → Parameter calibration and surrogate modeling  


------------------------------------------------------------------------------------------

## 1. Moisture Transport Simulations

Folder:
    Generate_Moisture_Profiles/

Purpose:
Simulates diffusion-driven moisture transport and generates spatial moisture masks used in the mechanical model.

### Key scripts

- Fiber_Grid.py  
  Core diffusion model (grid definition, transport, evolution)

- run_cycles_ramp.py  
  Runs cyclic drying/moistening simulations and builds fiber mappings

- create_average_profiles.py  
  Generates homogenized moisture profiles for comparison

- visualization.py  
  Produces images/videos of moisture evolution (Fig. 2)

### Outputs

- mask_*_cycle.csv  
  → These files are REQUIRED input for mechanical simulations

------------------------------------------------------------------------------------------

## 2. Characteristic Time Definition

Folder:
    Characteristic_time_definition/

Purpose:
Provides validation and interpretation tools for characteristic time scales governing the system, including diffusion and viscoelastic responses.

### Key scripts

- characteristic_Chi.py  
  Computes diffusion-related characteristic times  

- characteristic_J.py  
  Evaluates viscoelastic characteristic times  

- iso_effective.py, isotherms.py  
  Supporting analyses for finding the Fourier numbers in experimental examples

### Outputs

- Figures used for validation:
  - Fig. 3 (time-scale comparison)  
  - Fig. 4 (analytical vs numerical behavior)  

- Supporting dataset:
  - long_moistening.csv       (Long moisture simulation to have "equilibrium" case)
  
------------------------------------------------------------------------------------------

## 3. Mechanical Simulations

Folder:
    Generate_Data/

Purpose:
Implements the moisture-dependent stick–slip Fiber Bundle Model and generates all simulation data.

### Core Model (Model_files/)

- Model_class_copy_moist_grad_control_new.py  
  Full FBM implementation:
  elasticity, viscoelasticity, slip, moisture coupling

- Sim_class_moist_grad.py  
  Simulation engine:
  time evolution, cycles, loading

- model_visualizer.py  
  Debugging and visualization tools

- slip_checkpoints.zarr  
  Event-driven simulation data used as initialization reference

### Running Simulations

- run_all_new.py  
  MAIN script for generating full datasets

- single_run.py  
  Minimal simulation (for debugging and testing)

### Strain Decomposition Scripts

Located in:
    Generate_Data/Strain_Decomposition/

Includes:

- creep_analysis.py → viscoelastic response  
- overall_plasticity.py → plasticity behavior  
- compute_mech2.py → strain decomposition  
- non_rec_mech.py → recoverable vs non-recoverable mechanosorption  
- input_unitless_KV.txt → creep simulations  
- input_unitless_plastic.txt → plastic simulations  
- plot scripts for Fig. 5 and 6  

### Input Files

- input_many_run.txt → full parameter sweeps  
- input_single_run.txt → minimal test  

### Output

- Structured results:
    Simulations_folder/Structured_Data/
    
- Stain decomposition aux. files:
    Strain_Decomposition/moisture_sweep_results/ , Strain_Decomposition/plastic_analysis/ , Strain_Decomposition/mechsorption_analysis/


------------------------------------------------------------------------------------------

## 4. Post-processing & Analysis

Folder:
    Pos_Processing_Folder/

Purpose:
Processes simulation outputs and generates figures used in the publication.

### Subfolders

- fourier_*  
  Used for:
  - Parameter fitting  
  - Characteristic time extraction  
  - Logistic and linear fits  

### Key scripts

- Gather_Results_2_0.py  
  MAIN aggregation script  
  → builds master dataset

- Fo_eff_master_curve.py  
  Computes effective Fourier numbers and fits master curve (Fig. 7)

- Fo_eff_master_different_characteristic.py  
  Computes master curves for multiple characteristic definitions (Fig. 8)

- Analyze_Load_Degree.py  
  Load dependence analysis (Fig. 9)

- Compare_Avg_Full.py  
  Compares homogeneous vs heterogeneous moisture (Fig. 10)


------------------------------------------------------------------------------------------

## 5. Experimental Fit & Surrogate Modeling

Folder:
    Experimental_Fit/

Purpose:
Generates datasets for parameter calibration, fits the model to experimental data, and builds surrogate models for efficient exploration of the parameter space.

This module enables:
- Comparison between simulations and experimental mechanosorption data  
- Training of surrogate models
- Global sensitivity analysis (Sobol indices)  
- Identification of optimal parameter regions  

### Workflow

1. Generate dataset from simulations  
2. Merge results into a structured dataset  
3. Train surrogate models  
4. Perform sensitivity analysis  
5. Explore optimal parameter regions  

### Key scripts

- generate_scored_data.py  
  Runs simulations over a defined parameter space and evaluates fit quality  

- surrogate_sobol.py  
  Performs Sobol sensitivity analysis and train surrogate model to generate new input candidates

- new_clustering.py  
  Clusters optimal candidates in parameter space  

### Outputs

- Best-fit candidates and parameter sets  
- Sensitivity indices (Sobol)  
- Clustering of optimal regions  
- Final comparison plots:
  - Fig. 11 (fit to experimental data)  
  - Fig. 12 (model analysis)

------------------------------------------------------------------------------------------

## Workflow Summary

Moisture diffusion  
        ↓  
FBM simulations  
        ↓  
Strain decomposition  
        ↓  
Final figures & fits  


------------------------------------------------------------------------------------------

## Important Notes

- Moisture profiles MUST be generated before simulations.  

- Several scripts depend on previously generated data.  

- Large simulations can produce heavy datasets  
  → process data per cycle when possible.  


------------------------------------------------------------------------------------------

## Recommended Usage

- Debug / exploration:
    single_run.py  

- Full dataset generation:
    run_all_new.py  

- Final aggregation:
    Gather_Results_2_0.py  

- Plotting:
    plot_* scripts  
