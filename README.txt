################################################################
README — CODE STRUCTURE
“The Role of Moisture Gradients and Time Scales in Wood Mechanosorption”
J. Amando de Barros, F. Wittel
Wood Science and Technology (2026)
################################################################

############################
OVERVIEW
############################

This repository contains the full simulation and analysis framework used in:

“The Role of Moisture Gradients and Time Scales in Wood Mechanosorption”
(Wood Science and Technology, 2026)

The workflow is divided into three main steps:

Generation of moisture profiles

Mechanical simulations

Data analysis and figure generation

All scripts are written in Python and organized into dedicated folders corresponding to each stage of the workflow.

Running the full pipeline requires executing the steps in order.

################################################################
STEP 1 — GENERATING MOISTURE PROFILES
################################################################

Moisture transport simulations are located in the folder:

Generate_Moisture_Profiles/

These scripts simulate diffusion-driven moisture fields and prepare spatially resolved moisture masks used later in the mechanical simulations.

Main scripts

Fiber_Grid.py
Contains the main definitions for the moisture transport model and simulation class.
Includes:

Moisture diffusion coefficients

Stopping criteria

Gradient calculations

Grid definition and time evolution

run_cycles_ramp.py
Runs moisture transport simulations for multiple:

Cyclic times

Ramping times

Generates time-dependent moisture profiles for different transport conditions.

mask_generator.py
Maps simulated moisture profiles onto the fiber bundle geometry.
Interpolates results to fixed time steps and creates moisture masks used by the mechanical model.

create_average_profiles.py
Uses generated masks to compute average moisture profiles.
These profiles are used to analyse the effect of homogeneous vs heterogeneous moisture distributions.

visualization.py
Generates visual outputs of the moisture simulations:

Videos

Images

Cycle-by-cycle moisture evolution

################################################################
STEP 2 — RUNNING MECHANICAL SIMULATIONS
################################################################

All scripts used to run the fiber bundle mechanical model and generate the simulation data are located in:

Generate_Data/

These scripts produce the mechanical results used in the publication.

############################
MAIN SIMULATION SCRIPTS
############################

single_run.py
Runs a single simulation using a selected input file.
Useful for testing, debugging, or exploring parameter variations.

non_rec_mech.py
Runs simulations with increasing number of loaded cycles to compute:

Recoverable mechanosorption

Non-recoverable mechanosorption

creep_analysis.py
Runs creep simulations with fixed moisture profiles to compute:

Effective viscoelastic compliance

overall_plasticity.py
Runs mechanosorptive plastic simulations for:

Fixed load degrees

Fixed moisture contents

run_all_new.py
Master script that runs all simulations required for the study.
Executes combinations of:

Diffusion times

Ramping times

Viscoelastic characteristic times

Average vs full moisture profiles

IMPORTANT:
Moisture masks must be generated first (STEP 1) before running this script.

############################
INPUT FILES
############################

These files define parameter sets used by the simulation scripts.

input_mech_plot.py
Input parameters for simulations used in Figure 6.

input_unitless_KV.py
Input parameters for creep simulations.

input_unitless_plastic.py
Input parameters for mechanosorptive plastic simulations.

input_many_run.py
General input file used by run_all_new.py to generate full datasets across multiple systems.

input_single_run.py
Minimal input file for simple test simulations and debugging.

############################
MODEL IMPLEMENTATION (Model_files)
############################

The folder:

Generate_Data/Model_files/

contains the core model, simulation engine, and visualization tools.

Model_class_copy_moist_grad_control_new.py
Main model definition.
Includes:

Fiber bundle mechanics

Slip and failure logic

Moisture coupling

Strain calculations

Checkpoint and restart functions

Sim_class_moist_grad.py
Main simulation engine.
Handles:

Time evolution

Load control

Moisture evolution

Cycle structure

Output storage

model_visualizer.py
Visualization tools for model state and results:

Slip maps

Histograms

Spatial fields

Debug visualization

################################################################
STEP 3 — ANALYSIS
################################################################

Analysis scripts are located in:

Analysis_Folder/

These scripts process simulation outputs and generate the results used in the paper figures.

Auxiliary folder:
Fourier_Meta/
Contains calculations related to Fourier numbers and characteristic diffusion times.

############################
ANALYSIS SCRIPTS
############################

compute_mechsorption.py
Computes mechanosorptive strain by combining:

Creep simulations

Plastic simulations

Base model outputs

Implements strain decomposition described in Section 3.1 of the paper.

run_mech_all.py
Computes mechanosorptive strain for all systems generated by run_all_new.py.

Gather_Results_2_0.py
Collects results from all simulations and generates a summary CSV file including:

Mechanosorptive strains

Total strains before unloading

Number of slip events

Fourier numbers

Additional metadata

Fo_eff_fit_2_0.py
Uses the CSV file generated above to:

Fit logistic functions

Compute effective Fourier numbers

Generate results shown in Figure 7

fit_many_Fo.py
Performs simultaneous fitting for multiple characteristic times (63%, 95%, 99%).
Used for results shown in Figure 8.

linear_load_limits.py
Analyses dependence of mechanosorptive strain on load degree.
Used to verify linearity assumptions (Figure 9).

Compare_Avg_Full.py
Compares simulations with:

Homogeneous moisture profiles

Heterogeneous moisture profiles

Analyses mechanosorptive strain and number of slip events (Figure 10).

################################################################
GENERAL NOTES
################################################################

All scripts are written in Python.

Recommended workflow:

Generate moisture profiles (STEP 1)

Run mechanical simulations (STEP 2)

Run analysis scripts (STEP 3)

Some scripts depend on previously generated data.
Ensure required input files and moisture masks exist before running batch simulations.

For exploratory runs or debugging, use:
single_run.py with input_single_run.py.
