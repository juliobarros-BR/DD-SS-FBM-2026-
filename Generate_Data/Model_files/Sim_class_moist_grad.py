#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 10:25:24 2024

@author: jortiz
"""


import ast
import numpy as np
import os
import copy
import time
from tqdm import tqdm
from pathlib import Path
import pickle
import joblib

class Simulate:
    def __init__(self, model,plastic=False):
        self.model = model
        self.current_time = 0
        self.current_moist = int(self.model.sys_var["start_wet"])
        self.current_load = 0
        
        self.plastic = plastic
        
        # print(self.current_moist)
        if not model.sys_var["has_moist"] and not model.sys_var["has_load"]:
            self.moist_sequence, self.load_sequence, self.time_sequence = self.create_sim_sequence(model.sys_var["cycles_pre_load"], model.sys_var["cycles_loaded"], model.sys_var["cycles_unload"], model.sys_var["period"]**(1/model.sys_var["KV_num"]), model.sys_var["creep_test"])
        else:
            self.moist_sequence, self.load_sequence, self.time_sequence = self.get_from_data_file(model.sys_var["moisture_df"],model.sys_var["load_df"])
            
        self.load_sequence = list(np.array(self.load_sequence)[:]*self.model.critical_load*self.model.sys_var["load_d"])
        self.time_sequence = list(np.array(self.time_sequence)[:]*1)
        
        self.History = {
            "Total_strain": np.array([]),
            "Slip_strain": np.array([]),
            "Time": np.array([]),
            "Slip Count": np.array([]),
            "Creep": np.array([]),
            "Elastic": np.array([]),
            "Hygroexp": np.array([]),
            "Number_of_fibers": np.array([]),
            "Load": np.array([]),
            "Moisture": np.array([]),
            "Broken": [],
            "Hygro_weighted": np.array([]),
            "Plastic_like": np.array([]),
        }
        
        
        self.slip_hist_loaded   = []
        self.current_i=0
        if self.plastic and self.model.sys_var["cycles_loaded"] > 1:
            ckpt_file = Path(self.model.input_path).parent / "latest_checkpoint.pkl"
            # print(ckpt_file)
            if ckpt_file.exists():
                print(f"Loading checkpoint from {ckpt_file}")
                with open(ckpt_file, "rb") as f:
                    restored_model, restored_sim = pickle.load(f)
        
                # restore model
                self.model = restored_model
        
                # restore persistent sim state, but keep the new sequences
                for key, val in restored_sim.__dict__.items():
                    if key in ["moist_sequence", "load_sequence", "time_sequence"]:
                        continue  # keep freshly generated ones
                    self.__dict__[key] = val

        
    def update_history(self):
        # --- DD-SS weighted hygro and plastic-like strain ---
        intact = (self.model.local_intact == 1)
        phi = self.model.fiber_moisture[intact]
        D_i = self.model.D_min + self.model.D_lin_coeff * phi
        w = 1.0 / D_i

        eps_hyg_i = self.model.alpha * (phi - self.model.sys_var["start_wet"])
        eps_hyg_w = np.sum(w * eps_hyg_i) / np.sum(w)

        plastic_like = self.model.total_strain - eps_hyg_w

        self.History["Hygro_weighted"] = np.append(
            self.History["Hygro_weighted"], eps_hyg_w
        )
        self.History["Plastic_like"] = np.append(
            self.History["Plastic_like"], plastic_like
        )


        # ------------------------------------------------------------
        # Moisture + compliances for intact fibers (DD-SS weights)
        # ------------------------------------------------------------
        phi_intact = self.model.fiber_moisture[intact]
        D_i = self.model.D_min + self.model.D_lin_coeff * phi_intact
        wD = 1.0 / D_i  # DD-SS consistent weight

        # ------------------------------------------------------------
        # Elastic history (DD-SS-consistent weighted mean)
        # ------------------------------------------------------------
        elastic_i = self.model.eps_elastic[intact]  # already set in update_total_strain()
        elastic_weighted = np.sum(wD * elastic_i) / np.sum(wD)
        self.History["Elastic"] = np.append(self.History["Elastic"], elastic_weighted)

        # ------------------------------------------------------------
        # Creep diagnostic
        #   - creep per intact fiber: sum over KV elements
        #   - weight by J (as you intended)
        # ------------------------------------------------------------
        creep_i = self.model.local_creep[intact].sum(axis=1)

        # Use whichever name you actually have (J_lin_coeff vs J_lin)
        J_lin_coeff = getattr(self.model, "J_lin_coeff", None)
        if J_lin_coeff is None:
            J_lin_coeff = getattr(self.model, "J_lin", None)
        if J_lin_coeff is None:
            raise AttributeError("Model must have J_lin_coeff or J_lin for moisture-dependent J.")

        J_i = self.model.J_min + J_lin_coeff * phi_intact
        wJ = J_i

        creep_weighted = np.sum(wJ * creep_i) / np.sum(wJ)
        self.History["Creep"] = np.append(self.History["Creep"], creep_weighted)

        # ------------------------------------------------------------
        # Total strain (already computed by DD-SS update_total_strain)
        # ------------------------------------------------------------
        self.History["Total_strain"] = np.append(self.History["Total_strain"], self.model.total_strain)

        # ------------------------------------------------------------
        # Slip strain: mean slip over intact fibers (matches old "per intact fiber" idea)
        # ------------------------------------------------------------
        slip_i = self.model.local_slip[intact]
        slip_mean = np.sum(slip_i) / np.sum(intact)
        self.History["Slip_strain"] = np.append(self.History["Slip_strain"], slip_mean)

        # ------------------------------------------------------------
        # The rest unchanged
        # ------------------------------------------------------------
        self.History["Time"]     = np.append(self.History["Time"], self.current_time)
        self.History["Load"]     = np.append(self.History["Load"], self.model.load)
        self.History["Moisture"] = np.append(self.History["Moisture"], np.mean(self.model.fiber_moisture))

        self.History["Slip Count"] = np.append(self.History["Slip Count"], np.sum(self.model.slip_count))
        self.History["Hygroexp"]   = np.append(self.History["Hygroexp"], np.mean(self.model.eps_hyg))

        self.History["Number_of_fibers"] = np.append(
            self.History["Number_of_fibers"], np.sum(self.model.local_intact)
        )
        self.History["Broken"].append(self.model.broken)

        return

    
    


    def create_sim_sequence(self, cycles_pre_load, cycles_loaded, cycles_unload, period, creep_test=False):
        sequence_moist = []
        sequence_load = []
        sequence_time = []
        
        start_wet = int(self.model.sys_var.get("start_wet", 1))  # Default to 1 if missing
        moist_on = 1 - start_wet
        moist_off = start_wet
        # print(moist_off,moist_on)
        sequence_moist.extend([moist_off])
        sequence_load.extend([0])
        sequence_time.extend([5])
    
        if creep_test:
            # Load and hold for 100 * period
            
            sequence_moist.append(0)
            sequence_load.append(1)
            sequence_time.append(sequence_time[-1])
    
            sequence_moist.append(0)
            sequence_load.append(1)
            sequence_time.append(sequence_time[-1] + 1000 * period)
    
            # Unload and hold for 100 * period
            sequence_moist.append(0)
            sequence_load.append(0)
            sequence_time.append(sequence_time[-1])
    
            sequence_moist.append(0)
            sequence_load.append(0)
            sequence_time.append(sequence_time[-1] + 1000 * period)
    
            return sequence_moist, sequence_load, sequence_time
    
        # Default behavior with cyclic loading
        # print(self.model.sys_var.get("moist_grad"))
        if not self.model.sys_var.get("moist_grad"):
            for _ in range(cycles_pre_load):
                sequence_moist.extend([ moist_off, moist_on, moist_on, moist_on, moist_off, moist_off])
                sequence_load.extend([ 0, 0, 0, 0, 0, 0])
                last_time = sequence_time[-1]
                sequence_time.extend([last_time + period , last_time + period ,last_time + 3*period, last_time + 3 * period, last_time + 3 * period, last_time + 5 * period])
        
            sequence_moist.append(moist_off)
            sequence_load.append(1)
            sequence_time.append(sequence_time[-1])
        
            for _ in range(cycles_loaded):
                sequence_moist.extend([moist_off, moist_on, moist_on, moist_on, moist_off, moist_off])
                sequence_load.extend([1, 1, 1, 1, 1, 1])
                last_time = sequence_time[-1]
                sequence_time.extend([last_time + period, last_time + period ,last_time + 3*period, last_time + 3 * period, last_time + 3 * period, last_time + 5 * period])
                
            # sequence_time[-1]=last_time+4
            sequence_moist.append(moist_off)
            sequence_load.append(0)
            sequence_time.append(sequence_time[-1])
        
            for _ in range(cycles_unload):
                sequence_moist.extend([moist_off, moist_on, moist_on, moist_on, moist_off, moist_off])
                sequence_load.extend([0, 0, 0, 0, 0, 0])
                last_time = sequence_time[-1]
                sequence_time.extend([last_time + period , last_time + period ,last_time + 3*period, last_time + 3 * period, last_time + 3 * period, last_time + 5 * period])
        else:
            for _ in range(cycles_pre_load):
                sequence_moist.extend([0, 0, 1, 0])
                sequence_load.extend([0, 0, 0, 0])
                last_time = sequence_time[-1]
                sequence_time.extend([last_time, last_time , last_time + 11 * period, last_time + 20 * period])
        
            sequence_moist.append(0)
            sequence_load.append(1)
            sequence_time.append(sequence_time[-1])
            sequence_moist.append(0)
            sequence_load.append(1)
            sequence_time.append(sequence_time[-1]+4*period)
        
            for _ in range(cycles_loaded):
                sequence_moist.extend([0, 0, 1, 0])
                sequence_load.extend([1, 1, 1, 1])
                last_time = sequence_time[-1]
                sequence_time.extend([last_time, last_time , last_time + 11 * period, last_time + 20 * period])
        
            sequence_moist.append(0)
            sequence_load.append(0)
            sequence_time.append(sequence_time[-1])
            # sequence_moist.append(0)
            # sequence_load.append(0)
            # sequence_time.append(sequence_time[-1]+4*period)
        
            for _ in range(cycles_unload):
                sequence_moist.extend([0, 0, 1, 0])
                sequence_load.extend([0, 0, 0, 0])
                last_time = sequence_time[-1]
                sequence_time.extend([last_time, last_time , last_time + 11 * period, last_time + 20 * period])
        # 🔁 Invert moisture if start_wet is False (i.e., start_wet)
        if self.model.sys_var.get("start_wet", 1) == 1:
            sequence_moist = [1 - m for m in sequence_moist]
            # self.model.alpha *= -1
            # sequence_moist = [-x for x in sequence_moist]

        # print(sequence_moist)
        return sequence_moist, sequence_load, sequence_time

    
    def get_from_data_file(self,moisture_file_path,load_file_path):
        # Read moisture data file
        moisture_data_1 = np.loadtxt(moisture_file_path+".csv", delimiter=',', skiprows=1)
        moisture_time = np.sort(moisture_data_1[:, 0])  # Assuming time is in the second column
        moisture_values = moisture_data_1[:, 1]  # Assuming moisture values are in the first column
            # Read load data file
        load_data = np.loadtxt(load_file_path+".csv", delimiter=',', skiprows=1)
        load_time = load_data[:, 0]  # Assuming time is in the second column
        load_values = load_data[:, 1]  # Assuming load values are in the first column
        # Merge time arrays
        total_time = np.sort(np.concatenate((moisture_time, load_time)))
        moisture_array = []
        load_array = []
        for time_entry in total_time:
               # Find the index of the current time entry in moisture_time
               moisture_index = np.where(moisture_time == time_entry)[0]
               if moisture_index.size > 0:
                   # If the time entry is in moisture_time, use the corresponding value
                   moisture_array.append(moisture_values[moisture_index[0]])
               else:
                   # If the time entry is missing, repeat the previous entry
                   moisture_array.append(moisture_array[-1] if moisture_array else 0)

               # Find the index of the current time entry in load_time
               load_index = np.where(load_time == time_entry)[0]

               if load_index.size > 0:
                   # If the time entry is in load_time, use the corresponding value
                   load_array.append(load_values[load_index[0]])
               else:
                   # If the time entry is missing, repeat the previous entry
                   load_array.append(load_array[-1] if load_array else 0)
        return moisture_array, load_array, total_time
    
    

    def run(self, compare_n_cycles_ago=150):
        """
        Auto-skip stabilized cycles and separate handling of:
        - Moisture evolution: time-dependent via evolve_time
        - Load changes: handled quasi-statically using weakest fiber logic
        """
    
        # Histories and flags
        slip_hist_unloaded = []
        loaded_cycle_count   = 0
        unloaded_cycle_count = 0
        ignore_loaded   = False
        ignore_unloaded = False
        if not hasattr(self, "loaded_ignore_start_cycle"):
            self.loaded_ignore_start_cycle = None
    
        n = len(self.time_sequence)
        i = self.current_i
        # self.update_history()
        ja_unload = 0
        ja_plotou = 0
        # self.model.update_total_strain()
        # self.update_history()
        print("\n===== Starting simulation =====")

        while i < n:
            # print("PASS")
            self.current_time  = self.time_sequence[i]
            # self.update_history()
            self.current_load  = self.load_sequence[i]
            self.current_moist = self.moist_sequence[i]
    
            prev_moist = self.moist_sequence[i - 1]
            next_moist = self.moist_sequence[i]
            prev_load  = self.load_sequence[i - 1]
            next_load  = self.load_sequence[i]
    
            d_moisture = next_moist - prev_moist
            d_load     = next_load  - prev_load
            # self.model.update_total_strain()
            # self.update_history()
            # --- LOAD CHANGE (handled via weakest fiber evolution) ---
            if d_load:
                if next_load == 0:
                    print("\n--- Unloading ---")

                    # Reset UNLOADED phase tracking
                    
                    slip_hist_unloaded.clear()
                    unloaded_cycle_count = 0
                    ignore_unloaded = False
                    ja_unload +=1
                    if self.plastic:
                        self.current_i = i
                        ckpt_file = Path(self.model.input_path).parent / "latest_checkpoint.pkl"
                        with open(ckpt_file, "wb") as f:
                            pickle.dump((self.model, self), f)
                else:
                    print("\n--- Loading ---")
                    # Reset LOADED phase tracking
                    self.slip_hist_loaded.clear()
                    loaded_cycle_count = 0
                    ignore_loaded = False
    
                # Quasi-statically apply the load change in one go
                
                self.update_history()
                # self.model.apply_target_load_until_equilibrium(target_load=next_load,history_callback=self.update_history)
                self.model.apply_target_load_until_equilibrium(target_load=next_load)
                self.update_history()
    
            # --- MOISTURE CHANGE (handled via time evolution) ---
            elif d_moisture:
                if self.current_load and ignore_loaded:
                    # Skip ahead to unloading start
                    j = i + 1
                    while j < n and self.load_sequence[j] > 0.0:
                        j += 1
                    if j >= n:
                        del self.time_sequence[i+1:]
                        del self.load_sequence[i+1:]
                        del self.moist_sequence[i+1:]
                        break
                    dt = self.time_sequence[j] - self.time_sequence[i]
                    del self.time_sequence[i+1:j]
                    del self.load_sequence[i+1:j]
                    del self.moist_sequence[i+1:j]
                    for k in range(i+1, len(self.time_sequence)):
                        self.time_sequence[k] -= dt
                    i += 1
                    continue
    
                if self.current_load == 0 and ignore_unloaded:
                    self.time_sequence  = self.time_sequence[:i+1]
                    self.load_sequence  = self.load_sequence[:i+1]
                    self.moist_sequence = self.moist_sequence[:i+1]
                    break
    
                # Time-evolve system with fixed load
                self.evolve_time([self.time_sequence[i - 1], self.time_sequence[i]],
                                 self.model.sys_var.get("maximum_steps"), 1, next_moist)
                # if ja_unload and not ja_plotou:
                #     self.model.visualizer.plot_joint_bundle_state()

                    # joblib.dump(
                    #     {
                    #         "N": self.model.N,
                    #         "local_slip": self.model.local_slip.copy(),
                    #         "local_intact": self.model.local_intact.copy(),
                    #     },
                    #     f"avg_slip_snapshot_cycle_{ja_unload}.joblib",
                    #     compress=3
                    # )

                    # ja_plotou = 1

                # Slip stabilization check
                if next_moist == 0:
                    total_slip = float(getattr(self.model, "total_slip", 0.0))
                    if total_slip > 1e-12:
                        if self.current_load:
                            
                            loaded_cycle_count += 1
                            print(f"Loaded cycle {loaded_cycle_count}")

                            self.slip_hist_loaded.append(total_slip)
                            if (len(self.slip_hist_loaded) >= compare_n_cycles_ago + 1 and not ignore_loaded):
                                
                                ref = self.slip_hist_loaded[-(compare_n_cycles_ago + 1)]
                                if abs(total_slip - ref) < 0.00001 * total_slip:
                                    
                                    ignore_loaded = True
                                    if self.loaded_ignore_start_cycle is None:
                                        
                                        self.loaded_ignore_start_cycle = loaded_cycle_count
                        else:
                            unloaded_cycle_count += 1
                            print(f"Unloaded cycle {unloaded_cycle_count}")
                            slip_hist_unloaded.append(total_slip)
                            if (len(slip_hist_unloaded) >= compare_n_cycles_ago + 1 and not ignore_unloaded):
                                ref = slip_hist_unloaded[-(compare_n_cycles_ago + 1)]
                                if abs(total_slip - ref) < 0.00001 * total_slip:
                                    ignore_unloaded = True
    
            else:
                # Time evolution only (e.g., within a plateau region)
                self.evolve_time([self.time_sequence[i - 1], self.time_sequence[i]],
                                 self.model.sys_var.get("maximum_steps"), 0, next_moist)
    
            self.update_history()
            if self.model.broken:
                break
    
            i += 1
        return
     
    
    
    def complete_interval_fixed_steps(self, fixed, initial, final, flag, num_steps=100):
        """
        Completes the interval by splitting it into a fixed number of smaller steps and running each step.
    
        Parameters:
            fixed: Fixed parameters for the simulation.
            initial: Starting value of the interval.
            final: Ending value of the interval.
            flag: Additional flag for the interval check.
            num_steps: Number of steps to divide the interval into.
        """
        # Determine the step size and direction
        step_size = (final - initial) / num_steps
        direction = np.sign(step_size)
    
        # Initialize the current position
        current_position = initial
    
        # Iterate over the fixed number of steps
        for step in range(num_steps):
            next_position = current_position + step_size
    
            # Ensure we don't overshoot the final value due to numerical precision
            if direction * next_position > direction * final:
                next_position = final
    
            # Run the interval
            self.run_interval_fixed(next_position, fixed,flag)
    
            # Update the current position
            current_position = next_position
            
            
            self.update_history()
            # Break if we've reached the final position
            if direction * current_position >= direction * final:
                break
    
        return
    
    def run_interval_fixed(self, moist_value, load_value,flag):
        
        # Seems wrong, but I just change what means moist_value and load_value depending if Im
        # changing the moisture or load
        
        if not flag:
            
            self.model.normalized_moisture = (moist_value)
            self.model.fiber_moisture[...] = moist_value 
            # print("average",self.model.average_moisture)
            self.model.load = load_value
        else:
            self.model.normalized_moisture = load_value
            
            self.model.load = (moist_value)
        
        self.model.update_total_strain()
        
        self.model.slip_avalanche()
        
        if self.model.broken:
            return
        
        self.model.update_slip_strain()
        
        self.model.update_total_strain()
           
        return 
    
    
    
    
    
    def _precompute_kv_decay(self, dt):
        KV_num = self.model.sys_var.get("KV_num")
        base_tau = self.model.sys_var.get("tau")
        tau_list = base_tau * 10 ** np.arange(KV_num)
        decay_factors = 1 - np.exp(-dt / tau_list)
        return tau_list, decay_factors
    
    def _distribute_compliance(self, J_fiber, KV_num):
        weights = 2 ** np.arange(KV_num)   # exponential
        # weights = np.ones(KV_num)           # normal
        J_weights = weights / np.sum(weights)
        return J_fiber[:, None] * J_weights
    
    def _handle_moisture_counters(self, target_moist):
        if target_moist == 1:
            self.model.first_moist += 1
            self.model.first_dry += 1
            moist_count = sum(1 for k in self.model.moisture_profiles if k.startswith("moist_"))
            dry_count   = sum(1 for k in self.model.moisture_profiles if k.startswith("dry_"))
            self.model.first_moist = min(moist_count, self.model.first_moist)
            self.model.first_dry   = min(dry_count, self.model.first_dry)
    
    def _update_fiber_moisture_from_profile(self, target_moist, step_idx, alpha=None):
        prefix = 'moist_' if target_moist == 1 else 'dry_'
        index = self.model.first_moist if target_moist == 1 else self.model.first_dry
        key = f'{prefix}{index}'

        profile = self.model.moisture_profiles.get(key, [])

        if not profile:
            return

        tri_to_full_map = self.model.tri_to_full_map

        # --------------------------------------------------
        # Interpolate between two profile frames
        # --------------------------------------------------
        if alpha is not None and step_idx < len(profile) - 1:

            df_low = profile[step_idx].sort_values('fiber_index')
            df_high = profile[step_idx + 1].sort_values('fiber_index')

            tri_values = (
                (1 - alpha) * df_low['moisture'].values
                + alpha * df_high['moisture'].values
            )

        # --------------------------------------------------
        # Single frame
        # --------------------------------------------------
        elif step_idx < len(profile):

            df_step = profile[step_idx].sort_values('fiber_index')
            tri_values = df_step['moisture'].values

        else:
            return

        # --------------------------------------------------
        # Normalize moisture
        # --------------------------------------------------
        tri_values = np.clip((tri_values - 0.05) / (0.20 - 0.05), 0, 1)

        # --------------------------------------------------
        # Broadcast triangle → full bundle
        # --------------------------------------------------
        full_moisture = np.zeros(self.model.N)

        for tri_idx, full_ids in enumerate(tri_to_full_map):
            full_moisture[full_ids] = tri_values[tri_idx]

        self.model.fiber_moisture = full_moisture

    def evolve_time(self, interval, steps, change_moist, target_moist, max_recursion=10):
        dt = (interval[1] - interval[0]) / steps
        if dt * steps < 0.005:
            return
    
        self.model.normalized_moisture = self.current_moist
        self.current_time = interval[0]
        self._handle_moisture_counters(target_moist)
    
        J_min, J_lin = self.model.J_min, self.model.J_lin_coeff
        D_min, D_lin = self.model.D_min, self.model.D_lin_coeff
        KV_num = self.model.sys_var.get("KV_num")
        tau_list = self.model.sys_var.get("tau") * 10 ** np.arange(KV_num)
    
        def step_with_possible_subdivision(t0, t1, step_idx, recursion=10):
            dt_local = t1 - t0
        
            # ------------------------------------------------------------
            # Moisture interpolation (unchanged)
            # ------------------------------------------------------------
            if change_moist and self.model.sys_var.get("moist_grad"):
                alpha = (t1 - interval[0]) / (interval[1] - interval[0])
                self._update_fiber_moisture_from_profile(
                    target_moist, step_idx,
                    alpha if recursion > 0 else None
                )
        
            # ------------------------------------------------------------
            # 1) Update bundle strain (DD-SS-FBM inversion)
            # ------------------------------------------------------------
            self.model.update_total_strain()
        
            fiber_moisture = self.model.fiber_moisture
            D_fiber = D_min + D_lin * fiber_moisture
            J_fiber = J_min + J_lin * fiber_moisture
            J_i_fiber = self._distribute_compliance(J_fiber, KV_num)
            decay_factors = 1 - np.exp(-dt_local / tau_list)
        
            intact = (self.model.local_intact == 1)
        
            # ------------------------------------------------------------
            # 2) Compute forces CONSISTENT with update_total_strain()
            #    σ_i = ε_elastic_i / D_i  for intact fibers
            #    σ_i = 0                  for broken fibers
            # ------------------------------------------------------------
            forces = np.zeros_like(fiber_moisture)
        
            # D for intact fibers
            D_intact = D_fiber[intact]
        
            # elastic strain computed inside update_total_strain:
            # eps_elastic[intact] = total_strain - eps_NE_i
            eps_elastic_intact = self.model.eps_elastic[intact]
        
            # local forces
            forces[intact] = eps_elastic_intact / D_intact
            # broken fibers remain = 0
        
            self.model.local_force = forces
        
            # Debug print
            # print(
            #     "Total force:", np.sum(self.model.local_force),
            #     "Broken fiber force:", np.sum(self.model.local_force[~intact])
            # )
        
            # ------------------------------------------------------------
            # 3) Kelvin–Voigt creep update uses these forces
            # ------------------------------------------------------------
            for k in range(KV_num):
                self.model.local_creep[:, k] += (
                    J_i_fiber[:, k] * forces - self.model.local_creep[:, k]
                ) * decay_factors[k]
        
            # ------------------------------------------------------------
            # 4) Slip dynamics (stick–slip)
            # ------------------------------------------------------------
            slip_happened = self.model.slip_avalanche()
            if self.model.broken:
                return
        
            # If avalanche happened: subdivide
            if slip_happened and recursion < max_recursion:
                mid = (t0 + t1) / 2
                step_with_possible_subdivision(t0, mid, step_idx, recursion + 1)
                step_with_possible_subdivision(mid, t1, step_idx, recursion + 1)
                return
        
            # ------------------------------------------------------------
            # 5) Update slip strain and recompute total strain
            # ------------------------------------------------------------
            self.model.update_slip_strain()
            self.model.update_total_strain()
        
            # ------------------------------------------------------------
            # 6) Finalize
            # ------------------------------------------------------------
            self.current_time = t1
            self.update_history()

    
        # Perform all steps
        for i in range(steps):
            t0 = interval[0] + i * dt
            t1 = interval[0] + (i + 1) * dt
            step_with_possible_subdivision(t0, t1, i)
            # if i %5==0:
            #     self.model.visualizer.plot_joint_bundle_state()




    
    def run_strength(self):
        self.model.load = 0
        while np.sum(self.model.local_intact) != 0:
            self.model.load += 0.001
            self.model.update_total_strain()
            self.model.slip_avalanche()
            self.model.update_slip_strain()
            self.model.update_total_strain()
            self.update_history()
        
        
        return self.model.load, self.model.total_strain
        
        
        
        
        
    
    
            
            