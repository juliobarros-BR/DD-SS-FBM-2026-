#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 14:09:15 2024

@author: jortiz
"""

import ast
import numpy as np
import os
import time
import copy
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from pathlib import Path
import pandas as pd
import re
import matplotlib.gridspec as gridspec
import decimal
from scipy.signal import convolve2d
from .model_visualizer import ModelVisualizer
import zarr
from numcodecs import Blosc


class Model:
    def __init__(self, file_path, moisture_profiles = None):
        self.input_path = file_path
        self.sys_var = self.initialize_variables_from_file(self.input_path)
        self.sys_var['period'] = self.sys_var.get('period') 
        self.alpha = self.sys_var.get('alpha')
        # Initialize global properties
        self.total_slip = 0
        self.total_strain = 0
        self.N = self.sys_var.get('N')
        self.max_slip = int(-np.log(self.sys_var.get('failure_limit')) * self.sys_var.get('decay'))
        self.normalized_moisture = 0
        self.avg_moisture = 0
        self.load = 0
        self.checkpoint_path = self.sys_var.get('checkpoint_path', None)
        print("Checkpoint path:", self.checkpoint_path)

        self.broken = 0
        self.first_moist = 0
        self.first_dry = 0
        self.gamma_deg = self.sys_var.get('gamma_deg')
        self.cut_off = self.sys_var.get('cut_off')
        self.Z_deg = self.sys_var.get('Z')
        # Initialize local properties (for each fiber)
        self.slip_th = np.zeros(self.N)
        self.slip_count = np.zeros(self.N)
        self.local_slip = np.zeros(self.N)
        self.local_creep = np.zeros((self.N,self.sys_var.get('KV_num')))
        self.local_stress = np.zeros(self.N)
        self.local_intact = np.ones(self.N)
        self.local_force = np.ones(self.N)
        self.local_strain = np.ones(self.N)
        self.fiber_moisture = np.ones(self.N)*self.sys_var.get('start_wet')
        
        # print(self.sys_var.get('reverse_scale'), self.sys_var.get('wet_scale'))
        # Initialize thresholds
        np.random.seed(self.sys_var.get('seed'))
        self.threshold = (
            np.random.weibull(self.sys_var.get('m_Weibull'), self.N) 
            * self.sys_var.get('lambda_Weibull') 
            + self.sys_var.get("init_th")
        )
        self.export_thresholds("thresholds.npy")


        # Initialize fiber positions based on geometry
        self.geometry = self.sys_var.get('geometry', 0)  # Default to square
        self.positions = self.initialize_positions()
        self.kdtree = cKDTree(self.positions)
        self.neighbor_list = self.kdtree.query_ball_tree(self.kdtree, r=1)

        # Linear fitting for compliances between the driest and most moist states
        self.D_min = self.sys_var.get('D_d')
        self.D_lin_coeff = self.sys_var.get('D_w') - self.sys_var.get('D_d')
        self.J_min = self.sys_var.get('J_d')
        self.J_lin_coeff = self.sys_var.get('J_w') - self.sys_var.get('J_d')

        # Find critical load on a copy of the model
        self.history_critical_strain = np.array([])
        self.history_critical_load = np.array([])
        self.history_intact = np.array([])
        
        self.eps_hyg = np.zeros(self.N)
        self.eps_slip = np.zeros(self.N)
        self.eps_creep = np.zeros(self.N)        
        self.eps_elast = np.zeros(self.N)
        self.eps_elastic = np.zeros_like(self.local_strain)
        # self.critical_load, self.critical_strain = self.find_critical_load_slip_event_driven()
        self.critical_load , self.critical_strain = 1.8691471151149388,4.3117654332733
        
        self.critical_load = np.float64(self.critical_load)
        self.critical_strain = np.float64(self.critical_strain)
        self.moisture_profiles = {}
        folder_moist = self.sys_var.get('folder_moist', None)
        
        if folder_moist is not None and self.sys_var.get('moist_grad'):
            if moisture_profiles == None:
                self.load_moisture_profiles(folder_moist)
            else:
                self.moisture_profiles = moisture_profiles
        
        self.build_tri_to_full_map()
        self.average_moisture = np.mean(self.fiber_moisture[np.where(self.local_intact==1)])
        self.visualizer = ModelVisualizer(self)


        # if self.sys_var.get("use_moisture_profiles", 0):
        #     self.fiber_moisture = np.ones(self.N) * self.sys_var.get("start_wet")
        # else:
            # self.fiber_moisture = np.zeros(self.N)  # will be filled from profile
        if self.sys_var.get('schleronomic', None):
            self.apply_frozen_moisture_from_profiles()

    def export_thresholds(self, path):
        """
        Save the per-fiber thresholds (constant for the whole run).
        """
        np.save(path, self.threshold)

    def init_zarr_checkpoints(self, path):
        """
        Zarr v2 store (appendable) for checkpoints:
        - load: (k,)
        - slip_count: (k, N)
        """
        from pathlib import Path
        import zarr
        from numcodecs import Blosc

        path = Path(path)

        # IMPORTANT: force Zarr V2 format for numcodecs compatibility
        root = zarr.open_group(str(path), mode="w", zarr_format=2)

        comp = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)

        root.create_dataset(
            "load",
            shape=(0,),
            chunks=(8192,),
            dtype="f8",
            compressor=comp,   # v2 API
        )

        root.create_dataset(
            "slip_count",
            shape=(0, self.N),
            chunks=(64, self.N),
            dtype="i2",
            compressor=comp,   # v2 API
        )

        root.attrs["N"] = int(self.N)
        root.attrs["max_slip"] = int(self.max_slip)
        root.attrs["seed"] = int(self.sys_var.get("seed"))

        return root


    def append_zarr_checkpoint(self, root):
        """
        Append the current (load, slip_count) as a new row in the Zarr store.
        Compatible with Zarr v3 API (resize takes a shape tuple).
        """
        k = root["load"].shape[0]

        # resize using shape tuples
        root["load"].resize((k + 1,))
        root["slip_count"].resize((k + 1, self.N))

        # write data
        root["load"][k] = float(self.load)
        root["slip_count"][k, :] = self.slip_count.astype(np.int16, copy=False)

    def _find_ckpt_index(self, zroot, target_load: float) -> int:
        loads = zroot["load"][:]  # ok because it's small (only ~thousands)
        idx = int(np.searchsorted(loads, target_load, side="right") - 1)
        return max(idx, 0)

    def reconstruct_slip_from_slipcount(self):
        """
        Rebuild local_slip + local_intact from slip_count (dry monotone assumption).
        """
        decay = float(self.sys_var.get("decay"))
        mode = int(self.sys_var.get("degradation_mode", 0))

        n = self.slip_count.astype(np.int64)

        # intact derived
        self.local_intact = (n <= self.max_slip).astype(self.local_intact.dtype)

        # clamp effective slip count for summation
        n_eff = np.minimum(n, self.max_slip).astype(np.int64)

        if mode == 0:
            # exponential degradation: sum_{k=0}^{n-1} T*exp(-k/decay)
            r = np.exp(-1.0 / decay)
            # (1 - r^n)/(1 - r)
            self.local_slip = self.threshold * (1.0 - np.power(r, n_eff)) / (1.0 - r)
        else:
            # linear-to-min-factor degradation used in your code
            min_factor = np.exp(-self.max_slip / decay)
            # sum_{k=0}^{n-1} [1 - (1-min_factor)*(k/max_slip)]
            # = n - (1-min_factor)/(max_slip) * sum k
            ksum = (n_eff - 1) * n_eff / 2.0
            self.local_slip = self.threshold * (n_eff - (1.0 - min_factor) * (ksum / self.max_slip))

        # refresh derived cached strains
        self.update_slip_strain()
        self.update_total_strain()

    def fast_forward_from_checkpoints(
            self,
            target_load: float,
            checkpoint_zarr_path,
            thresholds_path=None,
            min_gain: float = 1e-3,
        ):

        import zarr
        from pathlib import Path

        target_load = float(target_load)

        # print(f"[FF] Requested target load: {target_load:.6f}")
        # print(f"[FF] Current load: {self.load:.6f}")

        if (target_load - float(self.load)) < float(min_gain):
            print("[FF] Skipping fast-forward: target too close.")
            return False

        checkpoint_zarr_path = Path(checkpoint_zarr_path)

        if checkpoint_zarr_path.suffix != ".zarr":
            checkpoint_zarr_path = checkpoint_zarr_path.with_suffix(".zarr")

        # print(f"[FF] Looking for checkpoint file: {checkpoint_zarr_path}")

        if not checkpoint_zarr_path.exists():
            # print("[FF] Checkpoint file NOT found.")
            return False

        # print("[FF] Checkpoint file found.")

        if thresholds_path is not None:
            thresholds_path = Path(thresholds_path)

            if thresholds_path.exists():
                # print(f"[FF] Loading thresholds from {thresholds_path}")
                self.threshold = np.load(thresholds_path)
            # else:
                # print("[FF] Threshold file not found, skipping.")

        zroot = zarr.open_group(str(checkpoint_zarr_path), mode="r")

        k = self._find_ckpt_index(zroot, target_load)

        ckpt_load = float(zroot["load"][k])

        # print(f"[FF] Selected checkpoint index: {k}")
        # print(f"[FF] Jumping to checkpoint load: {ckpt_load:.6f}")

        ckpt_sc = zroot["slip_count"][k, :].astype(
            self.slip_count.dtype,
            copy=True
        )

        self.load = ckpt_load
        self.slip_count[:] = ckpt_sc

        self.reconstruct_slip_from_slipcount()

        # print("[FF] Fast-forward successful.")

        return True

    
    def apply_frozen_moisture_from_profiles(self):
        """
        Apply a frozen moisture configuration from:
          - a given cycle
          - a given step inside that cycle
        using the SAME spatial mapping and normalization
        as during normal moisture evolution.
        """
        cycle = self.sys_var.get("frozen_moisture_cycle", None)
        step  = self.sys_var.get("frozen_moisture_step", None)
    
        if cycle is None or step is None:
            return
    
        key = f"moist_{cycle}"
        if key not in self.moisture_profiles:
            raise KeyError(f"Moisture profile {key} not found")
    
        profile = self.moisture_profiles[key]
    
        if step < 0 or step >= len(profile):
            raise IndexError(
                f"Requested step {step} but cycle {cycle} has {len(profile)} steps"
            )
    
        df = profile[step]
    
        grid_size = int(np.ceil(np.sqrt(self.N)))
        moisture_matrix = np.full((grid_size, grid_size), np.nan)
    
        x_idx = df["x"].astype(int).values
        y_idx = df["y"].astype(int).values
        moisture_matrix[y_idx, x_idx] = df["moisture"].values
    
        normalized = np.clip(
            (moisture_matrix - 0.05) / (0.2 - 0.05),
            0.0,
            1.0
        )
    
        self.fiber_moisture = normalized.flatten()
    
        # Freeze moisture evolution
        self.moisture_frozen = True
        # print(np.mean(self.fiber_moisture))
    
        # Optional visualization
        # if hasattr(self, "visualizer"):
        #     self.visualizer.plot_joint_bundle_state()
    
    
    
    
    
    def initialize_variables_from_file(self, file_path):
        variables = {}
        has_moisture_data = False
        has_load_data = False
        with open(file_path, 'r') as file:
            for line in file:
                # Remove leading and trailing whitespaces
                line = line.strip()

                # Skip empty lines or lines starting with '#' (comments)
                if not line or line.startswith('#'):
                    continue

                # Split the line into variable and value using '=' as delimiter
                variable, value_str = map(str.strip, line.split('=', 1))
                
                # Check if the line starts with "moisture_data"
                if variable == "moisture_df" and value_str:
                    # print("here")
                    has_moisture_data = True

                # Check if there's anything after the "=" sign for "load_data"
                elif variable == "load_df" and value_str:
                    has_load_data = True


                # Use ast.literal_eval to safely evaluate the value string
                try:
                    if os.path.isfile(value_str):
                        with open(value_str, 'r') as file_content:
                            value = file_content.read()
                    else:
                        # Use ast.literal_eval to safely evaluate other expressions
                        value = ast.literal_eval(value_str)
                except (SyntaxError, ValueError):
                    # If not a file path and not a literal expression, treat it as a string
                    value = value_str

                # Store the variable and its initialized value in the dictionary
                variables[variable] = value
        # print(has_moisture_data,has_load_data)
        variables["has_moist"] = has_moisture_data
        variables["has_load"] = has_load_data

        return variables    
    
    def load_moisture_profiles(self, folder_moist):
        
        folder_aux = Path(folder_moist)
        folder = Path.cwd() / folder_aux
        
        moist_files = sorted(folder.glob("mask_moistening_*_cycle.csv"),
                             key=lambda f: int(re.search(r"moistening_(\d+)_cycle", f.stem).group(1)))
        dry_files = sorted(folder.glob("mask_drying_*_cycle.csv"),
                           key=lambda f: int(re.search(r"drying_(\d+)_cycle", f.stem).group(1)))
        
        if not moist_files:
            print(f"\n --- No moistening files found in {folder} --- \n")
        else:
            print(f"\n --- Loading moisture files from {folder}: --- \n")
        
        for i, file in enumerate(moist_files):
            if i > self.sys_var.get("cycles_loaded") + self.sys_var.get("cycles_unload"):
                break
            # print(file)
            df = pd.read_csv(file)
            grouped = df.groupby('time_index')
            self.moisture_profiles[f'moist_{i+1}'] = [g for _, g in grouped]
            
        
        for i, file in enumerate(dry_files):
            if i > self.sys_var.get("cycles_loaded") + self.sys_var.get("cycles_unload"):
                break
            df = pd.read_csv(file)
            grouped = df.groupby('time_index')
            self.moisture_profiles[f'dry_{i+1}'] = [g for _, g in grouped]

    def build_tri_to_full_map(self):
        """
        Build mapping from compact triangular moisture fibers
        to full bundle fiber indices using the same symmetry
        operations used in the original make_masks().
        """

        # Use first moisture frame as geometric reference
        tri_df = self.moisture_profiles['moist_1'][0].copy()
        tri_df = tri_df.sort_values('fiber_index').reset_index(drop=True)

        tri_positions = tri_df[['x', 'y']].values.astype(int)

        # Infer size exactly like original make_masks
        size = tri_positions[:, 0].max() + 1

        # Full bundle assumed square lattice flattened row-major
        grid_size = int(np.ceil(np.sqrt(self.N)))

        def apply_transform(xy, matrix, offset):
            return (np.dot(xy, matrix) + offset).astype(int)

        transforms = [
            lambda xy: apply_transform(xy, [[1, 0], [0, 1]], [0, 0]),
            lambda xy: apply_transform(xy, [[0, -1], [1, 0]], [0, 2 * size - 1]),
            lambda xy: apply_transform(xy, [[-1, 0], [0, -1]], [2 * size - 1, 2 * size - 1]),
            lambda xy: apply_transform(xy, [[0, 1], [-1, 0]], [2 * size - 1, 0]),
        ]

        self.tri_to_full_map = []

        for x, y in tri_positions:

            # recreate mirrored triangle exactly as before
            if x != y:
                base_positions = np.array([
                    [x, y],
                    [y, x]
                ])
            else:
                base_positions = np.array([
                    [x, y]
                ])

            full_ids = []

            for transform in transforms:
                transformed = transform(base_positions)

                for xx, yy in transformed:
                    full_idx = yy * grid_size + xx
                    full_ids.append(full_idx)

            self.tri_to_full_map.append(np.unique(full_ids))


    
    def initialize_positions(self):
        if self.geometry == 0:
            # Square geometry
            grid_size = int(np.ceil(np.sqrt(self.N)))
            x, y = np.meshgrid(range(grid_size), range(grid_size))
            positions = np.column_stack((x.flatten(), y.flatten()))[:self.N]
        elif self.geometry == 1:
            # Circular geometry: grid points inside a circle
           grid_size = int(np.ceil(np.sqrt(self.N) * 1.5))  # safety margin
           center = grid_size // 2
           x, y = np.meshgrid(np.arange(grid_size), np.arange(grid_size))
           x = x - center
           y = y - center
           r = np.sqrt(x**2 + y**2)
    
           # Flatten and sort by distance from center
           flat_x = x.flatten()
           flat_y = y.flatten()
           flat_r = r.flatten()
           sorted_indices = np.argsort(flat_r)
            
           # Take only first N points (closest to center, i.e., circular)
           x_inside = flat_x[sorted_indices][:self.N]
           y_inside = flat_y[sorted_indices][:self.N]
           positions = np.column_stack((x_inside, y_inside))
        else:
            raise ValueError(f"Unsupported geometry type: {self.geometry}")
        return positions

    
        
    def update_slip_strain(self):
        self.total_slip     = np.sum(self.local_slip)
        return
    
    
    
    
    def update_total_strain(self):
        """
        Compute the total bundle strain according to the DD-SS-FBM formulation:
    
            ε = [ σ + Σ_i ( ε_NE_i / D_i ) ] / [ Σ_i (1 / D_i) ]
    
        where ε_NE_i = ε_VE_i + ε_H_i + ε_S_i
        and D_i = moisture-dependent elastic compliance.
        """
    
        intact = (self.local_intact == 1)
        if not np.any(intact):
            self.total_strain = np.nan
            return
    
        # ------------------------------------------------------------
        # 1) Moisture-dependent compliance of intact fibers
        # ------------------------------------------------------------
        phi = self.fiber_moisture[intact]
        D_i = self.D_min + self.D_lin_coeff * phi      # elastic compliance
    
        # ------------------------------------------------------------
        # 2) Non-elastic strains
        #    ε_NE = ε_slip + ε_creep + ε_hygro
        # ------------------------------------------------------------
        eps_slip_i = self.local_slip[intact]                    # slip strains
        eps_creep_i = self.local_creep[intact].sum(axis=1)    # VE strains
        eps_hygro_i = self.alpha * (phi - self.sys_var["start_wet"])
    
        eps_NE_i = eps_slip_i + eps_creep_i + eps_hygro_i
    
        # ------------------------------------------------------------
        # 3) Compute the two required sums:
        #    S1 = Σ_i (ε_NE_i / D_i)
        #    S2 = Σ_i (1 / D_i)
        # ------------------------------------------------------------
        S1 = np.sum(eps_NE_i / D_i)
        S2 = np.sum(1.0 / D_i)
    
        # ------------------------------------------------------------
        # 4) Full bundle stress = load * N
        # ------------------------------------------------------------
        sigma_bundle = self.load * self.N
    
        # ------------------------------------------------------------
        # 5) Total strain:
        #       ε = (sigma + S1) / S2
        # ------------------------------------------------------------
        self.total_strain = (sigma_bundle + S1) / S2
        
        self.eps_elastic[intact] = self.total_strain - eps_NE_i
        self.eps_slip[intact] = eps_slip_i
        self.eps_creep[intact] = eps_creep_i
        self.eps_hyg[intact] = eps_hygro_i
    
        return







    
    
   
    def get_next_slip_load(self, target_load, probe_dL=1e-9, tol=1e-14):
        """
        Predict next load (between current load and target_load direction) at which
        at least one intact fiber will satisfy the slip condition.

        Direction is determined from target_load - current_load (NOT load_d).
        Reverse penalty is handled per event sign (+ or - threshold), not frozen at current sign.
        """

        import numpy as np

        intact = (self.local_intact == 1)
        if not np.any(intact):
            return np.inf

        # ensure consistent state
        self.update_slip_strain()
        self.update_total_strain()

        L0 = float(self.load)
        target_load = float(target_load)
        dL = target_load - L0

        # ramp direction from target vs current
        ramp_dir = np.sign(dL) if abs(dL) > tol else 1.0
        ramp_dir = 1.0 if ramp_dir == 0 else ramp_dir

        decay = float(self.sys_var.get("decay"))
        degradation_mode = int(self.sys_var.get("degradation_mode"))
        reverse_scale = float(self.sys_var.get("reverse_scale"))
        wet_scale = float(self.sys_var.get("wet_scale"))
        hygro_slip = float(self.sys_var.get("hygro_slip"))

        # global material direction used ONLY for reverse penalty classification
        load_d = float(self.sys_var.get("load_d", 1.0))
        global_dir = np.sign(load_d) if load_d != 0 else 1.0
        global_dir = 1.0 if global_dir == 0 else global_dir

        # ---- eps_act at current load ----
        epsE = self.eps_elastic[intact]
        epsVE = self.local_creep[intact].sum(axis=1)
        epsHyg = self.eps_hyg[intact]
        eps_act0 = epsE + epsVE + epsHyg * hygro_slip

        # ---- baseline threshold magnitude (degradation + wet) ----
        slip_eff = np.minimum(self.slip_count[intact], self.max_slip)

        if degradation_mode == 1:
            min_factor = np.exp(-self.max_slip / decay)
            lin_factor = 1 - (1 - min_factor) * (slip_eff / self.max_slip)
            degrad_th = self.threshold[intact] * lin_factor
        else:
            degrad_th = self.threshold[intact] * np.exp(-slip_eff / decay)

        wet_sc = 1 + (wet_scale - 1) * self.fiber_moisture[intact]
        T = degrad_th * wet_sc  # positive

        # reverse penalty depends on the sign at the EVENT
        r_plus = reverse_scale if (1.0 != global_dir) else 1.0
        r_minus = reverse_scale if (-1.0 != global_dir) else 1.0

        T_plus = T * r_plus
        T_minus = T * r_minus

        # ---- slope d(eps_act)/dL ----
        # Use probe ALWAYS (robust; captures compliance/moisture effects in your update routines)
        self.load = L0 + ramp_dir * probe_dL
        self.update_total_strain()  # do NOT call slip_avalanche
        epsE1 = self.eps_elastic[intact]
        eps_act1 = epsE1 + epsVE + epsHyg * hygro_slip
        slope = (eps_act1 - eps_act0) / (ramp_dir * probe_dL)

        # restore
        self.load = L0
        self.update_total_strain()

        slope[np.abs(slope) < tol] = np.nan

        # ---- solve candidates for hitting +T_plus and -T_minus ----
        L_hit_plus  = L0 + ( +T_plus  - eps_act0) / slope
        L_hit_minus = L0 + ( -T_minus - eps_act0) / slope

        cand = np.concatenate([L_hit_plus, L_hit_minus])

        # must move in ramp direction AND must be between current and target (optional but recommended)
        valid = np.isfinite(cand) & ((cand - L0) * ramp_dir > tol)

        if not np.any(valid):
            return np.inf

        next_load = float(np.min(cand[valid]) if ramp_dir > 0 else np.max(cand[valid]))

        # If it lies past the target, treat as "no event before target"
        if np.isfinite(next_load) and ((next_load - target_load) * ramp_dir > 0):
            return np.inf

        # guard progress
        if abs(next_load - L0) < tol:
            next_load = L0 + ramp_dir * 10 * tol

        return next_load



    def apply_target_load_until_equilibrium(
        self,
        target_load,
        max_events=200000,
        max_steps=10_000,
        tol=None,
        target_tol=1e-10,
        history_callback=None,
        use_checkpoints=True,
        ckpt_min_gain=1e-3,
        ckpt_only_from_zero=True,
        zero_tol=1e-12,
    ):
        """
        Event-driven load advance: jump directly to the next predicted slip event
        (or to target_load), then relax with slip_avalanche().
        """
        import numpy as np

        target_load = float(target_load)

        # -----------------------------
        # (A) Optional checkpoint fast-forward (LOADING ONLY)
        # -----------------------------
        if use_checkpoints:
            cp_path = self.sys_var.get("checkpoint_path", None)
            th_path = self.sys_var.get("thresholds_path", None)

            starting_loading = (abs(float(self.load)) <= zero_tol) and (abs(target_load) > zero_tol)
            loading_direction = (target_load - float(self.load)) > 0

            allow_ckpt = (cp_path is not None) and loading_direction
            if ckpt_only_from_zero:
                allow_ckpt = allow_ckpt and starting_loading

            if allow_ckpt and (target_load - float(self.load)) >= ckpt_min_gain:
                did_ff = self.fast_forward_from_checkpoints(
                    target_load=target_load,
                    checkpoint_zarr_path=cp_path,
                    thresholds_path=th_path,
                    min_gain=ckpt_min_gain,
                )
                if did_ff and history_callback is not None:
                    history_callback()


        L0 = float(self.load)
        dL_target = target_load - L0

        ramp_dir = np.sign(dL_target) if abs(dL_target) > target_tol else 1.0
        ramp_dir = 1.0 if ramp_dir == 0 else ramp_dir  # ensure ±1

        if tol is None:
            event_tol = abs(dL_target) / float(max_steps) if abs(dL_target) > 0 else target_tol
        else:
            event_tol = float(tol)

        min_step = abs(dL_target) / float(max_steps) if abs(dL_target) > 0 else event_tol
        min_step = max(min_step, 10.0 * np.finfo(float).eps)

        if hasattr(self, "History") and isinstance(self.History, dict):
            self.History.setdefault("Event_tol", np.array([], dtype=float))
            self.History.setdefault("Target_tol", np.array([], dtype=float))
            self.History.setdefault("Min_step", np.array([], dtype=float))
            self.History["Event_tol"] = np.append(self.History["Event_tol"], event_tol)
            self.History["Target_tol"] = np.append(self.History["Target_tol"], target_tol)
            self.History["Min_step"] = np.append(self.History["Min_step"], min_step)

        events = 0
        steps_used = 0

        while ((target_load - self.load) * ramp_dir > target_tol) and (not self.broken) and (events < max_events):

            # FIX: pass ramp_dir into predictor
            next_load = self.get_next_slip_load(target_load=target_load)


            # print(
            #     f"Event {events}: Current load={self.load:.6e}, "
            #     f"Target={target_load:.6e}, Next slip load={next_load:.6e}, ramp_dir={ramp_dir:+.0f}"
            # )

            if (not np.isfinite(next_load)) or ((next_load - self.load) * ramp_dir <= 0):
                self.load = target_load

            else:
                dL_event = (next_load - self.load) * ramp_dir

                if dL_event <= event_tol:
                    self.load = float(self.load + ramp_dir * min_step)
                    steps_used += 1
                    if steps_used >= max_steps:
                        self.load = target_load
                else:
                    if (next_load - target_load) * ramp_dir > 0:
                        self.load = target_load
                    else:
                        self.load = float(next_load)

            self.update_total_strain()
            self.slip_avalanche()

            if history_callback is not None:
                history_callback()

            events += 1

        return (not self.broken)






    def slip_avalanche(self):
        """
        Slip avalanche with DD-SS-FBM variables.

        Reverse slips are allowed, BUT penalized by reverse_scale.
        The key fix is to make the slip direction + reverse_scale classification
        consistent and robust (no zero-sign issues).

        Slip triggers when: |eps_act| > slip_threshold
        where eps_act = eps_elastic + epsVE + epsHyg*hygro_slip
        """

        # Keep model consistent before beginning avalanche
        self.update_slip_strain()
        self.update_total_strain()

        decay = self.sys_var.get("decay")
        degradation_mode = self.sys_var.get("degradation_mode")  # 0 exp, 1 linear
        reverse_scale = float(self.sys_var.get("reverse_scale"))
        wet_scale = float(self.sys_var.get("wet_scale"))
        hygro_slip = float(self.sys_var.get("hygro_slip"))

        # Global loading direction (+1 / -1). If load_d=0, default to +1
        load_d = float(self.sys_var.get("load_d"))
        global_dir = np.sign(load_d) if load_d != 0 else 1.0
        global_dir = 1.0 if global_dir == 0 else global_dir

        i = 0
        # print("Starting slip avalanche...")
        while i < 1000:
            intact = (self.local_intact == 1)
            if not np.any(intact):
                self.broken = 1
                break

            # ------------------------------------------------------------
            # 1) Elastic and VE strains (DD-SS-FBM definitions)
            # ------------------------------------------------------------
            epsE = self.eps_elastic[intact]
            epsVE = self.local_creep[intact].sum(axis=1)
            epsHyg = self.eps_hyg[intact]
            eps_act = epsE + epsVE + epsHyg * hygro_slip  # driving strain for slip

            # robust sign for eps_act: treat 0 as +1 to avoid "0 direction"
            slip_dir = np.sign(eps_act)
            slip_dir[slip_dir == 0] = 1.0  # now in {+1, -1}

            # ------------------------------------------------------------
            # 2) Compute slip degradation thresholds (with saturation)
            # ------------------------------------------------------------
            slip_eff = np.minimum(self.slip_count[intact], self.max_slip)

            if degradation_mode == 1:
                min_factor = np.exp(-self.max_slip / decay)
                lin_factor = 1 - (1 - min_factor) * (slip_eff / self.max_slip)
                degrad_th = self.threshold[intact] * lin_factor
            else:
                degrad_th = self.threshold[intact] * np.exp(-slip_eff / decay)

            # ------------------------------------------------------------
            # 3) Environmental scaling
            # ------------------------------------------------------------
            wet_sc = 1 + (wet_scale - 1) * self.fiber_moisture[intact]

            # ------------------------------------------------------------
            # 4) Reverse scaling (penalize slips that oppose global_dir)
            # ------------------------------------------------------------
            # If slip_dir == global_dir  -> forward slip -> factor 1
            # If slip_dir != global_dir  -> reverse slip -> factor reverse_scale
            is_reverse = (slip_dir != global_dir)
            reverse_sc = np.where(is_reverse, reverse_scale, 1.0)

            slip_threshold = degrad_th * wet_sc * reverse_sc

            # ------------------------------------------------------------
            # 5) Slip condition: allow both directions, but threshold is scaled
            # ------------------------------------------------------------
            idx_slip_local = np.where(np.abs(eps_act) > slip_threshold)[0]
            if len(idx_slip_local) == 0:
                break

            idx_slip = np.where(intact)[0][idx_slip_local]

            # ------------------------------------------------------------
            # 6) Apply slip: sign matches eps_act (consistent with condition)
            # ------------------------------------------------------------
            slip_amount = slip_threshold[idx_slip_local] * slip_dir[idx_slip_local]
            new_slip = self.local_slip[idx_slip] + slip_amount

            self.local_slip[idx_slip] = np.round(new_slip, decimals=15)
            self.slip_count[idx_slip] += 1

            # ------------------------------------------------------------
            # 7) Break fibers exceeding limit
            # ------------------------------------------------------------
            broken_fibers = self.slip_count > self.max_slip
            self.local_intact[broken_fibers] = 0

            if np.sum(self.local_intact) == 0:
                self.broken = 1
                break

            # ------------------------------------------------------------
            # 8) Update after slip
            # ------------------------------------------------------------
            self.update_slip_strain()
            self.update_total_strain()

            i += 1
            
        return




    def run_slip_fbm_event_driven(
        self,
        start_load: float = 0.0,
        record_history: bool = True,
        tol: float = 1e-14,
        max_events: int = 1_000_000,
        save_load_step: float = 1e-3,   # save only when load increases by >= this amount
    ):
        """
        Quasi-static event-driven driver for your DD-SS + slip-avalanche model.

        Checkpointing:
        - If self.checkpoint_path is set, creates a Zarr store at <checkpoint_path>.zarr (folder).
        - Saves thresholds once next to it as <stem>_thresholds.npy
        - Appends (load, slip_count) whenever load advanced by >= save_load_step (and always at first save).
        """

        import numpy as np
        from pathlib import Path

        # --- Optional Zarr checkpointing setup (appendable, no rewrite) ---
        zroot = None
        last_saved_load = -np.inf

        cp_raw = getattr(self, "checkpoint_path", None)
        if cp_raw is not None and str(cp_raw).strip() != "":
            checkpoint_path = Path(cp_raw)

            # Ensure ".zarr" suffix (Zarr stores are directories)
            if checkpoint_path.suffix != ".zarr":
                checkpoint_path = checkpoint_path.with_suffix(".zarr")

            # Ensure parent directory exists
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

            print(f"[checkpoint] enabled: {checkpoint_path.resolve()} (Zarr folder store)")

            # Create store (NOTE: this overwrites if it already exists)
            zroot = self.init_zarr_checkpoints(checkpoint_path)

            # Save thresholds once, next to the zarr store
            th_path = checkpoint_path.with_name(checkpoint_path.stem + "_thresholds.npy")
            np.save(th_path, self.threshold)
            print(f"[checkpoint] thresholds saved: {th_path.resolve()}")

            # Force-save at first eligible moment
            last_saved_load = -np.inf
        else:
            print("[checkpoint] disabled (self.checkpoint_path is None/empty)")

        # --- History init ---
        if record_history:
            self.history_critical_load = np.array([], dtype=float)
            self.history_critical_strain = np.array([], dtype=float)
            self.history_intact = np.array([], dtype=int)

        # Initialize load
        self.load = float(start_load)

        # Define loading direction from your sys_var "load_d" (kept for consistency)
        load_d = float(self.sys_var.get("load_d", 1.0))
        g = np.sign(load_d) if load_d != 0 else 1.0
        g = 1.0 if g == 0 else g  # ensure ±1 (unused here but fine to keep)

        last_stable_eps = np.nan
        n_saved = 0

        for _event in range(max_events):

            intact = (self.local_intact == 1)
            n_intact = int(np.sum(intact))
            if n_intact == 0:
                break

            # --- 1) Relax at current load: resolve any pending slips/breaks ---
            self.update_slip_strain()
            self.update_total_strain()

            # If already unstable at this load, avalanche now
            _ = self.slip_avalanche()

            intact = (self.local_intact == 1)
            n_intact = int(np.sum(intact))
            if n_intact == 0 or getattr(self, "broken", 0) == 1:
                break

            # Update strains after avalanche
            self.update_slip_strain()
            self.update_total_strain()

            # --- Stable state before next load jump ---
            last_stable_eps = float(self.total_strain)
            # print(self.load, self.total_strain, n_intact, np.sum(self.slip_count))

            if record_history:
                self.history_critical_load = np.append(self.history_critical_load, self.load)
                self.history_critical_strain = np.append(self.history_critical_strain, self.total_strain)
                self.history_intact = np.append(self.history_intact, n_intact)

            # --- SAVE CHECKPOINT only when load advanced enough ---
            if zroot is not None:
                if (n_saved == 0) or ((self.load - last_saved_load) >= save_load_step):
                    self.append_zarr_checkpoint(zroot)
                    last_saved_load = float(self.load)
                    n_saved += 1
                    if n_saved <= 5 or (n_saved % 100 == 0):
                        print(f"[checkpoint] saved #{n_saved} at event {_event}, load={self.load:.6e}")

            # --- 2) Compute next event increment analytically ---
            next_load = self.get_next_slip_load(tol=tol)
            # print(f"Next event load: {next_load}")
            if not np.isfinite(next_load):
                break

            self.load = float(next_load)

        if zroot is not None:
            # Final report
            try:
                print(f"[checkpoint] done. total saved checkpoints: {zroot['load'].shape[0]}")
            except Exception:
                print(f"[checkpoint] done. total saved checkpoints: {n_saved}")

        return float(self.load), float(last_stable_eps)




    def find_critical_load_slip_event_driven(self, start_load=0.0):
        """
        Run the event-driven slip-FBM on a deep copy (like your traditional wrapper),
        and mirror histories back for plotting.

        IMPORTANT:
        - Checkpointing is DISABLED here to avoid writing files during deepcopy runs.
        """
        temp = copy.deepcopy(self)

        # Disable checkpointing on the temporary model
        # temp.checkpoint_path = None

        Lc, eps_c = temp.run_slip_fbm_event_driven(
            start_load=start_load,
            record_history=True,
        )

        self.history_critical_load = temp.history_critical_load.copy()
        self.history_critical_strain = temp.history_critical_strain.copy()
        self.history_intact = temp.history_intact.copy()

        return Lc, eps_c

