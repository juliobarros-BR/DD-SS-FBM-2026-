#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 10:03:06 2024

@author: jortiz
"""

import ast
import numpy as np
from Model_files.Model_class_copy_moist_grad_control_new import Model
from Model_files.Sim_class_moist_grad import Simulate
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import matplotlib

# matplotlib.use('Qt5Agg')




parser = argparse.ArgumentParser()
parser.add_argument("-f", "--filename", help="Input file name")
args = parser.parse_args()
file_path=args.filename
if file_path==None:
    file_path="input_single_run.txt"

main_model = Model(file_path)

main_sim = Simulate(main_model)

main_sim.run(compare_n_cycles_ago = 10)

main_sim.History["Time"][0]=0


df = pd.DataFrame(main_sim.History)

df.to_csv("all.csv")

fig, ax1 = plt.subplots(dpi=300,figsize=(16, 6))

# left axis: strains (your originals)
t = np.array(main_sim.History["Time"])
ax1.plot(t, np.array(main_sim.History["Total_strain"]) / main_sim.model.critical_strain, label=r"$<\varepsilon>$")
ax1.plot(t, np.array(main_sim.History["Hygroexp"])     / main_sim.model.critical_strain, label=r"$<\varepsilon_{H}>$")
ax1.plot(t, np.array(main_sim.History["Creep"])        / main_sim.model.critical_strain, label=r"$<\varepsilon_{C}>$")
ax1.plot(t, np.array(main_sim.History["Elastic"])      / main_sim.model.critical_strain, label=r"$<\varepsilon_{E}>$")
ax1.plot(t, np.array(main_sim.History["Slip_strain"])  / main_sim.model.critical_strain, label=r"$<\varepsilon_{S}>$")
ax1.plot(t, (np.array(main_sim.History["Hygroexp"])  +
              np.array(main_sim.History["Creep"]) +
              np.array(main_sim.History["Elastic"]) )/ main_sim.model.critical_strain, label=r"$<\varepsilon>$")



ax1.set_xlim(0, float(np.max(t)))
grid_spacing = 5
x_min, x_max = ax1.get_xlim()
custom_xticks = np.arange(x_min, x_max + grid_spacing, grid_spacing)
# ax1.set_xticks(custom_xticks)  # uncomment if you want fixed spacing
ax1.grid(True)
ax1.set_xlabel(r"$t/\tau_4$")
ax1.set_ylabel(r"$\varepsilon/\varepsilon_c$")

# right axis: moisture (gray, alpha=0.3)
moisture_color = (0.5, 0.5, 0.5)  # gray
ax2 = ax1.twinx()
line_moist, = ax2.plot(t, np.array(main_sim.History["Moisture"]),
                        label=r"$<\omega>$", color=moisture_color, alpha=0.3)

# style the moisture axis in gray with alpha=0.3
ax2.spines["right"].set_color(moisture_color)
ax2.spines["right"].set_alpha(0.3)
ax2.tick_params(axis="y", colors=moisture_color)
for lbl in ax2.get_yticklabels():
    lbl.set_alpha(1)                 # tick label alpha
for tick in ax2.yaxis.get_ticklines():
    tick.set_alpha(1)                # tick mark alpha
    tick.set_color(moisture_color)
ax2.set_ylabel(r"$\langle\omega\rangle/0.3$")
ax2.yaxis.label.set_color(moisture_color)
ax2.yaxis.label.set_alpha(1)

# draw ax2 behind ax1 so its line never sits above the legend

# combined legend, opaque and on top
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
leg = ax1.legend(lines1 + lines2, labels1 + labels2,
                  loc="upper left", bbox_to_anchor=(1.15, 1),
                  borderaxespad=0., framealpha=1, facecolor="white", edgecolor="black")

fig.tight_layout()
fig.savefig("result.png", dpi=300)
# plt.show()
        

plt.figure(dpi=300,figsize=(16, 6))
plt.title("Slip Strain")
plt.plot(main_sim.History["Time"],np.array(main_sim.History["Slip_strain"]/main_sim.model.critical_strain))
# plt.plot(main_sim.History["Time"])
plt.xlabel(r"Time ($t/\tau_1$)")
plt.grid()
# plt.xlim(0)
# plt.ylim(0,0.12)
plt.ylabel(r"Slip Strain ($\varepsilon_s/\varepsilon_c$)")
plt.savefig("slip.png")
# plt.show()

