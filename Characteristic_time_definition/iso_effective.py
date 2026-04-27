import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.ticker import NullLocator, NullFormatter

# -------------------- STYLE --------------------
plt.rcParams.update({
    'font.size': 28,
    'axes.labelsize': 28,
    'xtick.labelsize': 23,
    'ytick.labelsize': 23,
    'legend.fontsize': 24,
    'figure.titlesize': 25,
    'lines.linewidth': 3.5,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'Computer Modern Roman'],
    'mathtext.fontset': 'cm',
    'mathtext.default': 'it',
    'text.usetex': True,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.major.width': 1.5,
    'ytick.major.width': 1.5,
    'axes.linewidth': 1,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'legend.framealpha': 1.0,
    'legend.edgecolor': 'black',
    'savefig.dpi': 300,
    'savefig.pad_inches': 0.05,
    'figure.dpi': 150,
    'savefig.bbox': None,
})

def sci_tex(x):
    """LaTeX snippet without $...$; e.g. 1e-11 -> 10^{-11} or 2.5e-11 -> 2.5\\times10^{-11}"""
    if x == 0:
        return r"0"
    exp = int(np.floor(np.log10(abs(x))))
    mant = x / (10**exp)
    mant = float(np.round(mant, 3))
    if np.isclose(mant, 1.0):
        return rf"10^{{{exp}}}"
    return rf"{mant:g}\times 10^{{{exp}}}"

# ============================================================
# PART 1) VISCOELASTIC: KV chain -> T_tau^99
# ============================================================
J_i = np.array([2.1, 0.87, 1.8, 2.8]) * 1e-4   # [1/Pa]
tau_i = np.array([0.1, 1.0, 10.0, 100.0])      # [h]
J_inf = float(J_i.sum())

def J_of_t(t_h):
    t_h = np.asarray(t_h, dtype=float)
    return np.sum(J_i * (1.0 - np.exp(-t_h[:, None] / tau_i[None, :])), axis=1)

def find_T99_visco(target=0.99):
    # robust bracket + bisection in hours
    lo, hi = 0.0, 1.0
    while True:
        val = float(J_of_t(np.array([hi])) / J_inf)
        if val >= target:
            break
        hi *= 2.0
        if hi > 1e6:
            raise RuntimeError("Could not bracket T99 (hi too large).")

    for _ in range(80):
        mid = 0.5 * (lo + hi)
        val = float(J_of_t(np.array([mid])) / J_inf)
        if val < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)

Ttau99_h = find_T99_visco(target=0.99)

print(f"KV chain: J_inf = {J_inf:.3e} 1/Pa")
print(f"KV chain: T_tau^99 = {Ttau99_h:.3g} h")

# Optional: quick check plot (comment out if you don't want it)
t_plot = np.logspace(-3, 4, 800)  # hours
J_norm = J_of_t(t_plot) / J_inf
# plt.figure(figsize=(8,6))
# plt.xscale("log")
# plt.plot(t_plot, J_norm)
# plt.axhline(0.99, ls="--")
# plt.axvline(Ttau99_h, ls="--")
# plt.xlabel(r"$t$ [h]")
# plt.ylabel(r"$J(t)/J_{\infty}$ [-]")
# plt.tight_layout()
# plt.show()

# ============================================================
# PART 2) DIFFUSION: use your unitless simulation scaling -> T_chi^99(L,D)
# ============================================================
# Simulation reference
N_sim = 200
D_sim = 1.0
Tchi99_sim = 1865.025000002402  # sim units

# Choose physical mapping target (your choice)
L_ref = 4e-3     # [m] 4 mm corresponds to 200 cells
D_ref = 1e-11    # [m^2/s] (0.1*1e-10)

t0_seconds = (L_ref**2 / D_ref) * (D_sim / (N_sim**2))  # seconds per sim time unit
Tchi99_ref_h = (Tchi99_sim * t0_seconds) / 3600.0

def Tchi99_hours(L_m, D_m2s):
    return Tchi99_ref_h * (L_m / L_ref)**2 * (D_ref / D_m2s)

print(f"Diffusion scaling: t0 = {t0_seconds:.3g} s per sim unit")
print(f"Mapped diffusion T_chi^99(ref) = {Tchi99_ref_h:.3g} h for L_ref=4mm, D_ref=1e-11")

# ============================================================
# PART 3) FIX A CYCLE TIME and place two samples on effective map
# ============================================================
Tc_h = 20.0  # <-- set your experimental cycle time here (hours)

D_species = 1e-11         # same species -> same diffusion coefficient
samples = [
    {"name": "4 mm",  "L": 4e-3},
    {"name": "10 mm", "L": 10e-3},
]

# Effective model parameters (Tr=0)
a = 0.653
b = 0.064
Tr = 0.0  # ramping off -> exp(cTr)=1

# Compute points
pts = []
for s in samples:
    Lm = s["L"]
    Tchi = Tchi99_hours(Lm, D_species)
    Fo_chi = Tc_h / Tchi
    Fo_tau = Tc_h / Ttau99_h
    Fo_eff = (Fo_chi**a) * (Fo_tau**b)
    pts.append((Fo_tau, Fo_chi, Fo_eff, s["name"], Tchi))

    print(f"{s['name']}: T_chi^99={Tchi:.2f} h  -> Fo_chi^99={Fo_chi:.3g}")
print(f"Both samples share: Fo_tau^99 = {Tc_h:.3g}/{Ttau99_h:.3g} = {Tc_h/Ttau99_h:.3g}")

# ============================================================
# PART 4) PLOT effective iso-lines + the two samples
# ============================================================
Fo_tau_min, Fo_tau_max = 1e-3, 1e2
Fo_chi_min, Fo_chi_max = 1e-3, 1e2
FO_EFF_LEVELS = np.array([1e-2, 3e-2, 1e-1, 3e-1, 1, 3, 10])

Fo_tau_grid = np.logspace(np.log10(Fo_tau_min), np.log10(Fo_tau_max), 900)

fig, ax = plt.subplots(figsize=(8, 6))
ax.set_xscale("log")
ax.set_yscale("log")

norm = mcolors.LogNorm(vmin=FO_EFF_LEVELS.min(), vmax=FO_EFF_LEVELS.max())
cmap = cm.copper
x_label = 10

for fe in FO_EFF_LEVELS:
    color = cmap(norm(fe))
    Fo_chi_line = (fe / (Fo_tau_grid**b)) ** (1.0 / a)
    mask = (Fo_chi_line >= Fo_chi_min) & (Fo_chi_line <= Fo_chi_max)
    if np.any(mask):
        ax.plot(Fo_tau_grid[mask], Fo_chi_line[mask], lw=2.5, color=color)
        y_label = (fe / (x_label**b)) ** (1.0 / a)
        if Fo_chi_min < y_label < Fo_chi_max:
            ax.text(x_label, y_label * 1.1,
                    rf"$Fo_{{\mathrm{{eff}}}}={2*fe:g}$",
                    color=color, fontsize=18, ha="center", va="bottom")

# plot points
for (x, y, fe, name, Tchi) in pts:
    ax.scatter([x], [y], s=65, color="black", zorder=6)
    print("NAME", name, "Fo_tau", x, "Fo_chi", y, "Fo_eff", fe)
    if name =="4 mm":
        ax.text(x*1.12, y*1.10,
            rf"L={name}" "\n"
            # rf"$T_\chi^{{99}}\approx {Tchi:.0f}\,\mathrm{{h}}$" "\n"
            rf"$Fo_{{\mathrm{{eff}}}}^{{99}}={y*2:.2g}$",
             ha="left", va="bottom", fontsize=20,
            bbox=dict(facecolor="white", edgecolor="black", alpha=0.55))
    else:
        ax.text(x*1.12, y*0.19,
                rf"L={name}" "\n"
                # rf"$T_\chi^{{99}}\approx {Tchi:.0f}\,\mathrm{{h}}$" "\n"
                rf"$Fo_{{\mathrm{{eff}}}}^{{99}}={y:.2g}$",
                ha="left", va="bottom", fontsize=20,
                bbox=dict(facecolor="white", edgecolor="black", alpha=0.55))

ax.set_xlim(Fo_tau_min, Fo_tau_max)
ax.set_ylim(Fo_chi_min, Fo_chi_max)

ax.set_xticks([1e-3, 1e-2, 1e-1, 1, 10, 100])
ax.set_xticklabels([r"$10^{-3}$", r"$10^{-2}$", r"$10^{-1}$",
                    r"$10^{0}$", r"$10^{1}$", r"$10^{2}$"])
ax.xaxis.set_minor_locator(NullLocator())
ax.xaxis.set_minor_formatter(NullFormatter())
ax.yaxis.set_minor_locator(NullLocator())
ax.yaxis.set_minor_formatter(NullFormatter())
ax.tick_params(axis='x', which='major', pad=10)

ax.set_xlabel(r"$Fo_{\tau}^{99} [-]$")
ax.set_ylabel(r"$Fo_{\chi}^{99} [-]$")
ax.grid(True, which="both", alpha=0.25)

ax.text(0.03, 0.97,
        # rf"$T_r=0$" "\n"
        rf"$T_c={2*Tc_h:.0f}\,\mathrm{{h}}$",
        # rf"$T_\tau^{{99}}\approx {Ttau99_h:.0f}\,\mathrm{{h}}$",
        transform=ax.transAxes,
        ha="left", va="top",
        bbox=dict(facecolor="white", edgecolor="black"))


ax.set_position([0.15, 0.15, 0.72, 0.7])

plt.tight_layout()
plt.savefig("./Fig4b.png", dpi=300)
# plt.show()
