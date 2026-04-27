import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.ticker import LogLocator, NullFormatter

# =======================
# STYLE (unchanged)
# =======================
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

# ============================================================
# Time scaling using your *unitless simulation* T99_sim
# ============================================================
N_sim = 200
D_sim = 1.0
T99_sim = 1865.025000002402

L_phys_ref = 4e-3   # [m]
D_phys_ref = 1e-11  # [m^2/s]

t0_seconds = (L_phys_ref**2 / D_phys_ref) * (D_sim / (N_sim**2))
T99_ref_h = (T99_sim * t0_seconds) / 3600.0

L_ref = L_phys_ref
D_ref = D_phys_ref

def T99_hours(L_m, D_m2s):
    return T99_ref_h * (L_m / L_ref)**2 * (D_ref / D_m2s)

def Tcycle_for_Fo(L_m, D_m2s, Fo=1.0):
    return Fo * T99_hours(L_m, D_m2s)

def sci_tex(x):
    """Return LaTeX snippet WITHOUT surrounding $...$."""
    if x == 0:
        return r"0"
    exp = int(np.floor(np.log10(abs(x))))
    mant = x / (10**exp)
    mant = float(np.round(mant, 3))

    if np.isclose(mant, 1.0):
        return rf"10^{{{exp}}}"
    else:
        return rf"{mant:g}\times 10^{{{exp}}}"


print(f"t0 = {t0_seconds:.3g} s per sim unit")
print(f"T99_ref = {T99_ref_h:.3g} h for L_ref={L_ref*1e3:.1f} mm, D_ref={D_ref:.2e} m^2/s")

# =======================
# PLOT RANGES
# =======================
D_min, D_max = 5e-12, 5e-10
L_min, L_max = 0.5e-3, 0.5e-1

TLEVELS_H = np.array([1, 5,  20,  100])
D = np.logspace(np.log10(D_min), np.log10(D_max), 600)

fig, ax = plt.subplots(figsize=(8, 6))
ax.set_xscale("log")
ax.set_yscale("log")

# ---- keep full grid (major+minor), but NO tick labels on x/y ----
ax.xaxis.set_major_locator(LogLocator(base=10.0))
ax.yaxis.set_major_locator(LogLocator(base=10.0))
ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10)*0.1))
ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10)*0.1))
# ax.xaxis.set_major_formatter(NullFormatter())
# ax.yaxis.set_major_formatter(NullFormatter())
ax.xaxis.set_minor_formatter(NullFormatter())
ax.yaxis.set_minor_formatter(NullFormatter())
ax.tick_params(axis='x', which='major', pad=10)
ax.set_xticks([ 1e-11, 5e-11])
ax.set_xticklabels([ r"$10^{-11}$", r"$5\times 10^{-11}$"])

# ---- Colormap setup ----
norm = mcolors.LogNorm(vmin=TLEVELS_H.min(), vmax=TLEVELS_H.max())
cmap = cm.copper

D_label_pos = 12e-11  # choose something inside your x-limits

for Tc_h in TLEVELS_H:
    color = cmap(norm(Tc_h))

    # Fo_chi=1 => Tc = T99 => L(D) = L_ref * sqrt( (Tc/T99_ref) * (D/D_ref) )
    L_line = L_ref * np.sqrt((Tc_h / T99_ref_h) * (D / D_ref))
    mask = (L_line >= L_min) & (L_line <= L_max)

    if np.any(mask):
        ax.plot(D[mask], L_line[mask], lw=2.5, color=color)

        # label on the curve at D_label_pos
        L_label = L_ref * np.sqrt((Tc_h / T99_ref_h) * (D_label_pos / D_ref))
        if L_min < L_label < L_max:
            ax.text(D_label_pos, L_label * 0.6,
                    rf"$T_c={2*Tc_h:g}\,\mathrm{{h}}$",
                    color=color, fontsize=18, ha="center", va="bottom")

# =======================
# Experimental points
# =======================
D_pt = 1e-11

points = [
    (D_pt, 4e-3, "4 mm"),
    (D_pt, 10e-3, "10 mm"),
]

for (D_i, L_i, lab) in points:
    Tc_i = Tcycle_for_Fo(L_i, D_i, Fo=1.0)
    ax.scatter([D_i], [L_i], s=55, marker="o", zorder=6, color="black")
    s=sci_tex(D_i)
    txt = (
        # rf"$\chi_d={s}$" "\n"
        rf"$L={L_i*1e3:.0f}\,\mathrm{{mm}}$" "\n"
        rf"$T_c(Fo_\chi^{{99}}=2)\approx {int(2*round(Tc_i))}\,\mathrm{{h}}$"

    )

    ax.text(D_i * 1.025, L_i * 2, txt,
            ha="left", va="top", fontsize=20,
            bbox=dict(facecolor="white", edgecolor="black", alpha=0.55))

    print(f"{lab}: D={D_i:.2e}, L={L_i:.3e} -> T99={T99_hours(L_i,D_i):.3g} h, Tc={Tc_i:.3g} h")

# =======================
# Axes + grid + labels
# =======================
ax.set_xlim(D_min, D_max)
ax.set_ylim(L_min, L_max)

ax.text(0.03, 0.97, r"$Fo_{\chi}^{99}=2$",
        transform=ax.transAxes,
        ha="left", va="top",
        bbox=dict(facecolor="white", edgecolor="black"))

ax.set_xlabel(r"$\chi_d$  [m$^2$/s]")
ax.set_ylabel(r"$L$  [m]")

# full grid including minor
ax.grid(True, which="major", alpha=0.35)
ax.grid(True, which="minor", alpha=0.18)
ax.set_position([0.15, 0.15, 0.72, 0.7])


plt.tight_layout()
plt.savefig("./Fig4a.png", dpi=300)
# plt.show()
