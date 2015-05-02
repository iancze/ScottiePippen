import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.ticker import FormatStrFormatter as FSF
from matplotlib.ticker import MultipleLocator
from scipy.interpolate import InterpolatedUnivariateSpline as spline

from grids import DartmouthPMS, PISA, Baraffe15, Seiss

# Functon lifted from triangle.py: https://github.com/dfm/triangle.py/
def hist2d(ax, x, y, *args, **kwargs):
    """
    Plot a 2-D histogram of samples.
    """

    extent = [[x.min(), x.max()], [y.min(), y.max()]]
    bins = 50
    color = "k"
    linewidths = 0.8

    cmap = cm.get_cmap("gray")
    cmap._init()
    cmap._lut[:-3, :-1] = 0.
    cmap._lut[:-3, -1] = np.linspace(1, 0, cmap.N)

    X = np.linspace(extent[0][0], extent[0][1], bins + 1)
    Y = np.linspace(extent[1][0], extent[1][1], bins + 1)
    try:
        H, X, Y = np.histogram2d(x.flatten(), y.flatten(), bins=(X, Y))
    except ValueError:
        raise ValueError("It looks like at least one of your sample columns "
                         "have no dynamic range. You could try using the "
                         "`extent` argument.")

    # V = 1.0 - np.exp(-0.5 * np.array([1.0, 2.0, 3.0]) ** 2)
    V = 1.0 - np.exp(-0.5 * np.array([1.0, 2.0]) ** 2)
    #V = 1.0 - np.exp(-0.5 * np.arange(0.5, 2.1, 0.5) ** 2)
    Hflat = H.flatten()
    inds = np.argsort(Hflat)[::-1]
    Hflat = Hflat[inds]
    sm = np.cumsum(Hflat)
    sm /= sm[-1]

    for i, v0 in enumerate(V):
        try:
            V[i] = Hflat[sm <= v0][-1]
        except:
            V[i] = Hflat[0]

    X1, Y1 = 0.5 * (X[1:] + X[:-1]), 0.5 * (Y[1:] + Y[:-1])
    X, Y = X[:-1], Y[:-1]

    # Plot the contours
    # ax.pcolor(X, Y, H.max() - H.T, cmap=cmap)
    ax.contour(X1, Y1, H.T, V, colors=color, linewidths=linewidths)

    ax.set_xlim(extent[0])
    ax.set_ylim(extent[1])

TR = np.load("eparams_R.npy")
temp = TR[:,0]
radius = TR[:,5]

BARAFFE = np.load("plots/Baraffe15/eparams_emcee.npy")
DARTMOUTH = np.load("plots/DartmouthPMS/eparams_emcee.npy")
PISA_ = np.load("plots/PISA/eparams_emcee.npy")
SIESS = np.load("plots/Seiss/eparams_emcee.npy")

fig,ax = plt.subplots(nrows=2, ncols=4, figsize=(6.5,4.))

# Models are in order: Baraffe, Dartmouth, PISA, SIESS
labels = ["BCAH15", "Dartmouth14", "PISA", "Siess"]

# First row is T - R diagrams

# Second row is M - Age diagrams

def interp(T, R):
    interp = spline(T, R, k=5)
    Tfine = np.linspace(np.min(T), np.max(T))
    Rfine = interp(Tfine)
    return (Tfine, Rfine)


for i,a in enumerate(ax[0]):
    hist2d(a, temp, radius)

    a.xaxis.set_major_formatter(FSF("%.0f"))
    a.xaxis.set_major_locator(MultipleLocator(500))
    a.set_xlim(7100, 5200)
    a.annotate(labels[i], (0.05, 0.05), xycoords="axes fraction", size=6, backgroundcolor="w")

    if i != 0:
        a.xaxis.set_ticks([])
        a.yaxis.set_ticks([])

ax[0,0].set_ylabel(r"$R_\ast$ [$R_\odot$]")
ax[0,0].set_xlabel(r"$T_\textrm{eff}$ [K]")

# Break the tracks up into ages


# Plot Baraffe
# Baraffe15
grid = Baraffe15(age_range=[1, 50], mass_range=[0.9, 1.4])
grid.load()
masses = np.arange(1.2, 1.5, 0.1)
Ts = []
Rs = []
for mass in masses:
    inds = np.isclose(grid.masses, mass)
    tt = grid.temps[inds]
    rr = grid.radii[inds]
    # tfine, rfine = interp(tt, rr)
    Ts.append(tt)
    Rs.append(rr)

for T, R in zip(Ts, Rs):
    ax[0,0].plot(T, R, "-", color="0.5")

# Plot Dartmouth
grid = DartmouthPMS(age_range=[1, 100], mass_range=[0.5, 2.0])
grid.load()
masses = np.arange(1.2, 1.55, 0.1)
Ts = []
Rs = []
for mass in masses:
    inds = np.isclose(grid.masses, mass)
    tt = grid.temps[inds]
    rr = grid.radii[inds]
    # tfine, rfine = interp(tt, rr)
    Ts.append(tt)
    Rs.append(rr)

for T, R in zip(Ts, Rs):
    ax[0,1].plot(T, R, "-", color="0.5")

# Plot PISA
grid = PISA(age_range=[1, 100], mass_range=[0.5, 2.0])
grid.load()
masses = np.arange(1.2, 1.55, 0.1)
Ts = []
Rs = []
for mass in masses:
    inds = np.isclose(grid.masses, mass)
    tt = grid.temps[inds]
    rr = grid.radii[inds]
    # tfine, rfine = interp(tt, rr)
    Ts.append(tt)
    Rs.append(rr)

for T, R in zip(Ts, Rs):
    ax[0,2].plot(T, R, "-", color="0.5")

# Plot Siess
grid = Seiss(age_range=[1, 100], mass_range=[0.5, 2.0])
grid.load()
masses = np.arange(1.2, 1.55, 0.1)
Ts = []
Rs = []
for mass in masses:
    inds = np.isclose(grid.masses, mass)
    tt = grid.temps[inds]
    rr = grid.radii[inds]
    # tfine, rfine = interp(tt, rr)
    Ts.append(tt)
    Rs.append(rr)

for T, R in zip(Ts, Rs):
    ax[0,3].plot(T, R, "-", color="0.5")

hist2d(ax[1,0], BARAFFE[:,1], BARAFFE[:,0])
hist2d(ax[1,1], DARTMOUTH[:,1], DARTMOUTH[:,0])
hist2d(ax[1,2], PISA_[:,1], PISA_[:,0])
hist2d(ax[1,3], SIESS[:,1], SIESS[:,0])

for i,a in enumerate(ax[1]):
    a.set_xlim(1.1, 1.5)
    a.set_ylim(10, 25.)
    a.axvspan(1.17, 1.31, color="0.8")

    a.xaxis.set_major_formatter(FSF("%.1f"))
    a.xaxis.set_major_locator(MultipleLocator(0.1))

    if i != 0:
        a.xaxis.set_ticks([])
        a.yaxis.set_ticks([])


ax[1,0].axvline(1.4, color="0.5", linestyle=":")
ax[1,0].set_ylabel(r"$\tau$ [Myr]")
ax[1,0].set_xlabel(r"$M_\ast$ [$M_\odot$]")




fig.subplots_adjust(left=0.1, right=0.9, wspace=0.0, top=0.98, bottom=0.1, hspace=0.3)
fig.savefig("posterior.pdf")
