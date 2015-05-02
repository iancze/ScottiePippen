import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.ticker import FormatStrFormatter as FSF
from matplotlib.ticker import MultipleLocator
from scipy.interpolate import InterpolatedUnivariateSpline as spline

from grids import DartmouthPMS, PISA, Baraffe15, Seiss

# Functon lifted from triangle.py: https://github.com/dfm/triangle.py/
def hist2d(ax, x, y, sigs=[1,2], color="k", *args, **kwargs):
    """
    Plot a 2-D histogram of samples.
    """

    extent = [[x.min(), x.max()], [y.min(), y.max()]]
    bins = 50
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
    V = 1.0 - np.exp(-0.5 * np.array(sigs) ** 2)
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

fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(3.5,5.0))

# Models are in order: Baraffe, Dartmouth, PISA, SIESS
labels = ["BHAC15", "Dartmouth14", "PISA", "Siess"]

# First row is SED fit

# Second row is Teff/R posteriors overlaid with Dartmouth isomass and isochrones

# Third row is (1 sigma?) posteriors from all measurements with marginals overlaid on the side.

# We'll need to cook up a routine that can paint marginals on the side.

def interp(T, R):
    interp = spline(T, R, k=5)
    Tfine = np.linspace(np.min(T), np.max(T))
    Rfine = interp(Tfine)
    return (Tfine, Rfine)

hist2d(ax[0], temp, radius)
ax[0].xaxis.set_major_formatter(FSF("%.0f"))
ax[0].xaxis.set_major_locator(MultipleLocator(500))
ax[0].set_ylim(1.15, 1.65)
ax[0].set_xlim(7300, 5500)
ax[0].set_ylabel(r"$R_\ast$ [$R_\odot$]")
ax[0].set_xlabel(r"$T_\textrm{eff}$ [K]")

# Plot Dartmouth
grid = DartmouthPMS(age_range=[5, 40], mass_range=[0.5, 2.0])
grid.load()
masses = np.arange(1.2, 1.55, 0.1)
Ts = []
Rs = []
As = []
for mass in masses:
    inds = np.isclose(grid.masses, mass)
    tt = grid.temps[inds]
    rr = grid.radii[inds]
    aa = grid.ages[inds]
    # tfine, rfine = interp(tt, rr)
    Ts.append(tt)
    Rs.append(rr)
    As.append(aa)

age_bins = np.array([12.5, 15, 17.5, 20, 25])[::-1]
# colors = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#e6ab02", "#a6761d"]
colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00"]
age_labels = [r"$<$12.5 Myr", "12.5-15 Myr", "15-17.5 Myr", "17.5-20 Myr", "20-25  Myr"]

ax[0].annotate(r"1.2 $M_\odot$", xy=(0.8, 0.4), xycoords="axes fraction", rotation=-21, size=8)
ax[0].annotate(r"1.3 $M_\odot$", xy=(0.73, 0.78), xycoords="axes fraction", rotation=-37, size=8)
ax[0].annotate(r"1.4 $M_\odot$", xy=(0.14, 0.43), xycoords="axes fraction", size=8)
ax[0].annotate(r"1.5 $M_\odot$", xy=(0.07, 0.73), xycoords="axes fraction", rotation=18, size=8)
for T, R, A in zip(Ts, Rs, As):
    for i in range(len(colors)):
        ind = (A <= age_bins[i])
        ax[0].plot(T[ind], R[ind], "-", color=colors[::-1][i])

for i, label in enumerate(age_labels):
    ax[0].annotate(label, xy=(0.31, 0.33 - 0.07*i), xycoords="axes fraction", size=8, color=colors[i], backgroundcolor="w", ha="right")

hist_colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3"]

hist2d(ax[1], BARAFFE[:,1], BARAFFE[:,0], sigs=[1], color=hist_colors[0])
hist2d(ax[1], DARTMOUTH[:,1], DARTMOUTH[:,0], sigs=[1], color=hist_colors[1])
hist2d(ax[1], PISA_[:,1], PISA_[:,0], sigs=[1], color=hist_colors[2])
hist2d(ax[1], SIESS[:,1], SIESS[:,0], sigs=[1], color=hist_colors[3])

mass_low, mass_high = 0.9, 1.5
ax[1].set_xlim(mass_low, mass_high)
ax[1].set_ylim(0, 30.)
ax[1].axvspan(1.17, 1.31, color="0.8")

ax[1].xaxis.set_major_formatter(FSF("%.1f"))
ax[1].xaxis.set_major_locator(MultipleLocator(0.1))

ax[1].set_ylabel(r"$\tau_\ast$ [Myr]")
ax[1].set_xlabel(r"$M_\ast$ [$M_\odot$]")

# Define bins and get the line that corresponds to this
def plot_hist(ax, samples, bins, low, high, color, xaxis=True):
    hist, bin_edges = np.histogram(samples, bins)
    # Determine bin centers
    bin_centers = 0.5 * (bin_edges[0:-1] + bin_edges[1:])

    #Scale the hist output so that the maximum value of hist corresponds to high and 0 corresponds to low
    hist = hist * (high - low)/(np.max(hist) - 0.0) + low
    if xaxis:
        ax.plot(bin_centers, hist, color=color)
    else:
        ax.plot(hist, bin_centers, color=color)

mass_bins_BCAH = np.linspace(1.1, 1.4, num=20)
mass_bins = np.linspace(1.1, 1.6, num=30)


plot_hist(ax[1], BARAFFE[:,1], mass_bins_BCAH, low=0.0, high=10.0, color=hist_colors[0])
# Plot the closure to the BCAH models
BCAH_mass = 1.4 * np.ones((30,))
BCAH_age1 = np.linspace(18.2, 26.5, num=30)
BCAH_age2 = np.linspace(0, 4.5, num=30)
ax[1].plot(BCAH_mass, BCAH_age1, ls=":", color=hist_colors[0])
ax[1].plot(BCAH_mass, BCAH_age2, ls=":", color=hist_colors[0])

plot_hist(ax[1], DARTMOUTH[:,1], mass_bins, low=0.0, high=10.0, color=hist_colors[1])
plot_hist(ax[1], PISA_[:,1], mass_bins, low=0.0, high=10.0, color=hist_colors[2])
plot_hist(ax[1], SIESS[:,1], mass_bins, low=0.0, high=10.0, color=hist_colors[3])

age_bins = np.linspace(11, 35, num=30)
plot_hist(ax[1], BARAFFE[:,0], age_bins, low=mass_low, high=1.1, color=hist_colors[0], xaxis=False)
plot_hist(ax[1], DARTMOUTH[:,0], age_bins, low=mass_low, high=1.1, color=hist_colors[1], xaxis=False)
plot_hist(ax[1], PISA_[:,0], age_bins, low=mass_low, high=1.1, color=hist_colors[2], xaxis=False)
plot_hist(ax[1], SIESS[:,0], age_bins, low=mass_low, high=1.1, color=hist_colors[3], xaxis=False)

for i, label in enumerate(["BHAC15", "DartmouthPMS", "PISA", "Siess"]):
    ax[1].annotate(label, xy=(0.03, 0.27 - 0.07*i), xycoords="axes fraction", size=8, color=hist_colors[i], backgroundcolor="w", ha="left")

fig.subplots_adjust(left=0.19, right=0.81, top=0.98, bottom=0.1, hspace=0.3)
fig.savefig("posterior_column.pdf")
