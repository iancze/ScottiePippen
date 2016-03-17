#!/usr/bin/env python

import argparse

parser = argparse.ArgumentParser(description="Plot mass contours.")
parser.add_argument("--config", default="config.yaml", help="The config file specifying everything we need.")
args = parser.parse_args()

# Likelihood functions to convert posteriors in weird formats into posteriors on temp, log10 Luminosity for a single star (of a potential binary).

import yaml

f = open(args.config)
config = yaml.load(f)
f.close()


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.ticker import FormatStrFormatter as FSF
from matplotlib.ticker import MultipleLocator
from scipy.interpolate import InterpolatedUnivariateSpline as spline

from ScottiePippen.grids import model_dict

# Functon lifted from triangle.py: https://github.com/dfm/triangle.py/
def hist2d(ax, x, y, sigs=[1,2], color="k", *args, **kwargs):
    """
    Plot a 2-D histogram of samples.
    """

    extent = kwargs.get("extent", None)
    # if extent is None:
    #     extent = [[x.min(), x.max()], [y.min(), y.max()]]

    if extent is None:
        xmin = x.min()
        xmax = x.max()
        xrange = xmax - xmin
        xbuf = 0.2 * xrange
        ymin = y.min()
        ymax = y.max()
        yrange = ymax - ymin
        ybuf = 0.2 * yrange

        extent = [[xmin - xbuf, xmax + xbuf], [ymin - ybuf, ymax+ ybuf]]

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

    V = 1.0 - np.exp(-0.5 * np.array(sigs) ** 2)

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


fig, ax = plt.subplots(nrows=1, figsize=(3.5,2.2))

# Models are in order: Baraffe, Dartmouth, PISA, SIESS
# grid_names = config["grids"]
grid_name = "DartmouthPMS"

# hist_colors = {"Baraffe15":"#e41a1c", "DartmouthPMS":"#377eb8", "PISA":"#4daf4a", "Seiss":"#984ea3"}


# First, plot the 2D contours for each model

# This will be custom edited for each plot
# extent = [[0.3, 1.0], [0, 10]]

#
mass_low, mass_high = 0.35, 0.8
ax.set_xlim(mass_low, mass_high)
# ax[1].set_xlim(mass_low, mass_high)
#
age_low, age_high = 0.0, 40
ax.set_ylim(age_low, age_high)
# ax[1].set_ylim(age_low, age_high)

# tot_low, tot_high = 0.5, 1.5
# ax[2].set_xlim(tot_low, tot_high)
#
# q_low, q_high = 0.7, 1.15
# ax[2].set_ylim(q_low, q_high)


# Plot Willies RV ratio

ax.axvspan(0.61 - 0.14, 0.61 + 0.14, color="0.8")
ax.axvline(0.59, color="k", ls="-.")
ax.axvline(0.63, color="k", ls="-.")

# Plot the mass constraint
# ax.axvspan(0.66, 1.06, color="0.8")

#
# ax.xaxis.set_major_formatter(FSF("%.1f"))


ax.set_xlabel(r"$M$ [$M_\odot$]")
# ax[1].set_xlabel(r"$M_1$ [$M_\odot$]")

ax.set_ylabel(r"$\tau$ [Myr]")
# ax[1].set_ylabel(r"$\tau_1$ [Myr]")

# ax[2].set_xlabel(r"$M_\ast$ [$M_\odot$]")
# ax[2].set_ylabel(r"$q$")

# Define bins and get the line that corresponds to this
def plot_hist(ax, samples, bins, low, high, color, xaxis=True):
    hist, bin_edges = np.histogram(samples, bins)
    # Determine bin centers
    bin_centers = 0.5 * (bin_edges[0:-1] + bin_edges[1:])

    #Scale the hist output so that the maximum value of hist corresponds to high and 0 corresponds to low
    # hist = hist * (high - low)/(np.max(hist) - 0.0) + low

    # Scale the hist output so that the minimum value of hist corresponds to low and 0 corresponds to high
    hist = hist * (low - high)/(np.max(hist) - 0.0) + high

    if xaxis:
        ax.plot(bin_centers, hist, color=color)
    else:

        ax.plot(hist, bin_centers, color=color)

mass_bins = np.linspace(mass_low, mass_high, num=35)
age_bins = np.linspace(age_low, age_high, num=35)

# tot_bins = np.linspace(tot_low, tot_high, num=35)
# q_bins = np.linspace(q_low, q_high, num=35)

# for grid_name in grid_names:
samples = np.load("flatchain_{}.npy".format(grid_name))

age = samples[:,0]
mass0 = samples[:,1]

mass1 = samples[:,2]

# tot = mass0 + mass1
# q = mass1/mass0

# color = hist_colors[grid_name]

hist2d(ax, mass1, age, sigs=[1], color="r") #, extent=extent)
hist2d(ax, mass0, age, sigs=[1], color="b") #, extent=extent)
# hist2d(ax, tot, q, sigs=[1], color=color)

plot_hist(ax, mass1, mass_bins, low=30, high=age_high, color="r")
plot_hist(ax, age, age_bins, low=0.72, high=mass_high, color=(140/256, 40/256, 200/256), xaxis=False)

plot_hist(ax, mass0, mass_bins, low=30, high=age_high, color="b")
# plot_hist(ax, age, age_bins, low=0.72, high=mass_high, color="b", xaxis=False)


# plot_hist(ax[2], tot, tot_bins, low=1.03, high=q_high, color=color)
# plot_hist(ax[2], q, q_bins, low=1.3, high=tot_high, color=color, xaxis=False)



# for i, grid_name in enumerate(grid_names):
#     color = hist_colors[grid_name]
#     fig.text(0.16 + 0.24 * i, 0.92, grid_name, size=10, color=color, ha="center")

# ax.xaxis.set_major_locator(MultipleLocator(0.2))
# ax[1].xaxis.set_major_locator(MultipleLocator(0.2))
# ax[2].xaxis.set_major_locator(MultipleLocator(0.2))



# ax[2].axhline(0.936, ls="-.", color="k")

fig.subplots_adjust(left=0.19, right=0.81, top=0.96, bottom=0.19, wspace=0.35)
fig.savefig("age_mass.pdf")
