#!/usr/bin/env python

import argparse

parser = argparse.ArgumentParser(description="Convert literature estimates into T, log10L form.")
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
    if extent is None:
        extent = [[x.min(), x.max()], [y.min(), y.max()]]

    bins = 35
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

    ax.set_xlim(extent[0])
    ax.set_ylim(extent[1])


fig, ax = plt.subplots(figsize=(3.5,2.5))

# Plot Dartmouth
grid = model_dict["DartmouthPMS"](age_range=[0.01, 250], mass_range=[0.2, 1.5])

masses = np.array([0.3, 0.4, 0.5, 0.6, 0.7, 0.8])

ages = np.linspace(0.1, 200, num=200)

TT = []
LL = []

for mass in masses:
    Ts = []
    lLs = []

    for age in ages:
        p = np.array([age, mass])
        Ts.append(grid.interp_T(p))
        lLs.append(grid.interp_lL(p))

    TT.append(np.array(Ts))
    LL.append(10**np.array(lLs))

for Ts, LLs in zip(TT, LL):
    ax.plot(Ts, LLs, "-", color="k", lw=0.8)

masses_fine = np.linspace(0.25, 0.9, num=200)
ages_coarse = np.array([2.0, 5., 10., 30., 100.])

# Now plot the isochrones
for age in ages_coarse:
    TT = []
    LL = []
    for mass in masses_fine:
        p = np.array([age, mass])
        T = grid.interp_T(p)
        L = (10**grid.interp_lL(p))
        TT.append(T)
        LL.append(L)
    ax.plot(TT, LL, "k:", lw=0.5)

# TT = []
# LL = []
#
# for mass in masses:
#     Ts = []
#     lLs = []
#
#     for age in ages:
#         p = np.array([age, mass])
#         Ts.append(grid.interp_T(p))
#         lLs.append(grid.interp_lL(p))
#
#     TT.append(np.array(Ts))
#     LL.append(np.array(lLs))

# age_bins = np.array([1, 3, 5, 15, 60])[::-1]
# colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00"]
# age_labels = [r"$<$1 Myr", "1-3 Myr", "3-5 Myr", "5-15 Myr", "15-60 Myr"]


# for Ts, lLs in zip(TT, LL):
#     for i in range(len(colors)):
#         ind = (ages <= age_bins[i])
#         ax.plot(Ts[ind], 10**lLs[ind], "-", color=colors[::-1][i])

# for i, label in enumerate(age_labels):
#     ax.annotate(label, xy=(0.08, 0.91 - 0.07*i), xycoords="axes fraction", size=8, color=colors[i], backgroundcolor="w", ha="left")

TL = np.load("star1.npy") #config["TlL_samples"])
temp = TL[:,0]
L = 10**TL[:,1]
hist2d(ax, temp, L, sigs=[1.0], color="r")

TL = np.load("star0.npy") #config["TlL_samples"])
temp = TL[:,0]
L = 10**TL[:,1]
hist2d(ax, temp, L, sigs=[1.0], color="b")

ax.annotate(r"0.4 $M_\odot$", xy=(3600, 0.37), rotation=-75, size=8)
ax.annotate(r"0.6 $M_\odot$", xy=(3840, 0.37), rotation=-71, size=8)
ax.annotate(r"0.8 $M_\odot$", xy=(4090, 0.37), rotation=-69, size=8)

ax.annotate("2 Myr", xy=(3500, 0.23), rotation=-45, size=8)
ax.annotate("5 Myr", xy=(3900, 0.31), rotation=-40, size=8)
ax.annotate("10 Myr", xy=(4100, 0.28), rotation=-40, size=8)
ax.annotate("30 Myr", xy=(4080, 0.17), rotation=-22, size=8)
ax.annotate("100 Myr", xy=(3900, 0.04), rotation=-13, size=8)


ax.xaxis.set_major_formatter(FSF("%.0f"))
ax.xaxis.set_major_locator(MultipleLocator(200))
ax.set_ylim(0.0, 0.4)
ax.set_xlim(4100, 3300)
ax.set_ylabel(r"$L_\ast$ [$L_\odot$]")
ax.set_xlabel(r"$T_\textrm{eff}$ [K]")

fig.subplots_adjust(left=0.19, right=0.81, top=0.96, bottom=0.16, hspace=0.3)
fig.savefig("HR_contours.pdf")
