import numpy as np
# import matplotlib
# matplotlib.use("Qt4Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap as LSC
from matplotlib.ticker import FormatStrFormatter as FSF
from matplotlib.ticker import MultipleLocator

from matplotlib.colors import ColorConverter
from scipy.interpolate import InterpolatedUnivariateSpline as spline

from grids import DartmouthPMS, PISA, Baraffe15, Seiss


# Functon lifted from triangle.py: https://github.com/dfm/triangle.py/
def hist2d(ax, x, y, sigs=[1], color="k", pcolor="grey", *args, **kwargs):
    """
    Plot a 2-D histogram of samples.
    """

    extent = kwargs.get("extent", None)
    if extent is None:
        extent = [[x.min(), x.max()], [y.min(), y.max()]]

    bins = 45
    linewidths = 0.8

    # Instead of this, create a color map with the peak color.

    if pcolor != "grey":
        # print(pcolor)
        r,g,b = pcolor
        # print(r, g, b)

        # Make our custom intensity scale
        dict_cmap = {'red':[(0.0,  r, r),
                           (1.0,  1.0,  1.0)],

                 'green': [(0.0,  g, g),
                           (1.0,  1.0, 1.0)],

                 'blue':  [(0.0,  b, b),
                           (1.0,  1.0,  1.0)]}

        cmap = LSC("new", dict_cmap)
    else:
        cmap = cm.get_cmap("gray")

    cmap._init()

    # The only thing he's changing here is the alpha interpolator, I think

    # He's saying that we will have everything be black, and change alpha from 1 to 0.0

    # cmap._lut[:-3, :-1] = 0.
    cmap._lut[:-3, -1] = np.linspace(1, 0, cmap.N)

    # N is the number of levels in the colormap
    # Dunno what _lut is
    # look up table
    # Is he setting everything below some value to 0?


    X = np.linspace(extent[0][0], extent[0][1], bins + 1)
    # Y = np.linspace(extent[1][0], extent[1][1], bins + 1)
    Y = np.logspace(np.log10(extent[1][0]), np.log10(extent[1][1]), bins + 1)

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
    ax.pcolor(X, Y, H.max() - H.T, cmap=cmap)
    ax.contour(X1, Y1, H.T, V, colors=color, linewidths=linewidths)

    # ax.set_xlim(extent[0])
    # ax.set_ylim(extent[1])

# These are the luminosities for 1 star

TL = np.load("eparams_L.npy")
T_DQTAU = 10**TL[:,0] # [K]
L_DQTAU = 10**TL[:,1] # [L_sun]

TR = np.load("eparams_R.npy")
T_AKSCO = TR[:,0] # [K]
L_AKSCO =  0.5 * 10**TR[:,2] # [L_sun]

TL = np.load("eparams_LK5.npy")
T_K5 = 10**TL[:,0] # [K]
L_K5 = 10**TL[:,1] # [L_sun]

TL = np.load("eparams_LK7.npy")
T_K7 = 10**TL[:,0] # [K]
L_K7 = 10**TL[:,1] # [L_sun]

# From colorbrewer
# colors = ["#a6cee3", "#b2df8a", "#33a02c", "#1f78b4"]

labels = ["AK Sco", "V4046 Sgr A", "V4046 Sgr B", "DQ Tau"]
colors = ["#90c0e0", "#b2df8a", "#33a02c", "#1f78b4"]


fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(3.5,3.0))

# for i, (color, label) in enumerate(zip(colors, labels)):
#     ax.annotate(label, xy=(0.05, 0.26 - 0.05 * i), xycoords="axes fraction", size=8, color=color)



def interp(T, L):
    interp = spline(T, L, k=3)
    Tfine = np.linspace(np.min(T), np.max(T))
    Lfine = interp(Tfine)
    return (Tfine, Lfine)

CC = ColorConverter()

lc = "0.25"

hist2d(ax, T_AKSCO, L_AKSCO, color=lc, pcolor=CC.to_rgb(colors[0]))
hist2d(ax, T_K5, L_K5, color=lc, pcolor=CC.to_rgb(colors[1]))
hist2d(ax, T_K7, L_K7, color=lc, pcolor=CC.to_rgb(colors[2]))
hist2d(ax, T_DQTAU, L_DQTAU, color=lc, pcolor=CC.to_rgb(colors[3]))
# hist2d(ax, T_DQTAU, L_DQTAU, color=colors[3])

ax.xaxis.set_major_formatter(FSF("%.0f"))
ax.xaxis.set_major_locator(MultipleLocator(1000))
ax.set_yscale('log')
ax.set_ylim(0.09, 5.5)
ax.set_xlim(6900, 2900)
ax.set_ylabel(r"$L_\ast$ [$L_\odot$]")
ax.set_xlabel(r"$T_\textrm{eff}$ [K]")


# Now, load all of the grids we need

# Plot Dartmouth

grid_D = DartmouthPMS(age_range=[0.1, 200], mass_range=[0.3, 1.4])
# Has 0.4, 0.45, 0.5,  0.8, 0.85, 0.9,  1.2, 1.25, 1.3

grid_P = PISA(age_range=[0.1, 200], mass_range=[0.3, 1.4])
# # Has 0.4, 0.45, 0.5,  0.8, 0.85, 0.9,  1.2, 1.3
#
grid_B = Baraffe15(age_range=[0.1, 200], mass_range=[0.3, 1.4])
# # Has 0.4, 0.5,  0.8, 0.9,  1.2, 1.3
#
grid_S = Seiss(age_range=[0.1, 200], mass_range=[0.3, 1.4])

# # Has 0.4, 0.5,  0.8, 0.9,  1.2, 1.3
#
# # The masses we actually want to plot are:
# # AK Sco: 1.25 -> 1.2
# # DQ Tau: 0.45 or 0.5 -> 0.5
# # V4046: 0.9 and 0.85 -> 0.9

grids = [grid_D, grid_P, grid_B, grid_S]

npoints = 100
ages = np.linspace(0.5, 50., num=npoints) # [Myr]
tt = np.empty(npoints)
ll = np.empty(npoints)

for grid in grids:
    grid.load()
    grid.setup_interpolator()

lw = 0.8

for grid in grids:
    for i,age in enumerate(ages):
        p = np.array([age, 0.45])
        tt[i] = grid.interp_T_smooth(p)
        ll[i] = grid.interp_ll_smooth(p)

    ax.plot(tt, 10**ll, "w", lw=lw+0.1)
    ax.plot(tt, 10**ll, colors[3], lw=lw)

for grid in grids:
    for i,age in enumerate(ages):
        p = np.array([age, 0.85])
        tt[i] = grid.interp_T_smooth(p)
        ll[i] = grid.interp_ll_smooth(p)

    ax.plot(tt, 10**ll, "w", lw=lw+0.1)
    ax.plot(tt, 10**ll, colors[2], lw=lw)


for grid in grids:
    for i,age in enumerate(ages):
        p = np.array([age, 0.9])
        tt[i] = grid.interp_T_smooth(p)
        ll[i] = grid.interp_ll_smooth(p)

    ax.plot(tt, 10**ll, "w", lw=lw+0.1)
    ax.plot(tt, 10**ll, colors[1], lw=lw)


for grid in grids:
    for i,age in enumerate(ages):
        p = np.array([age, 1.24])
        tt[i] = grid.interp_T_smooth(p)
        ll[i] = grid.interp_ll_smooth(p)

    ax.plot(tt, 10**ll, "w", lw=lw+0.1)
    ax.plot(tt, 10**ll, colors[0], lw=lw)


ax.annotate("AK Sco", xy=(0.2, 0.85), xycoords="axes fraction", size=8)
ax.annotate(r"$1.24\,M_\odot$", xy=(0.26, 0.78), xycoords="axes fraction", size=8)

ax.annotate("V4046 Sgr A", xy=(0.32, 0.47), xycoords="axes fraction", size=8)
ax.annotate(r"$0.9\,M_\odot$", xy=(0.47, 0.4), xycoords="axes fraction", size=8)

ax.annotate("V4046 Sgr B", xy=(0.32, 0.17), xycoords="axes fraction", size=8)
ax.annotate(r"$0.85\,M_\odot$", xy=(0.4, 0.22), xycoords="axes fraction", size=8)

ax.annotate("DQ Tau", xy=(0.79, 0.68), xycoords="axes fraction", size=8)
ax.annotate(r"$0.45\,M_\odot$", xy=(0.79, 0.61), xycoords="axes fraction", size=8)

fig.subplots_adjust(left=0.17, right=0.83, top=0.98, bottom=0.15)
fig.savefig("HR_full.svg")
fig.savefig("HR_full.pdf")
