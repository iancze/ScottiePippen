# Write all of the mass tracks to a single CSV file.

import numpy as np
import csv
from scipy.interpolate import interp1d

from grids import PISA

pc = 3.086e18 # [cm]
R_sun = 6.96e10 # [cm]
L_sun = 3.9e33 # [erg/s]
M_sun = 1.99e33 # [g]
sigma_k = 5.67051e-5 # [erg cm-2 K-4 s-1] Stefan-Boltzman constant
G = 6.67259e-8 # [cm^3 /g /s^2] Gravitational constant

params = ["mass", "teff", "L", "radius", "age"]

f = open("PISA_tracks.csv", "w")
writer = csv.writer(f)

# write the header
writer.writerow(params)

grid = PISA(age_range=[0.0, 6000], mass_range=[0.1, 2.0])
grid.load()

# truncate at minimum radius
def simple_write():
    # For each mass, write these properties
    for mass in np.array([0.2, 0.6, 1.0, 1.4]):
        ind = np.isclose(mass, grid.masses)
        mm = grid.masses[ind]
        tt = grid.temps[ind]
        rr = grid.radii[ind]
        LL = 4 * np.pi * sigma_k * tt**4 * (rr * R_sun)**2 / L_sun # [L_sun]
        aa = grid.ages[ind]
        min_r = np.min(rr)
        for row in zip(mm, tt, LL, rr, aa):
            if row[3] == min_r:
                break
            else:
                writer.writerow(row)

    f.close()


# Use this to interpolate linearly
def linear_interpolate():
    npoints = 399
    # Ages to interpolate to
    ages = np.linspace(1, 200, num=npoints)

    #For each model, write these properties.
    for mass in np.unique(grid.masses):
        # Find all points that have this mass
        ind = grid.masses == mass
        rr = grid.radii[ind]
        tt = grid.temps[ind]
        aa = grid.ages[ind]

        # Create 1D interpolator for age vs. temp
        interp_temp = interp1d(aa, tt, kind="linear")
        temps = interp_temp(ages)

        # Create 1D interpolator for age vs. radius
        interp_radius = interp1d(aa, rr, kind="linear")
        radii = interp_radius(ages)

        masses = mass * np.ones((npoints))

        for mass, temp, radius, age in zip(masses, temps, radii, ages):
            L = 4 * np.pi * sigma_k * temp**4 * (radius * R_sun)**2 / L_sun # [L_sun]
            writer.writerow([mass, temp, L, age])

    f.close()

def main():
    simple_write()

if __name__=="__main__":
    main()
