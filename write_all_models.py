# Write all of the mass tracks to a single CSV file.

import numpy as np
import csv

from grids import DartmouthPMS, PISA, Baraffe15, Seiss

pc = 3.086e18 # [cm]
R_sun = 6.96e10 # [cm]
L_sun = 3.9e33 # [erg/s]
M_sun = 1.99e33 # [g]
sigma_k = 5.67051e-5 # [erg cm-2 K-4 s-1] Stefan-Boltzman constant
G = 6.67259e-8 # [cm^3 /g /s^2] Gravitational constant

params = ["model", "mass", "teff", "L", "age"]

f = open("tracks.csv", "w")
writer = csv.writer(f)

# write the header
writer.writerow(params)

model_names = ["Dartmouth", "PISA", "Baraffe", "Siess"]
models = [DartmouthPMS(age_range=[1, 100], mass_range=[0.1, 2.0]),
            PISA(age_range=[1, 100], mass_range=[0.1, 2.0]),
            Baraffe15(age_range=[1, 100], mass_range=[0.1, 1.4]),
            Seiss(age_range=[1, 100], mass_range=[0.1, 2.0])]

# Figure out which masses we have for all models
masses = [] #
for grid in models:
    grid.load()
    masses.append(set(["{:.2f}".format(mass) for mass in np.unique(grid.masses)]))

# Find the intersection of all of these sets
common_masses = masses[0].intersection(*masses[1:])
print("Intersection", common_masses)

last_temp = 0.0
last_L = 0.0
#For each model, write these properties.
for name, grid in zip(model_names, models):
    for mass, temp, radius, age in zip(grid.masses, grid.temps, grid.radii, grid.ages):
        # Only write the mass to the file if it is common to all models.
        if "{:.2f}".format(mass) in common_masses:
            L = 4 * np.pi * sigma_k * temp**4 * (radius * R_sun)**2 / L_sun # [L_sun]
            if temp != last_temp and L != last_L:
                writer.writerow([name, mass, temp, L, age])

            last_temp = temp
            last_L = L


f.close()
