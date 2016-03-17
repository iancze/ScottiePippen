#!/usr/bin/env python

import argparse

parser = argparse.ArgumentParser(description="Given a set of MCMC samples of T, log L, use scipy.kde to approximate the density field.")
parser.add_argument("--config", default="config.yaml", help="The config file specifying everything we need.")
args = parser.parse_args()

import yaml

f = open(args.config)
config = yaml.load(f)
f.close()

# Take the model grids, which provide discrete sets of (temp, R, M_star, and Age) and resample all of these to uniform grids.

import numpy as np
from scipy.stats import gaussian_kde

# import multiprocessing as mp

from emcee import EnsembleSampler

ndim = 3
nwalkers = config["walkers_per_dim"] * ndim

age_low, age_high = config["age_guess"]
mass_low, mass_high = config["mass_guess"]

p0 = np.array([np.random.uniform(age_low, age_high, nwalkers),
            np.random.uniform(mass_low, mass_high, nwalkers),
            np.random.uniform(mass_low, mass_high, nwalkers)]).T


# Load the samples from the CWD
samples = np.load(config["TlL_samples"]) # TlL.npy

cutoff = int(config["cutoff"]) #8000

# Otherwise we probably have too many
if len(samples) > cutoff:
    samples = samples[:cutoff].T
else:
    samples = samples.T

# temp0, temp1, lL0, lL1 = samples
kernel = gaussian_kde(samples)


def lnprob(p, grid):

    age, mass0, mass1 = p
    if np.any(p < 0.0):
        return -np.inf

    p0 = np.array([age, mass0])
    p1 = np.array([age, mass1])

    # Using smooth interpolators, determine temperature and log luminosity from age and mass
    temp0 = grid.interp_T(p0) # [K]
    lL0 = grid.interp_lL(p0) # Log10(L/L_sun) for a single star

    temp1 = grid.interp_T(p1) # [K]
    lL1 = grid.interp_lL(p1) # Log10(L/L_sun) for a single star


    # If we sampled outside of the grid, one of these values is NaN, so convert this to -Inf for MCMC sampling
    val = np.array([temp0, temp1, lL0, lL1])

    if np.any(np.isnan(val)):
        # return np.nan
        return -np.inf

    # Use the KDE kernel to evaluate how well this point fits based upon our temp, lL posterior.
    lnp = kernel.logpdf(val)
    return lnp


from ScottiePippen.grids import model_dict

for grid_name in config["grids"]:
    print(grid_name)
    grid = model_dict[grid_name](**config[grid_name])

    sampler = EnsembleSampler(nwalkers, ndim, lnprob, args=[grid])

    pos, prob, state = sampler.run_mcmc(p0, config["samples"])

    # Save the actual chain of samples
    np.save(config["outfile"].format(grid_name), sampler.chain)
    np.save("tauMass_lnp_{}".format(grid_name), sampler.lnprobability)
