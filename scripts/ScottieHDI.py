#!/usr/bin/env python

import argparse

parser = argparse.ArgumentParser(description="Measure statistics across multiple chains.")
parser.add_argument("--burn", type=int, default=0, help="How many samples to discard from the beginning of the chain for burn in.")
parser.add_argument("--config", default="config.yaml", help="The config file specifying everything we need.")
args = parser.parse_args()

import yaml

f = open(args.config)
config = yaml.load(f)
f.close()

import numpy as np
from scipy.optimize import fmin
import matplotlib.pyplot as plt

# Determine the highest density interval, a region that spans some percentage of the interval (e.g., 68%, or 95%), such that every value inside the interval has a higher probability than those outside of it.


# Let's simply do this numerically
def hdi(samples, bins=80):

    hist, bin_edges = np.histogram(samples, bins=bins, density=True)
    # convert bin_edges into bin centroids
    bin_centers = bin_edges[:-1] + np.diff(bin_edges)

    # Find the maximum from the histogram
    # indmax = np.argmax(hist)
    # binmax = bin_centers[indmax]

    dbin = bin_edges[1] - bin_edges[0]

    nbins = len(bin_centers)

    # Now, sort all of the bin heights in decreasing order
    indsort = np.argsort(hist)[::-1]
    histsort = hist[indsort]

    binmax = bin_centers[indsort][0]

    prob = histsort[0] * dbin
    i = 0
    while prob < 0.683:
        i += 1
        prob = np.sum(histsort[:i] * dbin)

    level = hist[i]

    indHDI = hist > level
    binHDI = bin_centers[indHDI]

    print("Ranges: low: {}, max: {}, high: {}".format(binHDI[0], binmax, binHDI[-1]))
    print("Diffs: max:{}, low:{}, high:{}, dbin:{}".format(binmax, binmax - binHDI[0], binHDI[-1]-binmax, dbin))
    print()



for grid_name in config["grids"]:

    print(grid_name)
    chain = np.load(config["outfile"].format(grid_name))

    # Truncate burn in from chain
    chain = chain[:, args.burn:, :]

    nwalkers, niter, ndim = chain.shape

    nsamples = nwalkers * niter
    # Flatchain is made after the walkers have been burned
    flatchain = np.reshape(chain, (nsamples, ndim))

    age, mass0, mass1 = flatchain.T

    q = mass1/mass0

    total = mass0 + mass1

    flatchain = np.array([age, total, mass0, mass1, q]).T

    for i in range(flatchain.shape[1]):
        hdi(flatchain[:,i])
