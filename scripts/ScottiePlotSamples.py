#!/usr/bin/env python

import argparse

parser = argparse.ArgumentParser(description="Measure statistics across multiple chains.")
parser.add_argument("--burn", type=int, default=0, help="How many samples to discard from the beginning of the chain for burn in.")
parser.add_argument("--config", default="config.yaml", help="The config file specifying everything we need.")
parser.add_argument("--no-tri", action="store_true", help="If true, skip plotting the triangle plots for speed's sake.")
args = parser.parse_args()

import yaml
import matplotlib.pyplot as plt
import numpy as np
import triangle

f = open(args.config)
config = yaml.load(f)
f.close()

def plot_walkers(chain, fname):

    nwalkers, niter, ndim = chain.shape

    # Plot the walkers
    fig, ax = plt.subplots(nrows=ndim, ncols=1, figsize=(10, 1.5 * ndim))

    iterations = np.arange(niter)

    for i in range(ndim):
        for j in range(nwalkers):
            ax[i].plot(iterations, chain[j, :, i], lw=0.2, color="k")


    ax[-1].set_xlabel("Iteration")

    fig.savefig(fname)


def plot_triangle(flatchain, fname):
    '''
    Make a triangle plot
    '''

    labels = [r"$\tau$ [Myr]", r"$M$ [$M_\odot$]"]
    figure = triangle.corner(flatchain, quantiles=[0.16, 0.5, 0.84],
        plot_contours=True, plot_datapoints=False, labels=labels, show_titles=True)
    figure.savefig(fname)


for grid_name in config["grids"]:

    chain = np.load(config["outfile"].format(grid_name))

    # Truncate burn in from chain
    chain = chain[:, args.burn:, :]

    nwalkers, niter, ndim = chain.shape

    nsamples = nwalkers * niter
    # Flatchain is made after the walkers have been burned
    flatchain = np.reshape(chain, (nsamples, ndim))

    # Save the flatchain
    np.save("flatchain_{}.npy".format(grid_name), flatchain)

    plot_walkers(chain, "walkers_{}.png".format(grid_name))

    if not args.no_tri:
        plot_triangle(flatchain, "triangle_{}.png".format(grid_name))
