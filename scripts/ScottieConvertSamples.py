#!/usr/bin/env python


import argparse

parser = argparse.ArgumentParser(description="Convert literature estimates into T, log10L form.")
parser.add_argument("--config", default="config.yaml", help="The config file specifying everything we need.")
args = parser.parse_args()

# Likelihood functions to convert posteriors in weird formats into posteriors on temp, log10 Luminosity for a single star (of a potential binary).

import yaml
import numpy as np
from emcee import EnsembleSampler

f = open(args.config)
config = yaml.load(f)
f.close()

# Now, we need 4 possible lnprob functions, that take into account each of these possibilities.

ndim = 2
nwalkers = 10 * ndim


if "logT" in config:
    logT, sigmalogT = config["logT"]

    if "L" in config:
        L, sigmaL = config["L"]

        mu = np.array([logT, L])
        Sigma = np.array([[sigmalogT**2, 0.0], [0.0, sigmaL**2]])

        p0 = np.array([np.random.uniform(10**(logT - sigmalogT), 10**(logT + sigmalogT), nwalkers),
                        np.random.uniform(np.log10(L - sigmaL), np.log10(L + sigmaL), nwalkers)]).T

        def lnprob(p):

            # Convert p to logT, L
            T, lL = p

            if T <= 0:
                return -np.inf

            logT = np.log10(T)
            L = 10**lL
            x = np.array([logT, L])

            R = x - mu # Residual vector
            invSigma = np.linalg.inv(Sigma)
            s, logdet = np.linalg.slogdet(Sigma)

            lnp = -0.5 * (R.dot(invSigma).dot(R) + logdet + 2 * np.log(2 * np.pi))
            return lnp

    else:
        lL, sigmalL = config["lL"]

        mu = np.array([logT, lL])
        Sigma = np.array([[sigmalogT**2, 0.0], [0.0, sigmalL**2]])

        p0 = np.array([np.random.uniform(10**(logT - sigmalogT), 10**(logT + sigmalogT), nwalkers),
                        np.random.uniform(lL - sigmalL, lL + sigmalL, nwalkers)]).T

        def lnprob(p):

            # Convert p to logT
            T, lL = p

            if T <= 0:
                return -np.inf

            logT = np.log10(T)
            x = np.array([logT, lL])

            R = x - mu # Residual vector
            invSigma = np.linalg.inv(Sigma)
            s, logdet = np.linalg.slogdet(Sigma)

            lnp = -0.5 * (R.dot(invSigma).dot(R) + logdet + 2 * np.log(2 * np.pi))
            return lnp

else:
    T, sigmaT = config["Teff"]

    if "L" in config:
        L, sigmaL = config["L"]

        # print("Using Teff, L values.")

        mu = np.array([T, L])
        # print("mu", mu)
        Sigma = np.array([[sigmaT**2, 0.0], [0.0, sigmaL**2]])

        p0 = np.array([np.random.uniform(T - sigmaT, T + sigmaT, nwalkers),
                        np.random.uniform(np.log10(L - sigmaL), np.log10(L + sigmaL), nwalkers)]).T

        # print("p0", p0)

        def lnprob(p):

            # print("mu", mu)
            # print("p", p)
            # Convert p to L
            T, lL = p

            # To prevent weird behavior from happening
            if lL < -4.0:
                return -np.inf

            L = 10**lL

            x = np.array([T, L])
            # print("x", x)

            if np.any(np.isinf(x)):
                return -np.inf

            R = x - mu # Residual vector
            # print("R: ", R)

            invSigma = np.linalg.inv(Sigma)
            s, logdet = np.linalg.slogdet(Sigma)

            lnp = -0.5 * (R.dot(invSigma).dot(R) + logdet + 2 * np.log(2 * np.pi))
            # print("lnp", lnp)
            # print()
            return lnp


    else:
        lL, sigmalL = config["lL"]

        mu = np.array([T, lL])
        Sigma = np.array([[sigmaT**2, 0.0], [0.0, sigmalL**2]])

        p0 = np.array([np.random.uniform(T - sigmaT, T + sigmaT, nwalkers),
                        np.random.uniform(lL - sigmalL, lL + sigmalL, nwalkers)]).T

        def lnprob(p):
            x = p
            R = x - mu # Residual vector
            invSigma = np.linalg.inv(Sigma)
            s, logdet = np.linalg.slogdet(Sigma)

            lnp = -0.5 * (R.dot(invSigma).dot(R) + logdet + 2 * np.log(2 * np.pi))
            return lnp



sampler = EnsembleSampler(nwalkers, ndim, lnprob)

pos, prob, state = sampler.run_mcmc(p0, 10000)
sampler.reset()
print("Burned in")

# actual run
pos, prob, state = sampler.run_mcmc(pos, 20000)

# Save the flatchain of samples
np.save(config["TlL_samples"], sampler.flatchain)



# Need option to say we only want some fraction of the Luminosity determined for the system

# To generate samples for our L contour.
# def lnpL(p):
#
#     # Determined using SED modeling for DQ Tau
#     # mu = np.array([3.5855, -0.395])
#     # Sigma = np.array([[0.0237**2, 0.0], [0.0, 0.164**2]])
#
#     # Determined using SED modeling for V4046 Sgr, K5
#     # mu = np.array([3.6385, -0.4560])
#     # Sigma = np.array([[0.0233**2, 0.0], [0.0, 0.109**2]])
#
#     # Determined using SED modeling for V4046 Sgr, K7
#     mu = np.array([3.6085, -.6021])
#     Sigma = np.array([[0.0219**2, 0.0], [0.0, 0.121**2]])
#
#
#     R = p - mu # Residual vector
#     invSigma = np.linalg.inv(Sigma)
#     s, logdet = np.linalg.slogdet(Sigma)
#
#     lnp = -0.5 * (R.dot(invSigma).dot(R) + logdet + 2 * np.log(2 * np.pi))
#     return lnp

# def lnprob_TL(p):
#
#     # p = np.array([age, mass])
#
#     # print("Sample p", p)
#     age, mass = p
#     if age < 0.0 or mass < 0.0:
#         return -np.inf
#
#
#     # Using smooth interpolators, determine temperature and log luminosity from age and mass
#     temp = grid.interp_T_smooth(p) # [K]
#     ll = grid.interp_ll_smooth(p) # Log10(L/L_sun) for a single star
#
#     # print("temp", temp)
#     # print("ll", ll)
#
#     # OK To return NaN for grid search, but not for MCMC sampling
#     if np.isnan(temp) or np.isnan(ll):
#         # return np.nan
#         return -np.inf
#
#     # Covariance matrix determined from SED modeling estimate
#     x = np.array([temp, ll])
#
#     # Determined using SED modeling from AK Sco
#     mu = np.array([6.36225305e+03, 7.75636071e-01 - np.log10(2)]) # Divide luminosity in half
#     Sigma = np.array([[2.40799517e+04, 6.53674282e+00],
#     [6.53674282e+00, 3.31446535e-03]])
#
#     #
#     # mu = np.array([3.5855, -0.395])
#     # Sigma = np.array([[0.0237**2, 0.0], [0.0, 0.164**2]])
#
#     R = x - mu # Residual vector
#     invSigma = np.linalg.inv(Sigma)
#     s, logdet = np.linalg.slogdet(Sigma)
#
#     lnp = -0.5 * (R.dot(invSigma).dot(R) + logdet + 2 * np.log(2 * np.pi))
#     return lnp
