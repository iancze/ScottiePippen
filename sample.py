# Take the model grids, which provide discrete sets of (temp, R, M_star, and Age) and resample all of these to uniform grids.

import numpy as np
from emcee import EnsembleSampler

from grids import DartmouthPMS, PISA, Baraffe15, Seiss

# grid = DartmouthPMS(age_range=[0.1, 100], mass_range=[0.1, 1.5])
# grid = PISA(age_range=[0.1, 100], mass_range=[0.1, 1.5])
# grid = Baraffe15(age_range=[0.1, 100], mass_range=[0.1, 1.4])
grid = Seiss(age_range=[0.1, 100], mass_range=[0.1, 1.4])
grid.load()

def lnprob_TR(p):

    # p = np.array([age, mass])

    # Using the linear interpolators, determine temperature and radius from age and mass
    temp = grid.interp_temp(p)
    radius = grid.interp_radius(p)

    # OK To return NaN for grid search, but not for MCMC sampling
    if np.isnan(temp) or np.isnan(radius):
        # return np.nan
        return -np.inf

    # Covariance matrix determined from SED modeling estimate
    # temp, radius
    x = np.concatenate([temp, radius])

    # Determined from old AV
    # mu = np.array([6385., 1.503])
    # Sigma = np.array([[2.342e+04, 2.5067], [2.5067, 5.452e-03]])

    # Determined using unred, Rv = 4.3
    mu = np.array([6367., 1.433])
    Sigma = np.array([[2.408e+04, -0.0543], [-0.0543, 4.172e-03]])


    R = x - mu # Residual vector
    invSigma = np.linalg.inv(Sigma)
    s, logdet = np.linalg.slogdet(Sigma)

    lnp = -0.5 * (R.dot(invSigma).dot(R) + logdet + 2 * np.log(2 * np.pi))
    return lnp

def lnprob_TL(p):

    # p = np.array([age, mass])

    # Using linear interpolators, determine temperature and luminosity from age and mass
    temp = grid.interp_temp(p)
    lum = grid.interp_L(p) #[L_sun]

    # OK To return NaN for grid search, but not for MCMC sampling
    if np.isnan(temp) or np.isnan(lum):
        # return np.nan
        return -np.inf

    # Covariance matrix determined from SED modeling estimate
    # log10(temp), log10(L)
    x = np.concatenate([np.log10(temp), lum])

    # Determined using SED modeling
    mu = np.array([3.5855, -0.395])
    Sigma = np.array([[0.0237**2, 0.0], [0.0, 0.164**2]])

    R = x - mu # Residual vector
    invSigma = np.linalg.inv(Sigma)
    s, logdet = np.linalg.slogdet(Sigma)

    lnp = -0.5 * (R.dot(invSigma).dot(R) + logdet + 2 * np.log(2 * np.pi))
    return lnp


def eval_lnp_TR():
    num = 100
    aa = np.linspace(5, 30, num=num)
    mm = np.linspace(1.1, 1.5, num=num)
    pp = cartesian([aa, mm]) # List of [[x_0, y_0], [x_1, y_1], ...]

    lnp = np.empty((num, num))

    for i in range(num):
        for j in range(num):
            ll = lnprob_TR(np.array([aa[i], mm[j]]))
            # print(ll)
            lnp[j,i] = ll

    # Normalize lnp
    P = np.e**lnp
    P /= np.nansum(P)

    fig, ax = plt.subplots(nrows=1, figsize=(4,4))
    ext = [np.min(aa), np.max(aa), np.min(mm), np.max(mm)]
    img = ax.imshow(P, origin="lower", extent=ext, interpolation="none", aspect="auto")

    # ax.plot(ages, masses, "k.", alpha=0.5, ms=1.0)
    ax.set_xlabel(r"$\tau$ [Myr]")
    ax.set_ylabel(r"$M$ [$M_\odot$]")
    cb = fig.colorbar(img ) #, format="%.1f")
    cb.set_label(r"$P$")

    fig.savefig("P.png")

def sample_lnp():

    ndim = 2
    nwalkers = 10 * ndim

    p0 = np.array([np.random.uniform(1, 10, nwalkers),
                    np.random.uniform(0.3, 0.7, nwalkers)]).T

    sampler = EnsembleSampler(nwalkers, ndim, lnprob_TL)

    pos, prob, state = sampler.run_mcmc(p0, 10000)
    sampler.reset()
    print("Burned in")

    # actual run
    pos, prob, state = sampler.run_mcmc(pos, 20000)

    # Save the flatchain of samples
    np.save("eparams_emcee.npy", sampler.flatchain)

# To generate samples for our L contour.
def lnpL(p):

    # Determined using SED modeling for DQ Tau
    # mu = np.array([3.5855, -0.395])
    # Sigma = np.array([[0.0237**2, 0.0], [0.0, 0.164**2]])

    # Determined using SED modeling for V4046 Sgr, K5
    # mu = np.array([3.6385, -0.4560])
    # Sigma = np.array([[0.0233**2, 0.0], [0.0, 0.109**2]])

    # Determined using SED modeling for V4046 Sgr, K7
    mu = np.array([3.6085, -.6021])
    Sigma = np.array([[0.0219**2, 0.0], [0.0, 0.121**2]])


    R = p - mu # Residual vector
    invSigma = np.linalg.inv(Sigma)
    s, logdet = np.linalg.slogdet(Sigma)

    lnp = -0.5 * (R.dot(invSigma).dot(R) + logdet + 2 * np.log(2 * np.pi))
    return lnp

def sample_TL():

    ndim = 2
    nwalkers = 10 * ndim

    p0 = np.array([np.random.uniform(3.4, 3.8, nwalkers),
                    np.random.uniform(-0.5, -0.1, nwalkers)]).T

    sampler = EnsembleSampler(nwalkers, ndim, lnpL)

    pos, prob, state = sampler.run_mcmc(p0, 10000)
    sampler.reset()
    print("Burned in")

    # actual run
    pos, prob, state = sampler.run_mcmc(pos, 20000)

    # Save the flatchain of samples
    np.save("eparams_LK7.npy", sampler.flatchain)


def main():
    # sample_lnp()
    sample_TL()

if __name__=="__main__":
    main()
