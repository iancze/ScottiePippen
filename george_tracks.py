import matplotlib.pyplot as plt
import numpy as np

from grids import DartmouthPMS, PISA, Baraffe15, Seiss

# grid = DartmouthPMS(age_range=[1, 200], mass_range=[0.5, 2.5])
# grid = PISA(age_range=[1, 100], mass_range=[0.5, 2.0])
grid = Baraffe15(age_range=[1, 100], mass_range=[0.5, 1.4])
# grid = Seiss(age_range=[1, 100], mass_range=[0.5, 2.0])
grid.load()


# Try computing a GP

import george
from george.kernels import ExpSquaredKernel

# Set up the Gaussian process.
A_temp, A_radius, tau_age, tau_mass = np.array([3057., 1.0, 3.3, 0.19])

temp_kernel = A_temp**2 * ExpSquaredKernel(metric=[tau_age**2, tau_mass**2], ndim=2)
radius_kernel = A_radius**2 * ExpSquaredKernel(metric=[tau_age**2, tau_mass**2], ndim=2)

N = len(grid.points)

temp_gp = george.GP(temp_kernel)
radius_gp = george.GP(radius_kernel)

# print(temp_gp.kernel.pars)
# temp_gp.kernel[:] = np.log(np.array([A_temp, tau_age, tau_mass, nugget_temp])**2)
#
# print(temp_gp.kernel.pars)

# plt.imshow(temp_gp.get_matrix(grid.points), interpolation="none", origin="upper")
# plt.colorbar()
# plt.savefig("matrix.png")


# Pre-compute the factorization of the matrix.
temp_gp.compute(grid.points, yerr=50.*np.ones(N))
radius_gp.compute(grid.points, yerr=0.3*np.ones(N))
#
# # # Compute the log likelihood.
# print(temp_gp.lnlikelihood(grid.temps))
# print(radius_gp.lnlikelihood(grid.radii))

# import sys
# sys.exit()

fig, ax = plt.subplots(nrows=1, figsize=(8,8))

blue = True

for mass in np.unique(grid.masses):

    track = np.array([np.linspace(2., 20., num=50), mass * np.ones(50)]).T
    mu_T, cov = temp_gp.predict(grid.temps, track)
    # Ts = temp_gp.sample_conditional(grid.temps, track, 5)
    # std_T = np.sqrt(np.diag(cov))
    mu_R, cov = radius_gp.predict(grid.radii, track)
    # std_R = np.sqrt(np.diag(cov))
    # Rs = radius_gp.sample_conditional(grid.radii, track, 5)

    # if blue:
    #     col = "b."
    # else:
    #     col = "k."
    # blue = not blue
    #
    # for T,R in zip(Ts, Rs):
    #     ax.plot(T, R, col, ms=1)
    ax.plot(mu_T, mu_R)

# Label the tau0, M points
masses = np.unique(grid.masses)

for mass in masses:
    # Find all T, R that have this mass
    ind = (grid.masses == mass)
    tt = grid.temps[ind]
    rr = grid.radii[ind]
    ax.plot(tt, rr, "g-")
    ax.plot(tt, rr, "go")
    # ax.annotate("{:.1f}".format(mass), (tt[0], rr[0]), size=5)

# ax.plot(grid.temps, grid.radii, "go")
ax.set_xlim(8000, 3000)
fig.savefig("TR.png")

def lnprob(p):

    A_temp, A_radius, tau_age, tau_mass = p

    if np.any(p <= 0):
        return -np.inf

    # Setting the "vector", so we need to use the natural log: http://dan.iel.fm/george/current/user/kernels/#implementation
    temp_gp.kernel[:] = np.log(np.array([A_temp, tau_age, tau_mass])**2)
    radius_gp.kernel[:] = np.log(np.array([A_radius, tau_age, tau_mass])**2)

    lnp = temp_gp.lnlikelihood(grid.temps, quiet=True) + radius_gp.lnlikelihood(grid.radii, quiet=True)
    # print("P:", p, "lnp:", lnp)
    return lnp


def optimize():
    from emcee import EnsembleSampler
    import multiprocessing as mp

    ndim = 4
    nwalkers = 4 * ndim

    p0 = np.array([np.random.uniform(1000, 5000, nwalkers),
                    np.random.uniform(0.1, 1.0, nwalkers),
                    np.random.uniform(2, 12, nwalkers),
                    np.random.uniform(0.1, 1.5, nwalkers)]).T

    sampler = EnsembleSampler(nwalkers, ndim, lnprob, threads=mp.cpu_count())

    pos, prob, state = sampler.run_mcmc(p0, 1000)
    sampler.reset()
    print("Burned in")

    # actual run
    pos, prob, state = sampler.run_mcmc(pos, 1000)

    # Save the last position of the walkers
    np.save("walkers_emcee.npy", pos)
    np.save("eparams_emcee.npy", sampler.flatchain)


def main():
    # optimize()
    pass

if __name__=="__main__":
    main()
