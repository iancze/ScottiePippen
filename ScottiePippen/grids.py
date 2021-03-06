import numpy as np
from scipy.interpolate import interp1d
from astropy.io import ascii

import ScottiePippen as SP

class IndexInterpolator:
    '''
    Object to return fractional distance between grid points of a single grid variable.

    :param parameter_list: list of parameter values
    :type parameter_list: 1-D list
    '''

    def __init__(self, parameter_list):
        self.parameter_list = np.unique(parameter_list)
        self.index_interpolator = interp1d(self.parameter_list, np.arange(len(self.parameter_list)), kind='linear')

    def __call__(self, value):
        '''
        Evaluate the interpolator at a parameter.

        :param value:
        :type value: float
        :raises C.InterpolationError: if *value* is out of bounds.

        :returns: ((low_val, high_val), (frac_low, frac_high)), the lower and higher bounding points in the grid
        and the fractional distance (0 - 1) between them and the value.
        '''
        try:
            index = self.index_interpolator(value)
        except ValueError as e:
            print("Requested value {} is out of bounds. {}".format(value, e))
            raise
        high = np.ceil(index)
        low = np.floor(index)
        frac_index = index - low
        return ((self.parameter_list[low], self.parameter_list[high]), ((1 - frac_index), frac_index))


class Base:
    def __init__(self, name, basefmt, age_range, mass_range):
        self.name = name
        self.basefmt = basefmt
        self.age_range = age_range
        self.mass_range = mass_range

        self.load()
        self.setup_interpolator()

    def load(self):
        print("Load function must be defined by a subclass.")
        raise NotImplementedError

        # Will load the various grids from file.

        # Then, form arrays of age, mass, radius, and temperature

        # Put these arrays into the interpolators. Interpolators always take an (Age, Mass) pair,
        # in that order.

    def setup_interpolator(self):
        '''
        Once all of the data files have been loaded into memory using grid.load() (implemented by the subclass), then call this function to actually setup the two stage interpolation necessary for smooth contours.

        This works this way because all grids provide stars at set mass locations, and then as the age evolves, predict different stellar properties, such as Teff and (log) Luminosity.

        This means if we want to query (T,L) for a given (tau, M), we shouldn't just do a bi-linear interpolation between (T,L) points like we were doing before.

        Instead, we should find the two mass points in the grid that bracket M, call them M_low and M_high. Then, interpolate T and L for (tau, M_low) and (tau, M_high).

        Finally, then interpolate T and L from the (T,L)'s corresponding to (tau, M_low) and (tau, M_high) for M.
        '''

        # Identify the unique masses
        umasses = np.unique(self.masses)

        # Create an index interpolator for umasses.
        self.mass_interp = IndexInterpolator(umasses)

        self.T_interpolators = {}
        self.lL_interpolators = {}

        for mass in umasses:

            ind = np.isclose(self.masses, mass)

            # Find all the ages that correspond to this mass
            ages = self.ages[ind]

            # Sort them in increasing order
            ind2 = np.argsort(ages)

            # Find all the temps that correspond to this mass
            # Find all the ll that correspond to this mass

            # Sort all of these according to increasing ages
            temps = self.temps[ind][ind2]
            lLs = self.lums[ind][ind2]

            # Fit a linear interpolator for t(age) and ll(age), store these in an array
            self.T_interpolators[mass] = interp1d(ages, temps)
            self.lL_interpolators[mass] = interp1d(ages, lLs)


    def interp_T(self, p):
        '''p is [age, mass] '''

        age, mass = p

        try:
            # First identify the upper and lower masses
            (low_val, high_val), (frac_low, frac_high) = self.mass_interp(mass)

            T_high = self.T_interpolators[high_val](age)
            T_low = self.T_interpolators[low_val](age)
        except ValueError:
            # This means we must be out of range of the grid.
            return np.nan

        # Weighted average estimates for age based on how close.
        T = frac_low * T_low + frac_high * T_high

        return T

    def interp_lL(self, p):
        '''p is [age, mass] '''

        age, mass = p

        try:
            # First identify the upper and lower masses
            (low_val, high_val), (frac_low, frac_high) = self.mass_interp(mass)

            lL_high = self.lL_interpolators[high_val](age)
            lL_low = self.lL_interpolators[low_val](age)
        except ValueError:
            # This means we must be out of range of the grid.
            return np.nan

        # Weighted average estimates for age based on how close.
        lL = frac_low * lL_low + frac_high * lL_high

        return lL


class DartmouthPMS(Base):
    def __init__(self, age_range, mass_range):
        super().__init__(name="DartmouthPMS", basefmt=SP.data_dir + "Dartmouth/PMS/fehp00afep0/m{:0>3.0f}fehp00afep0.jc2mass", age_range=age_range, mass_range=mass_range)

    def load(self):
        # Dartmouth masses
        masses = np.concatenate((np.arange(0.1, 1.8, 0.05), np.arange(1.8, 3., 0.1), np.arange(3., 5., 0.2)))

        ind = (masses >= self.mass_range[0]) & (masses <= self.mass_range[1])
        masses = masses[ind]

        # Go through all of the files, read the relevant properties, and then concatenate these
        # into 4, 1-D arrays
        mass_list = []
        age_list = []
        temp_list = []
        radius_list = []
        lum_list = []

        for mass in masses:

            fname = self.basefmt.format(100 * mass)

            data = ascii.read(fname, names=["age", "LTeff", "logg", "LL", "U", "B", "V", "R", "I", "J", "H", "Ks"])

            age = 1e-6 * data["age"] # [Myr]
            ind = (age >= self.age_range[0]) & (age <= self.age_range[1])

            age = age[ind]
            temp = 10**data["LTeff"][ind] # [K]
            radius = np.sqrt(G * mass * M_sun / (10**data["logg"][ind])) / R_sun # [R_sun]

            LL = data["LL"][ind] # [L_sun]

            mass_list.append(mass * np.ones(np.sum(ind)))
            age_list.append(age)
            temp_list.append(temp)
            radius_list.append(radius)
            lum_list.append(LL)

        self.masses = np.concatenate(mass_list)
        self.ages = np.concatenate(age_list)
        self.temps = np.concatenate(temp_list)
        self.radii = np.concatenate(radius_list)
        self.lums = np.concatenate(lum_list)

        self.points = np.array([self.ages, self.masses]).T


class PISA(Base):
    def __init__(self, age_range, mass_range):
        super().__init__(name="PISA", basefmt=SP.data_dir + "PISA/Z0.02000_Y0.2880_XD2E5_ML1.68_AS05/TRK_M{:.2f}_Z0.02000_Y0.2880_XD2E5_ML1.68_AS05.DAT", age_range=age_range, mass_range=mass_range)

    def load(self):
        masses = np.concatenate((np.arange(0.2, 1., 0.05), np.arange(1.0, 2., 0.1), np.arange(2., 4., 0.2), np.arange(4.0, 7.1, 0.5)))

        ind = (masses >= self.mass_range[0]) & (masses <= self.mass_range[1])
        masses = masses[ind]

        # Go through all of the files, read the relevant properties, and then concatenate these
        # into 4, 1-D arrays
        mass_list = []
        age_list = []
        temp_list = []
        radius_list = []
        lum_list = []

        for mass in masses:

            fname = self.basefmt.format(mass)

            data = ascii.read(fname, names=["NMD", "L_age", "Xc", "LL", "LTeff", "LTc", "LOG RHOc", "M-CC", "L-PP", "L-CNO", "L-GRA"])

            age = 1e-6 * 10**data["L_age"] # [Myr]
            ind = (age >= self.age_range[0]) & (age <= self.age_range[1])

            age = age[ind]
            temp = 10**data["LTeff"][ind] # [K]
            L = 10**data["LL"][ind] * L_sun # [ergs/s]

            radius = np.sqrt(L / (4 * np.pi * sigma_k * temp**4)) / R_sun # [R_sun]

            mass_list.append(mass * np.ones(np.sum(ind)))
            age_list.append(age)
            temp_list.append(temp)
            radius_list.append(radius)
            lum_list.append(data["LL"][ind]) # log10(L_sun)

        self.masses = np.concatenate(mass_list)
        self.ages = np.concatenate(age_list)
        self.temps = np.concatenate(temp_list)
        self.radii = np.concatenate(radius_list)
        self.lums = np.concatenate(lum_list)

        self.points = np.array([self.ages, self.masses]).T


class Baraffe15(Base):
    def __init__(self, age_range, mass_range):
        super().__init__(name="Baraffe15", basefmt=SP.data_dir + "Baraffe15/{:.4f}.dat", age_range=age_range, mass_range=mass_range)

    def load(self):
        # In Myr
        ages = np.array([0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 8.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0, 80.0, 100.0, 120.0, 200.0])

        # Grid filenames are in Gyr
        ind = (ages >= self.age_range[0]) & (ages <= self.age_range[1])
        ages = ages[ind]

        # Go through all of the files, read the relevant properties, and then concatenate these
        # into 4, 1-D arrays
        mass_list = []
        age_list = []
        temp_list = []
        radius_list = []
        lum_list = []

        for age in ages:

            fname = self.basefmt.format(1e-3 * age)

            data = ascii.read(fname, names=["mass", "Teff", "L", "g", "radius", "Li/Li0", "U*", "G", "R", "I", "12k.I", "ZPR", "Z", "Y", "J", "H", "Ks", "CH4_ON", "CH4_OFF"], comment="\!")

            mass = data["mass"] # [M_sun]
            ind = (mass >= self.mass_range[0]) & (mass <= self.mass_range[1])

            mass = mass[ind] # [M_sun]
            temp = data["Teff"][ind] # [K]
            radius = data["radius"][ind] # R_sun
            LL = data["L"][ind] # L_sun

            mass_list.append(mass)
            age_list.append(age * np.ones(np.sum(ind)))
            temp_list.append(temp)
            radius_list.append(radius)
            lum_list.append(LL)

        self.masses = np.concatenate(mass_list)
        self.ages = np.concatenate(age_list)
        self.temps = np.concatenate(temp_list)
        self.radii = np.concatenate(radius_list)
        self.lums = np.concatenate(lum_list)

        self.points = np.array([self.ages, self.masses]).T

class Seiss(Base):
    def __init__(self, age_range, mass_range):
        super().__init__(name="Seiss", basefmt=SP.data_dir + "Seiss/m{:.1f}z02.hrd", age_range=age_range, mass_range=mass_range)

    def load(self):
        # In Myr
        masses = np.concatenate((np.arange(0.3, 2.0, 0.1), np.array([2.0, 2.2, 2.5, 2.7, 3.0, 3.5, 4.0, 5.0, 6.0])))

        ind = (masses >= self.mass_range[0]) & (masses <= self.mass_range[1])
        masses = masses[ind]

        # Go through all of the files, read the relevant properties, and then concatenate these
        # into 4, 1-D arrays
        mass_list = []
        age_list = []
        temp_list = []
        radius_list = []
        lum_list = []

        for mass in masses:

            fname = self.basefmt.format(mass
            # model, L (Lo), Reff (Ro), Teff, log g, age (yr), phase, Mbol, R* (Ro), rho_eff, M (Mo)
            data = ascii.read(fname, names=["model", "phase", "L", "Mbol", "Reff", "radius", "Teff", "rho_eff", "log g", "M (Mo)", "age"])

            age = 1e-6 * data["age"] # [Myr]
            ind = (age >= self.age_range[0]) & (age <= self.age_range[1])

            age = age[ind]

            L = data["L"][ind] * L_sun # [ergs/s]
            radius = data["radius"][ind] # [R_sun]

            # Because T_eff is computed at tau=2/3, we'll try recomputing at the surface, using Rstar
            temp = (L / (4 * np.pi * (radius * R_sun)**2 * sigma_k))**0.25 # [K]

            mass_list.append(mass * np.ones(np.sum(ind)))
            age_list.append(age)
            temp_list.append(temp)
            radius_list.append(radius)
            lum_list.append(np.log10(data["L"][ind]))


        self.masses = np.concatenate(mass_list)
        self.ages = np.concatenate(age_list)
        self.temps = np.concatenate(temp_list)
        self.radii = np.concatenate(radius_list)
        self.lums = np.concatenate(lum_list)

        self.points = np.array([self.ages, self.masses]).T


model_dict = {"DartmouthPMS":DartmouthPMS, "PISA":PISA, "Baraffe15": Baraffe15, "Seiss":Seiss}


def cartesian(arrays, out=None):
    """
    Generate a cartesian product of input arrays.

    http://stackoverflow.com/questions/1208118/using-numpy-to-build-an-array-of-all-combinations-of-two-arrays

    Parameters
    ----------
    arrays : list of array-like
        1-D arrays to form the cartesian product of.
    out : ndarray
        Array to place the cartesian product in.

    Returns
    -------
    out : ndarray
        2-D array of shape (M, len(arrays)) containing cartesian products
        formed of input arrays.

    Examples
    --------
    >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6],
           [1, 4, 7],
           [1, 5, 6],
           [1, 5, 7],
           [2, 4, 6],
           [2, 4, 7],
           [2, 5, 6],
           [2, 5, 7],
           [3, 4, 6],
           [3, 4, 7],
           [3, 5, 6],
           [3, 5, 7]])

    """

    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    m = n / arrays[0].size
    out[:,0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m,1:])
        for j in range(1, arrays[0].size):
            out[j*m:(j+1)*m,1:] = out[0:m,1:]
    return out

def main():
    pass


if __name__=="__main__":
    main()
