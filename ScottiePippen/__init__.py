__version__ = '0.1'
__all__ = ["grids"]

# Figure out the basestring for these files
import os
data_dir = os.path.dirname(os.path.abspath(__file__)) + "/data/"

# Constants to be used
pc = 3.086e18 # [cm]
R_sun = 6.96e10 # [cm]
L_sun = 3.9e33 # [erg/s]
M_sun = 1.99e33 # [g]
sigma_k = 5.67051e-5 # [erg cm-2 K-4 s-1] Stefan-Boltzman constant
G = 6.67259e-8 # [cm^3 /g /s^2] Gravitational constant
