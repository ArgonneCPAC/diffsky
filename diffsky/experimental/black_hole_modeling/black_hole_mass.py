import numpy as np
from jax import random as jran

__all__ = ("bh_mass_from_bulge_mass", "monte_carlo_black_hole_mass")
fixed_seed = 43


def bh_mass_from_bulge_mass(bulge_mass):
    """
    Kormendy & Ho (2013) fitting function for the Mbh--Mbulge power law relation.

    Parameters
    ----------
    bulge_mass : ndarray
        Stellar mass of the bulge in units of Msun

    Returns
    -------
    bh_mass : ndarray
        Mass of the black hole in units of Msun

    Notes
    -----
    See arXiv:1304.7762, section 6.1 for power-law exponent and normalization

    """
    prefactor = 0.0013
    alpha = 1.0
    bh_mass = prefactor * bulge_mass**alpha
    return bh_mass


def monte_carlo_black_hole_mass(bulge_mass, ran_key):
    """
    Monte Carlo realization of the Kormendy & Ho (2013) fitting function
    for the Mbh--Mbulge power law relation.

    Parameters
    ----------
    bulge_mass : ndarray
        Stellar mass of the bulge in units of Msun

    ran_key : jax.random.key
        Random number seed

    Returns
    -------
    bh_mass : ndarray
        Mass of the black hole in units of Msun

    Notes
    -----
    See arXiv:1304.7762, section 6.1 for power-law exponent and normalization

    """
    loc = np.log10(bh_mass_from_bulge_mass(bulge_mass))
    scale = 0.27
    lg_bhm = jran.normal(ran_key, shape=bulge_mass.shape) * scale + loc
    bh_mass = 10**lg_bhm

    return bh_mass
