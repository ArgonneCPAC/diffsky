""" """

import numpy as np
from dsps.cosmology import flat_wcdm


def density_threshold(cosmo_params, redshift, mdef):
    """Calculate halo density threshold vs redshift

    Parameters
    ----------
    cosmo_params : namedtuple
        ('Om0', 'w0', 'wa', 'h')

    redshift : float or array

    mdef : string
        e.g., '200m' or '2500c'

    Returns
    -------
    rho_threshold : float or array
        Density threshold in units of [Msun/kpc**3]
        (n.b., not [Msun/kpc**3/h**2])

    """
    rho_crit = flat_wcdm.rho_crit(redshift, *cosmo_params)  # [Msun/kpc**3]

    if mdef[-1] == "c":
        delta = int(mdef[:-1])
        rho_threshold = rho_crit * delta
    elif mdef[-1] == "m":
        delta = int(mdef[:-1])
        rho_m = flat_wcdm._Om(redshift, *cosmo_params[:-1]) * rho_crit
        rho_threshold = rho_m * delta

    return rho_threshold


def halo_mass_to_halo_radius(mass, cosmo_params, redshift, mdef):
    """
    Calculate halo radius from halo mass

    Parameters
    ----------
    mass : float or array
        Units of Msun (not Msun/h)

    cosmo_params : namedtuple
        ('Om0', 'w0', 'wa', 'h')

    redshift : float or array

    mdef : string
        e.g., '200m' or '2500c'

    Returns
    -------
    radius : float or array
        Units of kpc (not kpc/h)

    """
    rho = density_threshold(cosmo_params, redshift, mdef)
    radius = (mass * 3.0 / 4.0 / np.pi / rho) ** (1.0 / 3.0)
    return radius
