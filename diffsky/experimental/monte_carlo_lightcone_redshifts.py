""""""

from functools import partial

import jax.numpy as jnp
from dsps.cosmology import flat_wcdm
from jax import jit as jjit
from jax import random, vmap

_CD = (0, None, None, None, None)
comoving_distance = jjit(vmap(flat_wcdm.comoving_distance_to_z, in_axes=_CD))


@partial(jjit, static_argnames=["n_pts"])
def mc_lightcone_redshift(ran_key, n_pts, z_min, z_max, cosmo_params, n_table=1_000):
    """Generate a realization of redshifts in a lightcone spanning the input z-range

    Parameters
    ----------
    ran_key : jax.random

    n_pts : int
        Number of points to generate

    z_min : float

    z_max : float

    cosmo_params : namedtuple
        dsps.cosmology.flat_wcdm cosmo_params = (Om0, w0, wa, h)

    n_table : int, optional
        Number of points in the lookup table used to numerically invert the cdf

    Returns
    -------
    mc_redshifts : ndarray, shape (n_pts, )

    """
    z_table = jnp.linspace(z_min, z_max, n_table)

    prefactor = 4.0 * jnp.pi / 3.0
    vol_com = prefactor * comoving_distance(z_table, *cosmo_params) ** 3

    weights = vol_com / vol_com.sum()
    cdf = jnp.cumsum(weights)

    uran = random.uniform(ran_key, shape=(n_pts,))
    mc_redshifts = jnp.interp(uran, cdf, z_table)

    return mc_redshifts
