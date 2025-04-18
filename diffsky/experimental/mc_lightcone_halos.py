""" """

from functools import partial

from dsps.cosmology import flat_wcdm
from jax import config, grad
from jax import jit as jjit
from jax import numpy as jnp
from jax import random as jran
from jax import vmap

from ..mass_functions import mc_hosts

FULL_SKY_AREA = (4 * jnp.pi) * (180 / jnp.pi) ** 2


config.update("jax_enable_x64", True)


_G = (0, None, None, 0, 0)
mc_logmp_vmap = jjit(vmap(mc_hosts._mc_host_halos_singlez_kern, in_axes=_G))

_Z = (0, None, None, None, None)
dist_com_grad_kern = jjit(
    vmap(grad(flat_wcdm.comoving_distance_to_z, argnums=0), in_axes=_Z)
)


@jjit
def _spherical_shell_comoving_volume(z_grid, cosmo_params):
    """Comoving volume of a spherical shell with width dR"""

    # Compute comoving distance to each grid point
    r_grid = flat_wcdm.comoving_distance(z_grid, *cosmo_params)

    # Compute dR = (dR/dz)*dz
    d_r_grid_dz = dist_com_grad_kern(z_grid, *cosmo_params)
    d_z_grid = z_grid[1] - z_grid[0]
    d_r_grid = d_r_grid_dz * d_z_grid

    # vol_shell_grid = 4π*R*R*dR
    vol_shell_grid = 4 * jnp.pi * r_grid * r_grid * d_r_grid
    return vol_shell_grid


@partial(jjit, static_argnames=["npts"])
def mc_lightcone_redshift(
    ran_key, npts, z_min, z_max, cosmo_params=flat_wcdm.PLANCK15, n_table=1000
):
    """Generate a realization of redshifts in a lightcone spanning the input z-range

    Parameters
    ----------
    ran_key : jax.random

    n_pts : int
        Number of points to generate

    z_min : float

    z_max : float

    cosmo_params : namedtuple
        dsps.cosmology.flat_wcdm cosmology
        cosmo_params = (Om0, w0, wa, h)

    n_table : int, optional
        Number of points in the lookup table used to numerically invert the cdf

    Returns
    -------
    mc_redshifts : ndarray, shape (n_pts, )
        Redshifts distributed randomly within the lightcone volume

    """
    # Set up a uniform grid in redshift
    z_grid = jnp.linspace(z_min, z_max, n_table)

    # Compute the comoving volume of a thin shell at each grid point
    vol_shell_grid = _spherical_shell_comoving_volume(z_grid, cosmo_params)

    weights_grid = vol_shell_grid / vol_shell_grid.sum()
    cdf_grid = jnp.cumsum(weights_grid)

    # Assign redshift via inverse transformation sampling of the shell volume CDF
    uran_z = jran.uniform(ran_key, minval=0, maxval=1, shape=(npts,))
    mc_redshifts = jnp.interp(uran_z, cdf_grid, z_grid)

    return mc_redshifts


def mc_lightcone_host_halos(
    ran_key,
    lgmp_min,
    z_min,
    z_max,
    sky_area_degsq,
    cosmo_params=flat_wcdm.PLANCK15,
    hmf_params=mc_hosts.DEFAULT_HMF_PARAMS,
    n_grid=2_000,
):
    """Generate a Monte Carlo realization of a lightcone of host halos

    Parameters
    ----------
    ran_key : jax.random.key

    lgmp_min : float
        Minimum halo mass

    z_min, z_max : float

    sky_area_degsq : float
        Sky area in units of deg^2

    cosmo_params : namedtuple
        dsps.cosmology.flat_wcdm cosmology
        cosmo_params = (Om0, w0, wa, h)

    Returns
    -------
    z_halopop : ndarray, shape (n_halos, )
        Redshifts distributed randomly within the lightcone volume

    logmp_halopop : ndarray, shape (n_halos, )
        Halo masses derived by Monte Carlo sampling the halo mass function
        at the appropriate redshift for each point

    """

    # Three randoms: one for Nhalos, one for halo mass, one for redshift
    halo_counts_key, m_key, z_key = jran.split(ran_key, 3)

    # Set up a uniform grid in redshift
    z_grid = jnp.linspace(z_min, z_max, n_grid)

    # Compute the comoving volume of a thin shell at each grid point
    fsky = sky_area_degsq / FULL_SKY_AREA
    vol_shell_grid = fsky * _spherical_shell_comoving_volume(z_grid, cosmo_params)

    # At each grid point, compute <Nhalos> for the shell volume
    mean_nhalos_grid = mc_hosts._compute_nhalos_tot(
        hmf_params, lgmp_min, z_grid, vol_shell_grid
    )

    # At each grid point, compute a Poisson realization of <Nhalos>
    nhalos_grid = jran.poisson(halo_counts_key, mean_nhalos_grid)
    nhalos_tot = nhalos_grid.sum()

    # Compute the CDF of the volume
    weights_grid = vol_shell_grid / vol_shell_grid.sum()
    cdf_grid = jnp.cumsum(weights_grid)

    # Assign redshift via inverse transformation sampling of the shell volume CDF
    uran_z = jran.uniform(z_key, minval=0, maxval=1, shape=(nhalos_tot,))
    z_halopop = jnp.interp(uran_z, cdf_grid, z_grid)

    # Randoms used in inverse transformation sampling halo mass
    uran_m = jran.uniform(m_key, minval=0, maxval=1, shape=(nhalos_tot,))

    # Compute the effective volume of each halo according to its redshift
    vol_galpop = jnp.interp(z_halopop, z_grid, vol_shell_grid)

    # Draw a halo mass from the HMF at the particular redshift of each halo
    logmp_halopop = mc_logmp_vmap(uran_m, hmf_params, lgmp_min, z_halopop, vol_galpop)

    return z_halopop, logmp_halopop
