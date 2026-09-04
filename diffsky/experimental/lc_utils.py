"""Utility functions for lightcone calculations"""

from functools import partial

from dsps.cosmology import flat_wcdm
from jax import grad
from jax import jit as jjit
from jax import numpy as jnp
from jax import random as jran
from jax import vmap

N_Z_GRID = 2_000
FULL_SKY_AREA = (4 * jnp.pi) * (180 / jnp.pi) ** 2

C_SPEED = 299792458.0  # m/s

_Z = (0, None, None, None, None)
d_Rcom_dz_func = jjit(
    vmap(grad(flat_wcdm.comoving_distance_to_z, argnums=0), in_axes=_Z)
)


@jjit
def _get_corrected_phi_bounds_for_last_journey_core_lc_7(phi_lo, phi_hi):
    """Fix bug in phi coordinate in LastJourney/core-lc-7 dataset"""
    phi_lo = jnp.mod(phi_lo + jnp.pi, 2 * jnp.pi)
    phi_hi = jnp.mod(phi_hi + jnp.pi, 2 * jnp.pi)
    return phi_lo, phi_hi


@partial(jjit, static_argnames=["npts"])
def mc_lightcone_random_ra_dec(ran_key, npts, ra_min, ra_max, dec_min, dec_max):
    """Generate random ra, dec in the input patch of sky

    Parameters
    ----------
    ran_key : jax.random.key
        JAX random seed

    npts : int
        Number of points to generate

    ra_min, ra_max : float
        min/max ra in degrees

    dec_min, dec_max : float
        min/max dec in degrees

    Returns
    -------
    ra, dec : arrays of shape (npts, )
        Random coords on the sphere within the input range

    """
    theta_min, theta_max, phi_min, phi_max = _get_theta_phi_minmax_from_ra_dec_minmax(
        ra_min, ra_max, dec_min, dec_max
    )

    theta, phi = mc_lightcone_random_theta_phi(
        ran_key, npts, theta_min, theta_max, phi_min, phi_max
    )
    ra, dec = _get_ra_dec_from_theta_phi(theta, phi)

    return ra, dec


@jjit
def _get_ra_dec_minmax_from_theta_phi_minmax(theta_min, theta_max, phi_min, phi_max):
    ra_min = jnp.rad2deg(phi_min)
    ra_max = jnp.rad2deg(phi_max)

    dec_min = 90.0 - jnp.rad2deg(theta_max)
    dec_max = 90.0 - jnp.rad2deg(theta_min)

    return ra_min, ra_max, dec_min, dec_max


@jjit
def _get_theta_phi_minmax_from_ra_dec_minmax(ra_min, ra_max, dec_min, dec_max):
    phi_min = jnp.deg2rad(ra_min)
    phi_max = jnp.deg2rad(ra_max)

    theta_min = jnp.deg2rad(90.0 - dec_max)
    theta_max = jnp.deg2rad(90.0 - dec_min)

    return theta_min, theta_max, phi_min, phi_max


@partial(jjit, static_argnames=["npts"])
def mc_lightcone_random_theta_phi(
    ran_key, npts, theta_min, theta_max, phi_min, phi_max
):
    """Generate random theta, phi in the input patch of sky"""
    theta_key, phi_key = jran.split(ran_key, 2)

    # Uniform random 0<Φ<2π, accounting for possible wrapping by 2π
    delta_phi = phi_max - phi_min
    uran = jran.uniform(phi_key, shape=(npts,), minval=0.0, maxval=delta_phi)
    phi = jnp.mod(uran + phi_min, 2 * jnp.pi)

    # Sample uniformly in cos(θ)
    cos_lo = jnp.cos(theta_max)
    cos_hi = jnp.cos(theta_min)
    cos_theta = jran.uniform(theta_key, (npts,), minval=cos_lo, maxval=cos_hi)
    theta = jnp.arccos(cos_theta)

    return theta, phi


@jjit
def _get_ra_dec_from_theta_phi(theta, phi):
    """Change sky coordinates from {theta, phi} (radians) to {ra, dec} (degrees)

    Copied from diffsky.data_loaders.hacc_utils.lightcone_utils.py
    to avoid circular imports. This function should be deleted once lc_utils.py
    is migrated out of diffsky.experimental
    """
    return _get_lon_lat_from_theta_phi(theta, phi)


@jjit
def _get_lon_lat_from_theta_phi(theta, phi):
    """Copied from diffsky.data_loaders.hacc_utils.lightcone_utils.py
    to avoid circular imports. This function should be deleted once lc_utils.py
    is migrated out of diffsky.experimental
    """
    lon = jnp.degrees(phi)
    lat = 90.0 - jnp.degrees(theta)
    return lon, lat


@partial(jjit, static_argnames=["npts"])
def mc_lightcone_random_redshift(
    ran_key, npts, z_min, z_max, cosmo_params, n_z_grid=N_Z_GRID
):
    """
    Monte Carlo generator of redshifts that randomly sample the comoving volume

    Parameters
    ----------
    ran_key : jax.random.key
        JAX random seed

    npts : int
        Number of points to generate

    z_min, z_max : float
        min/max of redshift in the lightcone

    cosmo_params : namedtuple
        dsps.cosmology.flat_wcdm cosmology
        cosmo_params = (Om0, w0, wa, h)

    Returns
    -------
    redshift : array of shape (npts, )

    Notes
    -----
    For deterministic behavior with precomputed uniform randoms spanning [0, 1],
    instead call the underlying kernel _mc_lightcone_random_redshifts_kern.

    """
    uran = jran.uniform(ran_key, minval=0.0, maxval=1.0, shape=(npts,))
    redshift = _mc_lightcone_random_redshifts_kern(
        uran, z_min, z_max, cosmo_params, n_z_grid
    )
    return redshift


@partial(jjit, static_argnames=["n_z_grid"])
def _mc_lightcone_random_redshifts_kern(uran, z_min, z_max, cosmo_params, n_z_grid):
    """Deterministic kernel to generate of redshifts that sample the comoving volume"""
    # Set up a uniform grid in redshift
    z_grid = jnp.linspace(z_min, z_max, n_z_grid)

    # Compute the comoving volume of a thin shell at each grid point
    vol_shell_grid_mpc = spherical_shell_comoving_volume(z_grid, cosmo_params)

    # Compute the CDF of the volume
    weights_grid = vol_shell_grid_mpc / vol_shell_grid_mpc.sum()
    cdf_grid = jnp.cumsum(weights_grid)

    # Assign redshift via inverse transformation sampling of the comoving volume CDF
    redshift = jnp.interp(uran, cdf_grid, z_grid)

    return redshift


@jjit
def spherical_shell_comoving_volume(z_grid, cosmo_params):
    """Comoving volume of a spherical shell with width ΔR"""

    # Compute comoving distance to each grid point
    r_grid = flat_wcdm.comoving_distance(z_grid, *cosmo_params)

    # Compute ΔR = (∂R/∂z)*Δz
    d_r_grid_dz = d_Rcom_dz_func(z_grid, *cosmo_params)
    d_z_grid = z_grid[1] - z_grid[0]
    d_r_grid = d_r_grid_dz * d_z_grid

    # vol_shell_grid = 4π*R*R*ΔR
    vol_shell_grid = 4 * jnp.pi * r_grid * r_grid * d_r_grid
    return vol_shell_grid


@jjit
def get_z_obs_from_z_true(z_true, v_pec_kms):
    """Calculate the observed redshift accounting for peculiar velocity

    Parameters
    ----------
    z_true : array
        True redshift

    v_pec_kms : array
        Peculiar velocity in units of km/s

    Returns
    -------
    z_obs : array
        Observed redshift, accounting for peculiar velocity

    Notes
    -----
    (1+z_obs) = (1+z_true) * (1+z_pec)

    1+z_pec = sqrt( (1+v/c) / (1-v/c) )

    """
    c_speed_kms = C_SPEED / 1000.0
    x = v_pec_kms / c_speed_kms
    num, denom = 1 - x, 1 + x
    z_true_plus1 = 1.0 + z_true
    zpec_plus1 = jnp.sqrt(num / denom)
    zobs_plus1 = z_true_plus1 * zpec_plus1
    z_obs = zobs_plus1 - 1.0
    return z_obs
