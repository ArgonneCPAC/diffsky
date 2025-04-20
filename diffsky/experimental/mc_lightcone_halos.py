""" """

from collections import namedtuple
from functools import partial

import numpy as np
from diffmah.diffmah_kernels import _log_mah_kern
from diffmah.diffmahpop_kernels.bimod_censat_params import DEFAULT_DIFFMAHPOP_PARAMS
from diffmah.diffmahpop_kernels.mc_bimod_cens import mc_cenpop
from diffstar.utils import cumulative_mstar_formed_galpop
from diffstarpop import mc_diffstarpop_cen_tpeak as mcdct
from diffstarpop import param_utils as dpu
from diffstarpop.defaults import DEFAULT_DIFFSTARPOP_PARAMS
from dsps.constants import T_TABLE_MIN
from dsps.cosmology import flat_wcdm
from dsps.sed.stellar_age_weights import calc_age_weights_from_sfh_table
from jax import config, grad
from jax import jit as jjit
from jax import numpy as jnp
from jax import random as jran
from jax import vmap

from ..mass_functions import mc_hosts

config.update("jax_enable_x64", True)

_AGEPOP = (None, 0, None, 0)
calc_age_weights_from_sfh_table_vmap = jjit(
    vmap(calc_age_weights_from_sfh_table, in_axes=_AGEPOP)
)

FULL_SKY_AREA = (4 * jnp.pi) * (180 / jnp.pi) ** 2

interp_vmap = jjit(vmap(jnp.interp, in_axes=(0, None, 0)))

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


def mc_lightcone_host_halo_mass_function(
    ran_key,
    lgmp_min,
    z_min,
    z_max,
    sky_area_degsq,
    cosmo_params=flat_wcdm.PLANCK15,
    hmf_params=mc_hosts.DEFAULT_HMF_PARAMS,
    n_grid=2_000,
):
    """Generate a Monte Carlo realization of a lightcone of host halo mass and redshift

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


def mc_lightcone_host_halo_diffmah(
    ran_key,
    lgmp_min,
    z_min,
    z_max,
    sky_area_degsq,
    cosmo_params=flat_wcdm.PLANCK15,
    hmf_params=mc_hosts.DEFAULT_HMF_PARAMS,
    diffmahpop_params=DEFAULT_DIFFMAHPOP_PARAMS,
    n_grid=2_000,
):
    """Generate mass assembly histories for host halos sampled from a lightcone

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
    cenpop : namedtuple
        z_obs, logmp_obs, mah_params, logmp0 = cenpop

        z_obs : narray, shape (n_halos, )
            Lightcone redshift

        logmp_obs : narray, shape (n_halos, )
            Halo mass at the lightcone redshift

        mah_params : namedtuple of diffmah params
            Each tuple entry is an ndarray with shape (n_halos, )

        logmp0 : narray, shape (n_halos, )
            Halo mass at z=0

    """

    lc_hmf_key, mah_key = jran.split(ran_key, 2)
    z_halopop, logmp_halopop = mc_lightcone_host_halo_mass_function(
        lc_hmf_key,
        lgmp_min,
        z_min,
        z_max,
        sky_area_degsq,
        cosmo_params=cosmo_params,
        hmf_params=hmf_params,
        n_grid=n_grid,
    )
    t_obs_halopop = flat_wcdm.age_at_z(z_halopop, *cosmo_params)
    t_0 = flat_wcdm.age_at_z0(*cosmo_params)
    lgt0 = jnp.log10(t_0)

    tarr = np.array((10**lgt0,))
    args = (diffmahpop_params, tarr, logmp_halopop, t_obs_halopop, mah_key, lgt0)
    halopop = mc_cenpop(*args)  # mah_params, dmhdt, log_mah
    logmp0_halopop = halopop.log_mah[:, 0]

    logmp_obs_halopop = _log_mah_kern(halopop.mah_params, t_obs_halopop, lgt0)

    colnames = ("z_obs", "t_obs", "logmp_obs", "mah_params", "logmp0")
    DiffmahCenPop = namedtuple("DiffmahCenPop", colnames)
    cenpop = DiffmahCenPop._make(
        (
            z_halopop,
            t_obs_halopop,
            logmp_obs_halopop,
            halopop.mah_params,
            logmp0_halopop,
        )
    )

    return cenpop


def mc_lightcone_diffstar_cens(
    ran_key,
    lgmp_min,
    z_min,
    z_max,
    sky_area_degsq,
    cosmo_params=flat_wcdm.PLANCK15,
    hmf_params=mc_hosts.DEFAULT_HMF_PARAMS,
    diffmahpop_params=DEFAULT_DIFFMAHPOP_PARAMS,
    diffstarpop_params=DEFAULT_DIFFSTARPOP_PARAMS,
    n_grid=2_000,
    n_t_table=100,
):
    """
    Generate halo and galaxy assembly histories for host halos sampled from a lightcone

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
    cenpop : namedtuple
        z_obs, logmp_obs, mah_params, logmp0 = cenpop

        z_obs : narray, shape (n_halos, )
            Lightcone redshift

        logmp_obs : narray, shape (n_halos, )
            Halo mass at the lightcone redshift

        mah_params : namedtuple of diffmah params
            Each tuple entry is an ndarray with shape (n_halos, )

        logmp0 : narray, shape (n_halos, )
            log10 of halo mass at z=0

        logsm_obs : narray, shape (n_halos, )
            log10(Mstar) at the time of observation

        logssfr_obs : narray, shape (n_halos, )
            log10(SFR/Mstar) at the time of observation

        sfh_params : namedtuple
            Diffstar params for every galaxy

        sfh_table : narray, shape (n_halos, n_times)
            Star formation rate in Msun/yr

        t_table : narray, shape (n_times, )

        diffstarpop_data : namedtuple
            ancillary diffstarpop data such as frac_q

    """
    cenpop = mc_lightcone_host_halo_diffmah(
        ran_key,
        lgmp_min,
        z_min,
        z_max,
        sky_area_degsq,
        cosmo_params=cosmo_params,
        hmf_params=hmf_params,
        diffmahpop_params=diffmahpop_params,
        n_grid=n_grid,
    )

    t0 = flat_wcdm.age_at_z0(*cosmo_params)

    t_table = jnp.linspace(T_TABLE_MIN, t0, n_t_table)
    args = (diffstarpop_params, cenpop.mah_params, cenpop.logmp0, ran_key, t_table)

    ddp_fields = "sfh_params_ms", "sfh_params_q", "sfh_ms", "sfh_q", "frac_q", "mc_is_q"
    DiffstarPopData = namedtuple("DiffstarPopData", ddp_fields)
    diffstarpop_data = DiffstarPopData(*mcdct.mc_diffstar_sfh_galpop_cen(*args))

    sfh_table = jnp.where(
        diffstarpop_data.mc_is_q.reshape((-1, 1)),
        diffstarpop_data.sfh_q,
        diffstarpop_data.sfh_ms,
    )
    sfh_params = dpu.mc_select_diffstar_params(
        diffstarpop_data.sfh_params_q,
        diffstarpop_data.sfh_params_ms,
        diffstarpop_data.mc_is_q,
    )

    logsmh_table = np.log10(cumulative_mstar_formed_galpop(t_table, sfh_table))
    logsm_obs = interp_vmap(cenpop.t_obs, t_table, logsmh_table)
    logsfr_obs = interp_vmap(cenpop.t_obs, t_table, np.log10(sfh_table))
    logssfr_obs = logsfr_obs - logsm_obs

    fields = (
        *cenpop._fields,
        "logsm_obs",
        "logssfr_obs",
        "sfh_params",
        "sfh_table",
        "t_table",
        "diffstarpop_data",
    )
    values = (
        *cenpop,
        logsm_obs,
        logssfr_obs,
        sfh_params,
        sfh_table,
        t_table,
        diffstarpop_data,
    )
    DiffstarCenPop = namedtuple("DiffstarCenPop", fields)
    sfh_cenpop = DiffstarCenPop(*values)

    return sfh_cenpop


def mc_lightcone_diffstar_stellar_ages_cens(
    ran_key,
    lgmp_min,
    z_min,
    z_max,
    sky_area_degsq,
    cosmo_params=flat_wcdm.PLANCK15,
    hmf_params=mc_hosts.DEFAULT_HMF_PARAMS,
    diffmahpop_params=DEFAULT_DIFFMAHPOP_PARAMS,
    diffstarpop_params=DEFAULT_DIFFSTARPOP_PARAMS,
    ssp_lg_age_gyr=np.linspace(5.0, 10.25, 90) - 9.0,
    n_grid=2_000,
    n_t_table=100,
):
    cenpop = mc_lightcone_diffstar_cens(
        ran_key,
        lgmp_min,
        z_min,
        z_max,
        sky_area_degsq,
        cosmo_params=cosmo_params,
        hmf_params=hmf_params,
        diffmahpop_params=diffmahpop_params,
        n_grid=n_grid,
    )
    age_weights_galpop = calc_age_weights_from_sfh_table_vmap(
        cenpop.t_table, cenpop.sfh_table, ssp_lg_age_gyr, cenpop.t_obs
    )
    fields = (*cenpop._fields, "age_weights", "ssp_lg_age_gyr")
    values = (*cenpop, age_weights_galpop, ssp_lg_age_gyr)
    DiffstarCenPop = namedtuple("DiffstarCenPop", fields)
    cenpop = DiffstarCenPop(*values)

    return cenpop
