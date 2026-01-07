# flake8: noqa: E402
"""Functions to generate Monte Carlo realizations of galaxies on a lightcone"""

from jax import config

config.update("jax_enable_x64", True)
import warnings
from collections import namedtuple

import numpy as np
from diffmah import logmh_at_t_obs
from diffmah.diffmah_kernels import _log_mah_kern
from diffmah.diffmahpop_kernels.bimod_censat_params import DEFAULT_DIFFMAHPOP_PARAMS
from diffmah.diffmahpop_kernels.mc_bimod_cens import mc_cenpop
from diffstar.defaults import FB
from diffstar.diffstarpop import mc_diffstar_sfh_galpop
from diffstar.diffstarpop import param_utils as dpu
from diffstar.diffstarpop.defaults import DEFAULT_DIFFSTARPOP_PARAMS
from diffstar.utils import cumulative_mstar_formed_galpop
from dsps.constants import T_TABLE_MIN
from dsps.cosmology import flat_wcdm
from dsps.sed import metallicity_weights as zmetw
from dsps.sed.stellar_age_weights import calc_age_weights_from_sfh_table
from jax import jit as jjit
from jax import numpy as jnp
from jax import random as jran
from jax import vmap
from jax.scipy.interpolate import RegularGridInterpolator
from scipy.stats import qmc

from ..burstpop import diffqburstpop_mono
from ..mass_functions import hmf_model, mc_hosts
from ..phot_utils import get_wave_eff_table
from . import precompute_ssp_phot as psspp
from .lc_utils import spherical_shell_comoving_volume

try:
    from mpi4py import MPI

    COMM = MPI.COMM_WORLD
except ImportError:
    MPI = COMM = None

N_HMF_GRID = 2_000
N_SFH_TABLE = 100
DEFAULT_LOGMP_CUTOFF = 10.0
DEFAULT_LOGMP_HIMASS_CUTOFF = 14.5


_AGEPOP = (None, 0, None, 0)
calc_age_weights_from_sfh_table_vmap = jjit(
    vmap(calc_age_weights_from_sfh_table, in_axes=_AGEPOP)
)

FULL_SKY_AREA = (4 * jnp.pi) * (180 / jnp.pi) ** 2

LSST_PHOTKEYS = ("lsst_u*", "lsst_g*", "lsst_r*", "lsst_i*", "lsst_z*", "lsst_y*")

interp_vmap = jjit(vmap(jnp.interp, in_axes=(0, None, 0)))

_A = (None, None, 0)
_B = (None, None, 1)
interp_vmap2 = jjit(vmap(jnp.interp, in_axes=_B, out_axes=1))


_G = (0, None, None, 0, None)
mc_logmp_vmap = jjit(vmap(mc_hosts._mc_host_halos_singlez_kern, in_axes=_G))

_LCA = (None, 0, None, None)
_compute_nhalos_tot_vmap = jjit(vmap(mc_hosts._compute_nhalos_tot, in_axes=_LCA))


# gal_lgmet, gal_lgmet_scatter, ssp_lgmet
_M = (0, None, None)
_calc_lgmet_weights_galpop = jjit(
    vmap(zmetw.calc_lgmet_weights_from_lognormal_mdf, in_axes=_M)
)

# diffburstpop_params, logsm, logssfr, ssp_lg_age_gyr, smooth_age_weights
_B = (None, 0, 0, None, 0)
_calc_bursty_age_weights_vmap = jjit(
    vmap(
        diffqburstpop_mono.calc_bursty_age_weights_from_diffburstpop_params, in_axes=_B
    )
)


def estimate_nhalos_in_lightcone(
    lgmp_min,
    z_min,
    z_max,
    sky_area_degsq,
    *,
    cosmo_params=flat_wcdm.PLANCK15,
    hmf_params=mc_hosts.DEFAULT_HMF_PARAMS,
    n_hmf_grid=N_HMF_GRID,
    lgmp_max=mc_hosts.LGMH_MAX,
):
    # Set up a uniform grid in redshift
    z_grid = jnp.linspace(z_min, z_max, n_hmf_grid)

    # Compute the comoving volume of a thin shell at each grid point
    fsky = sky_area_degsq / FULL_SKY_AREA
    vol_shell_grid_mpc = fsky * spherical_shell_comoving_volume(z_grid, cosmo_params)

    # At each grid point, compute <Nhalos> for the shell volume
    mean_nhalos_grid = mc_hosts._compute_nhalos_tot(
        hmf_params, lgmp_min, z_grid, vol_shell_grid_mpc
    )
    mean_nhalos_lgmp_max = mc_hosts._compute_nhalos_tot(
        hmf_params, lgmp_max, z_grid, vol_shell_grid_mpc
    )
    mean_nhalos_grid = mean_nhalos_grid - mean_nhalos_lgmp_max
    nhalos_tot = mean_nhalos_grid.sum()
    return nhalos_tot


def mc_lightcone_host_halo_mass_function(
    ran_key,
    lgmp_min,
    z_min,
    z_max,
    sky_area_degsq,
    cosmo_params=flat_wcdm.PLANCK15,
    hmf_params=mc_hosts.DEFAULT_HMF_PARAMS,
    n_hmf_grid=N_HMF_GRID,
    lgmp_max=mc_hosts.LGMH_MAX,
    nhalos_tot=None,
):
    """Generate a Monte Carlo realization of a lightcone of host halo mass and redshift

    Parameters
    ----------
    ran_key : jran.key

    lgmp_min : float
        Minimum halo mass in units of Msun (not Msun/h)

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

    Notes
    -----
    All mass quantities quoted in Msun (not Msun/h)

    """

    # Three randoms: one for Nhalos, one for halo mass, one for redshift
    halo_counts_key, m_key, z_key = jran.split(ran_key, 3)

    # Set up a uniform grid in redshift
    z_grid = jnp.linspace(z_min, z_max, n_hmf_grid)

    # Compute the comoving volume of a thin shell at each grid point
    fsky = sky_area_degsq / FULL_SKY_AREA
    vol_shell_grid_mpc = fsky * spherical_shell_comoving_volume(z_grid, cosmo_params)

    # At each grid point, compute <Nhalos> for the shell volume
    mean_nhalos_grid = mc_hosts._compute_nhalos_tot(
        hmf_params, lgmp_min, z_grid, vol_shell_grid_mpc
    )
    mean_nhalos_lgmp_max = mc_hosts._compute_nhalos_tot(
        hmf_params, lgmp_max, z_grid, vol_shell_grid_mpc
    )
    mean_nhalos_grid = mean_nhalos_grid - mean_nhalos_lgmp_max

    if nhalos_tot is None:
        # At each grid point, compute a Poisson realization of <Nhalos>
        nhalos_grid = jran.poisson(halo_counts_key, mean_nhalos_grid)
        nhalos_tot = nhalos_grid.sum()

    # Compute the CDF of the volume
    weights_grid = mean_nhalos_grid / mean_nhalos_grid.sum()
    cdf_grid = jnp.cumsum(weights_grid)

    # Assign redshift via inverse transformation sampling of the halo counts CDF
    uran_z = jran.uniform(z_key, minval=0, maxval=1, shape=(nhalos_tot,))
    z_halopop = jnp.interp(uran_z, cdf_grid, z_grid)

    # Randoms used in inverse transformation sampling halo mass
    uran_m = jran.uniform(m_key, minval=0, maxval=1, shape=(nhalos_tot,))

    # Draw a halo mass from the HMF at the particular redshift of each halo
    logmp_halopop = mc_logmp_vmap(uran_m, hmf_params, lgmp_min, z_halopop, lgmp_max)

    return z_halopop, logmp_halopop


@jjit
def pdf_weighted_lgmp_grid_singlez(hmf_params, lgmp_grid, redshift):
    weights_grid = hmf_model.predict_differential_hmf(hmf_params, lgmp_grid, redshift)
    weights_grid = weights_grid / weights_grid.sum()
    return weights_grid


_A = (None, None, 0)
pdf_weighted_lgmp_grid_vmap = jjit(vmap(pdf_weighted_lgmp_grid_singlez, in_axes=_A))


@jjit
def get_nhalo_weighted_lc_grid(
    lgmp_grid,
    z_grid,
    sky_area_degsq,
    hmf_params=mc_hosts.DEFAULT_HMF_PARAMS,
    cosmo_params=flat_wcdm.PLANCK15,
):
    """Compute the number of halos on the input grid of halo mass and redshift

    Parameters
    ----------
    lgmp_grid : array, shape (n_m, )
        Base-10 log halo mass in units of Msun (not Msun/h)

    z_grid : array, shape (n_z, )

    sky_area_degsq : float
        Sky area in units of deg^2

    cosmo_params : namedtuple
        dsps.cosmology.flat_wcdm cosmology
        cosmo_params = (Om0, w0, wa, h)

    Returns
    -------
    nhalo_weighted_lc_grid : array, shape (n_z, n_m)

    Notes
    -----
    All mass quantities quoted in Msun (not Msun/h)

    """
    # Compute the comoving volume of a thin shell at each grid point
    fsky = sky_area_degsq / FULL_SKY_AREA
    vol_shell_grid_mpc = fsky * spherical_shell_comoving_volume(z_grid, cosmo_params)

    # At each grid point, compute <Nhalos> for the shell volume
    mean_nhalos_lgmp_min = mc_hosts._compute_nhalos_tot(
        hmf_params, lgmp_grid[0], z_grid, vol_shell_grid_mpc
    )
    mean_nhalos_lgmp_max = mc_hosts._compute_nhalos_tot(
        hmf_params, lgmp_grid[-1], z_grid, vol_shell_grid_mpc
    )
    mean_nhalos_z_grid = mean_nhalos_lgmp_min - mean_nhalos_lgmp_max

    lgmp_weights = pdf_weighted_lgmp_grid_vmap(hmf_params, lgmp_grid, z_grid)

    n_z = z_grid.size
    nhalo_weighted_lc_grid = mean_nhalos_z_grid.reshape((n_z, 1)) * lgmp_weights

    return nhalo_weighted_lc_grid


def mc_lightcone_host_halo_diffmah(
    ran_key,
    lgmp_min,
    z_min,
    z_max,
    sky_area_degsq,
    cosmo_params=flat_wcdm.PLANCK15,
    hmf_params=mc_hosts.DEFAULT_HMF_PARAMS,
    diffmahpop_params=DEFAULT_DIFFMAHPOP_PARAMS,
    n_hmf_grid=N_HMF_GRID,
    logmp_cutoff=DEFAULT_LOGMP_CUTOFF,
    logmp_cutoff_himass=DEFAULT_LOGMP_HIMASS_CUTOFF,
    lgmp_max=mc_hosts.LGMH_MAX,
):
    """Generate halo MAHs for host halos sampled from a lightcone

    Parameters
    ----------
    ran_key : jran.key

    lgmp_min : float
        Minimum halo mass in units of Msun (not Msun/h)

    z_min, z_max : float

    sky_area_degsq : float
        Sky area in units of deg^2

    cosmo_params : namedtuple
        dsps.cosmology.flat_wcdm cosmology
        cosmo_params = (Om0, w0, wa, h)

    logmp_cutoff : float, optional
        Minimum halo mass for which DiffmahPop is used to generate MAHs.
        For logmp < logmp_cutoff, P(θ_MAH | logmp) = P(θ_MAH | logmp_cutoff)

    Returns
    -------
    cenpop : dict

        z_obs : narray, shape (n_halos, )
            Lightcone redshift

        logmp_obs : narray, shape (n_halos, )
            Halo mass at the lightcone redshift

        mah_params : namedtuple of diffmah params
            Each tuple entry is an ndarray with shape (n_halos, )

        logmp0 : narray, shape (n_halos, )
            Base-10 log of halo mass in units of Msun at z=0

    Notes
    -----
    All mass quantities quoted in Msun (not Msun/h)

    """

    lc_hmf_key, mah_key = jran.split(ran_key, 2)
    z_obs, logmp_obs_mf = mc_lightcone_host_halo_mass_function(
        lc_hmf_key,
        lgmp_min,
        z_min,
        z_max,
        sky_area_degsq,
        cosmo_params=cosmo_params,
        hmf_params=hmf_params,
        n_hmf_grid=n_hmf_grid,
        lgmp_max=lgmp_max,
    )
    t_obs = flat_wcdm.age_at_z(z_obs, *cosmo_params)
    t_0 = flat_wcdm.age_at_z0(*cosmo_params)
    lgt0 = jnp.log10(t_0)

    logmp_obs_mf_clipped = np.clip(logmp_obs_mf, logmp_cutoff, logmp_cutoff_himass)

    tarr = np.array((10**lgt0,))
    args = (diffmahpop_params, tarr, logmp_obs_mf_clipped, t_obs, mah_key, lgt0)
    mah_params_uncorrected = mc_cenpop(*args)[0]  # mah_params, dmhdt, log_mah

    logmp_obs_orig = _log_mah_kern(mah_params_uncorrected, t_obs, lgt0)
    delta_logmh_clip = logmp_obs_orig - logmp_obs_mf
    mah_params = mah_params_uncorrected._replace(
        logm0=mah_params_uncorrected.logm0 - delta_logmh_clip
    )

    logmp0 = _log_mah_kern(mah_params, 10**lgt0, lgt0)
    logmp_obs = _log_mah_kern(mah_params, t_obs, lgt0)

    fields = ("z_obs", "t_obs", "logmp_obs", "mah_params", "logmp0")
    values = (z_obs, t_obs, logmp_obs, mah_params, logmp0)
    cenpop_out = dict()
    for key, value in zip(fields, values):
        cenpop_out[key] = value

    return cenpop_out


def get_weighted_lightcone_grid_host_halo_diffmah(
    ran_key,
    lgmp_grid,
    z_grid,
    sky_area_degsq,
    cosmo_params=flat_wcdm.PLANCK15,
    hmf_params=mc_hosts.DEFAULT_HMF_PARAMS,
    diffmahpop_params=DEFAULT_DIFFMAHPOP_PARAMS,
    logmp_cutoff=DEFAULT_LOGMP_CUTOFF,
    logmp_cutoff_himass=DEFAULT_LOGMP_HIMASS_CUTOFF,
):
    """Compute the number of halos on the input grid of halo mass and redshift

    Parameters
    ----------
    ran_key : jran.key

    lgmp_grid : array, shape (n_m, )
        Grid of Base-10 log of halo mass in units of Msun (not Msun/h)

    z_grid : array, shape (n_z, )
        Grid of redshift

    sky_area_degsq : float
        Sky area in units of deg^2

    cosmo_params : namedtuple
        dsps.cosmology.flat_wcdm cosmology
        cosmo_params = (Om0, w0, wa, h)

    cosmo_params : namedtuple
        dsps.cosmology.flat_wcdm cosmology
        cosmo_params = (Om0, w0, wa, h)

    logmp_cutoff : float, optional
        Minimum halo mass for which DiffmahPop is used to generate MAHs.
        For logmp < logmp_cutoff, P(θ_MAH | logmp) = P(θ_MAH | logmp_cutoff)

    Returns
    -------
    cenpop : dict

        z_obs : narray, shape (n_z*n_m, )
            Lightcone redshift

        logmp_obs : narray, shape (n_z*n_m, )
            Base-10 log of halo mass in units of Msun at the lightcone redshift

        mah_params : namedtuple of diffmah params
            Each tuple entry is an ndarray with shape (n_z*n_m, )

        logmp0 : narray, shape (n_z*n_m, )
            Base-10 log of halo mass in units of Msun at z=0

        nhalos : array, shape (n_z*n_m, )
            Number of halos of this mass and redshift

    Notes
    -----
    All mass quantities quoted in Msun (not Msun/h)

    """
    nhalo_weighted_lc_grid = get_nhalo_weighted_lc_grid(
        lgmp_grid,
        z_grid,
        sky_area_degsq,
        hmf_params=hmf_params,
        cosmo_params=cosmo_params,
    )
    nhalo_weights = nhalo_weighted_lc_grid.flatten()
    z_obs = np.repeat(z_grid, lgmp_grid.size)
    logmp_obs_mf = np.tile(lgmp_grid, z_grid.size)

    t_obs = flat_wcdm.age_at_z(z_obs, *cosmo_params)
    t_0 = flat_wcdm.age_at_z0(*cosmo_params)
    lgt0 = jnp.log10(t_0)

    logmp_obs_mf_clipped = np.clip(logmp_obs_mf, logmp_cutoff, logmp_cutoff_himass)

    tarr = np.array((10**lgt0,))
    args = (diffmahpop_params, tarr, logmp_obs_mf_clipped, t_obs, ran_key, lgt0)
    mah_params_uncorrected = mc_cenpop(*args)[0]  # mah_params, dmhdt, log_mah

    logmp_obs_orig = _log_mah_kern(mah_params_uncorrected, t_obs, lgt0)
    delta_logmh_clip = logmp_obs_orig - logmp_obs_mf
    mah_params = mah_params_uncorrected._replace(
        logm0=mah_params_uncorrected.logm0 - delta_logmh_clip
    )

    logmp0 = _log_mah_kern(mah_params, 10**lgt0, lgt0)
    logmp_obs = _log_mah_kern(mah_params, t_obs, lgt0)

    fields = ("z_obs", "t_obs", "logmp_obs", "mah_params", "logmp0")
    values = (z_obs, t_obs, logmp_obs, mah_params, logmp0)
    cenpop_out = dict()
    for key, value in zip(fields, values):
        cenpop_out[key] = value
    cenpop_out["nhalos"] = nhalo_weights

    return cenpop_out


def mc_lightcone_diffstar_cens(
    ran_key,
    lgmp_min,
    z_min,
    z_max,
    sky_area_degsq,
    cosmo_params=flat_wcdm.PLANCK15,
    fb=FB,
    hmf_params=mc_hosts.DEFAULT_HMF_PARAMS,
    diffmahpop_params=DEFAULT_DIFFMAHPOP_PARAMS,
    diffstarpop_params=DEFAULT_DIFFSTARPOP_PARAMS,
    n_hmf_grid=N_HMF_GRID,
    n_sfh_table=N_SFH_TABLE,
    return_internal_quantities=False,
    logmp_cutoff=DEFAULT_LOGMP_CUTOFF,
    logmp_cutoff_himass=DEFAULT_LOGMP_HIMASS_CUTOFF,
):
    """
    Generate halo MAH and galaxy SFH for host halos sampled from a lightcone

    Parameters
    ----------
    ran_key : jran.key

    lgmp_min : float
        Base-10 log of minimum halo mass in units of Msun (not Msun/h)

    z_min, z_max : float

    sky_area_degsq : float
        Sky area in units of deg^2

    cosmo_params : namedtuple
        dsps.cosmology.flat_wcdm cosmology
        cosmo_params = (Om0, w0, wa, h)

    Returns
    -------
    cenpop : dict

        z_obs : narray, shape (n_halos, )
            Lightcone redshift

        logmp_obs : narray, shape (n_halos, )
            Halo mass at the lightcone redshift

        mah_params : namedtuple of diffmah params
            Each tuple entry is an ndarray with shape (n_halos, )

        logmp0 : narray, shape (n_halos, )
            Base-10 log of halo mass in units of Msun at z=0

        logsm_obs : narray, shape (n_halos, )
            log10(Mstar) at the time of observation

        logssfr_obs : narray, shape (n_halos, )
            log10(SFR/Mstar) at the time of observation

        sfh_params : namedtuple
            Diffstar params for every galaxy

        sfh_table : narray, shape (n_halos, n_times)
            Star formation rate in Msun/yr

        t_table : narray, shape (n_times, )

        diffstarpop_data : dict
            ancillary diffstarpop data such as frac_q

    Notes
    -----
    All mass quantities quoted in Msun (not Msun/h)

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
        n_hmf_grid=n_hmf_grid,
        logmp_cutoff=logmp_cutoff,
        logmp_cutoff_himass=logmp_cutoff_himass,
    )

    t0 = flat_wcdm.age_at_z0(*cosmo_params)

    t_table = jnp.linspace(T_TABLE_MIN, t0, n_sfh_table)

    upids = jnp.zeros_like(cenpop["logmp0"]).astype(int) - 1
    lgmu_infall = jnp.zeros_like(cenpop["logmp0"])
    logmhost_infall = jnp.zeros_like(cenpop["logmp0"]) + cenpop["logmp0"]
    gyr_since_infall = jnp.zeros_like(cenpop["logmp0"])
    args = (
        diffstarpop_params,
        cenpop["mah_params"],
        cenpop["logmp0"],
        upids,
        lgmu_infall,
        logmhost_infall,
        gyr_since_infall,
        ran_key,
        t_table,
    )

    ddp_fields = "sfh_params_ms", "sfh_params_q", "sfh_ms", "sfh_q", "frac_q", "mc_is_q"
    ddp_values = mc_diffstar_sfh_galpop(*args, lgt0=jnp.log10(t0), fb=FB)
    diffstarpop_data = dict()
    for key, value in zip(ddp_fields, ddp_values):
        diffstarpop_data[key] = value

    sfh_table = jnp.where(
        diffstarpop_data["mc_is_q"].reshape((-1, 1)),
        diffstarpop_data["sfh_q"],
        diffstarpop_data["sfh_ms"],
    )
    sfh_params = dpu.mc_select_diffstar_params(
        diffstarpop_data["sfh_params_q"],
        diffstarpop_data["sfh_params_ms"],
        diffstarpop_data["mc_is_q"],
    )

    logsmh_table = np.log10(cumulative_mstar_formed_galpop(t_table, sfh_table))
    logsm_obs = interp_vmap(cenpop["t_obs"], t_table, logsmh_table)
    logsfr_obs = interp_vmap(cenpop["t_obs"], t_table, np.log10(sfh_table))
    logssfr_obs = logsfr_obs - logsm_obs

    if return_internal_quantities:
        logsmh_table_q = np.log10(
            cumulative_mstar_formed_galpop(t_table, diffstarpop_data["sfh_q"])
        )
        logsm_obs_q = interp_vmap(cenpop["t_obs"], t_table, logsmh_table_q)
        logsfr_obs_q = interp_vmap(
            cenpop["t_obs"], t_table, np.log10(diffstarpop_data["sfh_q"])
        )
        logssfr_obs_q = logsfr_obs_q - logsm_obs_q

        logsmh_table_ms = np.log10(
            cumulative_mstar_formed_galpop(t_table, diffstarpop_data["sfh_ms"])
        )
        logsm_obs_ms = interp_vmap(cenpop["t_obs"], t_table, logsmh_table_ms)
        logsfr_obs_ms = interp_vmap(
            cenpop["t_obs"], t_table, np.log10(diffstarpop_data["sfh_ms"])
        )
        logssfr_obs_ms = logsfr_obs_ms - logsm_obs_ms

    fields = (
        *cenpop.keys(),
        "logsm_obs",
        "logssfr_obs",
        "sfh_params",
        "sfh_table",
        "t_table",
        "diffstarpop_data",
    )
    values = (
        *cenpop.values(),
        logsm_obs,
        logssfr_obs,
        sfh_params,
        sfh_table,
        t_table,
        diffstarpop_data,
    )

    if return_internal_quantities:
        fields = (
            *fields,
            "logsm_obs_ms",
            "logssfr_obs_ms",
            "sfh_params_ms",
            "sfh_table_ms",
            "logsm_obs_q",
            "logssfr_obs_q",
            "sfh_params_q",
            "sfh_table_q",
        )
        values = (
            *values,
            logsm_obs_ms,
            logssfr_obs_ms,
            diffstarpop_data["sfh_params_ms"],
            diffstarpop_data["sfh_ms"],
            logsm_obs_q,
            logssfr_obs_q,
            diffstarpop_data["sfh_params_q"],
            diffstarpop_data["sfh_q"],
        )

    cenpop_out = dict()
    for key, value in zip(fields, values):
        cenpop_out[key] = value

    return cenpop_out


def sobol_lightcone_diffstar_cens(
    ran_key,
    num_halos,
    z_min,
    z_max,
    lgmp_min,
    lgmp_max,
    sky_area_degsq,
    cosmo_params=flat_wcdm.PLANCK15,
    fb=FB,
    hmf_params=mc_hosts.DEFAULT_HMF_PARAMS,
    diffmahpop_params=DEFAULT_DIFFMAHPOP_PARAMS,
    diffstarpop_params=DEFAULT_DIFFSTARPOP_PARAMS,
    n_sfh_table=N_SFH_TABLE,
    return_internal_quantities=False,
    logmp_cutoff=DEFAULT_LOGMP_CUTOFF,
    logmp_cutoff_himass=DEFAULT_LOGMP_HIMASS_CUTOFF,
):
    """
    Generate halo MAH and galaxy SFH for host halos sampled from a lightcone

    Parameters
    ----------
    ran_key : jran.key

    num_halos : int
        Number of halos to generate

    z_min, z_max : float

    lgmp_min, lgmp_max : float
        Base-10 log of min/max halo mass in units of Msun (not Msun/h)

    sky_area_degsq : float
        Sky area in units of deg^2

    Returns
    -------
    cenpop : dict

        z_obs : narray, shape (n_halos, )
            Lightcone redshift

        logmp_obs : narray, shape (n_halos, )
            Halo mass at the lightcone redshift

        mah_params : namedtuple of diffmah params
            Each tuple entry is an ndarray with shape (n_halos, )

        logmp0 : narray, shape (n_halos, )
            Base-10 log of halo mass in units of Msun at z=0

        logsm_obs : narray, shape (n_halos, )
            log10(Mstar) at the time of observation

        logssfr_obs : narray, shape (n_halos, )
            log10(SFR/Mstar) at the time of observation

        sfh_params : namedtuple
            Diffstar params for every galaxy

        sfh_table : narray, shape (n_halos, n_times)
            Star formation rate in Msun/yr

        t_table : narray, shape (n_times, )

        diffstarpop_data : dict
            ancillary diffstarpop data such as frac_q

    Notes
    -----
    All mass quantities quoted in Msun (not Msun/h)

    """
    ran_key, lc_key = jran.split(ran_key, 2)
    args = (lc_key, num_halos, z_min, z_max, lgmp_min, lgmp_max, sky_area_degsq)
    cenpop = mc_weighted_halo_lightcone(
        *args,
        hmf_params=hmf_params,
        diffmahpop_params=diffmahpop_params,
        logmp_cutoff=logmp_cutoff,
        logmp_cutoff_himass=logmp_cutoff_himass,
    )

    t0 = flat_wcdm.age_at_z0(*cosmo_params)

    t_table = jnp.linspace(T_TABLE_MIN, t0, n_sfh_table)

    upids = jnp.zeros_like(cenpop["logmp0"]).astype(int) - 1
    lgmu_infall = jnp.zeros_like(cenpop["logmp0"])
    logmhost_infall = jnp.zeros_like(cenpop["logmp0"]) + cenpop["logmp0"]
    gyr_since_infall = jnp.zeros_like(cenpop["logmp0"])
    args = (
        diffstarpop_params,
        cenpop["mah_params"],
        cenpop["logmp0"],
        upids,
        lgmu_infall,
        logmhost_infall,
        gyr_since_infall,
        ran_key,
        t_table,
    )

    ddp_fields = "sfh_params_ms", "sfh_params_q", "sfh_ms", "sfh_q", "frac_q", "mc_is_q"
    ddp_values = mc_diffstar_sfh_galpop(*args, lgt0=jnp.log10(t0), fb=fb)
    diffstarpop_data = dict()
    for key, value in zip(ddp_fields, ddp_values):
        diffstarpop_data[key] = value

    sfh_table = jnp.where(
        diffstarpop_data["mc_is_q"].reshape((-1, 1)),
        diffstarpop_data["sfh_q"],
        diffstarpop_data["sfh_ms"],
    )
    sfh_params = dpu.mc_select_diffstar_params(
        diffstarpop_data["sfh_params_q"],
        diffstarpop_data["sfh_params_ms"],
        diffstarpop_data["mc_is_q"],
    )

    logsmh_table = np.log10(cumulative_mstar_formed_galpop(t_table, sfh_table))
    logsm_obs = interp_vmap(cenpop["t_obs"], t_table, logsmh_table)
    logsfr_obs = interp_vmap(cenpop["t_obs"], t_table, np.log10(sfh_table))
    logssfr_obs = logsfr_obs - logsm_obs

    if return_internal_quantities:
        logsmh_table_q = np.log10(
            cumulative_mstar_formed_galpop(t_table, diffstarpop_data["sfh_q"])
        )
        logsm_obs_q = interp_vmap(cenpop["t_obs"], t_table, logsmh_table_q)
        logsfr_obs_q = interp_vmap(
            cenpop["t_obs"], t_table, np.log10(diffstarpop_data["sfh_q"])
        )
        logssfr_obs_q = logsfr_obs_q - logsm_obs_q

        logsmh_table_ms = np.log10(
            cumulative_mstar_formed_galpop(t_table, diffstarpop_data["sfh_ms"])
        )
        logsm_obs_ms = interp_vmap(cenpop["t_obs"], t_table, logsmh_table_ms)
        logsfr_obs_ms = interp_vmap(
            cenpop["t_obs"], t_table, np.log10(diffstarpop_data["sfh_ms"])
        )
        logssfr_obs_ms = logsfr_obs_ms - logsm_obs_ms

    fields = (
        *cenpop.keys(),
        "logsm_obs",
        "logssfr_obs",
        "sfh_params",
        "sfh_table",
        "t_table",
        "diffstarpop_data",
    )
    values = (
        *cenpop.values(),
        logsm_obs,
        logssfr_obs,
        sfh_params,
        sfh_table,
        t_table,
        diffstarpop_data,
    )

    if return_internal_quantities:
        fields = (
            *fields,
            "logsm_obs_ms",
            "logssfr_obs_ms",
            "sfh_params_ms",
            "sfh_table_ms",
            "logsm_obs_q",
            "logssfr_obs_q",
            "sfh_params_q",
            "sfh_table_q",
        )
        values = (
            *values,
            logsm_obs_ms,
            logssfr_obs_ms,
            diffstarpop_data["sfh_params_ms"],
            diffstarpop_data["sfh_ms"],
            logsm_obs_q,
            logssfr_obs_q,
            diffstarpop_data["sfh_params_q"],
            diffstarpop_data["sfh_q"],
        )

    cenpop_out = dict()
    for key, value in zip(fields, values):
        cenpop_out[key] = value

    return cenpop_out


def get_nhalo_from_grid_interp(
    tot_num_halos,
    z_obs,
    logmp_obs_mf,
    z_min,
    z_max,
    lgmp_min,
    lgmp_max,
    sky_area_degsq,
    hmf_params,
    cosmo_params,
):
    ngrid_z = 200
    ngrid_m = 200
    ngrid_tot = ngrid_z * ngrid_m
    z_grid = jnp.linspace(z_min, z_max, ngrid_z)
    lgmp_grid = jnp.linspace(lgmp_min, lgmp_max, ngrid_m)
    nhalo_grid = get_nhalo_weighted_lc_grid(
        lgmp_grid, z_grid, sky_area_degsq, hmf_params, cosmo_params
    )

    interpolator = RegularGridInterpolator(
        (z_grid, lgmp_grid), nhalo_grid, bounds_error=False, fill_value=None
    )  # type: ignore

    interp = interpolator(jnp.column_stack([z_obs, logmp_obs_mf]))
    return interp * ngrid_tot / tot_num_halos


def mc_weighted_halo_lightcone(
    ran_key,
    num_halos,
    z_min,
    z_max,
    lgmp_min,
    lgmp_max,
    sky_area_degsq,
    hmf_params=mc_hosts.DEFAULT_HMF_PARAMS,
    diffmahpop_params=DEFAULT_DIFFMAHPOP_PARAMS,
    logmp_cutoff=DEFAULT_LOGMP_CUTOFF,
    logmp_cutoff_himass=DEFAULT_LOGMP_HIMASS_CUTOFF,
    comm=None,
):
    if comm is None:
        try:
            comm = MPI.COMM_WORLD
            # ONLY generate the halos necessary on this rank
            num_halos_on_rank = num_halos // comm.size + (
                1 if comm.rank < num_halos % comm.size else 0
            )
            starting_index = comm.rank * (num_halos // comm.size) + min(
                comm.rank, num_halos % comm.size
            )
        except AttributeError:
            num_halos_on_rank = num_halos
            starting_index = 0

    ran_key, ran_key_sobol = jran.split(ran_key, 2)

    # Generate Sobol sequence for halo masses and redshifts
    seed = int(jran.randint(ran_key_sobol, (), 0, 2**31 - 1))
    bits = None
    if num_halos > 1e9:
        # 64-bit sequence required to generate over 2^30 halos
        bits = 64
    sampler = qmc.Sobol(d=2, scramble=True, rng=seed, bits=bits)
    if starting_index > 0:
        sampler.fast_forward(starting_index)

    with warnings.catch_warnings():
        # Ignore warning about Sobol sequences not being fully balanced
        warnings.filterwarnings("ignore", category=UserWarning)
        sample = sampler.random(num_halos_on_rank)
    z_obs, logmp_obs_mf = qmc.scale(sample, (z_min, lgmp_min), (z_max, lgmp_max)).T

    mclh_args = (
        ran_key,
        num_halos,
        z_obs,
        logmp_obs_mf,
        z_min,
        z_max,
        lgmp_min,
        lgmp_max,
        sky_area_degsq,
    )
    mclh_kwargs = dict(
        hmf_params=hmf_params,
        diffmahpop_params=diffmahpop_params,
        logmp_cutoff=logmp_cutoff,
        logmp_cutoff_himass=logmp_cutoff_himass,
    )

    res = get_weighted_lightcone_sobol_host_halo_diffmah(
        *mclh_args, **mclh_kwargs
    )  # type: ignore

    return res


def get_weighted_lightcone_sobol_host_halo_diffmah(
    ran_key,
    num_halos,
    z_obs,
    logmp_obs_mf,
    z_min,
    z_max,
    lgmp_min,
    lgmp_max,
    sky_area_degsq,
    cosmo_params=flat_wcdm.PLANCK15,
    hmf_params=mc_hosts.DEFAULT_HMF_PARAMS,
    diffmahpop_params=DEFAULT_DIFFMAHPOP_PARAMS,
    logmp_cutoff=DEFAULT_LOGMP_CUTOFF,
    logmp_cutoff_himass=DEFAULT_LOGMP_HIMASS_CUTOFF,
):
    """
    Compute the number of halos on the input halo mass and redshift points
    """

    nhalo_weights = get_nhalo_from_grid_interp(
        num_halos,
        z_obs,
        logmp_obs_mf,
        z_min,
        z_max,
        lgmp_min,
        lgmp_max,
        sky_area_degsq,
        hmf_params=hmf_params,
        cosmo_params=cosmo_params,
    )
    t_obs = flat_wcdm.age_at_z(z_obs, *cosmo_params)
    t_0 = flat_wcdm.age_at_z0(*cosmo_params)
    lgt0 = jnp.log10(t_0)

    logmp_obs_mf_clipped = np.clip(logmp_obs_mf, logmp_cutoff, logmp_cutoff_himass)

    tarr = np.array((10**lgt0,))

    ran_key, mah_key = jran.split(ran_key, 2)
    args = (diffmahpop_params, tarr, logmp_obs_mf_clipped, t_obs, mah_key, lgt0)
    mah_params_uncorrected = mc_cenpop(*args)[0]
    msk_mah_params_nan = np.isnan(mah_params_uncorrected.logm0)

    # workaround to the problem of diffmahpop occasionally returning NaN for logm0
    has_logm0_nans = np.any(msk_mah_params_nan)
    while has_logm0_nans:
        ran_key, mah_key = jran.split(ran_key, 2)

        args = (diffmahpop_params, tarr, logmp_obs_mf_clipped, t_obs, mah_key, lgt0)
        mah_params_uncorrected_new = mc_cenpop(*args)[0]  # mah_params, dmhdt, log_mah
        mah_params_uncorrected_logm0 = jnp.where(
            msk_mah_params_nan,
            mah_params_uncorrected_new.logm0,
            mah_params_uncorrected.logm0,
        )
        mah_params_uncorrected = mah_params_uncorrected._replace(
            logm0=mah_params_uncorrected_logm0
        )

        msk_nan = np.isnan(mah_params_uncorrected.logm0)
        has_logm0_nans = np.sum(msk_nan) > 0

    logmp_obs_orig = _log_mah_kern(mah_params_uncorrected, t_obs, lgt0)
    delta_logmh_clip = logmp_obs_orig - logmp_obs_mf
    mah_params = mah_params_uncorrected._replace(
        logm0=mah_params_uncorrected.logm0 - delta_logmh_clip
    )

    logmp0 = _log_mah_kern(mah_params, 10**lgt0, lgt0)
    logmp_obs = _log_mah_kern(mah_params, t_obs, lgt0)

    fields = ("z_obs", "t_obs", "logmp_obs", "mah_params", "logmp0")
    values = (z_obs, t_obs, logmp_obs, mah_params, logmp0)
    cenpop_out = dict()
    for key, value in zip(fields, values):
        cenpop_out[key] = value
    cenpop_out["nhalos"] = nhalo_weights

    return cenpop_out


def mc_weighted_lightcone_data(
    ran_key,
    num_halos,
    z_min,
    z_max,
    lgmp_min,
    lgmp_max,
    sky_area_degsq,
    ssp_data,
    tcurves,
    z_phot_table,
    logmp_cutoff=11.0,
    cosmo_params=flat_wcdm.PLANCK15,
):
    """Monte Carlo generator of a weighted lightcone of halos

    Parameters
    ----------
    ran_key : jran.key

    num_halos : int
        Number of halos in the weighted grid

    lgmp_min, lgmp_max : float
        Minimum and maximum halo mass in units of Msun (not Msun/h)

    z_min, z_max : float

    sky_area_degsq : float
        Sky area in units of deg^2

    ssp_data : namedtuple

    tcurves : list

    z_phot_table : ndarray, shape (n_phot_table, )

    logmp_cutoff : float, optional
        Minimum halo mass for which DiffmahPop is used to generate MAHs.
        For logmp < logmp_cutoff, P(θ_MAH | logmp) = P(θ_MAH | logmp_cutoff)

    cosmo_params : namedtuple, optional
        dsps.cosmology.flat_wcdm cosmology
        cosmo_params = (Om0, w0, wa, h)

    Returns
    -------
    lc_data : namedtuple
        Stores all the data necessary to compute photometry

    """
    args = (
        ran_key,
        num_halos,
        z_min,
        z_max,
        lgmp_min,
        lgmp_max,
        sky_area_degsq,
    )

    lc_grid = mc_weighted_halo_lightcone(*args, logmp_cutoff=logmp_cutoff)

    t0 = flat_wcdm.age_at_z0(*cosmo_params)
    t_table = jnp.linspace(T_TABLE_MIN, t0, N_SFH_TABLE)
    logmp_obs = logmh_at_t_obs(lc_grid["mah_params"], lc_grid["t_obs"], jnp.log10(t0))

    precomputed_ssp_mag_table = psspp.get_precompute_ssp_mag_redshift_table(
        tcurves, ssp_data, z_phot_table, cosmo_params
    )
    wave_eff_table = get_wave_eff_table(z_phot_table, tcurves)

    lc_data = LCData(
        lc_grid["nhalos"],
        lc_grid["z_obs"],
        lc_grid["t_obs"],
        lc_grid["mah_params"],
        lc_grid["logmp0"],
        t_table,
        ssp_data,
        precomputed_ssp_mag_table,
        z_phot_table,
        wave_eff_table,
        logmp_obs,
    )
    return lc_data


def mc_lightcone_data(
    ran_key,
    z_min,
    z_max,
    lgmp_min,
    sky_area_degsq,
    ssp_data,
    tcurves,
    z_phot_table,
    cosmo_params=flat_wcdm.PLANCK15,
):
    """Monte Carlo generator of a lightcone of halos (unweighted)

    Parameters
    ----------
    ran_key : jran.key

    lgmp_min : float
        Minimum halo mass in units of Msun (not Msun/h)

    z_min, z_max : float

    sky_area_degsq : float
        Sky area in units of deg^2

    ssp_data : namedtuple

    tcurves : list

    z_phot_table : ndarray, shape (n_phot_table, )

    cosmo_params : namedtuple, optional
        dsps.cosmology.flat_wcdm cosmology
        cosmo_params = (Om0, w0, wa, h)

    Returns
    -------
    lc_data : namedtuple
        Stores all the data necessary to compute photometry

    """
    args = (ran_key, lgmp_min, z_min, z_max, sky_area_degsq)
    cenpop = mc_lightcone_host_halo_diffmah(*args)
    n_gals = len(cenpop["logmp0"])
    cenpop["nhalos"] = np.ones(n_gals).astype(int)

    t0 = flat_wcdm.age_at_z0(*cosmo_params)
    t_table = jnp.linspace(T_TABLE_MIN, t0, N_SFH_TABLE)
    logmp_obs = logmh_at_t_obs(cenpop["mah_params"], cenpop["t_obs"], jnp.log10(t0))

    precomputed_ssp_mag_table = psspp.get_precompute_ssp_mag_redshift_table(
        tcurves, ssp_data, z_phot_table, cosmo_params
    )
    wave_eff_table = get_wave_eff_table(z_phot_table, tcurves)

    lc_data = LCData(
        cenpop["nhalos"],
        cenpop["z_obs"],
        cenpop["t_obs"],
        cenpop["mah_params"],
        cenpop["logmp0"],
        t_table,
        ssp_data,
        precomputed_ssp_mag_table,
        z_phot_table,
        wave_eff_table,
        logmp_obs,
    )
    return lc_data


_LCDKEYS = (
    "nhalos",
    "z_obs",
    "t_obs",
    "mah_params",
    "logmp0",
    "t_table",
    "ssp_data",
    "precomputed_ssp_mag_table",
    "z_phot_table",
    "wave_eff_table",
    "logmp_obs",
)
LCData = namedtuple("LCData", _LCDKEYS)
