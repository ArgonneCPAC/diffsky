# flake8: noqa: E402
"""Functions to generate Monte Carlo realizations of galaxies on a lightcone"""

from jax import config

config.update("jax_enable_x64", True)

import numpy as np
from diffmah.diffmah_kernels import _log_mah_kern
from diffmah.diffmahpop_kernels.bimod_censat_params import DEFAULT_DIFFMAHPOP_PARAMS
from diffmah.diffmahpop_kernels.mc_bimod_cens import mc_cenpop
from diffstar.utils import cumulative_mstar_formed_galpop
from diffstarpop import mc_diffstar_sfh_galpop
from diffstarpop import param_utils as dpu
from diffstarpop.defaults import DEFAULT_DIFFSTARPOP_PARAMS
from dsps.constants import T_TABLE_MIN
from dsps.cosmology import flat_wcdm
from dsps.metallicity import umzr
from dsps.sed import metallicity_weights as zmetw
from dsps.sed.stellar_age_weights import calc_age_weights_from_sfh_table
from jax import jit as jjit
from jax import numpy as jnp
from jax import random as jran
from jax import vmap

from ..burstpop import diffqburstpop_mono
from ..dustpop import tw_dustpop_mono, tw_dustpop_mono_noise
from ..mass_functions import hmf_model, mc_hosts
from ..phot_utils import get_wave_eff_from_tcurves
from ..ssp_err_model import ssp_err_model
from . import photometry_interpolation as photerp
from . import precompute_ssp_phot as psp
from .lc_utils import spherical_shell_comoving_volume
from .scatter import DEFAULT_SCATTER_PARAMS

N_HMF_GRID = 2_000
N_SFH_TABLE = 100

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
    ran_key : jax.random.key

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
    logmp_cutoff=0.0,
    lgmp_max=mc_hosts.LGMH_MAX,
):
    """Generate halo MAHs for host halos sampled from a lightcone

    Parameters
    ----------
    ran_key : jax.random.key

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

    logmp_obs_mf_clipped = np.clip(logmp_obs_mf, logmp_cutoff, np.inf)

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
    logmp_cutoff=0.0,
):
    """Compute the number of halos on the input grid of halo mass and redshift

    Parameters
    ----------
    ran_key : jax.random.key

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

    logmp_obs_mf_clipped = np.clip(logmp_obs_mf, logmp_cutoff, np.inf)

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
    hmf_params=mc_hosts.DEFAULT_HMF_PARAMS,
    diffmahpop_params=DEFAULT_DIFFMAHPOP_PARAMS,
    diffstarpop_params=DEFAULT_DIFFSTARPOP_PARAMS,
    n_hmf_grid=N_HMF_GRID,
    n_sfh_table=N_SFH_TABLE,
    return_internal_quantities=False,
    logmp_cutoff=0.0,
):
    """
    Generate halo MAH and galaxy SFH for host halos sampled from a lightcone

    Parameters
    ----------
    ran_key : jax.random.key

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
    ddp_values = mc_diffstar_sfh_galpop(*args)
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


def mc_lightcone_diffstar_stellar_ages_cens(
    ran_key,
    lgmp_min,
    z_min,
    z_max,
    sky_area_degsq,
    ssp_data,
    cosmo_params=flat_wcdm.PLANCK15,
    hmf_params=mc_hosts.DEFAULT_HMF_PARAMS,
    diffmahpop_params=DEFAULT_DIFFMAHPOP_PARAMS,
    diffstarpop_params=DEFAULT_DIFFSTARPOP_PARAMS,
    n_hmf_grid=N_HMF_GRID,
    n_sfh_table=N_SFH_TABLE,
    return_internal_quantities=False,
):
    """
    Generate halo MAH and galaxy SFH and stellar age weights
    for host halos sampled from a lightcone

    Parameters
    ----------
    ran_key : jax.random.key

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

        age_weights : ndarray, shape (n_halos, n_ages)
            Stellar age PDF, P(τ), for every galaxy

        ssp_lg_age_gyr : ndarray, shape (n_ages, )
            log10 stellar age grid in Gyr

    Notes
    -----
    All mass quantities quoted in Msun (not Msun/h)

    """
    cenpop = mc_lightcone_diffstar_cens(
        ran_key,
        lgmp_min,
        z_min,
        z_max,
        sky_area_degsq,
        cosmo_params=cosmo_params,
        hmf_params=hmf_params,
        diffmahpop_params=diffmahpop_params,
        diffstarpop_params=diffstarpop_params,
        n_hmf_grid=n_hmf_grid,
        n_sfh_table=n_sfh_table,
        return_internal_quantities=return_internal_quantities,
    )
    age_weights_galpop = calc_age_weights_from_sfh_table_vmap(
        cenpop["t_table"], cenpop["sfh_table"], ssp_data.ssp_lg_age_gyr, cenpop["t_obs"]
    )
    fields = (*cenpop.keys(), "age_weights", "ssp_lg_age_gyr")
    values = (*cenpop.values(), age_weights_galpop, ssp_data.ssp_lg_age_gyr)

    if return_internal_quantities:
        age_weights_galpop_ms = calc_age_weights_from_sfh_table_vmap(
            cenpop["t_table"],
            cenpop["sfh_table_ms"],
            ssp_data.ssp_lg_age_gyr,
            cenpop["t_obs"],
        )
        age_weights_galpop_q = calc_age_weights_from_sfh_table_vmap(
            cenpop["t_table"],
            cenpop["sfh_table_q"],
            ssp_data.ssp_lg_age_gyr,
            cenpop["t_obs"],
        )

        fields = (*fields, "age_weights_ms", "age_weights_q")
        values = (*values, age_weights_galpop_ms, age_weights_galpop_q)

    cenpop_out = dict()
    for key, value in zip(fields, values):
        cenpop_out[key] = value

    return cenpop_out


def mc_lightcone_diffstar_ssp_weights_cens(
    ran_key,
    lgmp_min,
    z_min,
    z_max,
    sky_area_degsq,
    ssp_data,
    cosmo_params=flat_wcdm.PLANCK15,
    hmf_params=mc_hosts.DEFAULT_HMF_PARAMS,
    diffmahpop_params=DEFAULT_DIFFMAHPOP_PARAMS,
    diffstarpop_params=DEFAULT_DIFFSTARPOP_PARAMS,
    mzr_params=umzr.DEFAULT_MZR_PARAMS,
    lgmet_scatter=umzr.MZR_SCATTER,
    diffburstpop_params=diffqburstpop_mono.DEFAULT_DIFFBURSTPOP_PARAMS,
    n_hmf_grid=N_HMF_GRID,
    n_sfh_table=N_SFH_TABLE,
    return_internal_quantities=False,
):
    cenpop = mc_lightcone_diffstar_stellar_ages_cens(
        ran_key,
        lgmp_min,
        z_min,
        z_max,
        sky_area_degsq,
        ssp_data,
        cosmo_params=cosmo_params,
        hmf_params=hmf_params,
        diffmahpop_params=diffmahpop_params,
        diffstarpop_params=diffstarpop_params,
        n_hmf_grid=n_hmf_grid,
        n_sfh_table=n_sfh_table,
        return_internal_quantities=return_internal_quantities,
    )
    cenpop["smooth_age_weights"] = cenpop["age_weights"].copy()
    if return_internal_quantities:
        cenpop["smooth_age_weights_ms"] = cenpop["age_weights_ms"].copy()
        cenpop["smooth_age_weights_q"] = cenpop["age_weights_q"].copy()

    # Compute age weights with burstiness
    _args = (
        diffburstpop_params,
        cenpop["logsm_obs"],
        cenpop["logssfr_obs"],
        ssp_data.ssp_lg_age_gyr,
        cenpop["age_weights"],
    )
    bursty_age_weights, burst_params = _calc_bursty_age_weights_vmap(*_args)
    cenpop["age_weights"] = bursty_age_weights
    for param, pname in zip(burst_params, burst_params._fields):
        cenpop[pname] = param

    if return_internal_quantities:
        _args = (
            diffburstpop_params,
            cenpop["logsm_obs_ms"],
            cenpop["logssfr_obs_ms"],
            ssp_data.ssp_lg_age_gyr,
            cenpop["age_weights_ms"],
        )
        bursty_age_weights, burst_params = _calc_bursty_age_weights_vmap(*_args)
        cenpop["age_weights_ms"] = bursty_age_weights

        _args = (
            diffburstpop_params,
            cenpop["logsm_obs_q"],
            cenpop["logssfr_obs_q"],
            ssp_data.ssp_lg_age_gyr,
            cenpop["age_weights_q"],
        )
        bursty_age_weights, burst_params = _calc_bursty_age_weights_vmap(*_args)
        cenpop["age_weights_q"] = bursty_age_weights

    lgmet_med = umzr.mzr_model(cenpop["logsm_obs"], cenpop["t_obs"], *mzr_params)

    cenpop["lgmet_weights"] = _calc_lgmet_weights_galpop(
        lgmet_med, lgmet_scatter, ssp_data.ssp_lgmet
    )
    n_gals, n_met = cenpop["lgmet_weights"].shape
    n_age = len(ssp_data.ssp_lg_age_gyr)
    _w_lgmet = cenpop["lgmet_weights"].reshape((n_gals, n_met, 1))

    _w_age = cenpop["age_weights"].reshape((n_gals, 1, n_age))
    cenpop["ssp_weights"] = _w_lgmet * _w_age

    _w_age = cenpop["smooth_age_weights"].reshape((n_gals, 1, n_age))
    cenpop["smooth_ssp_weights"] = _w_lgmet * _w_age

    if return_internal_quantities:
        _w_age_ms = cenpop["age_weights_ms"].reshape((n_gals, 1, n_age))
        _w_age_q = cenpop["age_weights_q"].reshape((n_gals, 1, n_age))

        cenpop["ssp_weights_ms"] = _w_lgmet * _w_age_ms
        cenpop["ssp_weights_q"] = _w_lgmet * _w_age_q

        _w_age_smooth_ms = cenpop["smooth_age_weights_ms"].reshape((n_gals, 1, n_age))
        _w_age_smooth_q = cenpop["smooth_age_weights_q"].reshape((n_gals, 1, n_age))
        cenpop["smooth_ssp_weights_ms"] = _w_lgmet * _w_age_smooth_ms
        cenpop["smooth_ssp_weights_q"] = _w_lgmet * _w_age_smooth_q

    return cenpop


def get_precompute_ssp_mag_redshift_table(tcurves, ssp_data, z_phot_table):
    """Calculate precomputed SSP magnitude table to use for lightcone interpolation

    Parameters
    ----------
    tcurves : list of n_bands transmission curves

    ssp_data : namedtuple

    z_phot_table : ndarray, shape (n_phot_table, )

    Returns
    -------
    precomputed_ssp_mag_table : ndarray, shape (n_phot_table, n_bands, n_met, n_age)

    """
    collector = []
    for z_obs in z_phot_table:
        ssp_obsflux_table = psp.get_ssp_obsflux_table(
            ssp_data, tcurves, z_obs, flat_wcdm.PLANCK15
        )
        collector.append(ssp_obsflux_table)
    precomputed_ssp_mag_table = -2.5 * np.log10(np.array(collector))
    return precomputed_ssp_mag_table


def mc_lightcone_obs_mags_cens(
    ran_key,
    lgmp_min,
    z_min,
    z_max,
    sky_area_degsq,
    ssp_data,
    tcurves,
    precomputed_ssp_mag_table,
    z_phot_table,
    cosmo_params=flat_wcdm.PLANCK15,
    hmf_params=mc_hosts.DEFAULT_HMF_PARAMS,
    diffmahpop_params=DEFAULT_DIFFMAHPOP_PARAMS,
    diffstarpop_params=DEFAULT_DIFFSTARPOP_PARAMS,
    mzr_params=umzr.DEFAULT_MZR_PARAMS,
    lgmet_scatter=umzr.MZR_SCATTER,
    diffburstpop_params=diffqburstpop_mono.DEFAULT_DIFFBURSTPOP_PARAMS,
    dustpop_params=tw_dustpop_mono.DEFAULT_DUSTPOP_PARAMS,
    scatter_params=DEFAULT_SCATTER_PARAMS,
    ssp_err_pop_params=ssp_err_model.DEFAULT_SSPERR_PARAMS,
    n_hmf_grid=N_HMF_GRID,
    n_sfh_table=N_SFH_TABLE,
    return_internal_quantities=False,
):
    """
    Generate photometry for host halos sampled from a lightcone

    Parameters
    ----------
    ran_key : jax.random.key

    lgmp_min : float
        Minimum halo mass in units of Msun (not Msun/h)

    z_min, z_max : float

    sky_area_degsq : float
        Sky area in units of deg^2

    cosmo_params : namedtuple
        dsps.cosmology.flat_wcdm cosmology
        cosmo_params = (Om0, w0, wa, h)

    ssp_data : namedtuple

    precomputed_ssp_mag_table : ndarray, shape (n_phot_table, n_bands, n_met, n_age)
        Computed by the get_precompute_ssp_mag_redshift_table function

    z_phot_table : ndarray, shape (n_phot_table, )

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

        age_weights : ndarray, shape (n_halos, n_ages)
            Stellar age PDF, P(τ), for every galaxy

        ssp_lg_age_gyr : ndarray, shape (n_ages, )
            log10 stellar age grid in Gyr

        obs_mags : ndarray, shape (n_bands, n_gals)
            Apparent magnitude of each galaxy in each band

    Notes
    -----
    All mass quantities quoted in Msun (not Msun/h)

    """
    ran_key, cenpop_key = jran.split(ran_key, 2)
    cenpop = mc_lightcone_diffstar_ssp_weights_cens(
        cenpop_key,
        lgmp_min,
        z_min,
        z_max,
        sky_area_degsq,
        ssp_data,
        cosmo_params=cosmo_params,
        hmf_params=hmf_params,
        diffmahpop_params=diffmahpop_params,
        diffstarpop_params=diffstarpop_params,
        mzr_params=mzr_params,
        lgmet_scatter=lgmet_scatter,
        diffburstpop_params=diffburstpop_params,
        n_hmf_grid=n_hmf_grid,
        n_sfh_table=n_sfh_table,
        return_internal_quantities=return_internal_quantities,
    )

    photmag_table_galpop = photerp.interpolate_ssp_photmag_table(
        cenpop["z_obs"], z_phot_table, precomputed_ssp_mag_table
    )
    cenpop["precomputed_ssp_mag_table"] = precomputed_ssp_mag_table
    cenpop["ssp_photflux_table"] = 10 ** (-0.4 * photmag_table_galpop)
    cenpop["z_phot_table"] = z_phot_table

    collector = []
    for z_obs in z_phot_table:
        wave_eff = get_wave_eff_from_tcurves(tcurves, z_obs)
        collector.append(wave_eff)
    wave_eff_table = np.array(collector)

    cenpop["wave_eff"] = interp_vmap2(cenpop["z_obs"], z_phot_table, wave_eff_table)

    # Delta mags
    frac_ssp_err = get_frac_ssp_err_vmap(
        ssp_err_pop_params,
        cenpop["z_obs"],
        cenpop["logsm_obs"],
        cenpop["wave_eff"],
        ssp_err_model.LAMBDA_REST,
    )
    cenpop["frac_ssp_err"] = frac_ssp_err

    n_gals = cenpop["z_obs"].size
    ran_key, dust_key = jran.split(ran_key, 2)
    av_key, delta_key, funo_key = jran.split(dust_key, 3)
    uran_av = jran.uniform(av_key, shape=(n_gals,))
    uran_delta = jran.uniform(delta_key, shape=(n_gals,))
    uran_funo = jran.uniform(funo_key, shape=(n_gals,))

    ftrans_args = (
        dustpop_params,
        cenpop["wave_eff"],
        cenpop["logsm_obs"],
        cenpop["logssfr_obs"],
        cenpop["z_obs"],
        ssp_data.ssp_lg_age_gyr,
        uran_av,
        uran_delta,
        uran_funo,
        scatter_params,
    )
    _res = calc_dust_ftrans_vmap(*ftrans_args)
    ftrans_nonoise, ftrans, dust_params, noisy_dust_params = _res

    for param, pname in zip(dust_params, dust_params._fields):
        cenpop[pname + "_nonoise"] = param
    for param, pname in zip(noisy_dust_params, noisy_dust_params._fields):
        cenpop[pname] = param

    cenpop["ftrans_nonoise"] = ftrans_nonoise
    cenpop["ftrans"] = ftrans

    n_gals, n_bands, n_met, n_age = cenpop["ssp_photflux_table"].shape
    w = cenpop["ssp_weights"].reshape((n_gals, 1, n_met, n_age))
    sm = 10 ** cenpop["logsm_obs"].reshape((n_gals, 1))

    ran_key, ssp_key = jran.split(ran_key, 2)
    delta_scatter = ssp_err_model.compute_delta_scatter(ssp_key, frac_ssp_err)

    integrand = w * cenpop["ssp_photflux_table"]
    photflux_galpop = jnp.sum(integrand, axis=(2, 3)) * sm
    cenpop["obs_mags_nodust_noerr"] = -2.5 * np.log10(photflux_galpop)

    _ferr_ssp = cenpop["frac_ssp_err"].reshape((n_gals, n_bands, 1, 1))
    integrand = w * cenpop["ssp_photflux_table"] * _ferr_ssp
    photflux_galpop = jnp.sum(integrand, axis=(2, 3)) * sm
    cenpop["obs_mags_nodust"] = -2.5 * np.log10(photflux_galpop) + delta_scatter

    _ftrans = ftrans.reshape((n_gals, n_bands, 1, n_age))
    integrand = w * cenpop["ssp_photflux_table"] * _ftrans
    photflux_galpop = jnp.sum(integrand, axis=(2, 3)) * sm
    cenpop["obs_mags_noerr"] = -2.5 * np.log10(photflux_galpop)

    integrand = w * cenpop["ssp_photflux_table"] * _ftrans * _ferr_ssp
    photflux_galpop = jnp.sum(integrand, axis=(2, 3)) * sm
    cenpop["obs_mags"] = -2.5 * np.log10(photflux_galpop) + delta_scatter

    w_noburst = cenpop["smooth_ssp_weights"].reshape((n_gals, 1, n_met, n_age))
    integrand = w_noburst * cenpop["ssp_photflux_table"]
    photflux_galpop = jnp.sum(integrand, axis=(2, 3)) * sm
    cenpop["obs_mags_nodust_noerr_noburst"] = -2.5 * np.log10(photflux_galpop)

    integrand = w_noburst * cenpop["ssp_photflux_table"] * _ftrans
    photflux_galpop = jnp.sum(integrand, axis=(2, 3)) * sm
    cenpop["obs_mags_noerr_noburst"] = -2.5 * np.log10(photflux_galpop)

    integrand = w_noburst * cenpop["ssp_photflux_table"] * _ftrans * _ferr_ssp
    photflux_galpop = jnp.sum(integrand, axis=(2, 3)) * sm
    cenpop["obs_mags_noburst"] = -2.5 * np.log10(photflux_galpop) + delta_scatter

    if return_internal_quantities:
        w_ms = cenpop["ssp_weights_ms"].reshape((n_gals, 1, n_met, n_age))
        integrand = w_ms * cenpop["ssp_photflux_table"]
        photflux_galpop = jnp.sum(integrand, axis=(2, 3)) * sm
        cenpop["obs_mags_nodust_noerr_ms"] = -2.5 * np.log10(photflux_galpop)

        w_q = cenpop["ssp_weights_q"].reshape((n_gals, 1, n_met, n_age))
        integrand = w_q * cenpop["ssp_photflux_table"]
        photflux_galpop = jnp.sum(integrand, axis=(2, 3)) * sm
        cenpop["obs_mags_nodust_noerr_q"] = -2.5 * np.log10(photflux_galpop)

    return cenpop


_D = (None, 0, None, None, None, None, None, None, None, None)
vmap_kern1 = jjit(
    vmap(
        tw_dustpop_mono_noise.calc_ftrans_singlegal_singlewave_from_dustpop_params,
        in_axes=_D,
    )
)
_E = (None, 0, 0, 0, 0, None, 0, 0, 0, None)
calc_dust_ftrans_vmap = jjit(vmap(vmap_kern1, in_axes=_E))


_F = (None, None, None, 0, None)
_G = (None, 0, 0, 0, None)
get_frac_ssp_err_vmap = jjit(
    vmap(vmap(ssp_err_model.F_sps_err_lambda, in_axes=_F), in_axes=_G)
)
