"""Functions to generate Monte Carlo realizations of galaxies on a lightcone"""

from functools import partial

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
from dsps.data_loaders import load_transmission_curve
from dsps.metallicity import umzr
from dsps.sed import metallicity_weights as zmetw
from dsps.sed.stellar_age_weights import calc_age_weights_from_sfh_table
from jax import config, grad
from jax import jit as jjit
from jax import numpy as jnp
from jax import random as jran
from jax import vmap

from ..mass_functions import mc_hosts
from . import photometry_interpolation as photerp
from . import precompute_ssp_phot as psp

config.update("jax_enable_x64", True)

_AGEPOP = (None, 0, None, 0)
calc_age_weights_from_sfh_table_vmap = jjit(
    vmap(calc_age_weights_from_sfh_table, in_axes=_AGEPOP)
)

FULL_SKY_AREA = (4 * jnp.pi) * (180 / jnp.pi) ** 2

LSST_PHOTKEYS = ("lsst_u*", "lsst_g*", "lsst_r*", "lsst_i*", "lsst_z*", "lsst_y*")

interp_vmap = jjit(vmap(jnp.interp, in_axes=(0, None, 0)))

_G = (0, None, None, 0, 0)
mc_logmp_vmap = jjit(vmap(mc_hosts._mc_host_halos_singlez_kern, in_axes=_G))

_Z = (0, None, None, None, None)
dist_com_grad_kern = jjit(
    vmap(grad(flat_wcdm.comoving_distance_to_z, argnums=0), in_axes=_Z)
)

# gal_lgmet, gal_lgmet_scatter, ssp_lgmet
_M = (0, None, None)
_calc_lgmet_weights_galpop = jjit(
    vmap(zmetw.calc_lgmet_weights_from_lognormal_mdf, in_axes=_M)
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
    """Generate halo MAHs for host halos sampled from a lightcone

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
    cenpop : dict

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

    fields = ("z_obs", "t_obs", "logmp_obs", "mah_params", "logmp0")
    values = (
        z_halopop,
        t_obs_halopop,
        logmp_obs_halopop,
        halopop.mah_params,
        logmp0_halopop,
    )
    cenpop_out = dict()
    for key, value in zip(fields, values):
        cenpop_out[key] = value

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
    n_grid=2_000,
    n_t_table=100,
):
    """
    Generate halo MAH and galaxy SFH for host halos sampled from a lightcone

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
    cenpop : dict

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

        diffstarpop_data : dict
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
    lgmu_infall = jnp.zeros_like(cenpop["logmp0"])
    logmhost_infall = jnp.zeros_like(cenpop["logmp0"]) + cenpop["logmp0"]
    gyr_since_infall = jnp.zeros_like(cenpop["logmp0"])
    args = (
        diffstarpop_params,
        cenpop["mah_params"],
        cenpop["logmp0"],
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
    cosmo_params=flat_wcdm.PLANCK15,
    hmf_params=mc_hosts.DEFAULT_HMF_PARAMS,
    diffmahpop_params=DEFAULT_DIFFMAHPOP_PARAMS,
    diffstarpop_params=DEFAULT_DIFFSTARPOP_PARAMS,
    ssp_lg_age_gyr=np.linspace(5.0, 10.25, 90) - 9.0,
    n_grid=2_000,
    n_t_table=100,
):
    """
    Generate halo MAH and galaxy SFH and stellar age weights
    for host halos sampled from a lightcone

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
    cenpop : dict

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

        diffstarpop_data : dict
            ancillary diffstarpop data such as frac_q

        age_weights : ndarray, shape (n_halos, n_ages)
            Stellar age PDF, P(τ), for every galaxy

        ssp_lg_age_gyr : ndarray, shape (n_ages, )
            log10 stellar age grid in Gyr

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
        n_grid=n_grid,
        n_t_table=n_t_table,
    )
    age_weights_galpop = calc_age_weights_from_sfh_table_vmap(
        cenpop["t_table"], cenpop["sfh_table"], ssp_lg_age_gyr, cenpop["t_obs"]
    )
    fields = (*cenpop.keys(), "age_weights", "ssp_lg_age_gyr")
    values = (*cenpop.values(), age_weights_galpop, ssp_lg_age_gyr)
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
    cosmo_params=flat_wcdm.PLANCK15,
    hmf_params=mc_hosts.DEFAULT_HMF_PARAMS,
    diffmahpop_params=DEFAULT_DIFFMAHPOP_PARAMS,
    diffstarpop_params=DEFAULT_DIFFSTARPOP_PARAMS,
    mzr_params=umzr.DEFAULT_MZR_PARAMS,
    lgmet_scatter=umzr.MZR_SCATTER,
    ssp_lg_age_gyr=np.linspace(5.0, 10.25, 90) - 9.0,
    ssp_lgmet=np.linspace(-4, -1.3, 11),
    n_grid=2_000,
    n_t_table=100,
):
    cenpop = mc_lightcone_diffstar_stellar_ages_cens(
        ran_key,
        lgmp_min,
        z_min,
        z_max,
        sky_area_degsq,
        cosmo_params=cosmo_params,
        hmf_params=hmf_params,
        diffmahpop_params=diffmahpop_params,
        diffstarpop_params=diffstarpop_params,
        n_grid=n_grid,
        n_t_table=n_t_table,
        ssp_lg_age_gyr=ssp_lg_age_gyr,
    )

    lgmet_med = umzr.mzr_model(cenpop["logsm_obs"], cenpop["t_obs"], *mzr_params)
    cenpop["lgmet_weights"] = _calc_lgmet_weights_galpop(
        lgmet_med, lgmet_scatter, ssp_lgmet
    )
    n_gals, n_met = cenpop["lgmet_weights"].shape
    n_age = len(ssp_lg_age_gyr)
    _w_lgmet = cenpop["lgmet_weights"].reshape((n_gals, n_met, 1))
    _w_age = cenpop["age_weights"].reshape((n_gals, 1, n_age))
    cenpop["ssp_weights"] = _w_lgmet * _w_age
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
    precomputed_ssp_mag_table,
    z_phot_table,
    cosmo_params=flat_wcdm.PLANCK15,
    hmf_params=mc_hosts.DEFAULT_HMF_PARAMS,
    diffmahpop_params=DEFAULT_DIFFMAHPOP_PARAMS,
    diffstarpop_params=DEFAULT_DIFFSTARPOP_PARAMS,
    mzr_params=umzr.DEFAULT_MZR_PARAMS,
    lgmet_scatter=umzr.MZR_SCATTER,
    n_grid=2_000,
    n_t_table=100,
    phot_keys=LSST_PHOTKEYS,
):
    """
    Generate photometry for host halos sampled from a lightcone

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

        diffstarpop_data : dict
            ancillary diffstarpop data such as frac_q

        age_weights : ndarray, shape (n_halos, n_ages)
            Stellar age PDF, P(τ), for every galaxy

        ssp_lg_age_gyr : ndarray, shape (n_ages, )
            log10 stellar age grid in Gyr

        obs_mags : ndarray, shape (n_bands, n_gals)
            Apparent magnitude of each galaxy in each band

    """
    cenpop = mc_lightcone_diffstar_ssp_weights_cens(
        ran_key,
        lgmp_min,
        z_min,
        z_max,
        sky_area_degsq,
        cosmo_params=cosmo_params,
        hmf_params=hmf_params,
        diffmahpop_params=diffmahpop_params,
        diffstarpop_params=diffstarpop_params,
        mzr_params=mzr_params,
        lgmet_scatter=lgmet_scatter,
        n_grid=n_grid,
        n_t_table=n_t_table,
        ssp_lgmet=ssp_data.ssp_lgmet,
        ssp_lg_age_gyr=ssp_data.ssp_lg_age_gyr,
    )
    if precomputed_ssp_mag_table is None:
        tcurves = [load_transmission_curve(bn_pat=key) for key in phot_keys]

        collector = []
        for z_obs in z_phot_table:
            ssp_obsflux_table = psp.get_ssp_obsflux_table(
                ssp_data, tcurves, z_obs, flat_wcdm.PLANCK15
            )
            collector.append(ssp_obsflux_table)
        precomputed_ssp_mag_table = -2.5 * np.log10(np.array(collector))

    photmag_table_galpop = photerp.interpolate_ssp_photmag_table(
        cenpop["z_obs"], z_phot_table, precomputed_ssp_mag_table
    )
    cenpop["precomputed_ssp_mag_table"] = precomputed_ssp_mag_table
    cenpop["photflux_table"] = 10 ** (-0.4 * photmag_table_galpop)
    cenpop["z_phot_table"] = z_phot_table

    n_gals, n_bands, n_met, n_age = cenpop["photflux_table"].shape
    w = cenpop["ssp_weights"].reshape((n_gals, 1, n_met, n_age))
    sm = 10 ** cenpop["logmp_obs"].reshape((n_gals, 1))
    photflux_galpop = jnp.sum(w * cenpop["photflux_table"], axis=(2, 3)) * sm
    cenpop["obs_mags"] = -2.5 * np.log10(photflux_galpop)

    return cenpop
