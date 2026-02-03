""""""

from collections import namedtuple
from functools import partial

from diffstar import DEFAULT_DIFFSTAR_PARAMS
from dsps.sfh.diffburst import DEFAULT_BURST_PARAMS
from jax import jit as jjit
from jax import numpy as jnp
from jax import random as jran
from jax import vmap

from ...dustpop.tw_dust import DEFAULT_DUST_PARAMS
from ...ssp_err_model import ssp_err_model
from .. import mc_diffstarpop_wrappers as mcdw
from .. import photometry_interpolation as photerp
from ..disk_bulge_modeling import disk_bulge_kernels as dbk
from ..disk_bulge_modeling import disk_knots
from ..disk_bulge_modeling import mc_disk_bulge as mcdb
from . import dbk_kernels
from . import ssp_weight_kernels as sspwk

PHOT_RAN_KEYS = (
    "mc_is_q",
    "uran_av",
    "uran_delta",
    "uran_funo",
    "uran_pburst",
    "delta_mag_ssp_scatter",
)
PhotRandoms = namedtuple("PhotRandoms", PHOT_RAN_KEYS)


LGMET_SCATTER = 0.2


_B = (None, None, 1)
interp_vmap2 = jjit(vmap(jnp.interp, in_axes=_B, out_axes=1))


@jjit
def get_mc_phot_randoms(ran_key, diffstarpop_params, mah_params, cosmo_params):
    n_gals = mah_params.logm0.size

    # Monte Carlo diffstar params
    ran_key, sfh_key = jran.split(ran_key, 2)
    sfh_params, mc_is_q = mcdw.mc_diffstarpop_cens_wrapper(
        diffstarpop_params, sfh_key, mah_params, cosmo_params
    )
    # Generate randoms for stochasticity in dust attenuation curves
    ran_key, dust_key = jran.split(ran_key, 2)
    dust_randoms = sspwk.get_dust_randoms(dust_key, n_gals)

    # Randoms for burstiness
    ran_key, burst_key = jran.split(ran_key, 2)
    uran_pburst = sspwk.get_burstiness_randoms(burst_key, n_gals)

    # Scatter for SSP errors
    ran_key, ssp_key = jran.split(ran_key, 2)
    delta_mag_ssp_scatter = ssp_err_model.get_delta_mag_ssp_scatter(ssp_key, n_gals)

    phot_randoms = PhotRandoms(
        mc_is_q,
        dust_randoms.uran_av,
        dust_randoms.uran_delta,
        dust_randoms.uran_funo,
        uran_pburst,
        delta_mag_ssp_scatter,
    )
    return phot_randoms, sfh_params


@jjit
def _dbk_kern(
    t_obs, ssp_data, t_table, sfh_table, burst_params, lgmet_weights, dbk_randoms
):
    disk_bulge_history = mcdb.decompose_sfh_into_disk_bulge_sfh(t_table, sfh_table)

    args = (
        t_obs,
        ssp_data,
        t_table,
        sfh_table,
        burst_params,
        lgmet_weights,
        disk_bulge_history,
        dbk_randoms.fknot,
    )
    dbk_weights = dbk_kernels.get_dbk_weights(*args)

    return dbk_weights, disk_bulge_history


@partial(jjit, static_argnames=["n_t_table"])
def _mc_phot_kern(
    ran_key,
    z_obs,
    t_obs,
    mah_params,
    ssp_data,
    precomputed_ssp_mag_table,
    z_phot_table,
    wave_eff_table,
    diffstarpop_params,
    mzr_params,
    spspop_params,
    scatter_params,
    ssp_err_pop_params,
    cosmo_params,
    fb,
    n_t_table=mcdw.N_T_TABLE,
):
    phot_randoms, sfh_params = get_mc_phot_randoms(
        ran_key, diffstarpop_params, mah_params, cosmo_params
    )
    phot_kern_results = _phot_kern(
        phot_randoms,
        sfh_params,
        z_obs,
        t_obs,
        mah_params,
        ssp_data,
        precomputed_ssp_mag_table,
        z_phot_table,
        wave_eff_table,
        mzr_params,
        spspop_params,
        scatter_params,
        ssp_err_pop_params,
        cosmo_params,
        fb,
        n_t_table=n_t_table,
    )
    return phot_kern_results, phot_randoms


@partial(jjit, static_argnames=["n_t_table"])
def _phot_kern(
    phot_randoms,
    sfh_params,
    z_obs,
    t_obs,
    mah_params,
    ssp_data,
    precomputed_ssp_mag_table,
    z_phot_table,
    wave_eff_table,
    mzr_params,
    spspop_params,
    scatter_params,
    ssp_err_pop_params,
    cosmo_params,
    fb,
    n_t_table=mcdw.N_T_TABLE,
):
    """Populate the input lightcone with galaxy SEDs"""

    t_table, sfh_table, logsm_obs, logssfr_obs = mcdw.compute_diffstar_info(
        mah_params, sfh_params, t_obs, cosmo_params, fb, n_t_table
    )

    age_weights_smooth, lgmet_weights = sspwk.get_smooth_ssp_weights(
        t_table, sfh_table, logsm_obs, ssp_data, t_obs, mzr_params, LGMET_SCATTER
    )

    _res = sspwk.compute_burstiness(
        phot_randoms.uran_pburst,
        phot_randoms.mc_is_q,
        logsm_obs,
        logssfr_obs,
        age_weights_smooth,
        lgmet_weights,
        ssp_data,
        spspop_params.burstpop_params,
    )
    ssp_weights, burst_params, mc_sfh_type = _res

    # Interpolate SSP mag table to z_obs of each galaxy
    photmag_table_galpop = photerp.interpolate_ssp_photmag_table(
        z_obs, z_phot_table, precomputed_ssp_mag_table
    )
    ssp_photflux_table = 10 ** (-0.4 * photmag_table_galpop)

    # For each filter, calculate λ_eff in the restframe of each galaxy
    wave_eff_galpop = interp_vmap2(z_obs, z_phot_table, wave_eff_table)

    dust_frac_trans, dust_params = sspwk.compute_dust_attenuation(
        phot_randoms.uran_av,
        phot_randoms.uran_delta,
        phot_randoms.uran_funo,
        logsm_obs,
        logssfr_obs,
        ssp_data,
        z_obs,
        wave_eff_galpop,
        spspop_params.dustpop_params,
        scatter_params,
    )
    # dust_frac_trans.shape = (n_gals, n_bands, n_age)

    # Throw out redundant dust params repeated at each λ_eff
    dust_params = dust_params._replace(
        av=dust_params.av[:, 0, -1],
        delta=dust_params.delta[:, 0],
        funo=dust_params.funo[:, 0],
    )

    # Calculate mean fractional change to the SSP fluxes in each band for each galaxy
    # L'_SSP(λ_eff) = L_SSP(λ_eff) & F_SSP(λ_eff)
    frac_ssp_errors_nonoise = ssp_err_model.frac_ssp_err_at_z_obs_galpop(
        ssp_err_pop_params, logsm_obs, z_obs, wave_eff_galpop
    )
    frac_ssp_errors = ssp_err_model.get_noisy_frac_ssp_errors(
        wave_eff_galpop, frac_ssp_errors_nonoise, phot_randoms.delta_mag_ssp_scatter
    )

    obs_mags = sspwk._compute_obs_mags_from_weights(
        logsm_obs, dust_frac_trans, frac_ssp_errors, ssp_photflux_table, ssp_weights
    )

    phot_kern_results = PhotKernResults(
        obs_mags,
        t_table,
        *sfh_params,
        sfh_table,
        logsm_obs,
        logssfr_obs,
        mc_sfh_type,
        ssp_weights,
        lgmet_weights,
        *burst_params,
        *dust_params,
        dust_frac_trans,
        ssp_photflux_table,
        frac_ssp_errors,
        wave_eff_galpop,
    )
    return phot_kern_results


@partial(jjit, static_argnames=["n_t_table"])
def _mc_specphot_kern(
    ran_key,
    z_obs,
    t_obs,
    mah_params,
    ssp_data,
    precomputed_ssp_mag_table,
    ssp_lineflux_table,
    z_phot_table,
    wave_eff_table,
    diffstarpop_params,
    mzr_params,
    spspop_params,
    scatter_params,
    ssp_err_pop_params,
    cosmo_params,
    fb,
    n_t_table=mcdw.N_T_TABLE,
):
    phot_kern_results, phot_randoms = _mc_phot_kern(
        ran_key,
        z_obs,
        t_obs,
        mah_params,
        ssp_data,
        precomputed_ssp_mag_table,
        z_phot_table,
        wave_eff_table,
        diffstarpop_params,
        mzr_params,
        spspop_params,
        scatter_params,
        ssp_err_pop_params,
        cosmo_params,
        fb,
        n_t_table=n_t_table,
    )

    gal_linefluxes = sspwk._compute_obs_flux_from_weights(
        phot_kern_results.logsm_obs,
        dust_frac_trans,
        frac_ssp_errors,
        ssp_lineflux_table,
        phot_kern_results.ssp_weights,
    )
    return phot_kern_results, phot_randoms, gal_linefluxes


@jjit
def _mc_dbk_kern(
    t_obs, ssp_data, t_table, sfh_table, burst_params, lgmet_weights, dbk_key
):
    n_gals = t_obs.shape[0]
    dbk_randoms = get_mc_dbk_randoms(dbk_key, n_gals)
    dbk_weights, disk_bulge_history = _dbk_kern(
        t_obs, ssp_data, t_table, sfh_table, burst_params, lgmet_weights, dbk_randoms
    )
    return dbk_randoms, dbk_weights, disk_bulge_history


@partial(jjit, static_argnames=["n_t_table"])
def _sed_kern(
    phot_randoms,
    sfh_params,
    z_obs,
    t_obs,
    mah_params,
    ssp_data,
    mzr_params,
    spspop_params,
    scatter_params,
    ssp_err_pop_params,
    cosmo_params,
    fb,
    n_t_table=mcdw.N_T_TABLE,
):
    """"""

    t_table, sfh_table, logsm_obs, logssfr_obs = mcdw.compute_diffstar_info(
        mah_params, sfh_params, t_obs, cosmo_params, fb, n_t_table
    )

    age_weights_smooth, lgmet_weights = sspwk.get_smooth_ssp_weights(
        t_table, sfh_table, logsm_obs, ssp_data, t_obs, mzr_params, LGMET_SCATTER
    )

    _res = sspwk.compute_burstiness(
        phot_randoms.uran_pburst,
        phot_randoms.mc_is_q,
        logsm_obs,
        logssfr_obs,
        age_weights_smooth,
        lgmet_weights,
        ssp_data,
        spspop_params.burstpop_params,
    )
    ssp_weights, burst_params, mc_sfh_type = _res

    # For each filter, calculate λ_eff in the restframe of each galaxy
    n_gals = logsm_obs.shape[0]
    wave_eff_galpop = jnp.tile(ssp_data.ssp_wave, n_gals).reshape((n_gals, -1))

    dust_frac_trans, dust_params = sspwk.compute_dust_attenuation(
        phot_randoms.uran_av,
        phot_randoms.uran_delta,
        phot_randoms.uran_funo,
        logsm_obs,
        logssfr_obs,
        ssp_data,
        z_obs,
        wave_eff_galpop,
        spspop_params.dustpop_params,
        scatter_params,
    )
    dust_params = dust_params._replace(
        av=dust_params.av[:, 0, -1],
        delta=dust_params.delta[:, 0],
        funo=dust_params.funo[:, 0],
    )

    # Calculate mean fractional change to the SSP fluxes in each band for each galaxy
    # L'_SSP(λ_eff) = L_SSP(λ_eff) & F_SSP(λ_eff)
    frac_ssp_errors = ssp_err_model.frac_ssp_err_at_z_obs_galpop(
        ssp_err_pop_params, logsm_obs, z_obs, wave_eff_galpop
    )
    frac_ssp_errors = ssp_err_model.get_noisy_frac_ssp_errors(
        wave_eff_galpop, frac_ssp_errors, phot_randoms.delta_mag_ssp_scatter
    )

    n_met, n_age, n_wave = ssp_data.ssp_flux.shape
    dust_frac_trans = dust_frac_trans.swapaxes(1, 2)  # (n_gals, n_age, n_wave)

    a = dust_frac_trans.reshape((n_gals, 1, n_age, n_wave))
    b = frac_ssp_errors.reshape((n_gals, 1, 1, n_wave))
    c = ssp_weights.reshape((n_gals, n_met, n_age, 1))
    d = ssp_data.ssp_flux.reshape((1, n_met, n_age, n_wave))
    mstar = 10 ** logsm_obs.reshape((n_gals, 1))
    rest_sed = jnp.sum(a * b * c * d, axis=(1, 2)) * mstar

    sed_kern_results = (rest_sed, dust_frac_trans, frac_ssp_errors, ssp_weights)
    return sed_kern_results


@partial(jjit, static_argnames=["n_gals"])
def get_mc_dbk_randoms(dbk_key, n_gals):
    fknot = jran.uniform(
        dbk_key, minval=0, maxval=disk_knots.FKNOT_MAX, shape=(n_gals,)
    )
    return DBKRandoms(fknot=fknot)


@jjit
def _get_dbk_phot_from_dbk_weights(
    ssp_photflux_table, dbk_weights, dust_frac_trans, frac_ssp_err
):
    n_gals, n_bands, n_met, n_age = ssp_photflux_table.shape

    # Reshape arrays before calculating galaxy magnitudes
    _ftrans = dust_frac_trans.reshape((n_gals, n_bands, 1, n_age))

    _ferr_ssp = frac_ssp_err.reshape((n_gals, n_bands, 1, 1))

    _w_bulge = dbk_weights.ssp_weights_bulge.reshape((n_gals, 1, n_met, n_age))
    _w_dd = dbk_weights.ssp_weights_disk.reshape((n_gals, 1, n_met, n_age))
    _w_knot = dbk_weights.ssp_weights_knots.reshape((n_gals, 1, n_met, n_age))

    _mstar_bulge = dbk_weights.mstar_bulge.reshape((n_gals, 1))
    _mstar_disk = dbk_weights.mstar_disk.reshape((n_gals, 1))
    _mstar_knots = dbk_weights.mstar_knots.reshape((n_gals, 1))

    integrand_bulge = ssp_photflux_table * _w_bulge * _ftrans * _ferr_ssp
    flux_bulge = jnp.sum(integrand_bulge, axis=(2, 3)) * _mstar_bulge
    obs_mags_bulge = -2.5 * jnp.log10(flux_bulge)

    integrand_disk = ssp_photflux_table * _w_dd * _ftrans * _ferr_ssp
    flux_disk = jnp.sum(integrand_disk, axis=(2, 3)) * _mstar_disk
    obs_mags_disk = -2.5 * jnp.log10(flux_disk)

    integrand_knots = ssp_photflux_table * _w_knot * _ftrans * _ferr_ssp
    flux_knots = jnp.sum(integrand_knots, axis=(2, 3)) * _mstar_knots
    obs_mags_knots = -2.5 * jnp.log10(flux_knots)

    return obs_mags_bulge, obs_mags_disk, obs_mags_knots


def _mc_dbk_phot_kern(
    ran_key,
    z_obs,
    t_obs,
    mah_params,
    ssp_data,
    precomputed_ssp_mag_table,
    z_phot_table,
    wave_eff_table,
    diffstarpop_params,
    mzr_params,
    spspop_params,
    scatter_params,
    ssp_err_pop_params,
    cosmo_params,
    fb,
):
    phot_key, dbk_key = jran.split(ran_key, 2)
    phot_kern_results, phot_randoms = _mc_phot_kern(
        phot_key,
        z_obs,
        t_obs,
        mah_params,
        ssp_data,
        precomputed_ssp_mag_table,
        z_phot_table,
        wave_eff_table,
        diffstarpop_params,
        mzr_params,
        spspop_params,
        scatter_params,
        ssp_err_pop_params,
        cosmo_params,
        fb,
    )

    burst_params = DEFAULT_BURST_PARAMS._replace(
        lgfburst=phot_kern_results.lgfburst,
        lgyr_peak=phot_kern_results.lgyr_peak,
        lgyr_max=phot_kern_results.lgyr_max,
    )
    _ret2 = _mc_dbk_kern(
        t_obs,
        ssp_data,
        phot_kern_results.t_table,
        phot_kern_results.sfh_table,
        burst_params,
        phot_kern_results.lgmet_weights,
        dbk_key,
    )
    dbk_randoms, dbk_weights, disk_bulge_history = _ret2

    _ret3 = _get_dbk_phot_from_dbk_weights(
        phot_kern_results.ssp_photflux_table,
        dbk_weights,
        phot_kern_results.dust_frac_trans,
        phot_kern_results.frac_ssp_errors,
    )
    obs_mags_bulge, obs_mags_disk, obs_mags_knots = _ret3

    dbk_phot_info = MCDBKPhotInfo(
        **phot_kern_results._asdict(),
        **phot_randoms._asdict(),
        **dbk_randoms._asdict(),
        **disk_bulge_history.fbulge_params._asdict(),
        bulge_to_total_history=disk_bulge_history.bulge_to_total_history,
        obs_mags_bulge=obs_mags_bulge,
        obs_mags_disk=obs_mags_disk,
        obs_mags_knots=obs_mags_knots,
    )
    return dbk_phot_info, dbk_weights


def _mc_lc_dbk_sed_kern(
    dbk_phot_info,
    dbk_weights,
    z_obs,
    ssp_data,
    spspop_params,
    scatter_params,
    ssperr_params,
):
    n_gals = dbk_phot_info.logsm_obs.shape[0]
    wave_eff_galpop = jnp.tile(ssp_data.ssp_wave, n_gals).reshape((n_gals, -1))

    dust_frac_trans, dust_params = sspwk.compute_dust_attenuation(
        dbk_phot_info.uran_av,
        dbk_phot_info.uran_delta,
        dbk_phot_info.uran_funo,
        dbk_phot_info.logsm_obs,
        dbk_phot_info.logssfr_obs,
        ssp_data,
        z_obs,
        wave_eff_galpop,
        spspop_params.dustpop_params,
        scatter_params,
    )
    dust_params = dust_params._replace(
        av=dust_params.av[:, 0, -1],
        delta=dust_params.delta[:, 0],
        funo=dust_params.funo[:, 0],
    )

    # Calculate mean fractional change to the SSP fluxes in each band for each galaxy
    # L'_SSP(λ_eff) = L_SSP(λ_eff) & F_SSP(λ_eff)
    frac_ssp_errors_nonoise = ssp_err_model.frac_ssp_err_at_z_obs_galpop(
        ssperr_params, dbk_phot_info.logsm_obs, z_obs, wave_eff_galpop
    )
    frac_ssp_errors = ssp_err_model.get_noisy_frac_ssp_errors(
        wave_eff_galpop, frac_ssp_errors_nonoise, dbk_phot_info.delta_mag_ssp_scatter
    )

    n_met, n_age, n_wave = ssp_data.ssp_flux.shape
    dust_frac_trans = dust_frac_trans.swapaxes(1, 2)  # (n_gals, n_age, n_wave)

    a = dust_frac_trans.reshape((n_gals, 1, n_age, n_wave))
    b = frac_ssp_errors.reshape((n_gals, 1, 1, n_wave))
    d = ssp_data.ssp_flux.reshape((1, n_met, n_age, n_wave))

    _w_bulge = dbk_weights.ssp_weights_bulge.reshape((n_gals, n_met, n_age, 1))
    _w_dd = dbk_weights.ssp_weights_disk.reshape((n_gals, n_met, n_age, 1))
    _w_knot = dbk_weights.ssp_weights_knots.reshape((n_gals, n_met, n_age, 1))

    mb = dbk_phot_info.mstar_bulge.reshape((n_gals, 1))
    md = dbk_phot_info.mstar_disk.reshape((n_gals, 1))
    mk = dbk_phot_info.mstar_knots.reshape((n_gals, 1))

    sed_bulge = jnp.sum(a * b * _w_bulge * d, axis=(1, 2)) * mb
    sed_disk = jnp.sum(a * b * _w_dd * d, axis=(1, 2)) * md
    sed_knots = jnp.sum(a * b * _w_knot * d, axis=(1, 2)) * mk

    return sed_bulge, sed_disk, sed_knots


DBKRandoms = namedtuple("DBKRandoms", ("fknot",))
PHOT_KERN_KEYS = (
    "obs_mags",
    "t_table",
    *DEFAULT_DIFFSTAR_PARAMS._fields,
    "sfh_table",
    "logsm_obs",
    "logssfr_obs",
    "mc_sfh_type",
    "ssp_weights",
    "lgmet_weights",
    *DEFAULT_BURST_PARAMS._fields,
    *DEFAULT_DUST_PARAMS._fields,
    "dust_frac_trans",
    "ssp_photflux_table",
    "frac_ssp_errors",
    "wave_eff_galpop",
)
PhotKernResults = namedtuple("PhotKernResults", PHOT_KERN_KEYS)

DBK_EXTRA_FIELDS = (
    *dbk.FbulgeParams._fields,
    "bulge_to_total_history",
    "obs_mags_bulge",
    "obs_mags_disk",
    "obs_mags_knots",
)
MCDBKPhotInfo = namedtuple(
    "MCDBKPhotInfo",
    (
        *PhotKernResults._fields,
        *PhotRandoms._fields,
        *DBKRandoms._fields,
        *DBK_EXTRA_FIELDS,
    ),
)
