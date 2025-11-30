# flake8: noqa: E402
""""""

from jax import config

config.update("jax_enable_x64", True)

from collections import namedtuple
from functools import partial

from dsps.constants import T_TABLE_MIN
from dsps.cosmology import flat_wcdm
from jax import jit as jjit
from jax import numpy as jnp
from jax import random as jran

from ..ssp_err_model import ssp_err_model
from . import lc_phot_kern
from . import mc_diffstarpop_wrappers as mcdw
from . import photometry_interpolation as photerp
from .disk_bulge_modeling import disk_bulge_kernels as dbk
from .disk_bulge_modeling import disk_knots
from .disk_bulge_modeling import mc_disk_bulge as mcdb
from .kernels import dbk_kernels
from .kernels.ssp_weight_kernels_repro import (
    MCPhotInfo,
    _compute_obs_mags_from_weights,
    compute_burstiness,
    compute_dust_attenuation,
    compute_frac_ssp_errors,
    get_burstiness_randoms,
    get_dust_randoms,
    get_smooth_ssp_weights,
)

LGMET_SCATTER = 0.2

PHOT_RAN_KEYS = (
    "mc_is_q",
    "uran_av",
    "uran_delta",
    "uran_funo",
    "uran_pburst",
    "delta_mag_ssp_scatter",
)
PhotRandoms = namedtuple("PhotRandoms", PHOT_RAN_KEYS)


@jjit
def get_mc_phot_randoms(
    ran_key, diffstarpop_params, mah_params, cosmo_params, precomputed_ssp_mag_table
):
    n_gals = mah_params.logm0.size
    n_bands = precomputed_ssp_mag_table.shape[1]

    # Monte Carlo diffstar params
    ran_key, sfh_key = jran.split(ran_key, 2)
    sfh_params, mc_is_q = mcdw.mc_diffstarpop_cens_wrapper(
        diffstarpop_params, sfh_key, mah_params, cosmo_params
    )
    # Generate randoms for stochasticity in dust attenuation curves
    ran_key, dust_key = jran.split(ran_key, 2)
    dust_randoms = get_dust_randoms(dust_key, n_gals)

    # Randoms for burstiness
    ran_key, burst_key = jran.split(ran_key, 2)
    uran_pburst = get_burstiness_randoms(burst_key, n_gals)

    # Scatter for SSP errors
    ran_key, ssp_key = jran.split(ran_key, 2)
    ZZ = jnp.zeros((n_gals, n_bands))
    delta_mag_ssp_scatter = ssp_err_model.compute_delta_scatter(ssp_key, ZZ)

    phot_randoms = PhotRandoms(
        mc_is_q,
        dust_randoms.uran_av,
        dust_randoms.uran_delta,
        dust_randoms.uran_funo,
        uran_pburst,
        delta_mag_ssp_scatter,
    )
    return phot_randoms, sfh_params


@partial(jjit, static_argnames=["n_t_table"])
def _mc_phot_kern(
    ran_key,
    diffstarpop_params,
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
    phot_randoms, sfh_params = get_mc_phot_randoms(
        ran_key, diffstarpop_params, mah_params, cosmo_params, precomputed_ssp_mag_table
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
    return phot_kern_results


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

    age_weights_smooth, lgmet_weights = get_smooth_ssp_weights(
        t_table, sfh_table, logsm_obs, ssp_data, t_obs, mzr_params, LGMET_SCATTER
    )

    _res = compute_burstiness(
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
    wave_eff_galpop = lc_phot_kern.interp_vmap2(z_obs, z_phot_table, wave_eff_table)

    frac_trans, dust_params = compute_dust_attenuation(
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

    # Calculate mean fractional change to the SSP fluxes in each band for each galaxy
    # L'_SSP(λ_eff) = L_SSP(λ_eff) & F_SSP(λ_eff)
    frac_ssp_errors = compute_frac_ssp_errors(
        ssp_err_pop_params, z_obs, logsm_obs, wave_eff_galpop
    )

    obs_mags = _compute_obs_mags_from_weights(
        logsm_obs,
        frac_trans,
        frac_ssp_errors,
        ssp_photflux_table,
        ssp_weights,
        phot_randoms.delta_mag_ssp_scatter,
    )

    return (
        obs_mags,
        mc_sfh_type,
        ssp_weights,
        burst_params,
        dust_params,
        ssp_photflux_table,
        frac_ssp_errors,
    )


def mc_dbk_phot(
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
    _ret = _mc_phot_kern(
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
    phot_info, ssp_weights_smooth = _ret[:2]
    dust_att, ssp_photflux_table = _ret[3:5]
    frac_ssp_errors, delta_scatter_ms, delta_scatter_q = _ret[5:]
    _ret2 = _mc_dbk_kern(t_obs, ssp_data, phot_info, ssp_weights_smooth, dbk_key)
    dbk_weights, disk_bulge_history, fknot = _ret2

    _ret3 = get_dbk_phot(
        ssp_photflux_table,
        dbk_weights,
        dust_att,
        phot_info,
        frac_ssp_errors,
        delta_scatter_ms,
        delta_scatter_q,
    )
    obs_mags_bulge, obs_mags_disk, obs_mags_knots = _ret3

    dbk_phot_info = MCDBKPhotInfo(
        **phot_info._asdict(),
        **disk_bulge_history.fbulge_params._asdict(),
        bulge_to_total_history=disk_bulge_history.bulge_to_total_history,
        obs_mags_bulge=obs_mags_bulge,
        obs_mags_disk=obs_mags_disk,
        obs_mags_knots=obs_mags_knots,
        fknot=fknot,
    )
    return dbk_phot_info


@jjit
def _mc_dbk_kern(t_obs, ssp_data, phot_info, ssp_weights_smooth, dbk_key):
    disk_bulge_history = mcdb.decompose_sfh_into_disk_bulge_sfh(
        phot_info.t_table, phot_info.sfh_table
    )
    n_gals = t_obs.size
    fknot = jran.uniform(
        dbk_key, minval=0, maxval=disk_knots.FKNOT_MAX, shape=(n_gals,)
    )

    dbk_weights = dbk_kernels.get_dbk_weights(
        t_obs, ssp_data, phot_info, ssp_weights_smooth, disk_bulge_history, fknot
    )

    return dbk_weights, disk_bulge_history, fknot


@jjit
def get_dbk_phot(
    ssp_photflux_table,
    dbk_weights,
    dust_att,
    phot_info,
    frac_ssp_errors,
    delta_scatter_ms,
    delta_scatter_q,
):
    n_gals, n_bands, n_met, n_age = ssp_photflux_table.shape

    # Reshape arrays before calculating galaxy magnitudes
    _mc_q = phot_info.mc_sfh_type.reshape((n_gals, 1, 1, 1)) == 0
    _ftrans_ms = dust_att.frac_trans.ms.reshape((n_gals, n_bands, 1, n_age))
    _ftrans_q = dust_att.frac_trans.q.reshape((n_gals, n_bands, 1, n_age))
    _ftrans = jnp.where(_mc_q, _ftrans_q, _ftrans_ms)

    # Calculate fractional changes to SSP fluxes
    frac_ssp_err_ms = frac_ssp_errors.ms * 10 ** (-0.4 * delta_scatter_ms)
    frac_ssp_err_q = frac_ssp_errors.q * 10 ** (-0.4 * delta_scatter_q)

    _ferr_ssp_ms = frac_ssp_err_ms.reshape((n_gals, n_bands, 1, 1))
    _ferr_ssp_q = frac_ssp_err_q.reshape((n_gals, n_bands, 1, 1))
    _ferr_ssp = jnp.where(_mc_q, _ferr_ssp_q, _ferr_ssp_ms)

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


DBK_EXTRA_FIELDS = (
    *dbk.FbulgeParams._fields,
    "bulge_to_total_history",
    "obs_mags_bulge",
    "obs_mags_disk",
    "obs_mags_knots",
    "fknot",
)
MCDBKPhotInfo = namedtuple("MCDBKPhotInfo", (*MCPhotInfo._fields, *DBK_EXTRA_FIELDS))
