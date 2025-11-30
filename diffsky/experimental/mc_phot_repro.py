# flake8: noqa: E402
""""""

from jax import config

config.update("jax_enable_x64", True)


from collections import namedtuple

from diffstar.diffstarpop import mc_diffstar_params_galpop
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
from .kernels.ssp_weight_kernels import (
    MCPhotInfo,
    compute_burstiness,
    compute_dust_attenuation,
    compute_frac_ssp_errors,
    compute_mc_realization,
    compute_obs_mags_ms_q,
    get_dust_randoms,
    get_smooth_ssp_weights,
)
from .mc_diffstarpop_wrappers import N_T_TABLE

LGMET_SCATTER = 0.2


@jjit
def _get_diffmah_quantities(mah_params):
    pass


@jjit
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
    n_t_table=N_T_TABLE,
):
    """Populate the input lightcone with galaxy SEDs"""

    # Monte Carlo diffstar params
    ran_key, sfh_key = jran.split(ran_key, 2)
    sfh_params, mc_is_q = mcdw.mc_diffstarpop_cens_wrapper(
        diffstarpop_params, sfh_key, mah_params, cosmo_params
    )
    # Generate randoms for stochasticity in dust attenuation curves
    ran_key, dust_key = jran.split(ran_key, 2)
    dust_randoms = get_dust_randoms(dust_key, z_obs)

    # Scatter for SSP errors
    ran_key, ssp_key = jran.split(ran_key, 2)
    # delta_mag_ssp_scatter = ssp_err_model.compute_delta_scatter(
    #     ssp_key, frac_ssp_errors.ms
    # )

    smooth_ssp_weights = get_smooth_ssp_weights(
        diffstar_galpop, ssp_data, t_obs, mzr_params, LGMET_SCATTER
    )

    burstiness = compute_burstiness(
        diffstar_galpop, smooth_ssp_weights, ssp_data, spspop_params.burstpop_params
    )

    # Interpolate SSP mag table to z_obs of each galaxy
    photmag_table_galpop = photerp.interpolate_ssp_photmag_table(
        z_obs, z_phot_table, precomputed_ssp_mag_table
    )
    ssp_photflux_table = 10 ** (-0.4 * photmag_table_galpop)

    # For each filter, calculate λ_eff in the restframe of each galaxy
    wave_eff_galpop = lc_phot_kern.interp_vmap2(z_obs, z_phot_table, wave_eff_table)

    dust_att = compute_dust_attenuation(
        dust_randoms.uran_av,
        dust_randoms.uran_delta,
        dust_randoms.uran_funo,
        diffstar_galpop,
        ssp_data,
        z_obs,
        wave_eff_galpop,
        spspop_params.dustpop_params,
        scatter_params,
    )

    # Calculate mean fractional change to the SSP fluxes in each band for each galaxy
    # L'_SSP(λ_eff) = L_SSP(λ_eff) & F_SSP(λ_eff)
    frac_ssp_errors = compute_frac_ssp_errors(
        ssp_err_pop_params, z_obs, diffstar_galpop, wave_eff_galpop
    )

    obs_mags = compute_obs_mags_ms_q(
        diffstar_galpop,
        dust_att,
        frac_ssp_errors,
        ssp_photflux_table,
        smooth_ssp_weights.weights.ms,
        burstiness.weights.ms,
        smooth_ssp_weights.weights.q,
        delta_scatter_ms,
        delta_scatter_q,
    )
    phot_info = compute_mc_realization(
        diffstar_galpop,
        burstiness,
        smooth_ssp_weights,
        dust_att,
        obs_mags,
        delta_scatter_ms,
        delta_scatter_q,
        ran_key,
    )

    return (
        phot_info,
        smooth_ssp_weights,
        burstiness,
        dust_att,
        ssp_photflux_table,
        frac_ssp_errors,
        delta_scatter_ms,
        delta_scatter_q,
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
    phot_info, smooth_ssp_weights = _ret[:2]
    dust_att, ssp_photflux_table = _ret[3:5]
    frac_ssp_errors, delta_scatter_ms, delta_scatter_q = _ret[5:]
    _ret2 = _mc_dbk_kern(t_obs, ssp_data, phot_info, smooth_ssp_weights, dbk_key)
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
def _mc_dbk_kern(t_obs, ssp_data, phot_info, smooth_ssp_weights, dbk_key):
    disk_bulge_history = mcdb.decompose_sfh_into_disk_bulge_sfh(
        phot_info.t_table, phot_info.sfh_table
    )
    n_gals = t_obs.size
    fknot = jran.uniform(
        dbk_key, minval=0, maxval=disk_knots.FKNOT_MAX, shape=(n_gals,)
    )

    dbk_weights = dbk_kernels.get_dbk_weights(
        t_obs, ssp_data, phot_info, smooth_ssp_weights, disk_bulge_history, fknot
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
