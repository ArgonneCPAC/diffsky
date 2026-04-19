""""""

from collections import namedtuple

from dsps.sfh.diffburst import DEFAULT_BURST_PARAMS
from jax import jit as jjit
from jax import numpy as jnp
from jax import random as jran

from ...ssp_err_model import ssp_err_model
from ..disk_bulge_modeling import disk_bulge_kernels as dbk
from . import dbk_kernels, linelum_kernels, mc_randoms, phot_kernels
from . import ssp_weight_kernels as sspwk


@jjit
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
    phot_kern_results, phot_randoms = phot_kernels._mc_phot_kern(
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
    _ret2 = dbk_kernels._mc_dbk_kern(
        t_obs,
        ssp_data,
        phot_kern_results.t_table,
        phot_kern_results.sfh_table,
        burst_params,
        phot_kern_results.lgmet_weights,
        dbk_key,
    )
    dbk_randoms, dbk_weights, disk_bulge_history = _ret2

    _ret3 = dbk_kernels._get_dbk_phot_from_dbk_weights(
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


@jjit
def _mc_dbk_specphot_kern(
    ran_key,
    z_obs,
    t_obs,
    mah_params,
    ssp_data,
    precomputed_ssp_mag_table,
    z_phot_table,
    wave_eff_table,
    line_wave_table,
    diffstarpop_params,
    mzr_params,
    spspop_params,
    scatter_params,
    ssp_err_pop_params,
    cosmo_params,
    fb,
):
    phot_key, dbk_key = jran.split(ran_key, 2)
    phot_kern_results, phot_randoms, spec_kern_results = (
        linelum_kernels._mc_specphot_kern(
            phot_key,
            z_obs,
            t_obs,
            mah_params,
            ssp_data,
            precomputed_ssp_mag_table,
            z_phot_table,
            wave_eff_table,
            line_wave_table,
            diffstarpop_params,
            mzr_params,
            spspop_params,
            scatter_params,
            ssp_err_pop_params,
            cosmo_params,
            fb,
        )
    )

    burst_params = DEFAULT_BURST_PARAMS._replace(
        lgfburst=phot_kern_results.lgfburst,
        lgyr_peak=phot_kern_results.lgyr_peak,
        lgyr_max=phot_kern_results.lgyr_max,
    )
    _ret2 = dbk_kernels._mc_dbk_kern(
        t_obs,
        ssp_data,
        phot_kern_results.t_table,
        phot_kern_results.sfh_table,
        burst_params,
        phot_kern_results.lgmet_weights,
        dbk_key,
    )
    dbk_randoms, dbk_weights, disk_bulge_history = _ret2

    _ret3 = dbk_kernels._get_dbk_phot_from_dbk_weights(
        phot_kern_results.ssp_photflux_table,
        dbk_weights,
        phot_kern_results.dust_frac_trans,
        phot_kern_results.frac_ssp_errors,
    )
    obs_mags_bulge, obs_mags_disk, obs_mags_knots = _ret3

    _dbk_line_res = dbk_kernels._get_dbk_linelum_decomposition(
        dbk_weights, spec_kern_results, ssp_data
    )
    linelum_bulge, linelum_disk, linelum_knots = _dbk_line_res

    dbk_specphot_info = MCDBKSpecPhotInfo(
        **phot_kern_results._asdict(),
        **phot_randoms._asdict(),
        **dbk_randoms._asdict(),
        **disk_bulge_history.fbulge_params._asdict(),
        bulge_to_total_history=disk_bulge_history.bulge_to_total_history,
        obs_mags_bulge=obs_mags_bulge,
        obs_mags_disk=obs_mags_disk,
        obs_mags_knots=obs_mags_knots,
        linelum_gal=spec_kern_results.linelum_gal,
        linelum_bulge=linelum_bulge,
        linelum_disk=linelum_disk,
        linelum_knots=linelum_knots,
    )
    return dbk_specphot_info, dbk_weights


@jjit
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


DBK_PHOT_EXTRA_FIELDS = (
    *dbk.FbulgeParams._fields,
    "bulge_to_total_history",
    "obs_mags_bulge",
    "obs_mags_disk",
    "obs_mags_knots",
)
MCDBKPhotInfo = namedtuple(
    "MCDBKPhotInfo",
    (
        *phot_kernels.PhotKernResults._fields,
        *mc_randoms.PhotRandoms._fields,
        *mc_randoms.DBKRandoms._fields,
        *DBK_PHOT_EXTRA_FIELDS,
    ),
)

_dbk_specphot_keys = (
    *MCDBKPhotInfo._fields,
    *("linelum_gal", "linelum_bulge", "linelum_disk", "linelum_knots"),
)
MCDBKSpecPhotInfo = namedtuple("MCDBKSpecPhotInfo", _dbk_specphot_keys)
MCDBKSpecPhotInfo = namedtuple("MCDBKSpecPhotInfo", _dbk_specphot_keys)
