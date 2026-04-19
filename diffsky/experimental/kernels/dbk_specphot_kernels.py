""""""

from collections import namedtuple

from dsps.sfh.diffburst import DEFAULT_BURST_PARAMS
from jax import jit as jjit
from jax import numpy as jnp

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
    phot_randoms, sfh_params, dbk_randoms = mc_randoms.get_mc_dbk_phot_randoms(
        ran_key, diffstarpop_params, mah_params, cosmo_params
    )
    dbk_phot_info, dbk_weights = _dbk_phot_kern(
        phot_randoms,
        sfh_params,
        dbk_randoms,
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
    )

    return dbk_phot_info, dbk_weights


@jjit
def _dbk_phot_kern(
    phot_randoms,
    sfh_params,
    dbk_randoms,
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
):
    phot_kern_results = phot_kernels._phot_kern(
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
    )

    burst_params = DEFAULT_BURST_PARAMS._replace(
        lgfburst=phot_kern_results.lgfburst,
        lgyr_peak=phot_kern_results.lgyr_peak,
        lgyr_max=phot_kern_results.lgyr_max,
    )
    dbk_weights, disk_bulge_history = dbk_kernels._dbk_kern(
        t_obs,
        ssp_data,
        phot_kern_results.t_table,
        phot_kern_results.sfh_table,
        burst_params,
        phot_kern_results.lgmet_weights,
        dbk_randoms,
    )

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
    phot_randoms, sfh_params, dbk_randoms = mc_randoms.get_mc_dbk_phot_randoms(
        ran_key, diffstarpop_params, mah_params, cosmo_params
    )
    dbk_specphot_info, dbk_weights = _dbk_specphot_kern(
        phot_randoms,
        sfh_params,
        dbk_randoms,
        z_obs,
        t_obs,
        mah_params,
        ssp_data,
        precomputed_ssp_mag_table,
        z_phot_table,
        wave_eff_table,
        line_wave_table,
        mzr_params,
        spspop_params,
        scatter_params,
        ssp_err_pop_params,
        cosmo_params,
        fb,
    )
    return dbk_specphot_info, dbk_weights


@jjit
def _dbk_specphot_kern(
    phot_randoms,
    sfh_params,
    dbk_randoms,
    z_obs,
    t_obs,
    mah_params,
    ssp_data,
    precomputed_ssp_mag_table,
    z_phot_table,
    wave_eff_table,
    line_wave_table,
    mzr_params,
    spspop_params,
    scatter_params,
    ssp_err_pop_params,
    cosmo_params,
    fb,
):
    phot_kern_results, spec_kern_results = linelum_kernels._specphot_kern(
        phot_randoms,
        sfh_params,
        z_obs,
        t_obs,
        mah_params,
        ssp_data,
        precomputed_ssp_mag_table,
        z_phot_table,
        wave_eff_table,
        line_wave_table,
        mzr_params,
        spspop_params,
        scatter_params,
        ssp_err_pop_params,
        cosmo_params,
        fb,
    )
    dbk_phot_info, dbk_weights = _dbk_phot_kern(
        phot_randoms,
        sfh_params,
        dbk_randoms,
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
    )

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

    fbulge_params = dbk.FbulgeParams._make(
        [getattr(dbk_phot_info, p) for p in dbk.FbulgeParams._fields]
    )

    dbk_specphot_info = MCDBKSpecPhotInfo(
        **phot_kern_results._asdict(),
        **phot_randoms._asdict(),
        **dbk_randoms._asdict(),
        **fbulge_params._asdict(),
        bulge_to_total_history=dbk_phot_info.bulge_to_total_history,
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
def _dbk_sed_kern(
    mc_is_q,
    uran_av,
    uran_delta,
    uran_funo,
    uran_pburst,
    delta_mag_ssp_scatter,
    uran_fbulge,
    fknot,
    sfh_params,
    z_obs,
    t_obs,
    mah_params,
    ssp_data,
    mzr_params,
    spspop_params,
    scatter_params,
    ssperr_params,
    cosmo_params,
    fb,
):
    phot_randoms = mc_randoms.PhotRandoms(
        mc_is_q,
        uran_av,
        uran_delta,
        uran_funo,
        uran_pburst,
        delta_mag_ssp_scatter,
    )
    dbk_randoms = mc_randoms.DBKRandoms(fknot=fknot, uran_fbulge=uran_fbulge)
    n_gals = z_obs.size
    n_bands = 1
    n_z = 2
    z_phot_table = jnp.linspace(z_obs.min(), z_obs.max(), n_z)
    n_met = ssp_data.ssp_lgmet.size
    n_age = ssp_data.ssp_lg_age_gyr.size
    precomputed_ssp_mag_table = jnp.ones((n_z, n_bands, n_met, n_age))
    wave_eff_table = jnp.ones((n_z, n_bands))

    dbk_phot_info, dbk_weights = _dbk_phot_kern(
        phot_randoms,
        sfh_params,
        dbk_randoms,
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
        ssperr_params,
        cosmo_params,
        fb,
    )

    n_gals = dbk_phot_info.logsm_obs.shape[0]
    wave_eff_galpop = jnp.tile(ssp_data.ssp_wave, n_gals).reshape((n_gals, -1))

    dust_frac_trans, __ = sspwk.compute_dust_attenuation(
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

    mb = dbk_weights.mstar_bulge.reshape((n_gals, 1))
    md = dbk_weights.mstar_disk.reshape((n_gals, 1))
    mk = dbk_weights.mstar_knots.reshape((n_gals, 1))

    sed_bulge = jnp.sum(a * b * _w_bulge * d, axis=(1, 2)) * mb
    sed_disk = jnp.sum(a * b * _w_dd * d, axis=(1, 2)) * md
    sed_knots = jnp.sum(a * b * _w_knot * d, axis=(1, 2)) * mk

    sed_info = DBKSEDInfo(sed_bulge, sed_disk, sed_knots)

    return sed_info, dbk_weights


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

DBKSEDInfo = namedtuple("DBKSEDInfo", ("sed_bulge", "sed_disk", "sed_knots"))
