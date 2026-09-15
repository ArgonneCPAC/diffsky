""""""

from collections import namedtuple

from dsps.sfh.diffburst import DEFAULT_BURST_PARAMS
from jax import jit as jjit
from jax import numpy as jnp

from ...merging import merging_model
from ..disk_bulge_modeling import disk_bulge_kernels as dbk
from . import dbk_kernels, linelum_kernels_in_situ, mc_randoms, phot_kernels_in_situ


@jjit
def _mc_dbk_phot_kern(
    ran_key,
    z_obs,
    t_obs,
    mah_params,
    upid,
    lgmu_infall,
    logmhost_infall,
    gyr_since_infall,
    ssp_data,
    precomputed_ssp_mag_table,
    z_phot_table,
    wave_eff_table,
    diffstarpop_params,
    mzr_params,
    spspop_params,
    scatter_params,
    ssperr_params,
    merging_params,
    cosmo_params,
    fb,
):
    phot_randoms, diffstarpop_results, dbk_randoms = mc_randoms.get_dbk_phot_randoms(
        ran_key,
        diffstarpop_params,
        mah_params,
        upid,
        lgmu_infall,
        logmhost_infall,
        gyr_since_infall,
        cosmo_params,
    )

    t_infall = t_obs - gyr_since_infall
    logmp_infall = lgmu_infall + logmhost_infall
    p_merge_smooth = merging_model.get_p_merge_from_merging_params(
        merging_params, logmp_infall, logmhost_infall, t_obs, t_infall, upid
    )

    dbk_phot_info, dbk_weights = _dbk_phot_kern(
        phot_randoms,
        diffstarpop_results,
        dbk_randoms,
        z_obs,
        t_obs,
        mah_params,
        p_merge_smooth,
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

    return dbk_phot_info, dbk_weights


@jjit
def _dbk_phot_kern(
    phot_randoms,
    diffstarpop_results,
    dbk_randoms,
    z_obs,
    t_obs,
    mah_params,
    p_merge_smooth,
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
):
    phot_kern_results = phot_kernels_in_situ._phot_kern(
        phot_randoms,
        diffstarpop_results,
        z_obs,
        t_obs,
        mah_params,
        p_merge_smooth,
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

    burst_params = DEFAULT_BURST_PARAMS._replace(
        lgfburst=phot_kern_results.lgfburst,
        lgyr_peak=phot_kern_results.lgyr_peak,
        lgyr_max=phot_kern_results.lgyr_max,
    )
    age_weights = jnp.sum(phot_kern_results.ssp_weights, axis=1)
    dbk_weights, disk_bulge_history = dbk_kernels._dbk_kern(
        t_obs,
        ssp_data,
        phot_kern_results.t_table,
        phot_kern_results.sfh_table,
        burst_params,
        phot_kern_results.lgmet_weights,
        dbk_randoms,
        phot_kern_results.logsm_obs,
        age_weights,
        p_merge_smooth,
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
        logsm_bulge=jnp.log10(dbk_weights.mstar_bulge),
        logsm_disk=jnp.log10(dbk_weights.mstar_disk),
        logsm_knots=jnp.log10(dbk_weights.mstar_knots),
        obs_mags_bulge=obs_mags_bulge,
        obs_mags_disk=obs_mags_disk,
        obs_mags_knots=obs_mags_knots,
    )
    return dbk_phot_info, dbk_weights


@jjit
def _mc_dbk_photline_kern(
    ran_key,
    z_obs,
    t_obs,
    mah_params,
    upid,
    lgmu_infall,
    logmhost_infall,
    gyr_since_infall,
    ssp_data,
    precomputed_ssp_mag_table,
    z_phot_table,
    wave_eff_table,
    line_wave_table,
    diffstarpop_params,
    mzr_params,
    spspop_params,
    scatter_params,
    ssperr_params,
    merging_params,
    cosmo_params,
    fb,
):
    phot_randoms, diffstarpop_results, dbk_randoms = mc_randoms.get_dbk_phot_randoms(
        ran_key,
        diffstarpop_params,
        mah_params,
        upid,
        lgmu_infall,
        logmhost_infall,
        gyr_since_infall,
        cosmo_params,
    )

    t_infall = t_obs - gyr_since_infall
    logmp_infall = lgmu_infall + logmhost_infall
    p_merge_smooth = merging_model.get_p_merge_from_merging_params(
        merging_params, logmp_infall, logmhost_infall, t_obs, t_infall, upid
    )

    dbk_photline_info, dbk_weights = _dbk_photline_kern(
        phot_randoms,
        diffstarpop_results,
        dbk_randoms,
        z_obs,
        t_obs,
        mah_params,
        p_merge_smooth,
        ssp_data,
        precomputed_ssp_mag_table,
        z_phot_table,
        wave_eff_table,
        line_wave_table,
        mzr_params,
        spspop_params,
        scatter_params,
        ssperr_params,
        cosmo_params,
        fb,
    )
    return dbk_photline_info, dbk_weights


@jjit
def _dbk_photline_kern(
    phot_randoms,
    diffstarpop_results,
    dbk_randoms,
    z_obs,
    t_obs,
    mah_params,
    p_merge_smooth,
    ssp_data,
    precomputed_ssp_mag_table,
    z_phot_table,
    wave_eff_table,
    line_wave_table,
    mzr_params,
    spspop_params,
    scatter_params,
    ssperr_params,
    cosmo_params,
    fb,
):
    phot_kern_results, spec_kern_results = linelum_kernels_in_situ._photline_kern(
        phot_randoms,
        diffstarpop_results,
        z_obs,
        t_obs,
        mah_params,
        p_merge_smooth,
        ssp_data,
        precomputed_ssp_mag_table,
        z_phot_table,
        wave_eff_table,
        line_wave_table,
        mzr_params,
        spspop_params,
        scatter_params,
        ssperr_params,
        cosmo_params,
        fb,
    )
    dbk_phot_info, dbk_weights = _dbk_phot_kern(
        phot_randoms,
        diffstarpop_results,
        dbk_randoms,
        z_obs,
        t_obs,
        mah_params,
        p_merge_smooth,
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

    dbk_photline_info = MCDBKSpecPhotInfo(
        **phot_kern_results._asdict(),
        **phot_randoms._asdict(),
        **dbk_randoms._asdict(),
        **fbulge_params._asdict(),
        bulge_to_total_history=dbk_phot_info.bulge_to_total_history,
        logsm_bulge=jnp.log10(dbk_weights.mstar_bulge),
        logsm_disk=jnp.log10(dbk_weights.mstar_disk),
        logsm_knots=jnp.log10(dbk_weights.mstar_knots),
        obs_mags_bulge=obs_mags_bulge,
        obs_mags_disk=obs_mags_disk,
        obs_mags_knots=obs_mags_knots,
        linelum_gal=spec_kern_results.linelum_gal,
        linelum_weighted=spec_kern_results.linelum_weighted,
        linelum_bulge=linelum_bulge,
        linelum_disk=linelum_disk,
        linelum_knots=linelum_knots,
    )
    return dbk_photline_info, dbk_weights


DBK_PHOT_EXTRA_FIELDS = (
    *dbk.FbulgeParams._fields,
    "bulge_to_total_history",
    "logsm_bulge",
    "logsm_disk",
    "logsm_knots",
    "obs_mags_bulge",
    "obs_mags_disk",
    "obs_mags_knots",
)
MCDBKPhotInfo = namedtuple(
    "MCDBKPhotInfo",
    (
        *phot_kernels_in_situ.PhotKernResults._fields,
        *mc_randoms.PhotRandoms._fields,
        *mc_randoms.DBKRandoms._fields,
        *DBK_PHOT_EXTRA_FIELDS,
    ),
)

_dbk_photline_keys = (
    *MCDBKPhotInfo._fields,
    *(
        "linelum_gal",
        "linelum_weighted",
        "linelum_bulge",
        "linelum_disk",
        "linelum_knots",
    ),
)
MCDBKSpecPhotInfo = namedtuple("MCDBKSpecPhotInfo", _dbk_photline_keys)

DBKSEDInfo = namedtuple(
    "DBKSEDInfo", ("rest_sed_bulge", "rest_sed_disk", "rest_sed_knots")
)
