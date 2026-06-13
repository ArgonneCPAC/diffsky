""""""

from collections import namedtuple

from jax import jit as jjit

from ...merging import merging_model
from . import phot_kernels, mc_randoms
from . import ssp_weight_kernels as sspwk


@jjit
def _mc_specphot_kern(
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
    phot_randoms, diffstarpop_results = mc_randoms.get_phot_randoms(
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

    phot_kern_results, spec_kern_results = _specphot_kern(
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

    return phot_kern_results, phot_randoms, spec_kern_results


@jjit
def _specphot_kern(
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
):
    phot_kern_results = phot_kernels._phot_kern(
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

    _dust_res_mc = sspwk.compute_dust_attenuation_lines(
        phot_randoms.uran_av,
        phot_randoms.uran_delta,
        phot_randoms.uran_funo,
        phot_kern_results.logsm_obs,
        phot_kern_results.logssfr_obs,
        ssp_data,
        z_obs,
        line_wave_table,
        spspop_params.dustpop_params,
        scatter_params,
    )
    dust_ftrans_lines_mc = _dust_res_mc[0]

    linelum_gal = sspwk._compute_linelum_from_weights(
        phot_kern_results.logsm_obs,
        dust_ftrans_lines_mc,
        ssp_data,
        phot_kern_results.ssp_weights,
    )

    _dust_res_q = sspwk.compute_dust_attenuation_lines(
        phot_randoms.uran_av,
        phot_randoms.uran_delta,
        phot_randoms.uran_funo,
        phot_kern_results.diffstar_info_q.logsm_obs,
        phot_kern_results.diffstar_info_q.logssfr_obs,
        ssp_data,
        z_obs,
        line_wave_table,
        spspop_params.dustpop_params,
        scatter_params,
    )
    dust_ftrans_lines_q = _dust_res_q[0]

    linelum_q = sspwk._compute_linelum_from_weights(
        phot_kern_results.diffstar_info_q.logsm_obs,
        dust_ftrans_lines_q,
        ssp_data,
        phot_kern_results.burstiness_info_q.ssp_weights_smooth,
    )

    _dust_res_ms = sspwk.compute_dust_attenuation_lines(
        phot_randoms.uran_av,
        phot_randoms.uran_delta,
        phot_randoms.uran_funo,
        phot_kern_results.diffstar_info_ms.logsm_obs,
        phot_kern_results.diffstar_info_ms.logssfr_obs,
        ssp_data,
        z_obs,
        line_wave_table,
        spspop_params.dustpop_params,
        scatter_params,
    )
    dust_ftrans_lines_ms = _dust_res_ms[0]

    linelum_ms = sspwk._compute_linelum_from_weights(
        phot_kern_results.diffstar_info_ms.logsm_obs,
        dust_ftrans_lines_ms,
        ssp_data,
        phot_kern_results.burstiness_info_ms.ssp_weights_smooth,
    )

    linelum_bursty = sspwk._compute_linelum_from_weights(
        phot_kern_results.diffstar_info_ms.logsm_obs,
        dust_ftrans_lines_ms,
        ssp_data,
        phot_kern_results.burstiness_info_ms.ssp_weights_bursty,
    )

    n_gals = linelum_q.shape[0]
    fq = diffstarpop_results.frac_q
    weights_q = fq
    weights_ms = (1 - fq) * (1 - phot_kern_results.burstiness_info_ms.p_burst)
    weights_bursty = (1 - fq) * phot_kern_results.burstiness_info_ms.p_burst

    linelum_weighted = (
        (weights_q.reshape((n_gals, 1)) * linelum_q)
        + (weights_ms.reshape((n_gals, 1)) * linelum_ms)
        + (weights_bursty.reshape((n_gals, 1)) * linelum_bursty)
    )

    spec_kern_results = SpecKernResults(
        linelum_gal,
        linelum_weighted,
        dust_ftrans_lines_mc,
        linelum_ms,
        linelum_q,
        linelum_bursty,
    )

    return phot_kern_results, spec_kern_results


SpecKernResults = namedtuple(
    "SpecKernResults",
    (
        "linelum_gal",
        "linelum_weighted",
        "dust_ftrans_lines",
        "linelum_ms",
        "linelum_q",
        "linelum_bursty",
    ),
)
