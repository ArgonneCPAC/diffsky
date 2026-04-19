""""""

from collections import namedtuple

from jax import jit as jjit

from . import mc_randoms, phot_kernels
from . import ssp_weight_kernels as sspwk


@jjit
def _mc_specphot_kern(
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
    phot_randoms, sfh_params = mc_randoms.get_mc_phot_randoms(
        ran_key, diffstarpop_params, mah_params, cosmo_params
    )
    phot_kern_results, spec_kern_results = _specphot_kern(
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

    return phot_kern_results, phot_randoms, spec_kern_results


@jjit
def _specphot_kern(
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

    _dust_res = sspwk.compute_dust_attenuation_lines(
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
    dust_ftrans_lines = _dust_res[0]

    linelum_gal = sspwk._compute_linelum_from_weights(
        phot_kern_results.logsm_obs,
        dust_ftrans_lines,
        ssp_data,
        phot_kern_results.ssp_weights,
    )

    spec_kern_results = SpecKernResults(linelum_gal, dust_ftrans_lines)

    return phot_kern_results, spec_kern_results


SpecKernResults = namedtuple("SpecKernResults", ("linelum_gal", "dust_ftrans_lines"))
