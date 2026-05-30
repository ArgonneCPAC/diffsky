""""""

from collections import namedtuple

from jax import jit as jjit
from jax import numpy as jnp

from ...merging import compute_x_tot_from_x_in_situ, merging_model
from . import gd_linelum_kernels, gd_phot_kernels_merging, mc_randoms


@jjit
def _mc_specphot_kern_merging(
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
    ssperr_params,
    merging_params,
    cosmo_params,
    fb,
    logmp_infall,
    logmhost_infall,
    t_infall,
    is_central,
    sat_weights,
    halo_indx,
    mc_merge,
):
    upid = jnp.where(is_central == 1, -1, halo_indx)
    lgmu_infall = logmp_infall - logmhost_infall
    gyr_since_infall = t_obs - t_infall
    phot_randoms, diffstarpop_results, merging_randoms = (
        mc_randoms.get_phot_merge_randoms(
            ran_key,
            diffstarpop_params,
            mah_params,
            upid,
            lgmu_infall,
            logmhost_infall,
            gyr_since_infall,
            cosmo_params,
        )
    )

    phot_kern_results, spec_kern_results = _specphot_kern_merging(
        phot_randoms,
        merging_randoms,
        diffstarpop_results,
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
        ssperr_params,
        merging_params,
        cosmo_params,
        fb,
        logmp_infall,
        logmhost_infall,
        t_infall,
        is_central,
        sat_weights,
        halo_indx,
        mc_merge,
    )

    return phot_kern_results, phot_randoms, spec_kern_results


@jjit
def _specphot_kern_merging(
    phot_randoms,
    merging_randoms,
    diffstarpop_results,
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
    ssperr_params,
    merging_params,
    cosmo_params,
    fb,
    logmp_infall,
    logmhost_infall,
    t_infall,
    is_central,
    sat_weights,
    halo_indx,
    mc_merge,
):
    upids = jnp.where(is_central == 1, -1.0, 0.0)
    p_merge_smooth = merging_model.get_p_merge_from_merging_params(
        merging_params, logmp_infall, logmhost_infall, t_obs, t_infall, upids
    )

    _res = gd_linelum_kernels._specphot_kern(
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
    phot_kern_results, spec_kern_results = _res

    _res = gd_phot_kernels_merging._get_phot_kern_merging_quantities(
        phot_kern_results,
        merging_randoms,
        p_merge_smooth,
        sat_weights,
        halo_indx,
        mc_merge,
    )
    mstar_in_situ, mstar_obs, flux_in_situ, flux_obs, flux_obs_weighted, p_merge = _res

    args = (
        phot_kern_results,
        mstar_in_situ,
        mstar_obs,
        flux_in_situ,
        flux_obs,
        flux_obs_weighted,
        p_merge,
        merging_randoms.uran_pmerge,
    )
    func = gd_phot_kernels_merging._update_phot_kern_results_with_merging
    phot_kern_results = func(*args)

    args = phot_kern_results, spec_kern_results, sat_weights, halo_indx
    _res = _get_linelum_kern_merging_quantities(*args)
    linelums_obs, linelum_in_situ_mc, linelum_weighted, linelum_in_situ_weighted = _res

    spec_kern_results = _get_linelum_results_with_merging(
        spec_kern_results,
        linelums_obs,
        linelum_in_situ_mc,
        linelum_weighted,
        linelum_in_situ_weighted,
    )

    return phot_kern_results, spec_kern_results


@jjit
def _get_linelum_kern_merging_quantities(
    phot_kern_results, spec_kern_results, sat_weights, halo_indx
):
    linelum_in_situ_mc = spec_kern_results.linelum_gal
    linelums_obs = compute_x_tot_from_x_in_situ(
        linelum_in_situ_mc,
        phot_kern_results.p_merge[:, jnp.newaxis],
        sat_weights[:, jnp.newaxis],
        halo_indx,
    )

    linelum_in_situ_weighted = spec_kern_results.linelum_weighted
    linelum_weighted = compute_x_tot_from_x_in_situ(
        linelum_in_situ_weighted,
        phot_kern_results.p_merge[:, jnp.newaxis],
        sat_weights[:, jnp.newaxis],
        halo_indx,
    )

    return linelums_obs, linelum_in_situ_mc, linelum_weighted, linelum_in_situ_weighted


@jjit
def _get_linelum_results_with_merging(
    spec_kern_results,
    linelums_obs,
    linelum_in_situ,
    linelum_weighted,
    linelum_in_situ_weighted,
):
    spec_kern_results = spec_kern_results._replace(
        linelum_gal=linelums_obs, linelum_weighted=linelum_weighted
    )

    new_keys = ["linelum_gal_in_situ", "linelum_in_situ_weighted"]
    fields = list(spec_kern_results._fields) + new_keys
    SpecKernResults = namedtuple("SpecKernResults", fields)

    spec_kern_results = SpecKernResults(
        **spec_kern_results._asdict(),
        linelum_gal_in_situ=linelum_in_situ,
        linelum_in_situ_weighted=linelum_in_situ_weighted,
    )
    return spec_kern_results
