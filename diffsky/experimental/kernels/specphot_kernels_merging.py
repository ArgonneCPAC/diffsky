""""""

from collections import namedtuple

from jax import jit as jjit
from jax import numpy as jnp

from ...merging import compute_x_tot_from_x_in_situ
from . import linelum_kernels, mc_randoms, phot_kernels_merging


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
    ssp_err_pop_params,
    merge_params,
    cosmo_params,
    fb,
    logmp_infall,
    logmhost_infall,
    t_infall,
    is_central,
    nhalos_weights,
    halo_indx,
    mc_merge,
):
    phot_randoms, sfh_params, merging_randoms = mc_randoms.get_mc_phot_merge_randoms(
        ran_key, diffstarpop_params, mah_params, cosmo_params
    )

    phot_kern_results, spec_kern_results = _specphot_kern_merging(
        phot_randoms,
        merging_randoms,
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
        merge_params,
        cosmo_params,
        fb,
        logmp_infall,
        logmhost_infall,
        t_infall,
        is_central,
        nhalos_weights,
        halo_indx,
        mc_merge,
    )

    return phot_kern_results, phot_randoms, spec_kern_results


@jjit
def _specphot_kern_merging(
    phot_randoms,
    merging_randoms,
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
    merge_params,
    cosmo_params,
    fb,
    logmp_infall,
    logmhost_infall,
    t_infall,
    is_central,
    nhalos_weights,
    halo_indx,
    mc_merge,
):
    _res = linelum_kernels._specphot_kern(
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
    phot_kern_results, spec_kern_results = _res

    _res = phot_kernels_merging._get_phot_kern_merging_quantities(
        phot_kern_results,
        merging_randoms,
        t_obs,
        merge_params,
        logmp_infall,
        logmhost_infall,
        t_infall,
        is_central,
        nhalos_weights,
        halo_indx,
        mc_merge,
    )
    mstar_in_situ, mstar_obs, flux_in_situ, flux_obs, p_merge = _res

    args = (
        phot_kern_results,
        mstar_in_situ,
        mstar_obs,
        flux_in_situ,
        flux_obs,
        p_merge,
        merging_randoms.uran_pmerge,
    )
    phot_kern_results = phot_kernels_merging._get_phot_kern_results_with_merging(*args)

    args = phot_kern_results, spec_kern_results, nhalos_weights, halo_indx
    linelums_obs, linelum_in_situ = _get_linelum_kern_merging_quantities(*args)
    spec_kern_results = _get_linelum_results_with_merging(
        spec_kern_results, linelums_obs, linelum_in_situ
    )

    return phot_kern_results, spec_kern_results


@jjit
def _get_linelum_kern_merging_quantities(
    phot_kern_results, spec_kern_results, nhalos_weights, halo_indx
):
    linelum_in_situ = spec_kern_results.linelum_gal
    linelums_obs = compute_x_tot_from_x_in_situ(
        linelum_in_situ,
        phot_kern_results.p_merge[:, jnp.newaxis],
        nhalos_weights[:, jnp.newaxis],
        halo_indx,
    )
    return linelums_obs, linelum_in_situ


@jjit
def _get_linelum_results_with_merging(spec_kern_results, linelums_obs, linelum_in_situ):
    spec_kern_results = spec_kern_results._replace(linelum_gal=linelums_obs)

    new_keys = ["linelum_gal_in_situ"]
    fields = list(spec_kern_results._fields) + new_keys
    SpecKernResults = namedtuple("SpecKernResults", fields)
    spec_kern_results = SpecKernResults(
        **spec_kern_results._asdict(), linelum_gal_in_situ=linelum_in_situ
    )
    return spec_kern_results
