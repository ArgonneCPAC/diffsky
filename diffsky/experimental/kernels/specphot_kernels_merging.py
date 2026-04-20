""""""

from jax import jit as jjit
from jax import numpy as jnp

from ...merging import compute_x_tot_from_x_in_situ, merging_kernels, merging_model
from . import linelum_kernels, mc_randoms


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

    _res = _specphot_kern_merging(
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
    (
        phot_kern_results,
        linelums_in_situ,
        flux_obs,
        merge_prob,
        mstar_obs,
        linelums_obs,
    ) = _res

    return (
        phot_kern_results,
        linelums_in_situ,
        phot_randoms,
        flux_obs,
        merge_prob,
        mstar_obs,
        linelums_obs,
    )


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

    upids = jnp.where(is_central == 1, -1.0, 0.0)
    merge_prob = merging_model.get_p_merge_from_merging_params(
        merge_params, logmp_infall, logmhost_infall, t_obs, t_infall, upids
    )
    # If mc_merge=1, implement Monte Carlo merging, else merge_prob is a float
    mc_p_merge = merging_kernels.get_mc_p_merge(merging_randoms.uran_pmerge, merge_prob)
    merge_prob = jnp.where(mc_merge < 1, merge_prob, mc_p_merge)

    mstar_in_situ = 10**phot_kern_results.logsm_obs
    mstar_obs = compute_x_tot_from_x_in_situ(
        mstar_in_situ, merge_prob, nhalos_weights, halo_indx
    )

    flux_in_situ = 10 ** (-0.4 * phot_kern_results.obs_mags)
    flux_obs = compute_x_tot_from_x_in_situ(
        flux_in_situ,
        merge_prob[:, jnp.newaxis],
        nhalos_weights[:, jnp.newaxis],
        halo_indx,
    )

    linelums_obs = compute_x_tot_from_x_in_situ(
        spec_kern_results.linelum_gal,
        merge_prob[:, jnp.newaxis],
        nhalos_weights[:, jnp.newaxis],
        halo_indx,
    )

    return (
        phot_kern_results,
        spec_kern_results.linelum_gal,
        flux_obs,
        merge_prob,
        mstar_obs,
        linelums_obs,
    )
