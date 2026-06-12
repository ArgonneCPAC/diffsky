""""""

from collections import namedtuple
from functools import partial

from jax import jit as jjit
from jax import numpy as jnp

from ...merging import compute_x_tot_from_x_in_situ, merging_kernels, merging_model
from .. import mc_diffstarpop_wrappers as mcdw
from . import mc_randoms, gd_phot_kernels


@partial(jjit, static_argnames=["n_t_table"])
def _mc_phot_kern_merging(
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
    *,
    n_t_table=mcdw.N_T_TABLE,
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

    phot_kern_results = _phot_kern_merging(
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
        n_t_table=n_t_table,
    )
    return phot_kern_results, phot_randoms, merging_randoms


@partial(jjit, static_argnames=["n_t_table"])
def _phot_kern_merging(
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
    *,
    n_t_table=mcdw.N_T_TABLE,
):
    upids = jnp.where(is_central == 1, -1.0, 0.0)
    p_merge_smooth = merging_model.get_p_merge_from_merging_params(
        merging_params, logmp_infall, logmhost_infall, t_obs, t_infall, upids
    )

    phot_kern_results = gd_phot_kernels._phot_kern(
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
        n_t_table=n_t_table,
    )

    _res = _get_phot_kern_merging_quantities(
        phot_kern_results,
        merging_randoms,
        p_merge_smooth,
        sat_weights,
        halo_indx,
        mc_merge,
    )
    mstar_in_situ, mstar_obs, flux_in_situ, flux_obs, flux_obs_weighted, p_merge = _res
    phot_kern_results = _update_phot_kern_results_with_merging(
        phot_kern_results,
        mstar_in_situ,
        mstar_obs,
        flux_in_situ,
        flux_obs,
        flux_obs_weighted,
        p_merge,
        merging_randoms.uran_pmerge,
    )
    return phot_kern_results


@jjit
def _get_phot_kern_merging_quantities(
    phot_kern_results,
    merging_randoms,
    p_merge_smooth,
    sat_weights,
    halo_indx,
    mc_merge,
):

    # If mc_merge=1, implement Monte Carlo merging, else p_merge is a float
    mc_p_merge = merging_kernels.get_mc_p_merge(
        merging_randoms.uran_pmerge, p_merge_smooth
    )
    p_merge = jnp.where(mc_merge < 1, p_merge_smooth, mc_p_merge)

    mstar_in_situ = 10**phot_kern_results.logsm_obs
    mstar_obs = compute_x_tot_from_x_in_situ(
        mstar_in_situ, p_merge, sat_weights, halo_indx
    )

    flux_in_situ = 10 ** (-0.4 * phot_kern_results.obs_mags)
    flux_obs = compute_x_tot_from_x_in_situ(
        flux_in_situ, p_merge[:, jnp.newaxis], sat_weights[:, jnp.newaxis], halo_indx
    )
    weighted_flux_in_situ = 10 ** (-0.4 * phot_kern_results.obs_mags_weighted)
    flux_obs_weighted = compute_x_tot_from_x_in_situ(
        weighted_flux_in_situ,
        p_merge[:, jnp.newaxis],
        sat_weights[:, jnp.newaxis],
        halo_indx,
    )

    return mstar_in_situ, mstar_obs, flux_in_situ, flux_obs, flux_obs_weighted, p_merge


@jjit
def _update_phot_kern_results_with_merging(
    phot_kern_results,
    mstar_in_situ,
    mstar_obs,
    flux_in_situ,
    flux_obs,
    flux_obs_weighted,
    p_merge,
    uran_pmerge,
):
    ex_situ_dict = dict()
    ex_situ_dict["logsm_obs"] = jnp.log10(mstar_obs)
    ex_situ_dict["obs_mags"] = -2.5 * jnp.log10(flux_obs)
    ex_situ_dict["obs_mags_weighted"] = -2.5 * jnp.log10(flux_obs_weighted)

    phot_kern_results = phot_kern_results._replace(**ex_situ_dict)

    in_situ_dict = dict()
    in_situ_dict["logsm_obs" + "_in_situ"] = jnp.log10(mstar_in_situ)
    in_situ_dict["obs_mags" + "_in_situ"] = -2.5 * jnp.log10(flux_in_situ)

    new_keys = ["logsm_obs_in_situ", "obs_mags_in_situ", "p_merge", "uran_pmerge"]
    fields = list(phot_kern_results._fields) + new_keys
    PhotKernResults = namedtuple("PhotKernResults", fields)

    phot_kern_results = PhotKernResults(
        **phot_kern_results._asdict(),
        **in_situ_dict,
        p_merge=p_merge,
        uran_pmerge=uran_pmerge,
    )

    return phot_kern_results
