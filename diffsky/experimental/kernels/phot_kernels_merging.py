""""""

from collections import namedtuple
from functools import partial

from jax import jit as jjit
from jax import numpy as jnp

from ...merging import compute_x_tot_from_x_in_situ, merging_kernels, merging_model
from .. import mc_diffstarpop_wrappers as mcdw
from . import mc_randoms, phot_kernels


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
    *,
    n_t_table=mcdw.N_T_TABLE,
):
    phot_randoms, sfh_params, merging_randoms = mc_randoms.get_mc_phot_merge_randoms(
        ran_key, diffstarpop_params, mah_params, cosmo_params
    )

    phot_kern_results = _phot_kern_merging(
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
        n_t_table=n_t_table,
    )
    return phot_kern_results, phot_randoms


@partial(jjit, static_argnames=["n_t_table"])
def _phot_kern_merging(
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
    *,
    n_t_table=mcdw.N_T_TABLE,
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
        n_t_table=n_t_table,
    )

    upids = jnp.where(is_central == 1, -1.0, 0.0)
    p_merge = merging_model.get_p_merge_from_merging_params(
        merge_params, logmp_infall, logmhost_infall, t_obs, t_infall, upids
    )

    # If mc_merge=1, implement Monte Carlo merging, else p_merge is a float
    mc_p_merge = merging_kernels.get_mc_p_merge(merging_randoms.uran_pmerge, p_merge)
    p_merge = jnp.where(mc_merge < 1, p_merge, mc_p_merge)

    mstar_in_situ = 10**phot_kern_results.logsm_obs
    mstar_obs = compute_x_tot_from_x_in_situ(
        mstar_in_situ, p_merge, nhalos_weights, halo_indx
    )

    flux_in_situ = 10 ** (-0.4 * phot_kern_results.obs_mags)
    flux_obs = compute_x_tot_from_x_in_situ(
        flux_in_situ, p_merge[:, jnp.newaxis], nhalos_weights[:, jnp.newaxis], halo_indx
    )
    phot_kern_results = _get_phot_kern_results_with_merging(
        phot_kern_results, mstar_in_situ, mstar_obs, flux_in_situ, flux_obs, p_merge
    )
    return phot_kern_results


@jjit
def _get_phot_kern_results_with_merging(
    phot_kern_results, mstar_in_situ, mstar_obs, flux_in_situ, flux_obs, p_merge
):
    ex_situ_dict = dict()
    ex_situ_dict["logsm_obs"] = jnp.log10(mstar_obs)
    ex_situ_dict["obs_mags"] = -2.5 * jnp.log10(flux_obs)

    phot_kern_results = phot_kern_results._replace(**ex_situ_dict)

    in_situ_dict = dict()
    in_situ_dict["logsm_obs" + "_in_situ"] = jnp.log10(mstar_in_situ)
    in_situ_dict["obs_mags" + "_in_situ"] = -2.5 * jnp.log10(flux_in_situ)

    new_keys = list(in_situ_dict.keys()) + ["p_merge"]
    phot_kern_results_keys = list(phot_kern_results._fields) + new_keys
    PhotKernResults = namedtuple("PhotKernResults", phot_kern_results_keys)
    phot_kern_results = PhotKernResults(
        **phot_kern_results._asdict(), **in_situ_dict, p_merge=p_merge
    )

    return phot_kern_results
