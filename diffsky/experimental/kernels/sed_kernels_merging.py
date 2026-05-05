""""""

from collections import namedtuple
from . import sed_kernels
from functools import partial

from jax import jit as jjit
from jax import numpy as jnp


from .. import mc_diffstarpop_wrappers as mcdw
from ...merging import merging_model, merging_kernels


@partial(jjit, static_argnames=["n_t_table"])
def _sed_kern(
    phot_randoms,
    merging_randoms,
    sfh_params,
    z_obs,
    t_obs,
    mah_params,
    ssp_data,
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
    """"""
    in_situ_sed_results = sed_kernels._sed_kern(
        phot_randoms,
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
        n_t_table=n_t_table,
    )
    _res = _get_sed_kern_merging_quantities(
        in_situ_sed_results,
        merging_randoms,
        t_obs,
        merging_params,
        logmp_infall,
        logmhost_infall,
        t_infall,
        is_central,
        sat_weights,
        halo_indx,
        mc_merge,
    )
    mstar_in_situ, mstar_obs, rest_sed_in_situ, rest_sed, p_merge = _res

    sed_results = _update_sed_kern_results_with_merging(
        in_situ_sed_results,
        mstar_in_situ,
        mstar_obs,
        rest_sed_in_situ,
        rest_sed,
        p_merge,
        merging_randoms.uran_pmerge,
    )
    return sed_results


@jjit
def _get_sed_kern_merging_quantities(
    sed_kern_results,
    merging_randoms,
    t_obs,
    merging_params,
    logmp_infall,
    logmhost_infall,
    t_infall,
    is_central,
    sat_weights,
    halo_indx,
    mc_merge,
):
    upids = jnp.where(is_central == 1, -1.0, 0.0)
    p_merge = merging_model.get_p_merge_from_merging_params(
        merging_params, logmp_infall, logmhost_infall, t_obs, t_infall, upids
    )

    # If mc_merge=1, implement Monte Carlo merging, else p_merge is a float
    mc_p_merge = merging_kernels.get_mc_p_merge(merging_randoms.uran_pmerge, p_merge)
    p_merge = jnp.where(mc_merge < 1, p_merge, mc_p_merge)

    mstar_in_situ = 10**sed_kern_results.logsm_obs
    mstar_obs = merging_kernels.compute_x_tot_from_x_in_situ(
        mstar_in_situ, p_merge, sat_weights, halo_indx
    )

    rest_sed_in_situ = sed_kern_results.rest_sed
    rest_sed = merging_kernels.compute_x_tot_from_x_in_situ(
        rest_sed_in_situ,
        p_merge[:, jnp.newaxis],
        sat_weights[:, jnp.newaxis],
        halo_indx,
    )
    return mstar_in_situ, mstar_obs, rest_sed_in_situ, rest_sed, p_merge


@jjit
def _update_sed_kern_results_with_merging(
    in_situ_sed_results,
    mstar_in_situ,
    mstar_obs,
    rest_sed_in_situ,
    rest_sed,
    p_merge,
    uran_pmerge,
):
    ex_situ_dict = dict()
    ex_situ_dict["logsm_obs"] = jnp.log10(mstar_obs)
    ex_situ_dict["rest_sed"] = rest_sed

    in_situ_sed_results = in_situ_sed_results._replace(**ex_situ_dict)

    in_situ_dict = dict()
    in_situ_dict["logsm_obs" + "_in_situ"] = jnp.log10(mstar_in_situ)
    in_situ_dict["rest_sed" + "_in_situ"] = rest_sed_in_situ

    new_keys = ["logsm_obs_in_situ", "rest_sed_in_situ", "p_merge", "uran_pmerge"]
    fields = list(in_situ_sed_results._fields) + new_keys
    SEDResults = namedtuple("PhotKernResults", fields)

    sed_results = SEDResults(
        **in_situ_sed_results._asdict(),
        **in_situ_dict,
        p_merge=p_merge,
        uran_pmerge=uran_pmerge,
    )

    return sed_results
