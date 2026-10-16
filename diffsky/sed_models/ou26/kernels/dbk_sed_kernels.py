""""""

from collections import namedtuple
from functools import partial

from jax import jit as jjit
from jax import numpy as jnp

from .. import mc_diffstarpop_wrappers as mcdw
from . import dbk_sed_kernels_in_situ as gd_sedk
from . import sed_kernels


@partial(jjit, static_argnames=["n_t_table"])
def _dbk_sed_kern(
    phot_randoms,
    dbk_randoms,
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
    dbk_sed_info_in_situ = gd_sedk._dbk_sed_kern(
        phot_randoms,
        dbk_randoms,
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
        halo_indx,
        n_t_table=n_t_table,
    )

    dbk_sed_merging_quantities = _get_dbk_sed_kern_merging_quantities(
        dbk_sed_info_in_situ,
        merging_randoms,
        sat_weights,
        halo_indx,
        mc_merge,
    )

    dbk_sed_info = _update_dbk_sed_kern_results_with_merging(
        dbk_sed_info_in_situ, merging_randoms, *dbk_sed_merging_quantities
    )

    return dbk_sed_info


@jjit
def _get_dbk_sed_kern_merging_quantities(
    dbk_sed_info_in_situ,
    merging_randoms,
    sat_weights,
    halo_indx,
    mc_merge,
):
    _res = sed_kernels._get_sed_kern_merging_quantities(
        dbk_sed_info_in_situ, merging_randoms, sat_weights, halo_indx, mc_merge
    )
    (
        mstar_in_situ,
        mstar_obs,
        rest_sed_in_situ,
        rest_sed,
        p_merge,
        ssp_weights,
        ssp_weights_in_situ,
    ) = _res

    mb_in_situ = dbk_sed_info_in_situ.mstar_bulge
    md_in_situ = dbk_sed_info_in_situ.mstar_disk
    mk_in_situ = dbk_sed_info_in_situ.mstar_knots
    mstar_in_situ = mb_in_situ + md_in_situ + mk_in_situ
    mstar_obs = 10**dbk_sed_info_in_situ.logsm_obs
    mass_ratio = mstar_obs / mstar_in_situ

    mstar_bulge = mass_ratio * mb_in_situ
    mstar_disk = mass_ratio * md_in_situ
    mstar_knots = mass_ratio * mk_in_situ

    frac_sed_bulge = dbk_sed_info_in_situ.rest_sed_bulge / rest_sed_in_situ
    frac_sed_disk = dbk_sed_info_in_situ.rest_sed_disk / rest_sed_in_situ
    frac_sed_knots = dbk_sed_info_in_situ.rest_sed_knots / rest_sed_in_situ

    rest_sed_bulge = frac_sed_bulge * rest_sed
    rest_sed_disk = frac_sed_disk * rest_sed
    rest_sed_knots = frac_sed_knots * rest_sed

    return (
        mstar_in_situ,
        mstar_obs,
        rest_sed_in_situ,
        rest_sed,
        p_merge,
        mstar_bulge,
        mstar_disk,
        mstar_knots,
        rest_sed_bulge,
        rest_sed_disk,
        rest_sed_knots,
    )


@jjit
def _update_dbk_sed_kern_results_with_merging(
    dbk_sed_info_in_situ,
    merging_randoms,
    mstar_in_situ,
    mstar_obs,
    rest_sed_in_situ,
    rest_sed,
    p_merge,
    mstar_bulge,
    mstar_disk,
    mstar_knots,
    rest_sed_bulge,
    rest_sed_disk,
    rest_sed_knots,
):
    ex_situ_dict = dict()
    ex_situ_dict["logsm_obs"] = jnp.log10(mstar_obs)
    ex_situ_dict["rest_sed"] = rest_sed
    ex_situ_dict["mstar_bulge"] = mstar_bulge
    ex_situ_dict["mstar_disk"] = mstar_disk
    ex_situ_dict["mstar_knots"] = mstar_knots

    ex_situ_dict["rest_sed_bulge"] = rest_sed_bulge
    ex_situ_dict["rest_sed_disk"] = rest_sed_disk
    ex_situ_dict["rest_sed_knots"] = rest_sed_knots

    dbk_sed_info = dbk_sed_info_in_situ._replace(**ex_situ_dict)

    in_situ_dict = dict()
    in_situ_dict["logsm_obs" + "_in_situ"] = jnp.log10(mstar_in_situ)
    in_situ_dict["rest_sed_in_situ"] = rest_sed_in_situ

    new_keys = ["logsm_obs_in_situ", "rest_sed_in_situ", "p_merge", "uran_pmerge"]
    fields = list(dbk_sed_info._fields) + new_keys
    DBKSEDInfo = namedtuple("DBKSEDInfo", fields)

    dbk_sed_info = DBKSEDInfo(
        **dbk_sed_info._asdict(),
        **in_situ_dict,
        p_merge=p_merge,
        uran_pmerge=merging_randoms.uran_pmerge,
    )
    return dbk_sed_info
