""""""

from collections import namedtuple
from functools import partial

from jax import jit as jjit
from jax import numpy as jnp

from ...merging import merging_model
from .. import mc_diffstarpop_wrappers as mcdw
from . import dbk_kernels, sed_kernels


@partial(jjit, static_argnames=["n_t_table"])
def _dbk_sed_kern(
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
    *,
    n_t_table=mcdw.N_T_TABLE,
):
    """"""
    upid = jnp.where(is_central == 1, -1, halo_indx).astype(int)
    lgmu_infall = logmp_infall - logmhost_infall
    gyr_since_infall = t_obs - t_infall

    sed_info = sed_kernels._sed_kern(
        phot_randoms,
        sfh_params,
        z_obs,
        t_obs,
        mah_params,
        upid,
        lgmu_infall,
        logmhost_infall,
        gyr_since_infall,
        ssp_data,
        mzr_params,
        spspop_params,
        scatter_params,
        ssperr_params,
        merging_params,
        cosmo_params,
        fb,
        n_t_table=n_t_table,
    )

    p_merge_smooth = merging_model.get_p_merge_from_merging_params(
        merging_params, logmp_infall, logmhost_infall, t_obs, t_infall, upid
    )

    age_weights = jnp.sum(sed_info.ssp_weights, axis=1)
    args = (
        t_obs,
        ssp_data,
        sed_info.t_table,
        sed_info.sfh_table,
        sed_info.burst_params,
        sed_info.lgmet_weights,
        dbk_randoms,
        sed_info.logsm_obs,
        age_weights,
        p_merge_smooth,
    )
    dbk_weights, disk_bulge_history = dbk_kernels._dbk_kern(*args)

    n_gals = z_obs.size
    n_met, n_age, n_wave = ssp_data.ssp_flux.shape

    a = sed_info.dust_frac_trans.reshape((n_gals, 1, n_age, n_wave))
    b = sed_info.frac_ssp_errors.reshape((n_gals, 1, 1, n_wave))
    d = ssp_data.ssp_flux.reshape((1, n_met, n_age, n_wave))

    c_b = dbk_weights.ssp_weights_bulge.reshape((n_gals, n_met, n_age, 1))
    mb = dbk_weights.mstar_bulge.reshape((n_gals, 1))
    sed_bulge = jnp.sum(a * b * c_b * d, axis=(1, 2)) * mb

    c_dd = dbk_weights.ssp_weights_disk.reshape((n_gals, n_met, n_age, 1))
    mdd = dbk_weights.mstar_disk.reshape((n_gals, 1))
    sed_disk = jnp.sum(a * b * c_dd * d, axis=(1, 2)) * mdd

    c_k = dbk_weights.ssp_weights_knots.reshape((n_gals, n_met, n_age, 1))
    mk = dbk_weights.mstar_knots.reshape((n_gals, 1))
    sed_knots = jnp.sum(a * b * c_k * d, axis=(1, 2)) * mk

    sed_info = DBKSEDInfo(
        **sed_info._asdict(),
        rest_sed_bulge=sed_bulge,
        rest_sed_disk=sed_disk,
        rest_sed_knots=sed_knots,
        mstar_bulge=dbk_weights.mstar_bulge,
        mstar_disk=dbk_weights.mstar_disk,
        mstar_knots=dbk_weights.mstar_knots,
    )
    return sed_info


_DBK_SED_EXTRA_FIELDS = [
    "rest_sed_bulge",
    "rest_sed_disk",
    "rest_sed_knots",
    "mstar_bulge",
    "mstar_disk",
    "mstar_knots",
]
_DBK_SED_FIELDS = list(sed_kernels.SEDKernResults._fields) + _DBK_SED_EXTRA_FIELDS
DBKSEDInfo = namedtuple("DBKSEDInfo", _DBK_SED_FIELDS)
