""""""

from collections import namedtuple
from . import sed_kernels_merging as sedkm
from .. import mc_diffstarpop_wrappers as mcdw

from collections import namedtuple
from functools import partial

from jax import jit as jjit
from jax import numpy as jnp


from .. import mc_diffstarpop_wrappers as mcdw
from . import dbk_kernels


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
    sed_info = sedkm._sed_kern(
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
        n_t_table=n_t_table,
    )

    dbk_weights, disk_bulge_history = dbk_kernels._dbk_kern(
        t_obs,
        ssp_data,
        sed_info.t_table,
        sed_info.sfh_table,
        sed_info.burst_params,
        sed_info.lgmet_weights,
        dbk_randoms,
    )

    n_gals = z_obs.size
    n_met, n_age, n_wave = ssp_data.ssp_flux.shape

    _w_bulge = dbk_weights.ssp_weights_bulge.reshape((n_gals, n_met, n_age, 1))
    _w_dd = dbk_weights.ssp_weights_disk.reshape((n_gals, n_met, n_age, 1))
    _w_knot = dbk_weights.ssp_weights_knots.reshape((n_gals, n_met, n_age, 1))

    mb = dbk_weights.mstar_bulge.reshape((n_gals, 1))
    md = dbk_weights.mstar_disk.reshape((n_gals, 1))
    mk = dbk_weights.mstar_knots.reshape((n_gals, 1))

    a = sed_info.dust_frac_trans.reshape((n_gals, 1, n_age, n_wave))
    b = sed_info.frac_ssp_errors.reshape((n_gals, 1, 1, n_wave))
    d = ssp_data.ssp_flux.reshape((1, n_met, n_age, n_wave))

    sed_bulge = jnp.sum(a * b * _w_bulge * d, axis=(1, 2)) * mb
    sed_disk = jnp.sum(a * b * _w_dd * d, axis=(1, 2)) * md
    sed_knots = jnp.sum(a * b * _w_knot * d, axis=(1, 2)) * mk

    new_keys = ["rest_sed_bulge", "rest_sed_disk", "rest_sed_knots"]

    fields = list(sed_info._fields) + new_keys
    SEDInfo = namedtuple("SEDInfo", fields)
    sed_info = SEDInfo(
        **sed_info._asdict(),
        rest_sed_bulge=sed_bulge,
        rest_sed_disk=sed_disk,
        rest_sed_knots=sed_knots,
    )
    return sed_info
