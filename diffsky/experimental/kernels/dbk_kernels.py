""" """

from collections import namedtuple

from dsps.sfh import diffburst
from jax import jit as jjit
from jax import numpy as jnp
from jax import vmap

from ..disk_bulge_modeling import disk_knots
from . import ssp_weight_kernels as sspwk

interp_vmap = jjit(vmap(jnp.interp, in_axes=(0, None, 0)))

_BPOP = (None, 0, 0)
get_pureburst_age_weights = jjit(
    vmap(diffburst._pureburst_age_weights_from_params, in_axes=_BPOP)
)

DBKWeights = namedtuple(
    "DBKWeights",
    (
        "ssp_weights_bulge",
        "ssp_weights_disk",
        "ssp_weights_knots",
        "mstar_bulge",
        "mstar_disk",
        "mstar_knots",
    ),
)


@jjit
def get_bulge_weights(
    t_obs, ssp_data, phot_info, disk_bulge_history, smooth_ssp_weights
):
    n_gals = phot_info.logmp_obs.size
    logsm_obs_bulge = interp_vmap(
        t_obs, phot_info.t_table, jnp.log10(disk_bulge_history.smh_bulge)
    )
    mstar_obs_bulge = 10 ** logsm_obs_bulge.reshape((n_gals, 1))

    age_weights_bulge = sspwk.calc_age_weights_from_sfh_table_vmap(
        phot_info.t_table, disk_bulge_history.sfh_bulge, ssp_data.ssp_lg_age_gyr, t_obs
    )
    n_met = ssp_data.ssp_lgmet.size
    n_age = ssp_data.ssp_lg_age_gyr.size

    msk_q = phot_info.mc_sfh_type.reshape((n_gals, 1)) == 0
    lgmet_weights_bulge = jnp.where(
        msk_q,
        smooth_ssp_weights.lgmet_weights.q,
        smooth_ssp_weights.lgmet_weights.ms,
    )
    _w_age_bulge = age_weights_bulge.reshape((n_gals, 1, n_age))
    _w_lgmet_bulge = lgmet_weights_bulge.reshape((n_gals, n_met, 1))
    ssp_weights_bulge = _w_lgmet_bulge * _w_age_bulge
    return ssp_weights_bulge, mstar_obs_bulge


@jjit
def get_disk_weights(t_obs, ssp_data, phot_info, disk_bulge_history, fknot):
    ssp_lg_age_yr = ssp_data.ssp_lg_age_gyr + 9.0
    age_weights_pureburst = get_pureburst_age_weights(
        ssp_lg_age_yr,
        phot_info.burstiness.burst_params.lgyr_peak,
        phot_info.burstiness.burst_params.lgyr_max,
    )

    fburst = jnp.where(
        phot_info.mc_sfh_type < 2, 0.0, 10**phot_info.burstiness.burst_params.lgfburst
    )
    _res = disk_knots._disk_knot_vmap(
        phot_info.t_table,
        t_obs,
        phot_info.sfh_table,
        phot_info.sfh_table - disk_bulge_history.sfh_bulge,
        fburst,
        fknot,
        age_weights_pureburst,
        ssp_data.ssp_lg_age_gyr,
    )
    mstar_disk, mstar_knots, age_weights_disk, age_weights_knots = _res[2:]

    n_gals = t_obs.size
    n_met = ssp_data.ssp_lgmet.size
    n_age = ssp_data.ssp_lg_age_gyr.size

    _w_age_dd = age_weights_disk.reshape((n_gals, 1, n_age))
    _w_lgmet_dd = phot_info.lgmet_weights.reshape((n_gals, n_met, 1))
    ssp_weights_disk = _w_lgmet_dd * _w_age_dd

    _w_age_knot = age_weights_knots.reshape((n_gals, 1, n_age))
    _w_lgmet_knot = phot_info.lgmet_weights.reshape((n_gals, n_met, 1))
    ssp_weights_knots = _w_lgmet_knot * _w_age_knot

    return (ssp_weights_disk, ssp_weights_knots, mstar_disk, mstar_knots)


@jjit
def get_dbk_weights(
    t_obs, ssp_data, phot_info, smooth_ssp_weights, disk_bulge_history, fknot
):
    ssp_weights_bulge, mstar_bulge = get_bulge_weights(
        t_obs, ssp_data, phot_info, disk_bulge_history, smooth_ssp_weights
    )

    _res = get_disk_weights(t_obs, ssp_data, phot_info, disk_bulge_history, fknot)
    ssp_weights_disk, ssp_weights_knots, mstar_disk, mstar_knots = _res

    return DBKWeights(
        ssp_weights_bulge=ssp_weights_bulge,
        ssp_weights_disk=ssp_weights_disk,
        ssp_weights_knots=ssp_weights_knots,
        mstar_bulge=mstar_bulge,
        mstar_disk=mstar_disk,
        mstar_knots=mstar_knots,
    )
