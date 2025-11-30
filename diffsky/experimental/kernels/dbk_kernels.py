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
def get_bulge_weights(t_obs, ssp_data, t_table, disk_bulge_history, lgmet_weights):
    n_gals = t_obs.size
    logsm_obs_bulge = interp_vmap(
        t_obs, t_table, jnp.log10(disk_bulge_history.smh_bulge)
    )
    mstar_obs_bulge = 10 ** logsm_obs_bulge.reshape((n_gals, 1))
    age_weights_bulge = sspwk.calc_age_weights_from_sfh_table_vmap(
        t_table, disk_bulge_history.sfh_bulge, ssp_data.ssp_lg_age_gyr, t_obs
    )
    ssp_weights_bulge = sspwk.combine_age_met_weights(age_weights_bulge, lgmet_weights)

    return ssp_weights_bulge, mstar_obs_bulge


@jjit
def get_disk_weights(
    t_obs,
    ssp_data,
    t_table,
    sfh_table,
    lgmet_weights,
    burst_params,
    disk_bulge_history,
    fknot,
):
    ssp_lg_age_yr = ssp_data.ssp_lg_age_gyr + 9.0
    age_weights_pureburst = get_pureburst_age_weights(
        ssp_lg_age_yr, burst_params.lgyr_peak, burst_params.lgyr_max
    )

    _res = disk_knots._disk_knot_vmap(
        t_table,
        t_obs,
        sfh_table,
        sfh_table - disk_bulge_history.sfh_bulge,
        10**burst_params.lgfburst,
        fknot,
        age_weights_pureburst,
        ssp_data.ssp_lg_age_gyr,
    )
    mstar_disk, mstar_knots, age_weights_disk, age_weights_knots = _res[2:]

    ssp_weights_disk = sspwk.combine_age_met_weights(age_weights_disk, lgmet_weights)
    ssp_weights_knots = sspwk.combine_age_met_weights(age_weights_knots, lgmet_weights)

    return (ssp_weights_disk, ssp_weights_knots, mstar_disk, mstar_knots)


@jjit
def get_dbk_weights(
    t_obs,
    ssp_data,
    t_table,
    sfh_table,
    burst_params,
    lgmet_weights,
    disk_bulge_history,
    fknot,
):
    ssp_weights_bulge, mstar_bulge = get_bulge_weights(
        t_obs, ssp_data, t_table, disk_bulge_history, lgmet_weights
    )

    _res = get_disk_weights(
        t_obs,
        ssp_data,
        t_table,
        sfh_table,
        lgmet_weights,
        burst_params,
        disk_bulge_history,
        fknot,
    )
    ssp_weights_disk, ssp_weights_knots, mstar_disk, mstar_knots = _res

    return DBKWeights(
        ssp_weights_bulge=ssp_weights_bulge,
        ssp_weights_disk=ssp_weights_disk,
        ssp_weights_knots=ssp_weights_knots,
        mstar_bulge=mstar_bulge,
        mstar_disk=mstar_disk,
        mstar_knots=mstar_knots,
    )
