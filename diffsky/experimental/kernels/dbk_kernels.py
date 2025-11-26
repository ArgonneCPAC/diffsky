""" """

from jax import jit as jjit
from jax import numpy as jnp
from jax import vmap

from . import ssp_weight_kernels as sspwk

interp_vmap = jjit(vmap(jnp.interp, in_axes=(0, None, 0)))


@jjit
def mc_ssp_weights_bulge(
    t_obs, ssp_data, phot_info, disk_bulge_history, smooth_ssp_weights
):
    n_gals = phot_info.logmp_obs.size

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
    return ssp_weights_bulge
