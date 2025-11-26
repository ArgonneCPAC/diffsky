""" """

from jax import jit as jjit
from jax import numpy as jnp
from jax import vmap

from . import ssp_weight_kernels as sspwk

interp_vmap = jjit(vmap(jnp.interp, in_axes=(0, None, 0)))


def get_dbk_ssp_weights(
    t_obs, ssp_data, phot_info, disk_bulge_history, smooth_ssp_weights, burstiness
):
    n_gals = phot_info.logmp_obs.size

    # logsm_obs_bulge = interp_vmap(
    #     t_obs, phot_info.t_table, jnp.log10(disk_bulge_history.smh_bulge)
    # )
    # mstar_obs_bulge = 10 ** logsm_obs_bulge.reshape((n_gals, 1))
    age_weights_bulge = sspwk.calc_age_weights_from_sfh_table_vmap(
        phot_info.t_table, disk_bulge_history.sfh_bulge, ssp_data.ssp_lg_age_gyr, t_obs
    )
    n_met = ssp_data.ssp_lgmet.size
    n_age = ssp_data.ssp_lg_age_gyr.size
    lgmet_weights_bulge = jnp.where(
        phot_info.mc_sfh_type.reshape((n_gals, 1)) == 0,
        smooth_ssp_weights.lgmet_weights.q,
        smooth_ssp_weights.lgmet_weights.ms,
    )
    _w_age_bulge = age_weights_bulge.reshape((n_gals, 1, n_age))
    _w_lgmet_bulge = lgmet_weights_bulge.reshape((n_gals, n_met, 1))
    ssp_weights_bulge = _w_lgmet_bulge * _w_age_bulge
    return ssp_weights_bulge
