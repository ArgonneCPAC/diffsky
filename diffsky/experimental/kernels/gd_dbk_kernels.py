""""""

from jax import jit as jjit
from jax import numpy as jnp
from jax import vmap

from ..disk_bulge_modeling import dbpop
from . import dbk_kernels
from . import rapid_quenching as rq
from . import ssp_weight_kernels as sspwk

interp_vmap = jjit(vmap(jnp.interp, in_axes=(0, None, 0)))


@jjit
def _dbk_kern(
    t_obs,
    ssp_data,
    t_table,
    sfh_table,
    burst_params,
    lgmet_weights,
    dbk_randoms,
    logsm_obs,
    age_weights,
    p_merge_smooth,
):
    disk_bulge_history = dbpop.decompose_sfh_into_disk_bulge_sfh(
        dbk_randoms.uran_fbulge, t_table, sfh_table, t_obs
    )

    args = (
        t_obs,
        ssp_data,
        t_table,
        burst_params,
        lgmet_weights,
        disk_bulge_history,
        dbk_randoms.fknot,
        logsm_obs,
        age_weights,
        p_merge_smooth,
    )
    dbk_weights = get_dbk_weights_rq(*args)

    return dbk_weights, disk_bulge_history


@jjit
def get_dbk_weights_rq(
    t_obs,
    ssp_data,
    t_table,
    burst_params,
    lgmet_weights,
    disk_bulge_history,
    fknot,
    logsm_obs,
    age_weights_tot,
    p_merge_smooth,
):
    age_weights_bulge, mstar_bulge = get_bulge_age_weights_rq(
        t_obs, ssp_data, t_table, disk_bulge_history, p_merge_smooth
    )
    n_gals = mstar_bulge.size

    mstar_tot = 10**logsm_obs  # total stellar mass in galaxy
    mstar_burst = mstar_tot * 10**burst_params.lgfburst  # mass in burst
    mstar_ddk = mstar_tot - mstar_bulge  # mass of diffuse disk + knots
    mstar_knots = fknot * mstar_ddk  # mass of knots
    mstar_dd = mstar_ddk - mstar_knots  # mass of diffuse disk

    # m_ddk*W_ddk = m_dd*W_dd + m_k*W_k
    _A = mstar_tot.reshape((n_gals, 1)) * age_weights_tot
    _B = mstar_bulge.reshape((n_gals, 1)) * age_weights_bulge
    age_weights_ddk = (_A - _B) / mstar_ddk.reshape((n_gals, 1))

    ssp_lg_age_yr = ssp_data.ssp_lg_age_gyr + 9.0
    age_weights_pureburst = dbk_kernels.get_pureburst_age_weights(
        ssp_lg_age_yr, burst_params.lgyr_peak, burst_params.lgyr_max
    )

    # m_ddk*W_ddk = m_ddk_smooth*W_ddk_smooth + m_burst*W_burst
    _C = mstar_ddk.reshape((n_gals, 1)) * age_weights_ddk
    _D = mstar_burst.reshape((n_gals, 1)) * age_weights_pureburst
    mstar_ddk_smooth = mstar_ddk - mstar_burst
    age_weights_ddk_smooth = (_C - _D) / mstar_ddk_smooth.reshape((n_gals, 1))

    # m_k*W_k = m_ks_smooth*W_ddk_smooth + m_burst_knot*W_burst
    mburst_by_mknot = mstar_burst / mstar_knots
    mstar_knots_burst = jnp.where(mburst_by_mknot > 1, mstar_knots, mstar_burst)
    mstar_knots_smooth = mstar_knots - mstar_knots_burst  # possibly zero
    _E = mstar_knots_smooth.reshape((n_gals, 1)) * age_weights_ddk_smooth
    _F = mstar_knots_burst.reshape((n_gals, 1)) * age_weights_pureburst
    age_weights_knots = (_E + _F) / mstar_knots.reshape((n_gals, 1))

    mstar_dd_burst = jnp.where(mburst_by_mknot > 1, mstar_burst - mstar_knots, 0.0)
    mstar_dd_smooth = mstar_dd - mstar_dd_burst
    _G = mstar_dd_smooth.reshape((n_gals, 1)) * age_weights_ddk_smooth
    _H = mstar_dd_burst.reshape((n_gals, 1)) * age_weights_pureburst
    age_weights_dd = (_G + _H) / mstar_dd.reshape((n_gals, 1))

    ssp_weights_bulge = sspwk.combine_age_met_weights(age_weights_bulge, lgmet_weights)
    ssp_weights_disk = sspwk.combine_age_met_weights(age_weights_dd, lgmet_weights)
    ssp_weights_knots = sspwk.combine_age_met_weights(age_weights_knots, lgmet_weights)

    dbk_weights_rq = dbk_kernels.DBKWeights(
        ssp_weights_bulge=ssp_weights_bulge,
        ssp_weights_disk=ssp_weights_disk,
        ssp_weights_knots=ssp_weights_knots,
        mstar_bulge=mstar_bulge,
        mstar_disk=mstar_dd,
        mstar_knots=mstar_knots,
    )
    return dbk_weights_rq


@jjit
def get_bulge_age_weights_rq(
    t_obs, ssp_data, t_table, disk_bulge_history, p_merge_smooth
):
    logsm_obs_bulge = interp_vmap(
        t_obs, t_table, jnp.log10(disk_bulge_history.smh_bulge)
    )
    mstar_bulge = 10**logsm_obs_bulge
    age_weights_bulge = sspwk.calc_age_weights_from_sfh_table_vmap(
        t_table, disk_bulge_history.sfh_bulge, ssp_data.ssp_lg_age_gyr, t_obs
    )
    age_weights_bulge_rq, __ = rq.get_age_weights_rq_vmap(
        age_weights_bulge,
        p_merge_smooth,
        ssp_data.ssp_lg_age_gyr,
        rq.DEFAULT_RQ_PARAMS,
    )

    return age_weights_bulge_rq, mstar_bulge
