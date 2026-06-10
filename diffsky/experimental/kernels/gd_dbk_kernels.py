""""""

from collections import namedtuple

from dsps.sfh import diffburst
from jax import jit as jjit
from jax import numpy as jnp
from jax import vmap

from ..disk_bulge_modeling import dbpop
from . import rapid_quenching as rq
from . import ssp_weight_kernels as sspwk

interp_vmap = jjit(vmap(jnp.interp, in_axes=(0, None, 0)))
_BPOP = (None, 0, 0)
get_pureburst_age_weights = jjit(
    vmap(diffburst._pureburst_age_weights_from_params, in_axes=_BPOP)
)


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

    mstar_ddk_smooth = mstar_ddk - mstar_burst

    # m_tot*W_tot = m_ddk*W_ddk + m_bulge*W_bulge
    _A = mstar_tot.reshape((n_gals, 1)) * age_weights_tot
    _B = mstar_bulge.reshape((n_gals, 1)) * age_weights_bulge
    age_weights_ddk = (_A - _B) / mstar_ddk.reshape((n_gals, 1))

    ssp_lg_age_yr = ssp_data.ssp_lg_age_gyr + 9.0
    age_weights_pureburst = get_pureburst_age_weights(
        ssp_lg_age_yr, burst_params.lgyr_peak, burst_params.lgyr_max
    )

    # m_ddk*W_ddk = m_ddk_smooth*W_ddk_smooth + m_burst*W_burst
    _C = mstar_ddk.reshape((n_gals, 1)) * age_weights_ddk
    _D = mstar_burst.reshape((n_gals, 1)) * age_weights_pureburst
    age_weights_ddk_smooth = (_C - _D) / mstar_ddk_smooth.reshape((n_gals, 1))

    # m_k*W_k = m_ks_smooth*W_ddk_smooth + m_burst_knot*W_burst
    mburst_by_mknot = mstar_burst / mstar_knots
    mstar_knots_burst = jnp.where(mburst_by_mknot > 1, mstar_knots, mstar_burst)
    mstar_knots_smooth = mstar_knots - mstar_knots_burst  # possibly zero
    _E = mstar_knots_smooth.reshape((n_gals, 1)) * age_weights_ddk_smooth
    _F = mstar_knots_burst.reshape((n_gals, 1)) * age_weights_pureburst
    age_weights_knots = (_E + _F) / mstar_knots.reshape((n_gals, 1))

    # m_dd*W_dd = m_dds*W_ddk_smooth + m_ddb*W_burst
    mstar_dd_burst = jnp.where(mburst_by_mknot > 1, mstar_burst - mstar_knots, 0.0)
    mstar_dd_smooth = mstar_dd - mstar_dd_burst
    _G = mstar_dd_smooth.reshape((n_gals, 1)) * age_weights_ddk_smooth
    _H = mstar_dd_burst.reshape((n_gals, 1)) * age_weights_pureburst
    age_weights_dd = (_G + _H) / mstar_dd.reshape((n_gals, 1))

    ssp_weights_bulge = sspwk.combine_age_met_weights(age_weights_bulge, lgmet_weights)
    ssp_weights_disk = sspwk.combine_age_met_weights(age_weights_dd, lgmet_weights)
    ssp_weights_knots = sspwk.combine_age_met_weights(age_weights_knots, lgmet_weights)

    dbk_weights_rq = DBKWeights(
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


@jjit
def _get_dbk_phot_from_dbk_weights(
    ssp_photflux_table, dbk_weights, dust_frac_trans, frac_ssp_err
):
    n_gals, n_bands, n_met, n_age = ssp_photflux_table.shape

    # Reshape arrays before calculating galaxy magnitudes
    _ftrans = dust_frac_trans.reshape((n_gals, n_bands, 1, n_age))

    _ferr_ssp = frac_ssp_err.reshape((n_gals, n_bands, 1, 1))

    _w_bulge = dbk_weights.ssp_weights_bulge.reshape((n_gals, 1, n_met, n_age))
    _w_dd = dbk_weights.ssp_weights_disk.reshape((n_gals, 1, n_met, n_age))
    _w_knot = dbk_weights.ssp_weights_knots.reshape((n_gals, 1, n_met, n_age))

    _mstar_bulge = dbk_weights.mstar_bulge.reshape((n_gals, 1))
    _mstar_disk = dbk_weights.mstar_disk.reshape((n_gals, 1))
    _mstar_knots = dbk_weights.mstar_knots.reshape((n_gals, 1))

    integrand_bulge = ssp_photflux_table * _w_bulge * _ftrans * _ferr_ssp
    flux_bulge = jnp.sum(integrand_bulge, axis=(2, 3)) * _mstar_bulge
    obs_mags_bulge = -2.5 * jnp.log10(flux_bulge)

    integrand_disk = ssp_photflux_table * _w_dd * _ftrans * _ferr_ssp
    flux_disk = jnp.sum(integrand_disk, axis=(2, 3)) * _mstar_disk
    obs_mags_disk = -2.5 * jnp.log10(flux_disk)

    integrand_knots = ssp_photflux_table * _w_knot * _ftrans * _ferr_ssp
    flux_knots = jnp.sum(integrand_knots, axis=(2, 3)) * _mstar_knots
    obs_mags_knots = -2.5 * jnp.log10(flux_knots)

    return DBKPhotInfo(obs_mags_bulge, obs_mags_disk, obs_mags_knots)


@jjit
def _get_dbk_linelum_decomposition(dbk_weights, spec_kern_results, ssp_data):
    linelum_bulge = sspwk._compute_linelum_from_weights(
        jnp.log10(dbk_weights.mstar_bulge),
        spec_kern_results.dust_ftrans_lines,
        ssp_data,
        dbk_weights.ssp_weights_bulge,
    )
    linelum_disk = sspwk._compute_linelum_from_weights(
        jnp.log10(dbk_weights.mstar_disk),
        spec_kern_results.dust_ftrans_lines,
        ssp_data,
        dbk_weights.ssp_weights_disk,
    )
    linelum_knots = sspwk._compute_linelum_from_weights(
        jnp.log10(dbk_weights.mstar_knots),
        spec_kern_results.dust_ftrans_lines,
        ssp_data,
        dbk_weights.ssp_weights_knots,
    )
    return DBKSpecInfo(linelum_bulge, linelum_disk, linelum_knots)


DBKPhotInfo = namedtuple(
    "DBKPhotInfo", ("obs_mags_bulge", "obs_mags_disk", "obs_mags_knots")
)
DBKSpecInfo = namedtuple(
    "DBKSpecInfo", ("linelum_bulge", "linelum_disk", "linelum_knots")
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
