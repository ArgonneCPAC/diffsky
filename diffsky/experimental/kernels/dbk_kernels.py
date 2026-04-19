""" """

from collections import namedtuple

from dsps.sfh import diffburst
from jax import jit as jjit
from jax import numpy as jnp
from jax import vmap

from ..disk_bulge_modeling import dbpop, disk_knots
from . import mc_randoms
from . import ssp_weight_kernels as sspwk

interp_vmap = jjit(vmap(jnp.interp, in_axes=(0, None, 0)))

_BPOP = (None, 0, 0)
get_pureburst_age_weights = jjit(
    vmap(diffburst._pureburst_age_weights_from_params, in_axes=_BPOP)
)


@jjit
def _mc_dbk_kern(
    t_obs, ssp_data, t_table, sfh_table, burst_params, lgmet_weights, dbk_key
):
    n_gals = t_obs.shape[0]
    dbk_randoms = mc_randoms.get_mc_dbk_randoms(dbk_key, n_gals)
    dbk_weights, disk_bulge_history = _dbk_kern(
        t_obs, ssp_data, t_table, sfh_table, burst_params, lgmet_weights, dbk_randoms
    )
    return dbk_randoms, dbk_weights, disk_bulge_history


@jjit
def _dbk_kern(
    t_obs, ssp_data, t_table, sfh_table, burst_params, lgmet_weights, dbk_randoms
):
    disk_bulge_history = dbpop.decompose_sfh_into_disk_bulge_sfh(
        dbk_randoms.uran_fbulge, t_table, sfh_table, t_obs
    )

    args = (
        t_obs,
        ssp_data,
        t_table,
        sfh_table,
        burst_params,
        lgmet_weights,
        disk_bulge_history,
        dbk_randoms.fknot,
    )
    dbk_weights = get_dbk_weights(*args)

    return dbk_weights, disk_bulge_history


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

    return obs_mags_bulge, obs_mags_disk, obs_mags_knots


@jjit
def _get_dbk_linelum_decomposition(dbk_weights, spec_kern_results, ssp_data):
    linelums_bulge = sspwk._compute_linelum_from_weights(
        jnp.log10(dbk_weights.mstar_bulge),
        spec_kern_results.dust_ftrans_lines,
        ssp_data,
        dbk_weights.ssp_weights_bulge,
    )
    linelums_disk = sspwk._compute_linelum_from_weights(
        jnp.log10(dbk_weights.mstar_disk),
        spec_kern_results.dust_ftrans_lines,
        ssp_data,
        dbk_weights.ssp_weights_disk,
    )
    linelums_knots = sspwk._compute_linelum_from_weights(
        jnp.log10(dbk_weights.mstar_knots),
        spec_kern_results.dust_ftrans_lines,
        ssp_data,
        dbk_weights.ssp_weights_knots,
    )
    return linelums_bulge, linelums_disk, linelums_knots


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
