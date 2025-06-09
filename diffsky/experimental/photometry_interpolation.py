# flake8: noqa: E402
""" """

from jax import config

config.update("jax_enable_x64", True)

from dsps.cosmology import age_at_z
from dsps.sed import calc_ssp_weights_sfh_table_lognormal_mdf
from jax import jit as jjit
from jax import numpy as jnp
from jax import vmap

_a = (None, 0, 0, None, None, None, 0)
calc_ssp_weights_sfh_table_lognormal_mdf_vmap = jjit(
    vmap(calc_ssp_weights_sfh_table_lognormal_mdf, in_axes=_a)
)


def get_interpolated_photometry(
    ssp_z_table,
    ssp_restmag_table,
    ssp_obsmag_table,
    ssp_lgmet,
    ssp_lg_age,
    gal_t_table,
    gal_z_obs,
    gal_logsm_obs,
    gal_sfr_table,
    gal_lgmet_obs,
    gal_lgmet_scatter,
    cosmo_params,
    dust_trans_factors_obs=1.0,
    dust_trans_factors_rest=1.0,
):
    """Calculate restframe and observed photometry of galaxies in a lightcone

    Method is to interpolate precomputed photometry of SSP SEDs

    Parameters
    ----------
    ssp_z_table : array of shape (n_z_table_ssp, )
        Array must be monotonically increasing and bracket the range of gal_z_obs

    ssp_restmag_table : array of shape (n_z_table_ssp, n_met, n_age, n_rest_filters)

    ssp_obsmag_table : array of shape (n_z_table_ssp, n_met, n_age, n_obs_filters)

    ssp_lgmet : array of shape (n_met, )
        Array of log10(Z) of the SSP templates

    ssp_lg_age : array of shape (n_ages, )
        Array of log10(age/Gyr) of the SSP templates

    gal_t_table : array of shape (n_t_table_gals, )
        Age of the universe in Gyr

    gal_z_obs : array of shape (n_gals, )
        Redshift of each galaxy

    gal_logsm_obs : array of shape (n_gals, )
        Base-10 log of the stellar mass of each galaxy at z_obs

    gal_sfr_table : ndarray of shape (n_gals, n_t_table_gals)
        Star formation history of each galaxy in units of Msun/yr,
        tabulated at the input gal_t_table

    gal_lgmet_obs : array of shape (n_gals, )
        log10(Z) of each galaxy at z_obs

    gal_lgmet_scatter : float
        Lognormal scatter in the metallicity distribution function

    cosmo_params : 4-element tuple
        NamedTuple of cosmological parameters Om0, w0, wa, h

    dust_trans_factors_obs : array of shape (n_gals, n_obs_filters), optional
        Fraction of the flux transmitted by dust through each observer-frame filter
        Default behavior is 100% transmission in all bands

    dust_trans_factors_rest : array of shape (n_gals, n_rest_filters), optional
        Fraction of the flux transmitted by dust through each restframe filter
        Default behavior is 100% transmission in all bands

    Returns
    -------
    gal_obsmags : array of shape (n_gals, n_obs_filters)

    gal_restmags : array of shape (n_gals, n_rest_filters)

    gal_obsmags_nodust : array of shape (n_gals, n_obs_filters)

    gal_restmags_nodust : array of shape (n_gals, n_rest_filters)

    """
    msg = "ssp_z_table must be monotonically increasing"
    assert jnp.all(jnp.diff(ssp_z_table) > 0), msg

    msg = "Must have ssp_z_table.min() < gal_z_obs.min()"
    assert jnp.all(ssp_z_table.min() < gal_z_obs.min()), msg

    msg = "Must have ssp_z_table.max() > gal_z_obs.max()"
    assert jnp.all(ssp_z_table.max() > gal_z_obs.max()), msg

    gal_t_obs = age_at_z(gal_z_obs, *cosmo_params)

    _res = calc_ssp_weights_sfh_table_lognormal_mdf_vmap(
        gal_t_table,
        gal_sfr_table,
        gal_lgmet_obs,
        gal_lgmet_scatter,
        ssp_lgmet,
        ssp_lg_age,
        gal_t_obs,
    )
    gal_weights, gal_lgmet_weights, gal_age_weights = _res

    ssp_obsmag_table_pergal = interpolate_ssp_photmag_table(
        gal_z_obs, ssp_z_table, ssp_obsmag_table
    )
    n_gals, n_met, n_age, n_obs_filters = ssp_obsmag_table_pergal.shape

    _w = gal_weights.reshape((n_gals, n_met, n_age, 1))
    ssp_obsflux_table_pergal = 10 ** (-0.4 * ssp_obsmag_table_pergal)

    gal_mstar_obs = (10**gal_logsm_obs).reshape((n_gals, 1))
    gal_obsflux_nodust = (
        jnp.sum(_w * ssp_obsflux_table_pergal, axis=(1, 2)) * gal_mstar_obs
    )
    gal_obsmags_nodust = -2.5 * jnp.log10(gal_obsflux_nodust)

    dust_trans_factors_obs = jnp.ones_like(gal_obsflux_nodust) * dust_trans_factors_obs

    gal_obsflux = gal_obsflux_nodust * dust_trans_factors_obs
    gal_obsmags = -2.5 * jnp.log10(gal_obsflux)

    n_met, n_age, n_rest_filters = ssp_restmag_table.shape
    ssp_restmag_table = ssp_restmag_table.reshape((1, n_met, n_age, n_rest_filters))
    ssp_restflux_table = 10 ** (-0.4 * ssp_restmag_table)

    gal_restflux_nodust = jnp.sum(_w * ssp_restflux_table, axis=(1, 2)) * gal_mstar_obs
    gal_restmags_nodust = -2.5 * jnp.log10(gal_restflux_nodust)

    dust_trans_factors_rest = (
        jnp.ones_like(gal_restflux_nodust) * dust_trans_factors_rest
    )
    gal_restflux = gal_restflux_nodust * dust_trans_factors_rest
    gal_restmags = -2.5 * jnp.log10(gal_restflux)

    return gal_obsmags, gal_restmags, gal_obsmags_nodust, gal_restmags_nodust


@jjit
def interpolate_ssp_photmag_table(z_gals, z_table, ssp_photmag_table):
    iz_hi = jnp.searchsorted(z_table, z_gals)
    iz_lo = iz_hi - 1
    z_lo = z_table[iz_lo]
    z_hi = z_table[iz_hi]
    dz_bin = z_hi - z_lo
    dz = z_gals - z_lo
    w_lo = 1 - (dz / dz_bin)

    ssp_table_zlo = ssp_photmag_table[iz_lo]
    ssp_table_zhi = ssp_photmag_table[iz_hi]

    s = ssp_table_zlo.shape
    outshape = [s[0], *[1 for x in s[1:]]]
    w_lo = w_lo.reshape(outshape)

    gal_photmags = w_lo * ssp_table_zlo + (1 - w_lo) * ssp_table_zhi
    return gal_photmags
