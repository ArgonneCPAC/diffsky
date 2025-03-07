"""
"""

from collections import OrderedDict, namedtuple

from diffstarpop.defaults import DEFAULT_DIFFSTARPOP_PARAMS
from dsps.cosmology.defaults import DEFAULT_COSMOLOGY
from dsps.metallicity import umzr
from dsps.photometry import photpop
from dsps.sed import metallicity_weights as zmetw
from dsps.sed import stellar_age_weights as saw
from jax import jit as jjit
from jax import nn
from jax import numpy as jnp
from jax import random as jran
from jax import vmap

from .. import mc_diffsky as mcd
from ..burstpop import diffqburstpop
from ..dustpop import avpop_mono, deltapop, funopop_ssfr, tw_dust, tw_dustpop_mono
from ..phot_utils import get_wave_eff_from_tcurves, load_interpolated_lsst_curves
from ..utils import _inverse_sigmoid
from . import ssp_err_pop

DEFAULT_SCATTER_PDICT = OrderedDict(
    delta_scatter=5.0,
    av_scatter=5.0,
    lgfburst_scatter=5.0,
    lgmet_scatter=5.0,
    funo_scatter=5.0,
)
ScatterParams = namedtuple("ScatterParams", list(DEFAULT_SCATTER_PDICT.keys()))
DEFAULT_SCATTER_PARAMS = ScatterParams(*DEFAULT_SCATTER_PDICT.values())


# gal_t_table, gal_sfr_table, ssp_lg_age_gyr, t_obs, sfr_min
_A = (None, 0, None, None, None)
_calc_age_weights_galpop = jjit(vmap(saw.calc_age_weights_from_sfh_table, in_axes=_A))

# gal_lgmet, gal_lgmet_scatter, ssp_lgmet
_M = (0, None, None)
_calc_lgmet_weights_galpop = jjit(
    vmap(zmetw.calc_lgmet_weights_from_lognormal_mdf, in_axes=_M)
)

# diffburstpop_params, logsm, logssfr, ssp_lg_age_gyr, smooth_age_weights
_B = (None, 0, 0, None, 0)
_calc_bursty_age_weights_vmap = jjit(
    vmap(diffqburstpop.calc_bursty_age_weights_from_diffburstpop_params, in_axes=_B)
)


def mc_diffsky_galpop_lsst_phot(
    ran_key,
    z_obs,
    lgmp_min,
    volume_com,
    ssp_data,
    cosmo_params=DEFAULT_COSMOLOGY,
    diffstarpop_params=DEFAULT_DIFFSTARPOP_PARAMS,
    dustpop_params=tw_dustpop_mono.DEFAULT_DUSTPOP_PARAMS,
    mzr_params=umzr.DEFAULT_MZR_PARAMS,
    lgmet_scatter=umzr.MZR_SCATTER,
    diffburstpop_params=diffqburstpop.DEFAULT_DIFFBURSTPOP_PARAMS,
    scatter_params=DEFAULT_SCATTER_PARAMS,
    ssp_err_pop_params=ssp_err_pop.DEFAULT_SSP_ERR_POP_PARAMS,
    n_t=mcd.N_T,
    drn_ssp_data=mcd.DSPS_DATA_DRN,
    return_internal_quantities=False,
):
    diffstar_key, ran_key = jran.split(ran_key, 2)
    diffstar_data = mcd.mc_diffstar_galpop(
        diffstar_key,
        z_obs,
        lgmp_min,
        volume_com=volume_com,
        cosmo_params=cosmo_params,
        diffstarpop_params=diffstarpop_params,
        n_t=n_t,
        return_internal_quantities=return_internal_quantities,
    )
    diffsky_data = predict_lsst_phot_from_diffstar(
        diffstar_data,
        ran_key,
        z_obs,
        ssp_data,
        dustpop_params=dustpop_params,
        mzr_params=mzr_params,
        lgmet_scatter=lgmet_scatter,
        diffburstpop_params=diffburstpop_params,
        scatter_params=scatter_params,
        ssp_err_pop_params=ssp_err_pop_params,
        drn_ssp_data=drn_ssp_data,
        return_internal_quantities=return_internal_quantities,
    )
    return diffsky_data


def mc_diffsky_cenpop_lsst_phot(
    ran_key,
    z_obs,
    hosts_logmh_at_z,
    ssp_data,
    cosmo_params=DEFAULT_COSMOLOGY,
    diffstarpop_params=DEFAULT_DIFFSTARPOP_PARAMS,
    dustpop_params=tw_dustpop_mono.DEFAULT_DUSTPOP_PARAMS,
    mzr_params=umzr.DEFAULT_MZR_PARAMS,
    lgmet_scatter=umzr.MZR_SCATTER,
    diffburstpop_params=diffqburstpop.DEFAULT_DIFFBURSTPOP_PARAMS,
    scatter_params=DEFAULT_SCATTER_PARAMS,
    ssp_err_pop_params=ssp_err_pop.DEFAULT_SSP_ERR_POP_PARAMS,
    n_t=mcd.N_T,
    drn_ssp_data=mcd.DSPS_DATA_DRN,
    return_internal_quantities=False,
):
    diffstar_key, ran_key = jran.split(ran_key, 2)
    diffstar_data = mcd.mc_diffstar_cenpop(
        diffstar_key,
        z_obs,
        hosts_logmh_at_z=hosts_logmh_at_z,
        cosmo_params=cosmo_params,
        diffstarpop_params=diffstarpop_params,
        n_t=n_t,
        return_internal_quantities=return_internal_quantities,
    )
    diffsky_data = predict_lsst_phot_from_diffstar(
        diffstar_data,
        ran_key,
        z_obs,
        ssp_data,
        dustpop_params=dustpop_params,
        mzr_params=mzr_params,
        lgmet_scatter=lgmet_scatter,
        diffburstpop_params=diffburstpop_params,
        scatter_params=scatter_params,
        ssp_err_pop_params=ssp_err_pop_params,
        drn_ssp_data=drn_ssp_data,
        return_internal_quantities=return_internal_quantities,
    )
    return diffsky_data


def predict_lsst_phot_from_diffstar(
    diffstar_data,
    ran_key,
    z_obs,
    ssp_data,
    dustpop_params=tw_dustpop_mono.DEFAULT_DUSTPOP_PARAMS,
    mzr_params=umzr.DEFAULT_MZR_PARAMS,
    lgmet_scatter=umzr.MZR_SCATTER,
    diffburstpop_params=diffqburstpop.DEFAULT_DIFFBURSTPOP_PARAMS,
    scatter_params=DEFAULT_SCATTER_PARAMS,
    ssp_err_pop_params=ssp_err_pop.DEFAULT_SSP_ERR_POP_PARAMS,
    drn_ssp_data=mcd.DSPS_DATA_DRN,
    return_internal_quantities=False,
):
    diffsky_data = diffstar_data.copy()

    lgmet_key, ran_key = jran.split(ran_key, 2)
    lgmet_med = umzr.mzr_model(
        diffsky_data["logsm_obs"], diffsky_data["t_obs"], *mzr_params
    )
    unorm = jran.normal(lgmet_key, shape=lgmet_med.shape) * lgmet_scatter
    diffsky_data["lgmet_med"] = lgmet_med + unorm

    diffsky_data["smooth_age_weights"] = _calc_age_weights_galpop(
        diffsky_data["t_table"],
        diffsky_data["sfh"],
        ssp_data.ssp_lg_age_gyr,
        diffsky_data["t_obs"],
        saw.SFR_MIN,
    )

    if return_internal_quantities:
        diffsky_data["smooth_age_weights_ms"] = _calc_age_weights_galpop(
            diffsky_data["t_table"],
            diffsky_data["sfh_ms"],
            ssp_data.ssp_lg_age_gyr,
            diffsky_data["t_obs"],
            saw.SFR_MIN,
        )
        diffsky_data["smooth_age_weights_q"] = _calc_age_weights_galpop(
            diffsky_data["t_table"],
            diffsky_data["sfh_q"],
            ssp_data.ssp_lg_age_gyr,
            diffsky_data["t_obs"],
            saw.SFR_MIN,
        )

    diffsky_data["lgmet_weights"] = _calc_lgmet_weights_galpop(
        diffsky_data["lgmet_med"], lgmet_scatter, ssp_data.ssp_lgmet
    )

    _args = (
        diffburstpop_params,
        diffsky_data["logsm_obs"],
        diffsky_data["logssfr_obs"],
        ssp_data.ssp_lg_age_gyr,
        diffsky_data["smooth_age_weights"],
    )
    bursty_age_weights, burst_params = _calc_bursty_age_weights_vmap(*_args)
    diffsky_data["bursty_age_weights"] = bursty_age_weights

    if return_internal_quantities:
        _args = (
            diffburstpop_params,
            diffsky_data["logsm_obs"],
            diffsky_data["logssfr_obs"],
            ssp_data.ssp_lg_age_gyr,
            diffsky_data["smooth_age_weights_ms"],
        )
        bursty_age_weights, burst_params = _calc_bursty_age_weights_vmap(*_args)
        diffsky_data["bursty_age_weights_ms"] = bursty_age_weights

        _args = (
            diffburstpop_params,
            diffsky_data["logsm_obs"],
            diffsky_data["logssfr_obs"],
            ssp_data.ssp_lg_age_gyr,
            diffsky_data["smooth_age_weights_q"],
        )
        bursty_age_weights, burst_params = _calc_bursty_age_weights_vmap(*_args)
        diffsky_data["bursty_age_weights_q"] = bursty_age_weights

    n_gals = diffsky_data["sfh"].shape[0]
    n_met, n_age = ssp_data.ssp_flux.shape[0:2]

    _wmet = diffsky_data["lgmet_weights"].reshape((n_gals, n_met, 1))

    _amet = diffsky_data["smooth_age_weights"].reshape((n_gals, 1, n_age))
    smooth_weights = _wmet * _amet
    _norm = jnp.sum(smooth_weights, axis=(1, 2))
    smooth_weights = smooth_weights / _norm.reshape((n_gals, 1, 1))
    diffsky_data["smooth_ssp_weights"] = smooth_weights

    if return_internal_quantities:
        _amet = diffsky_data["smooth_age_weights_ms"].reshape((n_gals, 1, n_age))
        smooth_weights = _wmet * _amet
        _norm = jnp.sum(smooth_weights, axis=(1, 2))
        smooth_weights = smooth_weights / _norm.reshape((n_gals, 1, 1))
        diffsky_data["smooth_ssp_weights_ms"] = smooth_weights

        _amet = diffsky_data["smooth_age_weights_q"].reshape((n_gals, 1, n_age))
        smooth_weights = _wmet * _amet
        _norm = jnp.sum(smooth_weights, axis=(1, 2))
        smooth_weights = smooth_weights / _norm.reshape((n_gals, 1, 1))
        diffsky_data["smooth_ssp_weights_q"] = smooth_weights

    _bmet = diffsky_data["bursty_age_weights"].reshape((n_gals, 1, n_age))
    bursty_weights = _wmet * _bmet
    _norm = jnp.sum(bursty_weights, axis=(1, 2))
    bursty_weights = bursty_weights / _norm.reshape((n_gals, 1, 1))
    diffsky_data["bursty_ssp_weights"] = bursty_weights

    if return_internal_quantities:
        _bmet = diffsky_data["bursty_age_weights_ms"].reshape((n_gals, 1, n_age))
        bursty_weights = _wmet * _bmet
        _norm = jnp.sum(bursty_weights, axis=(1, 2))
        bursty_weights = bursty_weights / _norm.reshape((n_gals, 1, 1))
        diffsky_data["bursty_ssp_weights_ms"] = bursty_weights

        _bmet = diffsky_data["bursty_age_weights_q"].reshape((n_gals, 1, n_age))
        bursty_weights = _wmet * _bmet
        _norm = jnp.sum(bursty_weights, axis=(1, 2))
        bursty_weights = bursty_weights / _norm.reshape((n_gals, 1, 1))
        diffsky_data["bursty_ssp_weights_q"] = bursty_weights

    lsst_tcurves_interp, lsst_tcurves_sparse = load_interpolated_lsst_curves(
        ssp_data.ssp_wave, drn_ssp_data=drn_ssp_data
    )
    wave_eff_ugrizy_aa = get_wave_eff_from_tcurves(lsst_tcurves_sparse, z_obs)
    n_bands = wave_eff_ugrizy_aa.size

    X = jnp.array([ssp_data.ssp_wave] * 6)
    Y = jnp.array([x.transmission for x in lsst_tcurves_interp])

    _ssp_flux_table = 10 ** (
        -0.4
        * photpop.precompute_ssp_restmags(ssp_data.ssp_wave, ssp_data.ssp_flux, X, Y)
    )
    ssp_flux_table_multiband = jnp.swapaxes(jnp.swapaxes(_ssp_flux_table, 0, 2), 1, 2)

    av_key, delta_key, funo_key = jran.split(ran_key, 3)
    uran_av = jran.uniform(av_key, shape=(n_gals,))
    uran_delta = jran.uniform(delta_key, shape=(n_gals,))
    uran_funo = jran.uniform(funo_key, shape=(n_gals,))

    frac_trans_nonoise, frac_trans_noisy = calc_dust_ftrans_vmap(
        dustpop_params,
        wave_eff_ugrizy_aa,
        diffsky_data["logsm_obs"],
        diffsky_data["logssfr_obs"],
        z_obs,
        ssp_data.ssp_lg_age_gyr,
        uran_av,
        uran_delta,
        uran_funo,
        scatter_params,
    )

    logsm_obs = diffsky_data["logsm_obs"].reshape((n_gals, 1, 1, 1))
    gal_flux_table_nodust = ssp_flux_table_multiband * 10**logsm_obs

    flux_factor = ssp_err_pop.get_flux_factor_from_lgssfr_vmap(
        ssp_err_pop_params, diffsky_data["logssfr_obs"], wave_eff_ugrizy_aa
    )
    _ff = flux_factor.reshape((n_gals, n_bands, 1, 1))
    gal_flux_table_nodust = gal_flux_table_nodust * _ff

    n_gals, n_filters, n_met, n_age = gal_flux_table_nodust.shape
    _s = (n_gals, n_filters, 1, n_age)
    gal_flux_table_dust = gal_flux_table_nodust * frac_trans_noisy.reshape(_s)

    w = diffsky_data["smooth_ssp_weights"].reshape((n_gals, 1, n_met, n_age))
    flux = jnp.sum(gal_flux_table_nodust * w, axis=(2, 3))
    mag_smooth_nodust = -2.5 * jnp.log10(flux)
    flux = jnp.sum(gal_flux_table_dust * w, axis=(2, 3))
    mag_smooth_dust = -2.5 * jnp.log10(flux)

    w = diffsky_data["bursty_ssp_weights"].reshape((n_gals, 1, n_met, n_age))
    flux = jnp.sum(gal_flux_table_nodust * w, axis=(2, 3))
    mag_bursty_nodust = -2.5 * jnp.log10(flux)
    flux = jnp.sum(gal_flux_table_dust * w, axis=(2, 3))
    mag_bursty_dust = -2.5 * jnp.log10(flux)

    diffsky_data["rest_ugrizy_smooth_nodust"] = mag_smooth_nodust
    diffsky_data["rest_ugrizy_bursty_nodust"] = mag_bursty_nodust
    diffsky_data["rest_ugrizy_smooth_dust"] = mag_smooth_dust
    diffsky_data["rest_ugrizy_bursty_dust"] = mag_bursty_dust

    if return_internal_quantities:
        w = diffsky_data["smooth_ssp_weights_ms"].reshape((n_gals, 1, n_met, n_age))
        mags_nodust = -2.5 * jnp.log10(jnp.sum(gal_flux_table_nodust * w, axis=(2, 3)))
        mags_dust = -2.5 * jnp.log10(jnp.sum(gal_flux_table_dust * w, axis=(2, 3)))
        diffsky_data["rest_ugrizy_smooth_nodust_ms"] = mags_nodust
        diffsky_data["rest_ugrizy_smooth_dust_ms"] = mags_dust

        w = diffsky_data["smooth_ssp_weights_q"].reshape((n_gals, 1, n_met, n_age))
        mags_nodust = -2.5 * jnp.log10(jnp.sum(gal_flux_table_nodust * w, axis=(2, 3)))
        mags_dust = -2.5 * jnp.log10(jnp.sum(gal_flux_table_dust * w, axis=(2, 3)))
        diffsky_data["rest_ugrizy_smooth_nodust_q"] = mags_nodust
        diffsky_data["rest_ugrizy_smooth_dust_q"] = mags_dust

        w = diffsky_data["bursty_ssp_weights_ms"].reshape((n_gals, 1, n_met, n_age))
        mags_nodust = -2.5 * jnp.log10(jnp.sum(gal_flux_table_nodust * w, axis=(2, 3)))
        mags_dust = -2.5 * jnp.log10(jnp.sum(gal_flux_table_dust * w, axis=(2, 3)))
        diffsky_data["rest_ugrizy_bursty_nodust_ms"] = mags_nodust
        diffsky_data["rest_ugrizy_bursty_dust_ms"] = mags_dust

        w = diffsky_data["bursty_ssp_weights_q"].reshape((n_gals, 1, n_met, n_age))
        mags_nodust = -2.5 * jnp.log10(jnp.sum(gal_flux_table_nodust * w, axis=(2, 3)))
        mags_dust = -2.5 * jnp.log10(jnp.sum(gal_flux_table_dust * w, axis=(2, 3)))
        diffsky_data["rest_ugrizy_bursty_nodust_q"] = mags_nodust
        diffsky_data["rest_ugrizy_bursty_dust_q"] = mags_dust

        diffsky_data["frac_trans_nonoise"] = frac_trans_nonoise
        diffsky_data["frac_trans_noisy"] = frac_trans_noisy
        diffsky_data["wave_eff_ugrizy_aa"] = wave_eff_ugrizy_aa
        diffsky_data['flux_factor'] = flux_factor

    return diffsky_data


@jjit
def calc_dust_ftrans(
    dustpop_params,
    wave_aa,
    logsm,
    logssfr,
    redshift,
    ssp_lg_age_gyr,
    uran_av,
    uran_delta,
    uran_funo,
    scatter_params,
):
    av = avpop_mono.get_av_from_avpop_params_singlegal(
        dustpop_params.avpop_params, logsm, logssfr, redshift, ssp_lg_age_gyr
    )
    delta = deltapop.get_delta_from_deltapop_params(
        dustpop_params.deltapop_params, logsm, logssfr
    )
    funo = funopop_ssfr.get_funo_from_funopop_params(
        dustpop_params.funopop_params, logssfr
    )

    dust_params = tw_dust.DustParams(av, delta, funo)
    ftrans = tw_dust.calc_dust_frac_trans(wave_aa, dust_params)

    suav = jnp.log(jnp.exp(av) - 1)
    noisy_suav = _inverse_sigmoid(uran_av, suav, scatter_params.av_scatter, 0.0, 1.0)
    noisy_av = nn.softplus(noisy_suav)

    udelta = deltapop._get_unbounded_deltapop_param(delta, deltapop.DELTAPOP_BOUNDS)
    noisy_udelta = _inverse_sigmoid(
        uran_delta, udelta, scatter_params.delta_scatter, 0.0, 1.0
    )
    noisy_delta = deltapop._get_bounded_deltapop_param(
        noisy_udelta, deltapop.DELTAPOP_BOUNDS
    )

    ufuno = funopop_ssfr._get_u_p_from_p_scalar(funo, funopop_ssfr.FUNO_BOUNDS)
    noisy_ufuno = _inverse_sigmoid(
        uran_funo, ufuno, scatter_params.funo_scatter, 0.0, 1.0
    )
    noisy_funo = funopop_ssfr._get_p_from_u_p_scalar(
        noisy_ufuno, funopop_ssfr.FUNO_BOUNDS
    )

    noisy_dust_params = tw_dust.DustParams(noisy_av, noisy_delta, noisy_funo)
    ftrans_noisy = tw_dust.calc_dust_frac_trans(wave_aa, noisy_dust_params)

    return ftrans, ftrans_noisy


_A = (None, 0, None, None, None, None, None, None, None, None)
_B = [None, None, 0, 0, None, None, 0, 0, 0, None]

_f = jjit(vmap(calc_dust_ftrans, in_axes=_A))
calc_dust_ftrans_vmap = jjit(vmap(_f, in_axes=_B))
