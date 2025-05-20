""" """

from collections import namedtuple

from diffstar.utils import cumulative_mstar_formed_galpop
from diffstarpop import mc_diffstar_sfh_galpop
from dsps.sed.stellar_age_weights import calc_age_weights_from_sfh_table
from jax import jit as jjit
from jax import numpy as jnp
from jax import random as jran
from jax import vmap

from ..burstpop import diffqburstpop_mono
from ..dustpop import tw_dustpop_mono, tw_dustpop_mono_noise
from ..ssp_err_model import ssp_err_model
from . import photometry_interpolation as photerp

_B = (None, 0, 0, None, 0)
_calc_bursty_age_weights_vmap = jjit(
    vmap(
        diffqburstpop_mono.calc_bursty_age_weights_from_diffburstpop_params, in_axes=_B
    )
)


_AGEPOP = (None, 0, None, 0)
calc_age_weights_from_sfh_table_vmap = jjit(
    vmap(calc_age_weights_from_sfh_table, in_axes=_AGEPOP)
)

interp_vmap = jjit(vmap(jnp.interp, in_axes=(0, None, 0)))
_B = (None, None, 1)
interp_vmap2 = jjit(vmap(jnp.interp, in_axes=_B, out_axes=1))

_F = (None, None, None, 0, None)
_G = (None, 0, 0, 0, None)
get_frac_ssp_err_vmap = jjit(
    vmap(vmap(ssp_err_model.F_sps_err_lambda, in_axes=_F), in_axes=_G)
)

_D = (None, 0, None, None, None, None, None, None, None, None)
vmap_kern1 = jjit(
    vmap(
        tw_dustpop_mono_noise.calc_ftrans_singlegal_singlewave_from_dustpop_params,
        in_axes=_D,
    )
)
_E = (None, 0, 0, 0, 0, None, 0, 0, 0, None)
calc_dust_ftrans_vmap = jjit(vmap(vmap_kern1, in_axes=_E))

_DPKEYS = (
    "frac_q",
    "sfh_ms",
    "logsm_obs_ms",
    "logssfr_obs_ms",
    "sfh_q",
    "logsm_obs_q",
    "logssfr_obs_q",
)
DiffstarPopQuantities = namedtuple("DiffstarPopQuantities", _DPKEYS)
DPQ_EMPTY = DiffstarPopQuantities._make([None] * len(_DPKEYS))


@jjit
def diffstarpop_lc_cen_wrapper(
    diffstarpop_params, ran_key, mah_params, logmp0, t_table, t_obs
):
    n_gals = logmp0.size
    upids = jnp.zeros(n_gals).astype(int) - 1
    lgmu_infall = jnp.zeros(n_gals) - 1.0
    logmhost_infall = jnp.copy(logmp0)
    lgmu_infall = jnp.zeros(n_gals) - 1.0
    gyr_since_infall = jnp.zeros(n_gals)

    args = (
        diffstarpop_params,
        mah_params,
        logmp0,
        upids,
        lgmu_infall,
        logmhost_infall,
        gyr_since_infall,
        ran_key,
        t_table,
    )
    dgp = mc_diffstar_sfh_galpop(*args)

    logsmh_table_ms = jnp.log10(cumulative_mstar_formed_galpop(t_table, dgp.sfh_ms))
    logsm_obs_ms = interp_vmap(t_obs, t_table, logsmh_table_ms)
    logsfr_obs_ms = interp_vmap(t_obs, t_table, jnp.log10(dgp.sfh_ms))
    logssfr_obs_ms = logsfr_obs_ms - logsm_obs_ms

    logsmh_table_q = jnp.log10(cumulative_mstar_formed_galpop(t_table, dgp.sfh_q))
    logsm_obs_q = interp_vmap(t_obs, t_table, logsmh_table_q)
    logsfr_obs_q = interp_vmap(t_obs, t_table, jnp.log10(dgp.sfh_q))
    logssfr_obs_q = logsfr_obs_q - logsm_obs_q

    diffstar_galpop = DPQ_EMPTY._replace(
        frac_q=dgp.frac_q,
        sfh_ms=dgp.sfh_ms,
        logsm_obs_ms=logsm_obs_ms,
        logssfr_obs_ms=logssfr_obs_ms,
        sfh_q=dgp.sfh_q,
        logsm_obs_q=logsm_obs_q,
        logssfr_obs_q=logssfr_obs_q,
    )

    return diffstar_galpop


@jjit
def multiband_lc_phot_kern(
    ran_key,
    z_obs,
    t_obs,
    mah_params,
    logmp0,
    t_table,
    ssp_data,
    precomputed_ssp_mag_table,
    z_phot_table,
    wave_eff_table,
    diffstarpop_params,
    mzr_params,
    lgmet_scatter,
    diffburstpop_params,
    dustpop_params,
    dustpop_scatter_params,
    ssp_err_pop_params,
):
    diffstar_galpop = diffstarpop_lc_cen_wrapper(
        diffstarpop_params, ran_key, mah_params, logmp0, t_table, t_obs
    )

    smooth_age_weights_ms = calc_age_weights_from_sfh_table_vmap(
        t_table, diffstar_galpop.sfh_ms, ssp_data.ssp_lg_age_gyr, t_obs
    )
    smooth_age_weights_q = calc_age_weights_from_sfh_table_vmap(
        t_table, diffstar_galpop.sfh_q, ssp_data.ssp_lg_age_gyr, t_obs
    )

    _args = (
        diffburstpop_params,
        diffstar_galpop.logsm_obs_ms,
        diffstar_galpop.logssfr_obs_ms,
        ssp_data.ssp_lg_age_gyr,
        smooth_age_weights_ms,
    )
    age_weights_ms, burst_params = _calc_bursty_age_weights_vmap(*_args)

    photmag_table_galpop = photerp.interpolate_ssp_photmag_table(
        z_obs, z_phot_table, precomputed_ssp_mag_table
    )
    ssp_photflux_table = 10 ** (-0.4 * photmag_table_galpop)

    wave_eff_galpop = interp_vmap2(z_obs, z_phot_table, wave_eff_table)

    # Delta mags
    frac_ssp_err = get_frac_ssp_err_vmap(
        ssp_err_pop_params,
        z_obs,
        logsm_obs,
        wave_eff_galpop,
        ssp_err_model.LAMBDA_REST,
    )

    n_gals = z_obs.size
    ran_key, dust_key = jran.split(ran_key, 2)
    av_key, delta_key, funo_key = jran.split(dust_key, 3)
    uran_av = jran.uniform(av_key, shape=(n_gals,))
    uran_delta = jran.uniform(delta_key, shape=(n_gals,))
    uran_funo = jran.uniform(funo_key, shape=(n_gals,))

    ftrans_args = (
        dustpop_params,
        wave_eff_galpop,
        logsm_obs,
        logssfr_obs,
        z_obs,
        ssp_data.ssp_lg_age_gyr,
        uran_av,
        uran_delta,
        uran_funo,
        dustpop_scatter_params,
    )
    _res = calc_dust_ftrans_vmap(*ftrans_args)
    ftrans_nonoise, ftrans, dust_params, noisy_dust_params = _res

    n_gals, n_bands, n_met, n_age = ssp_photflux_table.shape

    _w = ssp_weights.reshape((n_gals, 1, n_met, n_age))
    _sm = 10 ** logmp_obs.reshape((n_gals, 1))
    _ferr_ssp = frac_ssp_err.reshape((n_gals, n_bands, 1, 1))
    _ftrans = ftrans.reshape((n_gals, n_bands, 1, n_age))

    integrand = _w * ssp_photflux_table * _ftrans * _ferr_ssp
    photflux_galpop = jnp.sum(integrand, axis=(2, 3)) * _sm
    obs_mags = -2.5 * jnp.log10(photflux_galpop)

    return obs_mags
