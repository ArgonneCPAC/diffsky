# flake8: noqa: E402
""" """
import numpy as np
from jax import config

config.update("jax_enable_x64", True)


from collections import namedtuple

from diffstar.utils import cumulative_mstar_formed_galpop
from diffstarpop import mc_diffstar_sfh_galpop
from dsps.constants import T_TABLE_MIN
from dsps.cosmology import flat_wcdm
from dsps.metallicity import umzr
from dsps.sed import metallicity_weights as zmetw
from dsps.sed.stellar_age_weights import calc_age_weights_from_sfh_table
from jax import jit as jjit
from jax import numpy as jnp
from jax import random as jran
from jax import vmap

from ..burstpop import diffqburstpop_mono, freqburst_mono
from ..dustpop import tw_dustpop_mono_noise
from ..param_utils import diffsky_param_wrapper as dpw
from ..phot_utils import get_wave_eff_from_tcurves
from ..ssp_err_model import ssp_err_model
from . import mc_lightcone_halos as mclh
from . import photometry_interpolation as photerp

_M = (0, None, None)
_calc_lgmet_weights_galpop = jjit(
    vmap(zmetw.calc_lgmet_weights_from_lognormal_mdf, in_axes=_M)
)

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


_LCPHOT_RET_KEYS = (
    "obs_mags_bursty_ms",
    "obs_mags_smooth_ms",
    "obs_mags_q",
    "weights_bursty_ms",
    "weights_smooth_ms",
    "weights_q",
)
LCPhot = namedtuple("LCPhot", _LCPHOT_RET_KEYS)
LCPHOT_EMPTY = LCPhot._make([None] * len(LCPhot._fields))

LGMET_SCATTER = 0.2
N_SFH_TABLE = 100


def get_wave_eff_table(z_phot_table, tcurves):
    collector = []
    for z_obs in z_phot_table:
        wave_eff = get_wave_eff_from_tcurves(tcurves, z_obs)
        collector.append(wave_eff)
    wave_eff_table = jnp.array(collector)
    return wave_eff_table


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
    _res = mc_diffstar_sfh_galpop(*args)
    diffstar_params_ms, diffstar_params_q, sfh_ms, sfh_q, frac_q, mc_is_q = _res

    logsmh_table_ms = jnp.log10(cumulative_mstar_formed_galpop(t_table, sfh_ms))
    logsm_obs_ms = interp_vmap(t_obs, t_table, logsmh_table_ms)
    logsfr_obs_ms = interp_vmap(t_obs, t_table, jnp.log10(sfh_ms))
    logssfr_obs_ms = logsfr_obs_ms - logsm_obs_ms

    logsmh_table_q = jnp.log10(cumulative_mstar_formed_galpop(t_table, sfh_q))
    logsm_obs_q = interp_vmap(t_obs, t_table, logsmh_table_q)
    logsfr_obs_q = interp_vmap(t_obs, t_table, jnp.log10(sfh_q))
    logssfr_obs_q = logsfr_obs_q - logsm_obs_q

    diffstar_galpop = DPQ_EMPTY._replace(
        frac_q=frac_q,
        sfh_ms=sfh_ms,
        logsm_obs_ms=logsm_obs_ms,
        logssfr_obs_ms=logssfr_obs_ms,
        sfh_q=sfh_q,
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
    spspop_params,
    scatter_params,
    ssp_err_pop_params,
):
    n_z_table, n_bands, n_met, n_age = precomputed_ssp_mag_table.shape
    n_gals = logmp0.size

    ran_key, sfh_key = jran.split(ran_key, 2)
    diffstar_galpop = diffstarpop_lc_cen_wrapper(
        diffstarpop_params, sfh_key, mah_params, logmp0, t_table, t_obs
    )

    smooth_age_weights_ms = calc_age_weights_from_sfh_table_vmap(
        t_table, diffstar_galpop.sfh_ms, ssp_data.ssp_lg_age_gyr, t_obs
    )
    smooth_age_weights_q = calc_age_weights_from_sfh_table_vmap(
        t_table, diffstar_galpop.sfh_q, ssp_data.ssp_lg_age_gyr, t_obs
    )

    _args = (
        spspop_params.burstpop_params,
        diffstar_galpop.logsm_obs_ms,
        diffstar_galpop.logssfr_obs_ms,
        ssp_data.ssp_lg_age_gyr,
        smooth_age_weights_ms,
    )
    bursty_age_weights_ms, burst_params = _calc_bursty_age_weights_vmap(*_args)

    p_burst_ms = freqburst_mono.get_freqburst_from_freqburst_params(
        spspop_params.burstpop_params.freqburst_params,
        diffstar_galpop.logsm_obs_ms,
        diffstar_galpop.logssfr_obs_ms,
    )

    lgmet_med_ms = umzr.mzr_model(diffstar_galpop.logsm_obs_ms, t_obs, *mzr_params)
    lgmet_med_q = umzr.mzr_model(diffstar_galpop.logsm_obs_q, t_obs, *mzr_params)

    lgmet_weights_ms = _calc_lgmet_weights_galpop(
        lgmet_med_ms, LGMET_SCATTER, ssp_data.ssp_lgmet
    )
    lgmet_weights_q = _calc_lgmet_weights_galpop(
        lgmet_med_q, LGMET_SCATTER, ssp_data.ssp_lgmet
    )

    _w_age_q = smooth_age_weights_q.reshape((n_gals, 1, n_age))
    _w_lgmet_q = lgmet_weights_q.reshape((n_gals, n_met, 1))
    ssp_weights_q = _w_lgmet_q * _w_age_q

    _w_age_ms = smooth_age_weights_ms.reshape((n_gals, 1, n_age))
    _w_lgmet_ms = lgmet_weights_ms.reshape((n_gals, n_met, 1))
    ssp_weights_smooth_ms = _w_lgmet_ms * _w_age_ms

    _w_age_bursty_ms = bursty_age_weights_ms.reshape((n_gals, 1, n_age))
    ssp_weights_bursty_ms = _w_lgmet_ms * _w_age_bursty_ms

    photmag_table_galpop = photerp.interpolate_ssp_photmag_table(
        z_obs, z_phot_table, precomputed_ssp_mag_table
    )
    ssp_photflux_table = 10 ** (-0.4 * photmag_table_galpop)

    wave_eff_galpop = interp_vmap2(z_obs, z_phot_table, wave_eff_table)

    # Delta mags
    frac_ssp_err_q = get_frac_ssp_err_vmap(
        ssp_err_pop_params,
        z_obs,
        diffstar_galpop.logsm_obs_q,
        wave_eff_galpop,
        ssp_err_model.LAMBDA_REST,
    )
    frac_ssp_err_ms = get_frac_ssp_err_vmap(
        ssp_err_pop_params,
        z_obs,
        diffstar_galpop.logsm_obs_ms,
        wave_eff_galpop,
        ssp_err_model.LAMBDA_REST,
    )

    ran_key, dust_key = jran.split(ran_key, 2)
    av_key, delta_key, funo_key = jran.split(dust_key, 3)
    uran_av = jran.uniform(av_key, shape=(n_gals,))
    uran_delta = jran.uniform(delta_key, shape=(n_gals,))
    uran_funo = jran.uniform(funo_key, shape=(n_gals,))

    ftrans_args_q = (
        spspop_params.dustpop_params,
        wave_eff_galpop,
        diffstar_galpop.logsm_obs_q,
        diffstar_galpop.logssfr_obs_q,
        z_obs,
        ssp_data.ssp_lg_age_gyr,
        uran_av,
        uran_delta,
        uran_funo,
        scatter_params,
    )
    _res = calc_dust_ftrans_vmap(*ftrans_args_q)
    ftrans_q = _res[1]

    ftrans_args_ms = (
        spspop_params.dustpop_params,
        wave_eff_galpop,
        diffstar_galpop.logsm_obs_ms,
        diffstar_galpop.logssfr_obs_ms,
        z_obs,
        ssp_data.ssp_lg_age_gyr,
        uran_av,
        uran_delta,
        uran_funo,
        scatter_params,
    )
    _res = calc_dust_ftrans_vmap(*ftrans_args_ms)
    ftrans_ms = _res[1]

    _mstar_ms = 10 ** diffstar_galpop.logsm_obs_ms.reshape((n_gals, 1))
    _mstar_q = 10 ** diffstar_galpop.logsm_obs_q.reshape((n_gals, 1))

    _w_smooth_ms = ssp_weights_smooth_ms.reshape((n_gals, 1, n_met, n_age))
    _w_bursty_ms = ssp_weights_bursty_ms.reshape((n_gals, 1, n_met, n_age))
    _w_q = ssp_weights_q.reshape((n_gals, 1, n_met, n_age))

    _ferr_ssp_ms = frac_ssp_err_ms.reshape((n_gals, n_bands, 1, 1))
    _ferr_ssp_q = frac_ssp_err_q.reshape((n_gals, n_bands, 1, 1))

    ran_key, ssp_q_key, ssp_ms_key = jran.split(ran_key, 3)
    delta_scatter_q = ssp_err_model.compute_delta_scatter(ssp_q_key, frac_ssp_err_q)
    delta_scatter_ms = ssp_err_model.compute_delta_scatter(ssp_ms_key, frac_ssp_err_ms)

    _ftrans_ms = ftrans_ms.reshape((n_gals, n_bands, 1, n_age))
    _ftrans_q = ftrans_q.reshape((n_gals, n_bands, 1, n_age))

    integrand_q = ssp_photflux_table * _w_q * _ftrans_q * _ferr_ssp_q
    photflux_galpop_q = jnp.sum(integrand_q, axis=(2, 3)) * _mstar_q
    obs_mags_q = -2.5 * jnp.log10(photflux_galpop_q) + delta_scatter_q

    integrand_smooth_ms = ssp_photflux_table * _w_smooth_ms * _ftrans_ms * _ferr_ssp_ms
    photflux_galpop_smooth_ms = jnp.sum(integrand_smooth_ms, axis=(2, 3)) * _mstar_ms
    obs_mags_smooth_ms = -2.5 * jnp.log10(photflux_galpop_smooth_ms) + delta_scatter_ms

    integrand_bursty_ms = ssp_photflux_table * _w_bursty_ms * _ftrans_ms * _ferr_ssp_ms
    photflux_galpop_bursty_ms = jnp.sum(integrand_bursty_ms, axis=(2, 3)) * _mstar_ms
    obs_mags_bursty_ms = -2.5 * jnp.log10(photflux_galpop_bursty_ms) + delta_scatter_ms

    weights_q = diffstar_galpop.frac_q
    weights_smooth_ms = (1 - diffstar_galpop.frac_q) * (1 - p_burst_ms)
    weights_bursty_ms = (1 - diffstar_galpop.frac_q) * p_burst_ms

    lc_phot = LCPHOT_EMPTY._replace(
        obs_mags_bursty_ms=obs_mags_bursty_ms,
        obs_mags_smooth_ms=obs_mags_smooth_ms,
        obs_mags_q=obs_mags_q,
        weights_bursty_ms=weights_bursty_ms,
        weights_smooth_ms=weights_smooth_ms,
        weights_q=weights_q,
    )

    return lc_phot


@jjit
def multiband_lc_phot_kern_u_param_arr(u_param_arr, ran_key, lc_data):
    """Kernel for KDE-based loss of lightcone photometry predictions

    Parameters
    ----------
    u_param_arr : ndarray, shape (n_params,)
        Flat array of unbounded parameters

    ran_key : jran.key(seed)

    lc_data : namedtuple
        Precomputed lightcone data generated by the generate_lc_data function

    Returns
    -------
    lc_phot : namedtuple
        Fields:
            obs_mags_bursty_ms, array with shape (n_gals, n_bands)
            obs_mags_smooth_ms, array with shape (n_gals, n_bands)
            obs_mags_q, array with shape (n_gals, n_bands)
            weights_bursty_ms, array with shape (n_gals, )
            weights_smooth_ms, array with shape (n_gals, )
            weights_q, array with shape (n_gals, )

    """
    u_param_collection = dpw.get_u_param_collection_from_u_param_array(u_param_arr)
    param_collection = dpw.get_param_collection_from_u_param_collection(
        *u_param_collection
    )
    lc_phot = multiband_lc_phot_kern(ran_key, *lc_data[1:], *param_collection)
    return lc_phot


def generate_lc_data(
    ran_key,
    lgmp_min,
    z_min,
    z_max,
    sky_area_degsq,
    ssp_data,
    cosmo_params,
    tcurves,
    z_phot_table,
):
    mclh_args = (ran_key, lgmp_min, z_min, z_max, sky_area_degsq)
    lc_halopop = mclh.mc_lightcone_host_halo_diffmah(*mclh_args)

    t0 = flat_wcdm.age_at_z0(*cosmo_params)
    t_table = jnp.linspace(T_TABLE_MIN, t0, N_SFH_TABLE)

    precomputed_ssp_mag_table = mclh.get_precompute_ssp_mag_redshift_table(
        tcurves, ssp_data, z_phot_table
    )
    wave_eff_table = get_wave_eff_table(z_phot_table, tcurves)
    nhalos = np.ones(len(lc_halopop["z_obs"]))

    lc_data = LCData(
        nhalos,
        lc_halopop["z_obs"],
        lc_halopop["t_obs"],
        lc_halopop["mah_params"],
        lc_halopop["logmp0"],
        t_table,
        ssp_data,
        precomputed_ssp_mag_table,
        z_phot_table,
        wave_eff_table,
    )
    return lc_data


def generate_weighted_grid_lc_data(
    ran_key,
    lgmp_grid,
    z_grid,
    sky_area_degsq,
    ssp_data,
    cosmo_params,
    tcurves,
    z_phot_table,
):
    args = (ran_key, lgmp_grid, z_grid, sky_area_degsq)
    lc_grid = mclh.get_weighted_lightcone_grid_host_halo_diffmah(*args)

    t0 = flat_wcdm.age_at_z0(*cosmo_params)
    t_table = jnp.linspace(T_TABLE_MIN, t0, N_SFH_TABLE)

    precomputed_ssp_mag_table = mclh.get_precompute_ssp_mag_redshift_table(
        tcurves, ssp_data, z_phot_table
    )
    wave_eff_table = get_wave_eff_table(z_phot_table, tcurves)

    lc_data = LCData(
        lc_grid["nhalos"],
        lc_grid["z_obs"],
        lc_grid["t_obs"],
        lc_grid["mah_params"],
        lc_grid["logmp0"],
        t_table,
        ssp_data,
        precomputed_ssp_mag_table,
        z_phot_table,
        wave_eff_table,
    )
    return lc_data


_LCDKEYS = (
    "nhalos",
    "z_obs",
    "t_obs",
    "mah_params",
    "logmp0",
    "t_table",
    "ssp_data",
    "precomputed_ssp_mag_table",
    "z_phot_table",
    "wave_eff_table",
)
LCData = namedtuple("LCData", _LCDKEYS)
