""" """

from jax import jit as jjit
from jax import numpy as jnp
from jax import random as jran
from jax import vmap

from ..dustpop import tw_dustpop_mono, tw_dustpop_mono_noise
from ..ssp_err_model import ssp_err_model
from . import photometry_interpolation as photerp

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


@jjit
def multiband_lc_phot_kern(
    ran_key,
    lgmp_min,
    z_min,
    z_max,
    sky_area_degsq,
    z_obs,
    lgmp_obs,
    mah_params,
    ssp_data,
    tcurves,
    precomputed_ssp_mag_table,
    z_phot_table,
    wave_eff_table,
    cosmo_params,
    hmf_params,
    diffmahpop_params,
    diffstarpop_params,
    mzr_params,
    lgmet_scatter,
    diffburstpop_params,
    dustpop_params,
    dustpop_scatter_params,
    ssp_err_pop_params,
    n_hmf_grid,
    n_sfh_table,
    return_internal_quantities,
):
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
