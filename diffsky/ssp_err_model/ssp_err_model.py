""" """

from dsps.utils import _tw_sigmoid
from jax import jit as jjit
from jax import numpy as jnp
from jax import random as jran
from jax import vmap

from ..utils import _sigmoid, _tw_interp_kern
from .defaults import (  # noqa
    DEFAULT_SSPERR_PARAMS,
    DEFAULT_SSPERR_U_PARAMS,
    SSPERR_PBOUNDS,
    ZERO_SSPERR_PARAMS,
    get_bounded_ssperr_params,
    get_unbounded_ssperr_params,
)

K_LOGSM = 5.0

LAMBDA_REST = jnp.array([1500.0, 2500.0, 3831.0, 4777.0, 6289.0, 7684.0])

Z_CONTROL = jnp.array([0.0, 0.5, 1.1])

SSP_SCATTER = 0.05


@jjit
def _delta_mag_r_kern(logsm, x0, ylo, yhi):
    delta_mag_r = _sigmoid(logsm, x0, K_LOGSM, ylo, yhi)
    return delta_mag_r


@jjit
def _delta_color_kern(logsm, x0, ylo, yhi):
    delta_color = _sigmoid(logsm, x0, K_LOGSM, ylo, yhi)
    return delta_color


@jjit
def _delta_mag_x_kern(delta_mag_r, logsm, x0, ylo, yhi):
    delta_color_x = _delta_color_kern(logsm, x0, ylo, yhi)
    delta_band_x = delta_color_x + delta_mag_r
    return delta_band_x


@jjit
def compute_delta_mags_all_bands(logsm, z_obs, ssperr_params):

    delta_mag_r_z0p0 = _delta_mag_r_kern(
        logsm,
        ssperr_params.z0p0_dr_logsm0,
        ssperr_params.z0p0_dr_ylo,
        ssperr_params.z0p0_dr_yhi,
    )
    delta_mag_r_z0p5 = _delta_mag_r_kern(
        logsm,
        ssperr_params.z0p5_dr_logsm0,
        ssperr_params.z0p5_dr_ylo,
        ssperr_params.z0p5_dr_yhi,
    )
    delta_mag_r_z1p1 = _delta_mag_r_kern(
        logsm,
        ssperr_params.z1p1_dr_logsm0,
        ssperr_params.z1p1_dr_ylo,
        ssperr_params.z1p1_dr_yhi,
    )

    delta_mag_fuv_z0p0 = _delta_mag_x_kern(
        delta_mag_r_z0p0,
        logsm,
        ssperr_params.z0p0_dfr_logsm0,
        ssperr_params.z0p0_dfr_ylo,
        ssperr_params.z0p0_dfr_yhi,
    )
    delta_mag_fuv_z0p5 = _delta_mag_x_kern(
        delta_mag_r_z0p5,
        logsm,
        ssperr_params.z0p5_dfr_logsm0,
        ssperr_params.z0p5_dfr_ylo,
        ssperr_params.z0p5_dfr_yhi,
    )
    delta_mag_fuv_z1p1 = _delta_mag_x_kern(
        delta_mag_r_z1p1,
        logsm,
        ssperr_params.z1p1_dfr_logsm0,
        ssperr_params.z1p1_dfr_ylo,
        ssperr_params.z1p1_dfr_yhi,
    )

    delta_mag_nuv_z0p0 = _delta_mag_x_kern(
        delta_mag_r_z0p0,
        logsm,
        ssperr_params.z0p0_dnr_logsm0,
        ssperr_params.z0p0_dnr_ylo,
        ssperr_params.z0p0_dnr_yhi,
    )
    delta_mag_nuv_z0p5 = _delta_mag_x_kern(
        delta_mag_r_z0p5,
        logsm,
        ssperr_params.z0p5_dnr_logsm0,
        ssperr_params.z0p5_dnr_ylo,
        ssperr_params.z0p5_dnr_yhi,
    )
    delta_mag_nuv_z1p1 = _delta_mag_x_kern(
        delta_mag_r_z1p1,
        logsm,
        ssperr_params.z1p1_dnr_logsm0,
        ssperr_params.z1p1_dnr_ylo,
        ssperr_params.z1p1_dnr_yhi,
    )

    delta_mag_u_z0p0 = _delta_mag_x_kern(
        delta_mag_r_z0p0,
        logsm,
        ssperr_params.z0p0_dur_logsm0,
        ssperr_params.z0p0_dur_ylo,
        ssperr_params.z0p0_dur_yhi,
    )
    delta_mag_u_z0p5 = _delta_mag_x_kern(
        delta_mag_r_z0p5,
        logsm,
        ssperr_params.z0p5_dur_logsm0,
        ssperr_params.z0p5_dur_ylo,
        ssperr_params.z0p5_dur_yhi,
    )
    delta_mag_u_z1p1 = _delta_mag_x_kern(
        delta_mag_r_z1p1,
        logsm,
        ssperr_params.z1p1_dur_logsm0,
        ssperr_params.z1p1_dur_ylo,
        ssperr_params.z1p1_dur_yhi,
    )

    delta_mag_g_z0p0 = _delta_mag_x_kern(
        delta_mag_r_z0p0,
        logsm,
        ssperr_params.z0p0_dgr_logsm0,
        ssperr_params.z0p0_dgr_ylo,
        ssperr_params.z0p0_dgr_yhi,
    )
    delta_mag_g_z0p5 = _delta_mag_x_kern(
        delta_mag_r_z0p5,
        logsm,
        ssperr_params.z0p5_dgr_logsm0,
        ssperr_params.z0p5_dgr_ylo,
        ssperr_params.z0p5_dgr_yhi,
    )
    delta_mag_g_z1p1 = _delta_mag_x_kern(
        delta_mag_r_z1p1,
        logsm,
        ssperr_params.z1p1_dgr_logsm0,
        ssperr_params.z1p1_dgr_ylo,
        ssperr_params.z1p1_dgr_yhi,
    )

    delta_mag_i_z0p0 = _delta_mag_x_kern(
        delta_mag_r_z0p0,
        logsm,
        ssperr_params.z0p0_dir_logsm0,
        ssperr_params.z0p0_dir_ylo,
        ssperr_params.z0p0_dir_yhi,
    )
    delta_mag_i_z0p5 = _delta_mag_x_kern(
        delta_mag_r_z0p5,
        logsm,
        ssperr_params.z0p5_dir_logsm0,
        ssperr_params.z0p5_dir_ylo,
        ssperr_params.z0p5_dir_yhi,
    )
    delta_mag_i_z1p1 = _delta_mag_x_kern(
        delta_mag_r_z1p1,
        logsm,
        ssperr_params.z1p1_dir_logsm0,
        ssperr_params.z1p1_dir_ylo,
        ssperr_params.z1p1_dir_yhi,
    )

    delta_mags_fuv = jnp.array(
        (delta_mag_fuv_z0p0, delta_mag_fuv_z0p5, delta_mag_fuv_z1p1)
    )
    delta_mags_nuv = jnp.array(
        (delta_mag_nuv_z0p0, delta_mag_nuv_z0p5, delta_mag_nuv_z1p1)
    )
    delta_mags_u = jnp.array((delta_mag_u_z0p0, delta_mag_u_z0p5, delta_mag_u_z1p1))
    delta_mags_g = jnp.array((delta_mag_g_z0p0, delta_mag_g_z0p5, delta_mag_g_z1p1))
    delta_mags_r = jnp.array((delta_mag_r_z0p0, delta_mag_r_z0p5, delta_mag_r_z1p1))
    delta_mags_i = jnp.array((delta_mag_i_z0p0, delta_mag_i_z0p5, delta_mag_i_z1p1))

    delta_mag_fuv_zobs = _redshift_interpolation_kern(z_obs, *delta_mags_fuv)
    delta_mag_nuv_zobs = _redshift_interpolation_kern(z_obs, *delta_mags_nuv)
    delta_mag_u_zobs = _redshift_interpolation_kern(z_obs, *delta_mags_u)
    delta_mag_g_zobs = _redshift_interpolation_kern(z_obs, *delta_mags_g)
    delta_mag_r_zobs = _redshift_interpolation_kern(z_obs, *delta_mags_r)
    delta_mag_i_zobs = _redshift_interpolation_kern(z_obs, *delta_mags_i)

    delta_mags = jnp.array(
        (
            delta_mag_fuv_zobs,
            delta_mag_nuv_zobs,
            delta_mag_u_zobs,
            delta_mag_g_zobs,
            delta_mag_r_zobs,
            delta_mag_i_zobs,
        )
    )

    return delta_mags


@jjit
def _redshift_interpolation_kern(zarr, y0, y1, y2, k=5.0, xa=0.5, xb=1.5):
    dxab = xb - xa
    xab = xa + 0.5 * dxab
    w0 = _sigmoid(zarr, xa, k, y0, y1)
    w1 = _sigmoid(zarr, xb, k, y1, y2)
    w01 = _sigmoid(zarr, xab, k, w0, w1)
    return w01


@jjit
def _tw_wave_interp_kern(wave, y_table, x_table=LAMBDA_REST):
    x0, x1, x2, x3, x4, x5 = x_table
    y0, y1, y2, y3, y4, y5 = y_table

    w02 = _tw_interp_kern(wave, x0, x1, x2, y0, y1, y2)
    w24 = _tw_interp_kern(wave, x2, x3, x4, y2, y3, y4)

    dx13 = (x3 - x1) / 3
    w04 = _tw_sigmoid(wave, x2, dx13, w02, w24)

    dx45 = (x5 - x4) / 3
    x45 = 0.5 * (x4 + x5)
    w05 = _tw_sigmoid(wave, x45, dx45, w04, y5)

    return w05


@jjit
def F_sps_err_from_delta_mag(delta_mag):

    F_ssp_err = 10 ** (-0.4 * delta_mag)

    return F_ssp_err


@jjit
def delta_mag_from_F_sps_err(F_ssp_err):

    delta_mag = -2.5 * jnp.log10(F_ssp_err)

    return delta_mag


@jjit
def F_sps_err_lambda(ssperr_params, logsm, z_obs, wave_obs, wave_eff_rest):

    delta_mags_rest = compute_delta_mags_all_bands(logsm, z_obs, ssperr_params)

    F_sps_err_wave_eff_rest = F_sps_err_from_delta_mag(delta_mags_rest)

    # F_sps_err_z_obs = jnp.interp(wave_obs, wave_eff_rest, F_sps_err_wave_eff_rest)
    F_sps_err_z_obs = _tw_wave_interp_kern(
        wave_obs, F_sps_err_wave_eff_rest, wave_eff_rest
    )

    return F_sps_err_z_obs


_A = (None, 0, None, None, None)
F_sps_err_lambda_galpop = jjit(vmap(F_sps_err_lambda, in_axes=_A))


@jjit
def delta_mag_from_lambda_rest(ssperr_params, z_obs, logsm, wave_obs, wave_eff_rest):

    F_sps_err_z_obs = F_sps_err_lambda_galpop(
        ssperr_params, logsm, z_obs, wave_obs, wave_eff_rest
    )

    delta_mag_z_obs = delta_mag_from_F_sps_err(F_sps_err_z_obs)

    return delta_mag_z_obs


@jjit
def compute_delta_scatter(ran_key, delta_mag):

    delta_scatter = jran.normal(ran_key, delta_mag.shape) * SSP_SCATTER

    return delta_scatter


@jjit
def noisy_delta_mag(
    ssperr_params,
    z_obs,
    logsm,
    wave_eff_aa_obs,
    wave_eff_rest,
    ran_key,
):

    delta_mag = delta_mag_from_lambda_rest(
        ssperr_params, z_obs, logsm, wave_eff_aa_obs, wave_eff_rest
    )

    delta_scatter = compute_delta_scatter(ran_key, delta_mag)

    noisy_delta_mag = delta_scatter + delta_mag

    return noisy_delta_mag


@jjit
def add_delta_mag_to_photometry(
    ssperr_params,
    z_obs,
    logsm_q,
    logsm_ms,
    wave_eff_aa_obs,
    wave_eff_rest,
    q_key,
    ms_key,
    mags_q_smooth,
    mags_q_bursty,
    mags_ms_smooth,
    mags_ms_bursty,
):

    noisy_delta_mag_q = noisy_delta_mag(
        ssperr_params, z_obs, logsm_q, wave_eff_aa_obs, wave_eff_rest, q_key
    )

    noisy_delta_mag_ms = noisy_delta_mag(
        ssperr_params, z_obs, logsm_ms, wave_eff_aa_obs, wave_eff_rest, ms_key
    )

    new_mags_q_smooth = mags_q_smooth + noisy_delta_mag_q
    new_mags_q_bursty = mags_q_bursty + noisy_delta_mag_q

    new_mags_ms_smooth = mags_ms_smooth + noisy_delta_mag_ms
    new_mags_ms_bursty = mags_ms_bursty + noisy_delta_mag_ms

    return new_mags_q_smooth, new_mags_q_bursty, new_mags_ms_smooth, new_mags_ms_bursty
