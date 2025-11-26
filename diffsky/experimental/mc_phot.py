# flake8: noqa: E402
""""""

from jax import config

config.update("jax_enable_x64", True)


from jax import jit as jjit
from jax import random as jran

from ..ssp_err_model import ssp_err_model
from . import lc_phot_kern
from . import mc_diffstarpop_wrappers as mcdw
from . import photometry_interpolation as photerp
from .disk_bulge_modeling import mc_disk_bulge as mcdb
from .kernels import dbk_kernels
from .kernels.ssp_weight_kernels import (
    compute_burstiness,
    compute_dust_attenuation,
    compute_frac_ssp_errors,
    compute_mc_realization,
    compute_obs_mags_ms_q,
    get_smooth_ssp_weights,
)
from .mc_diffstarpop_wrappers import N_T_TABLE

LGMET_SCATTER = 0.2


@jjit
def _mc_phot_kern(
    ran_key,
    z_obs,
    t_obs,
    mah_params,
    ssp_data,
    precomputed_ssp_mag_table,
    z_phot_table,
    wave_eff_table,
    diffstarpop_params,
    mzr_params,
    spspop_params,
    scatter_params,
    ssp_err_pop_params,
    cosmo_params,
    fb,
    n_t_table=N_T_TABLE,
):
    """Populate the input lightcone with galaxy SEDs"""

    ran_key, sfh_key = jran.split(ran_key, 2)
    args = (diffstarpop_params, sfh_key, mah_params, t_obs, cosmo_params, fb)
    diffstar_galpop = mcdw.diffstarpop_cen_wrapper(*args, n_t_table=n_t_table)

    smooth_ssp_weights = get_smooth_ssp_weights(
        diffstar_galpop, ssp_data, t_obs, mzr_params, LGMET_SCATTER
    )

    burstiness = compute_burstiness(
        diffstar_galpop, smooth_ssp_weights, ssp_data, spspop_params.burstpop_params
    )

    # Interpolate SSP mag table to z_obs of each galaxy
    photmag_table_galpop = photerp.interpolate_ssp_photmag_table(
        z_obs, z_phot_table, precomputed_ssp_mag_table
    )
    ssp_photflux_table = 10 ** (-0.4 * photmag_table_galpop)

    # For each filter, calculate λ_eff in the restframe of each galaxy
    wave_eff_galpop = lc_phot_kern.interp_vmap2(z_obs, z_phot_table, wave_eff_table)

    # Generate randoms for stochasticity in dust attenuation curves
    ran_key, dust_key = jran.split(ran_key, 2)
    dust_att = compute_dust_attenuation(
        dust_key,
        diffstar_galpop,
        ssp_data,
        z_obs,
        wave_eff_galpop,
        spspop_params.dustpop_params,
        scatter_params,
    )

    # Calculate mean fractional change to the SSP fluxes in each band for each galaxy
    # L'_SSP(λ_eff) = L_SSP(λ_eff) & F_SSP(λ_eff)
    frac_ssp_errors = compute_frac_ssp_errors(
        ssp_err_pop_params, z_obs, diffstar_galpop, wave_eff_galpop
    )
    ran_key, ssp_q_key, ssp_ms_key = jran.split(ran_key, 3)
    delta_scatter_ms = ssp_err_model.compute_delta_scatter(
        ssp_ms_key, frac_ssp_errors.ms
    )
    delta_scatter_q = ssp_err_model.compute_delta_scatter(ssp_q_key, frac_ssp_errors.q)

    obs_mags = compute_obs_mags_ms_q(
        diffstar_galpop,
        dust_att,
        frac_ssp_errors,
        ssp_photflux_table,
        smooth_ssp_weights.weights.ms,
        burstiness.weights.ms,
        smooth_ssp_weights.weights.q,
        delta_scatter_ms,
        delta_scatter_q,
    )
    phot_info = compute_mc_realization(
        diffstar_galpop,
        burstiness,
        smooth_ssp_weights,
        dust_att,
        obs_mags,
        delta_scatter_ms,
        delta_scatter_q,
        ran_key,
    )

    return phot_info, smooth_ssp_weights, burstiness


@jjit
def _mc_dbk_kern(t_obs, ssp_data, phot_info, smooth_ssp_weights, burstiness):
    disk_bulge_history = mcdb.decompose_sfh_into_disk_bulge_sfh(
        phot_info.t_table, phot_info.sfh_table
    )
    ssp_weights_bulge = dbk_kernels.mc_ssp_weights_bulge(
        t_obs, ssp_data, phot_info, disk_bulge_history, smooth_ssp_weights
    )
    return ssp_weights_bulge
