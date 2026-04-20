""""""

from collections import namedtuple
from functools import partial

from diffstar import DEFAULT_DIFFSTAR_PARAMS
from dsps.sfh.diffburst import DEFAULT_BURST_PARAMS
from jax import jit as jjit
from jax import numpy as jnp
from jax import vmap

from ...dustpop.tw_dust import DEFAULT_DUST_PARAMS
from ...ssp_err_model import ssp_err_model
from .. import mc_diffstarpop_wrappers as mcdw
from .. import photometry_interpolation as photerp
from . import constants, mc_randoms
from . import ssp_weight_kernels as sspwk

LGMET_SCATTER = constants.LGMET_SCATTER


_B = (None, None, 1)
interp_vmap2 = jjit(vmap(jnp.interp, in_axes=_B, out_axes=1))


@partial(jjit, static_argnames=["n_t_table"])
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
    ssperr_params,
    cosmo_params,
    fb,
    *,
    n_t_table=mcdw.N_T_TABLE,
):
    phot_randoms, sfh_params = mc_randoms.get_mc_phot_randoms(
        ran_key, diffstarpop_params, mah_params, cosmo_params
    )
    phot_kern_results = _phot_kern(
        phot_randoms,
        sfh_params,
        z_obs,
        t_obs,
        mah_params,
        ssp_data,
        precomputed_ssp_mag_table,
        z_phot_table,
        wave_eff_table,
        mzr_params,
        spspop_params,
        scatter_params,
        ssperr_params,
        cosmo_params,
        fb,
        n_t_table=n_t_table,
    )
    return phot_kern_results, phot_randoms


@partial(jjit, static_argnames=["n_t_table"])
def _phot_kern(
    phot_randoms,
    sfh_params,
    z_obs,
    t_obs,
    mah_params,
    ssp_data,
    precomputed_ssp_mag_table,
    z_phot_table,
    wave_eff_table,
    mzr_params,
    spspop_params,
    scatter_params,
    ssperr_params,
    cosmo_params,
    fb,
    *,
    n_t_table=mcdw.N_T_TABLE,
):
    """Populate the input lightcone with galaxy SEDs"""

    t_table, sfh_table, logsm_obs, logssfr_obs = mcdw.compute_diffstar_info(
        mah_params, sfh_params, t_obs, cosmo_params, fb, n_t_table
    )

    age_weights_smooth, lgmet_weights = sspwk.get_smooth_ssp_weights(
        t_table, sfh_table, logsm_obs, ssp_data, t_obs, mzr_params, LGMET_SCATTER
    )

    _res = sspwk.compute_burstiness(
        phot_randoms.uran_pburst,
        phot_randoms.mc_is_q,
        logsm_obs,
        logssfr_obs,
        age_weights_smooth,
        lgmet_weights,
        ssp_data,
        spspop_params.burstpop_params,
    )
    ssp_weights, burst_params, mc_sfh_type = _res

    # Interpolate SSP mag table to z_obs of each galaxy
    photmag_table_galpop = photerp.interpolate_ssp_photmag_table(
        z_obs, z_phot_table, precomputed_ssp_mag_table
    )
    ssp_photflux_table = 10 ** (-0.4 * photmag_table_galpop)

    # For each filter, calculate λ_eff in the restframe of each galaxy
    wave_eff_galpop = interp_vmap2(z_obs, z_phot_table, wave_eff_table)

    dust_frac_trans, dust_params = sspwk.compute_dust_attenuation(
        phot_randoms.uran_av,
        phot_randoms.uran_delta,
        phot_randoms.uran_funo,
        logsm_obs,
        logssfr_obs,
        ssp_data,
        z_obs,
        wave_eff_galpop,
        spspop_params.dustpop_params,
        scatter_params,
    )
    # dust_frac_trans.shape = (n_gals, n_bands, n_age)

    # Throw out redundant dust params repeated at each λ_eff
    dust_params = dust_params._replace(
        av=dust_params.av[:, 0, -1],
        delta=dust_params.delta[:, 0],
        funo=dust_params.funo[:, 0],
    )

    # Calculate mean fractional change to the SSP fluxes in each band for each galaxy
    # L'_SSP(λ_eff) = L_SSP(λ_eff) & F_SSP(λ_eff)
    frac_ssp_errors_nonoise = ssp_err_model.frac_ssp_err_at_z_obs_galpop(
        ssperr_params, logsm_obs, z_obs, wave_eff_galpop
    )
    frac_ssp_errors = ssp_err_model.get_noisy_frac_ssp_errors(
        wave_eff_galpop, frac_ssp_errors_nonoise, phot_randoms.delta_mag_ssp_scatter
    )

    obs_mags = sspwk._compute_obs_mags_from_weights(
        logsm_obs, dust_frac_trans, frac_ssp_errors, ssp_photflux_table, ssp_weights
    )

    phot_kern_results = PhotKernResults(
        obs_mags,
        t_table,
        *sfh_params,
        sfh_table,
        logsm_obs,
        logssfr_obs,
        mc_sfh_type,
        ssp_weights,
        lgmet_weights,
        *burst_params,
        *dust_params,
        dust_frac_trans,
        ssp_photflux_table,
        frac_ssp_errors,
        wave_eff_galpop,
    )
    return phot_kern_results


PHOT_KERN_KEYS = (
    "obs_mags",
    "t_table",
    *DEFAULT_DIFFSTAR_PARAMS._fields,
    "sfh_table",
    "logsm_obs",
    "logssfr_obs",
    "mc_sfh_type",
    "ssp_weights",
    "lgmet_weights",
    *DEFAULT_BURST_PARAMS._fields,
    *DEFAULT_DUST_PARAMS._fields,
    "dust_frac_trans",
    "ssp_photflux_table",
    "frac_ssp_errors",
    "wave_eff_galpop",
)
PhotKernResults = namedtuple("PhotKernResults", PHOT_KERN_KEYS)
