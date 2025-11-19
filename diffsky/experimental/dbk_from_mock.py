""""""

from collections import namedtuple

from diffmah import logmh_at_t_obs
from diffstar import calc_sfh_galpop
from dsps.cosmology import age_at_z0
from dsps.metallicity import umzr
from dsps.sfh import diffburst
from dsps.sfh.diffburst import DEFAULT_BURST_PARAMS
from jax import jit as jjit
from jax import numpy as jnp
from jax import vmap

from ..dustpop.tw_dust import DEFAULT_DUST_PARAMS
from ..ssp_err_model import ssp_err_model
from . import lc_phot_kern
from . import photometry_interpolation as photerp
from .disk_bulge_modeling import disk_bulge_kernels as dbk
from .disk_bulge_modeling import disk_knots
from .disk_bulge_modeling import mc_disk_bulge as mcdb

_BPOP = (None, 0, 0)
_pureburst_age_weights_from_params_vmap = jjit(
    vmap(diffburst._pureburst_age_weights_from_params, in_axes=_BPOP)
)
DBK_PHOT_INFO_KEYS = (
    "logmp_obs",
    "logsm_obs",
    "logssfr_obs",
    "sfh_table",
    "obs_mags",
    "obs_mags_bulge",
    "obs_mags_disk",
    "obs_mags_knots",
    *dbk.FbulgeParams._fields,
    "eff_bulge_history",
    "sfh_bulge",
    "smh_bulge",
    "bulge_to_total_history",
    *DEFAULT_BURST_PARAMS._fields,
    *DEFAULT_DUST_PARAMS._fields,
    "ssp_weights",
)
DBK_PhotInfo = namedtuple("DBK_PhotInfo", DBK_PHOT_INFO_KEYS)


@jjit
def _disk_bulge_knot_phot_from_mock(
    z_obs,
    t_obs,
    mah_params,
    logmp0,
    t_table,
    ssp_data,
    precomputed_ssp_mag_table,
    z_phot_table,
    wave_eff_table,
    diffstar_params,
    mzr_params,
    spspop_params,
    scatter_params,
    ssp_err_pop_params,
    cosmo_params,
    fb,
    uran_av,
    uran_delta,
    uran_funo,
    delta_scatter,
    mc_sfh_type,
    fknot,
):
    """Populate the input lightcone with galaxy SEDs"""
    n_z_table, n_bands, n_met, n_age = precomputed_ssp_mag_table.shape
    n_gals = logmp0.size

    # Calculate halo mass at the observed redshift
    t0 = age_at_z0(*cosmo_params)
    lgt0 = jnp.log10(t0)
    logmp_obs = logmh_at_t_obs(mah_params, t_obs, lgt0)

    # Calculate SFH with diffstarpop
    t0 = age_at_z0(*cosmo_params)

    sfh_table = calc_sfh_galpop(
        diffstar_params,
        mah_params,
        t_table,
        lgt0=jnp.log10(t0),
        fb=fb,
        return_smh=False,
    )
    logsm_obs, logssfr_obs = lc_phot_kern._get_sfh_info_at_t_obs(
        t_table, sfh_table, t_obs
    )

    # Calculate stellar age PDF weights from SFH
    smooth_age_weights = lc_phot_kern.calc_age_weights_from_sfh_table_vmap(
        t_table, sfh_table, ssp_data.ssp_lg_age_gyr, t_obs
    )
    # smooth_age_weights_ms.shape = (n_gals, n_age)

    # Calculate stellar age PDF weights from SFH + burstiness
    _args = (
        spspop_params.burstpop_params,
        logsm_obs,
        logssfr_obs,
        ssp_data.ssp_lg_age_gyr,
        smooth_age_weights,
    )
    _res = lc_phot_kern._calc_bursty_age_weights_vmap(*_args)
    bursty_age_weights = _res[0]  # bursty_age_weights_ms.shape = (n_gals, age)
    burst_params = _res[1]  # ('lgfburst', 'lgyr_peak', 'lgyr_max')

    # Calculate mean metallicity of the population
    lgmet_med = umzr.mzr_model(logsm_obs, t_obs, *mzr_params)

    # Calculate metallicity distribution function
    # lgmet_weights_q.shape = (n_gals, n_met)
    lgmet_weights = lc_phot_kern._calc_lgmet_weights_galpop(
        lgmet_med, lc_phot_kern.LGMET_SCATTER, ssp_data.ssp_lgmet
    )

    # Calculate SSP weights = P_SSP = P_met * P_age
    _w_age = smooth_age_weights.reshape((n_gals, 1, n_age))
    _w_lgmet = lgmet_weights.reshape((n_gals, n_met, 1))
    ssp_weights_smooth = _w_lgmet * _w_age

    _w_age_bursty = bursty_age_weights.reshape((n_gals, 1, n_age))
    ssp_weights_bursty = _w_lgmet * _w_age_bursty

    ssp_weights = jnp.where(
        mc_sfh_type.reshape((n_gals, 1, 1)) == 2, ssp_weights_bursty, ssp_weights_smooth
    )

    # Interpolate SSP mag table to z_obs of each galaxy
    photmag_table_galpop = photerp.interpolate_ssp_photmag_table(
        z_obs, z_phot_table, precomputed_ssp_mag_table
    )
    ssp_photflux_table = 10 ** (-0.4 * photmag_table_galpop)

    # For each filter, calculate λ_eff in the restframe of each galaxy
    wave_eff_galpop = lc_phot_kern.interp_vmap2(z_obs, z_phot_table, wave_eff_table)

    # Calculate mean fractional change to the SSP fluxes in each band for each galaxy
    # L'_SSP(λ_eff) = L_SSP(λ_eff) & F_SSP(λ_eff)
    frac_ssp_err = lc_phot_kern.get_frac_ssp_err_vmap(
        ssp_err_pop_params,
        z_obs,
        logsm_obs,
        wave_eff_galpop,
        ssp_err_model.LAMBDA_REST,
    )

    ftrans_args = (
        spspop_params.dustpop_params,
        wave_eff_galpop,
        logsm_obs,
        logssfr_obs,
        z_obs,
        ssp_data.ssp_lg_age_gyr,
        uran_av,
        uran_delta,
        uran_funo,
        scatter_params,
    )
    _res = lc_phot_kern.calc_dust_ftrans_vmap(*ftrans_args)
    ftrans = _res[1]
    dust_params = _res[3]

    # Calculate fractional changes to SSP fluxes
    exparg = -0.4 * delta_scatter
    frac_ssp_err = frac_ssp_err * 10**exparg

    # Reshape arrays before calculating galaxy magnitudes
    _ferr_ssp = frac_ssp_err.reshape((n_gals, n_bands, 1, 1))

    _ftrans = ftrans.reshape((n_gals, n_bands, 1, n_age))

    _mstar = 10 ** logsm_obs.reshape((n_gals, 1))

    _w = ssp_weights.reshape((n_gals, 1, n_met, n_age))

    # Calculate galaxy magnitudes as PDF-weighted sums
    integrand = ssp_photflux_table * _w * _ftrans * _ferr_ssp
    photflux_galpop = jnp.sum(integrand, axis=(2, 3)) * _mstar
    obs_mags = -2.5 * jnp.log10(photflux_galpop)

    # Begin calculation of disk/bulge/knot quantities
    # Compute restframe SED of bulge
    disk_bulge_history = mcdb.decompose_sfh_into_disk_bulge_sfh(t_table, sfh_table)

    # Calculate restframe SED of disk and knots
    ssp_lg_age_yr = ssp_data.ssp_lg_age_gyr + 9.0
    age_weights_pureburst = _pureburst_age_weights_from_params_vmap(
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
    mstar_tot, mburst, mdd, mknot, age_weights_dd, age_weights_knot = _res
    mstar_obs_dd = mdd.reshape((n_gals, 1))
    mstar_obs_knot = mknot.reshape((n_gals, 1))

    logsm_obs_bulge = lc_phot_kern.interp_vmap(
        t_obs, t_table, jnp.log10(disk_bulge_history.smh_bulge)
    )
    mstar_obs_bulge = 10 ** logsm_obs_bulge.reshape((n_gals, 1))
    age_weights_bulge = lc_phot_kern.calc_age_weights_from_sfh_table_vmap(
        t_table, disk_bulge_history.sfh_bulge, ssp_data.ssp_lg_age_gyr, t_obs
    )

    lgmet_med_obs = umzr.mzr_model(logsm_obs, t_obs, *mzr_params)
    lgmet_weights_obs = lc_phot_kern._calc_lgmet_weights_galpop(
        lgmet_med_obs, lc_phot_kern.LGMET_SCATTER, ssp_data.ssp_lgmet
    )

    _w_age_bulge = age_weights_bulge.reshape((n_gals, 1, n_age))
    _w_lgmet_bulge = lgmet_weights_obs.reshape((n_gals, n_met, 1))
    ssp_weights_bulge = _w_lgmet_bulge * _w_age_bulge

    _w_age_knot = age_weights_knot.reshape((n_gals, 1, n_age))
    _w_lgmet_knot = lgmet_weights_obs.reshape((n_gals, n_met, 1))
    ssp_weights_knot = _w_lgmet_knot * _w_age_knot

    _w_age_disk = age_weights_dd.reshape((n_gals, 1, n_age))
    _w_lgmet_disk = lgmet_weights_obs.reshape((n_gals, n_met, 1))
    ssp_weights_dd = _w_lgmet_disk * _w_age_disk

    _w_bulge = ssp_weights_bulge.reshape((n_gals, 1, n_met, n_age))
    _w_dd = ssp_weights_dd.reshape((n_gals, 1, n_met, n_age))
    _w_knot = ssp_weights_knot.reshape((n_gals, 1, n_met, n_age))

    # Calculate bulge magnitudes as PDF-weighted sums
    integrand_bulge = ssp_photflux_table * _w_bulge * _ftrans * _ferr_ssp
    photflux_galpop_bulge = jnp.sum(integrand_bulge, axis=(2, 3)) * mstar_obs_bulge
    obs_mags_bulge = -2.5 * jnp.log10(photflux_galpop_bulge)

    integrand_disk = ssp_photflux_table * _w_dd * _ftrans * _ferr_ssp
    photflux_galpop_disk = jnp.sum(integrand_disk, axis=(2, 3)) * mstar_obs_dd
    obs_mags_disk = -2.5 * jnp.log10(photflux_galpop_disk)

    integrand_knot = ssp_photflux_table * _w_knot * _ftrans * _ferr_ssp
    photflux_galpop_knot = jnp.sum(integrand_knot, axis=(2, 3)) * mstar_obs_knot
    obs_mags_knots = -2.5 * jnp.log10(photflux_galpop_knot)

    phot_info = DBK_PhotInfo(
        logmp_obs=logmp_obs,
        logsm_obs=logsm_obs,
        logssfr_obs=logssfr_obs,
        sfh_table=sfh_table,
        obs_mags=obs_mags,
        obs_mags_bulge=obs_mags_bulge,
        obs_mags_disk=obs_mags_disk,
        obs_mags_knots=obs_mags_knots,
        fbulge_tcrit=disk_bulge_history.fbulge_params.fbulge_tcrit,
        fbulge_early=disk_bulge_history.fbulge_params.fbulge_early,
        fbulge_late=disk_bulge_history.fbulge_params.fbulge_late,
        eff_bulge_history=disk_bulge_history.eff_bulge_history,
        sfh_bulge=disk_bulge_history.sfh_bulge,
        smh_bulge=disk_bulge_history.smh_bulge,
        bulge_to_total_history=disk_bulge_history.bulge_to_total_history,
        **burst_params._asdict(),
        **dust_params._asdict(),
        ssp_weights=ssp_weights,
    )

    return phot_info._asdict()
