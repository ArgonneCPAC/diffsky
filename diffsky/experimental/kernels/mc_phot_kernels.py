from collections import namedtuple
from functools import partial

from dsps.sfh.diffburst import DEFAULT_BURST_PARAMS
from jax import jit as jjit
from jax import numpy as jnp
from jax import random as jran
from jax import vmap

from ...merging import compute_x_tot_from_x_in_situ, merging_model
from ...ssp_err_model import ssp_err_model
from .. import mc_diffstarpop_wrappers as mcdw
from ..disk_bulge_modeling import disk_bulge_kernels as dbk
from . import dbk_kernels, linelum_kernels, mc_randoms, phot_kernels
from . import ssp_weight_kernels as sspwk

LGMET_SCATTER = 0.2


_B = (None, None, 1)
interp_vmap2 = jjit(vmap(jnp.interp, in_axes=_B, out_axes=1))


# kernels
_mc_phot_kern = phot_kernels._mc_phot_kern
_phot_kern = phot_kernels._phot_kern
_mc_dbk_kern = dbk_kernels._mc_dbk_kern
_dbk_kern = dbk_kernels._dbk_kern  # noqa
_mc_specphot_kern = linelum_kernels._mc_specphot_kern
_specphot_kern = linelum_kernels._specphot_kern

# randoms
get_mc_phot_randoms = mc_randoms.get_mc_phot_randoms
get_mc_dbk_randoms = mc_randoms.get_mc_dbk_randoms

# namedtuple containers
PhotRandoms = mc_randoms.PhotRandoms
SpecKernResults = linelum_kernels.SpecKernResults
PhotKernResults = phot_kernels.PhotKernResults
DBKRandoms = mc_randoms.DBKRandoms


@partial(jjit, static_argnames=["n_t_table"])
def _mc_phot_kern_merging(
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
    merge_params,
    cosmo_params,
    fb,
    logmp_infall,
    logmhost_infall,
    t_infall,
    is_central,
    nhalos_weights,
    halo_indx,
    n_t_table=mcdw.N_T_TABLE,
):
    phot_randoms, sfh_params = get_mc_phot_randoms(
        ran_key, diffstarpop_params, mah_params, cosmo_params
    )
    phot_kern_results, flux_obs, merge_prob, mstar_obs = _phot_kern_merging(
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
        ssp_err_pop_params,
        merge_params,
        cosmo_params,
        fb,
        logmp_infall,
        logmhost_infall,
        t_infall,
        is_central,
        nhalos_weights,
        halo_indx,
        n_t_table=n_t_table,
    )
    return phot_kern_results, phot_randoms, flux_obs, merge_prob, mstar_obs


@partial(jjit, static_argnames=["n_t_table"])
def _phot_kern_merging(
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
    ssp_err_pop_params,
    merge_params,
    cosmo_params,
    fb,
    logmp_infall,
    logmhost_infall,
    t_infall,
    is_central,
    nhalos_weights,
    halo_indx,
    n_t_table=mcdw.N_T_TABLE,
):
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
        ssp_err_pop_params,
        cosmo_params,
        fb,
        n_t_table=n_t_table,
    )

    upids = jnp.where(is_central == 1, -1.0, 0.0)
    merge_prob = merging_model.get_p_merge_from_merging_params(
        merge_params, logmp_infall, logmhost_infall, t_obs, t_infall, upids
    )

    mstar_in_situ = 10**phot_kern_results.logsm_obs
    mstar_obs = compute_x_tot_from_x_in_situ(
        mstar_in_situ, merge_prob, nhalos_weights, halo_indx
    )

    flux_in_situ = 10 ** (-0.4 * phot_kern_results.obs_mags)
    flux_obs = compute_x_tot_from_x_in_situ(
        flux_in_situ,
        merge_prob[:, jnp.newaxis],
        nhalos_weights[:, jnp.newaxis],
        halo_indx,
    )

    return phot_kern_results, flux_obs, merge_prob, mstar_obs


@jjit
def _mc_specphot_kern_merging(
    ran_key,
    phot_randoms,
    sfh_params,
    z_obs,
    t_obs,
    mah_params,
    ssp_data,
    precomputed_ssp_mag_table,
    z_phot_table,
    wave_eff_table,
    line_wave_table,
    diffstarpop_params,
    mzr_params,
    spspop_params,
    scatter_params,
    ssp_err_pop_params,
    merge_params,
    cosmo_params,
    fb,
    logmp_infall,
    logmhost_infall,
    t_infall,
    is_central,
    nhalos_weights,
    halo_indx,
):
    phot_randoms, sfh_params = get_mc_phot_randoms(
        ran_key, diffstarpop_params, mah_params, cosmo_params
    )

    (
        phot_kern_results,
        linelums_in_situ,
        flux_obs,
        merge_prob,
        mstar_obs,
        linelums_obs,
    ) = _specphot_kern_merging(
        phot_randoms,
        sfh_params,
        z_obs,
        t_obs,
        mah_params,
        ssp_data,
        precomputed_ssp_mag_table,
        z_phot_table,
        wave_eff_table,
        line_wave_table,
        mzr_params,
        spspop_params,
        scatter_params,
        ssp_err_pop_params,
        merge_params,
        cosmo_params,
        fb,
        logmp_infall,
        logmhost_infall,
        t_infall,
        is_central,
        nhalos_weights,
        halo_indx,
    )
    return (
        phot_kern_results,
        linelums_in_situ,
        phot_randoms,
        flux_obs,
        merge_prob,
        mstar_obs,
        linelums_obs,
    )


@jjit
def _specphot_kern_merging(
    phot_randoms,
    sfh_params,
    z_obs,
    t_obs,
    mah_params,
    ssp_data,
    precomputed_ssp_mag_table,
    z_phot_table,
    wave_eff_table,
    line_wave_table,
    mzr_params,
    spspop_params,
    scatter_params,
    ssp_err_pop_params,
    merge_params,
    cosmo_params,
    fb,
    logmp_infall,
    logmhost_infall,
    t_infall,
    is_central,
    nhalos_weights,
    halo_indx,
):
    phot_kern_results, linelums_in_situ, dust_ftrans_lines = _specphot_kern(
        phot_randoms,
        sfh_params,
        z_obs,
        t_obs,
        mah_params,
        ssp_data,
        precomputed_ssp_mag_table,
        z_phot_table,
        wave_eff_table,
        line_wave_table,
        mzr_params,
        spspop_params,
        scatter_params,
        ssp_err_pop_params,
        cosmo_params,
        fb,
    )

    upids = jnp.where(is_central == 1, -1.0, 0.0)
    merge_prob = merging_model.get_p_merge_from_merging_params(
        merge_params, logmp_infall, logmhost_infall, t_obs, t_infall, upids
    )

    mstar_in_situ = 10**phot_kern_results.logsm_obs
    mstar_obs = compute_x_tot_from_x_in_situ(
        mstar_in_situ, merge_prob, nhalos_weights, halo_indx
    )

    flux_in_situ = 10 ** (-0.4 * phot_kern_results.obs_mags)
    flux_obs = compute_x_tot_from_x_in_situ(
        flux_in_situ,
        merge_prob[:, jnp.newaxis],
        nhalos_weights[:, jnp.newaxis],
        halo_indx,
    )

    linelums_obs = compute_x_tot_from_x_in_situ(
        linelums_in_situ,
        merge_prob[:, jnp.newaxis],
        nhalos_weights[:, jnp.newaxis],
        halo_indx,
    )

    return (
        phot_kern_results,
        linelums_in_situ,
        flux_obs,
        merge_prob,
        mstar_obs,
        linelums_obs,
    )


@partial(jjit, static_argnames=["n_t_table"])
def _sed_kern(
    phot_randoms,
    sfh_params,
    z_obs,
    t_obs,
    mah_params,
    ssp_data,
    mzr_params,
    spspop_params,
    scatter_params,
    ssp_err_pop_params,
    cosmo_params,
    fb,
    n_t_table=mcdw.N_T_TABLE,
):
    """"""

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

    # For each filter, calculate λ_eff in the restframe of each galaxy
    n_gals = logsm_obs.shape[0]
    wave_eff_galpop = jnp.tile(ssp_data.ssp_wave, n_gals).reshape((n_gals, -1))

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
    dust_params = dust_params._replace(
        av=dust_params.av[:, 0, -1],
        delta=dust_params.delta[:, 0],
        funo=dust_params.funo[:, 0],
    )

    # Calculate mean fractional change to the SSP fluxes in each band for each galaxy
    # L'_SSP(λ_eff) = L_SSP(λ_eff) & F_SSP(λ_eff)
    frac_ssp_errors = ssp_err_model.frac_ssp_err_at_z_obs_galpop(
        ssp_err_pop_params, logsm_obs, z_obs, wave_eff_galpop
    )
    frac_ssp_errors = ssp_err_model.get_noisy_frac_ssp_errors(
        wave_eff_galpop, frac_ssp_errors, phot_randoms.delta_mag_ssp_scatter
    )

    n_met, n_age, n_wave = ssp_data.ssp_flux.shape
    dust_frac_trans = dust_frac_trans.swapaxes(1, 2)  # (n_gals, n_age, n_wave)

    a = dust_frac_trans.reshape((n_gals, 1, n_age, n_wave))
    b = frac_ssp_errors.reshape((n_gals, 1, 1, n_wave))
    c = ssp_weights.reshape((n_gals, n_met, n_age, 1))
    d = ssp_data.ssp_flux.reshape((1, n_met, n_age, n_wave))
    mstar = 10 ** logsm_obs.reshape((n_gals, 1))
    rest_sed = jnp.sum(a * b * c * d, axis=(1, 2)) * mstar

    sed_kern_results = (rest_sed, dust_frac_trans, frac_ssp_errors, ssp_weights)
    return sed_kern_results


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


def _mc_dbk_phot_kern(
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
):
    phot_key, dbk_key = jran.split(ran_key, 2)
    phot_kern_results, phot_randoms = _mc_phot_kern(
        phot_key,
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
    )

    burst_params = DEFAULT_BURST_PARAMS._replace(
        lgfburst=phot_kern_results.lgfburst,
        lgyr_peak=phot_kern_results.lgyr_peak,
        lgyr_max=phot_kern_results.lgyr_max,
    )
    _ret2 = _mc_dbk_kern(
        t_obs,
        ssp_data,
        phot_kern_results.t_table,
        phot_kern_results.sfh_table,
        burst_params,
        phot_kern_results.lgmet_weights,
        dbk_key,
    )
    dbk_randoms, dbk_weights, disk_bulge_history = _ret2

    _ret3 = _get_dbk_phot_from_dbk_weights(
        phot_kern_results.ssp_photflux_table,
        dbk_weights,
        phot_kern_results.dust_frac_trans,
        phot_kern_results.frac_ssp_errors,
    )
    obs_mags_bulge, obs_mags_disk, obs_mags_knots = _ret3

    dbk_phot_info = MCDBKPhotInfo(
        **phot_kern_results._asdict(),
        **phot_randoms._asdict(),
        **dbk_randoms._asdict(),
        **disk_bulge_history.fbulge_params._asdict(),
        bulge_to_total_history=disk_bulge_history.bulge_to_total_history,
        obs_mags_bulge=obs_mags_bulge,
        obs_mags_disk=obs_mags_disk,
        obs_mags_knots=obs_mags_knots,
    )
    return dbk_phot_info, dbk_weights


def _mc_dbk_specphot_kern(
    ran_key,
    z_obs,
    t_obs,
    mah_params,
    ssp_data,
    precomputed_ssp_mag_table,
    z_phot_table,
    wave_eff_table,
    line_wave_table,
    diffstarpop_params,
    mzr_params,
    spspop_params,
    scatter_params,
    ssp_err_pop_params,
    cosmo_params,
    fb,
):
    phot_key, dbk_key = jran.split(ran_key, 2)
    phot_kern_results, phot_randoms, spec_kern_results = _mc_specphot_kern(
        phot_key,
        z_obs,
        t_obs,
        mah_params,
        ssp_data,
        precomputed_ssp_mag_table,
        z_phot_table,
        wave_eff_table,
        line_wave_table,
        diffstarpop_params,
        mzr_params,
        spspop_params,
        scatter_params,
        ssp_err_pop_params,
        cosmo_params,
        fb,
    )

    burst_params = DEFAULT_BURST_PARAMS._replace(
        lgfburst=phot_kern_results.lgfburst,
        lgyr_peak=phot_kern_results.lgyr_peak,
        lgyr_max=phot_kern_results.lgyr_max,
    )
    _ret2 = _mc_dbk_kern(
        t_obs,
        ssp_data,
        phot_kern_results.t_table,
        phot_kern_results.sfh_table,
        burst_params,
        phot_kern_results.lgmet_weights,
        dbk_key,
    )
    dbk_randoms, dbk_weights, disk_bulge_history = _ret2

    _ret3 = _get_dbk_phot_from_dbk_weights(
        phot_kern_results.ssp_photflux_table,
        dbk_weights,
        phot_kern_results.dust_frac_trans,
        phot_kern_results.frac_ssp_errors,
    )
    obs_mags_bulge, obs_mags_disk, obs_mags_knots = _ret3

    line_names = []
    for name in ssp_data.ssp_emline_wave._fields:
        line_names.append(name)
        line_names.append(name + "_bulge")
        line_names.append(name + "_disk")
        line_names.append(name + "_knots")

    dbk_specphot_keys = (*MCDBKPhotInfo._fields, *line_names)
    MCDBKSpecPhotInfo = namedtuple("MCDBKSpecPhotInfo", dbk_specphot_keys)

    _dbk_line_res = _get_dbk_linelum_decomposition(
        dbk_weights, spec_kern_results, ssp_data
    )
    linelums_bulge, linelums_disk, linelums_knots = _dbk_line_res
    linelum_dict = dict()
    for i, name in enumerate(ssp_data.ssp_emline_wave._fields):
        linelum_dict[name] = spec_kern_results.gal_linelums[:, i]
        linelum_dict[name + "_bulge"] = linelums_bulge[:, i]
        linelum_dict[name + "_disk"] = linelums_disk[:, i]
        linelum_dict[name + "_knots"] = linelums_knots[:, i]

    dbk_specphot_info = MCDBKSpecPhotInfo(
        **phot_kern_results._asdict(),
        **phot_randoms._asdict(),
        **dbk_randoms._asdict(),
        **disk_bulge_history.fbulge_params._asdict(),
        bulge_to_total_history=disk_bulge_history.bulge_to_total_history,
        obs_mags_bulge=obs_mags_bulge,
        obs_mags_disk=obs_mags_disk,
        obs_mags_knots=obs_mags_knots,
        **linelum_dict,
    )
    return dbk_specphot_info, dbk_weights


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


def _mc_lc_dbk_sed_kern(
    dbk_phot_info,
    dbk_weights,
    z_obs,
    ssp_data,
    spspop_params,
    scatter_params,
    ssperr_params,
):
    n_gals = dbk_phot_info.logsm_obs.shape[0]
    wave_eff_galpop = jnp.tile(ssp_data.ssp_wave, n_gals).reshape((n_gals, -1))

    dust_frac_trans, dust_params = sspwk.compute_dust_attenuation(
        dbk_phot_info.uran_av,
        dbk_phot_info.uran_delta,
        dbk_phot_info.uran_funo,
        dbk_phot_info.logsm_obs,
        dbk_phot_info.logssfr_obs,
        ssp_data,
        z_obs,
        wave_eff_galpop,
        spspop_params.dustpop_params,
        scatter_params,
    )
    dust_params = dust_params._replace(
        av=dust_params.av[:, 0, -1],
        delta=dust_params.delta[:, 0],
        funo=dust_params.funo[:, 0],
    )

    # Calculate mean fractional change to the SSP fluxes in each band for each galaxy
    # L'_SSP(λ_eff) = L_SSP(λ_eff) & F_SSP(λ_eff)
    frac_ssp_errors_nonoise = ssp_err_model.frac_ssp_err_at_z_obs_galpop(
        ssperr_params, dbk_phot_info.logsm_obs, z_obs, wave_eff_galpop
    )
    frac_ssp_errors = ssp_err_model.get_noisy_frac_ssp_errors(
        wave_eff_galpop, frac_ssp_errors_nonoise, dbk_phot_info.delta_mag_ssp_scatter
    )

    n_met, n_age, n_wave = ssp_data.ssp_flux.shape
    dust_frac_trans = dust_frac_trans.swapaxes(1, 2)  # (n_gals, n_age, n_wave)

    a = dust_frac_trans.reshape((n_gals, 1, n_age, n_wave))
    b = frac_ssp_errors.reshape((n_gals, 1, 1, n_wave))
    d = ssp_data.ssp_flux.reshape((1, n_met, n_age, n_wave))

    _w_bulge = dbk_weights.ssp_weights_bulge.reshape((n_gals, n_met, n_age, 1))
    _w_dd = dbk_weights.ssp_weights_disk.reshape((n_gals, n_met, n_age, 1))
    _w_knot = dbk_weights.ssp_weights_knots.reshape((n_gals, n_met, n_age, 1))

    mb = dbk_phot_info.mstar_bulge.reshape((n_gals, 1))
    md = dbk_phot_info.mstar_disk.reshape((n_gals, 1))
    mk = dbk_phot_info.mstar_knots.reshape((n_gals, 1))

    sed_bulge = jnp.sum(a * b * _w_bulge * d, axis=(1, 2)) * mb
    sed_disk = jnp.sum(a * b * _w_dd * d, axis=(1, 2)) * md
    sed_knots = jnp.sum(a * b * _w_knot * d, axis=(1, 2)) * mk

    return sed_bulge, sed_disk, sed_knots


DBK_PHOT_EXTRA_FIELDS = (
    *dbk.FbulgeParams._fields,
    "bulge_to_total_history",
    "obs_mags_bulge",
    "obs_mags_disk",
    "obs_mags_knots",
)
MCDBKPhotInfo = namedtuple(
    "MCDBKPhotInfo",
    (
        *PhotKernResults._fields,
        *PhotRandoms._fields,
        *DBKRandoms._fields,
        *DBK_PHOT_EXTRA_FIELDS,
    ),
)
