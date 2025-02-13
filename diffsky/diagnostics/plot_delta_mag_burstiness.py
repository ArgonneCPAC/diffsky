"""
"""

import os

import numpy as np
from diffmah.diffmahpop_kernels.bimod_censat_params import DEFAULT_DIFFMAHPOP_PARAMS
from diffmah.diffmahpop_kernels.mc_bimod_cens import mc_cenpop
from diffstarpop.defaults import DEFAULT_DIFFSTARPOP_PARAMS
from diffstarpop.mc_diffstarpop_tpeak import mc_diffstar_sfh_galpop
from dsps.constants import T_TABLE_MIN
from dsps.cosmology import DEFAULT_COSMOLOGY
from dsps.cosmology.flat_wcdm import _age_at_z_kern, age_at_z0
from dsps.data_loaders import load_filter_data, load_ssp_templates
from dsps.data_loaders.retrieve_fake_fsps_data import (
    load_fake_filter_transmission_curves,
    load_fake_ssp_data,
)
from dsps.dust.utils import get_filter_effective_wavelength
from dsps.photometry import photpop
from dsps.utils import cumulative_mstar_formed
from jax import jit as jjit
from jax import random as jran
from jax import vmap

from diffsky import tw_photgrad

try:
    from matplotlib import pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    DEFAULT_DSPS_DRN = os.environ["DSPS_DRN"]
except KeyError:
    DEFAULT_DSPS_DRN = ""


_POP = (None, None, None, None, None, None, None, 0)
_MAG = (None, None, None, 0, 0, None, None, None)
kern = jjit(
    vmap(vmap(tw_photgrad.calc_approx_singlemag_singlegal, in_axes=_MAG), in_axes=_POP)
)

cumulative_mstar_formed_galpop = jjit(vmap(cumulative_mstar_formed, in_axes=(None, 0)))

mpurple, mgreen, morange = ("#9467bd", "#2ca02c", "#ff7f0e")


def get_interpolated_lsst_tcurves(ssp_wave, drn_ssp_data=DEFAULT_DSPS_DRN):
    try:
        tcurve_u = load_filter_data.load_transmission_curve(
            bn_pat="lsst_u*", drn=drn_ssp_data
        )
        tcurve_g = load_filter_data.load_transmission_curve(
            bn_pat="lsst_g*", drn=drn_ssp_data
        )
        tcurve_r = load_filter_data.load_transmission_curve(
            bn_pat="lsst_r*", drn=drn_ssp_data
        )
        tcurve_i = load_filter_data.load_transmission_curve(
            bn_pat="lsst_i*", drn=drn_ssp_data
        )
        tcurve_z = load_filter_data.load_transmission_curve(
            bn_pat="lsst_z*", drn=drn_ssp_data
        )
        tcurve_y = load_filter_data.load_transmission_curve(
            bn_pat="lsst_y*", drn=drn_ssp_data
        )
    except (ImportError, OSError, ValueError, AssertionError):
        _res = load_fake_filter_transmission_curves()
        wave, u, g, r, i, z, y = _res
        tcurve_u = load_filter_data.TransmissionCurve(wave, u)
        tcurve_g = load_filter_data.TransmissionCurve(wave, g)
        tcurve_r = load_filter_data.TransmissionCurve(wave, r)
        tcurve_i = load_filter_data.TransmissionCurve(wave, i)
        tcurve_z = load_filter_data.TransmissionCurve(wave, z)
        tcurve_y = load_filter_data.TransmissionCurve(wave, y)

    tcurve_u = tcurve_u._replace(
        transmission=np.interp(ssp_wave, tcurve_u.wave, tcurve_u.transmission)
    )
    tcurve_g = tcurve_g._replace(
        transmission=np.interp(ssp_wave, tcurve_g.wave, tcurve_g.transmission)
    )
    tcurve_r = tcurve_r._replace(
        transmission=np.interp(ssp_wave, tcurve_r.wave, tcurve_r.transmission)
    )
    tcurve_i = tcurve_i._replace(
        transmission=np.interp(ssp_wave, tcurve_i.wave, tcurve_i.transmission)
    )
    tcurve_z = tcurve_z._replace(
        transmission=np.interp(ssp_wave, tcurve_z.wave, tcurve_z.transmission)
    )
    tcurve_y = tcurve_y._replace(
        transmission=np.interp(ssp_wave, tcurve_y.wave, tcurve_y.transmission)
    )

    tcurve_u = tcurve_u._replace(wave=ssp_wave)
    tcurve_g = tcurve_g._replace(wave=ssp_wave)
    tcurve_r = tcurve_r._replace(wave=ssp_wave)
    tcurve_i = tcurve_i._replace(wave=ssp_wave)
    tcurve_z = tcurve_z._replace(wave=ssp_wave)
    tcurve_y = tcurve_y._replace(wave=ssp_wave)

    tcurves = list((tcurve_u, tcurve_g, tcurve_r, tcurve_i, tcurve_z, tcurve_y))
    return tcurves


def get_wave_eff(tcurves, z_obs):
    wave_eff_arr = []
    for tcurve in tcurves:
        waveff = get_filter_effective_wavelength(
            tcurve.wave, tcurve.transmission, z_obs
        )
        wave_eff_arr.append(waveff)
    wave_eff_arr = np.array(wave_eff_arr)
    return wave_eff_arr


def get_burstiness_delta_mag_quantities(
    z_obs, drn_ssp_data=DEFAULT_DSPS_DRN, n_halos=2_000
):
    ran_key = jran.key(0)
    try:
        ssp_data = load_ssp_templates(drn=drn_ssp_data)
    except (AssertionError, OSError, ValueError):
        ssp_data = load_fake_ssp_data()
        print(f"{drn_ssp_data} directory not found. Using fake SSP SEDs")

    lsst_tcurves = get_interpolated_lsst_tcurves(
        ssp_data.ssp_wave, drn_ssp_data=drn_ssp_data
    )
    wave_eff_arr = get_wave_eff(lsst_tcurves, z_obs)

    X = np.array([ssp_data.ssp_wave] * 6)
    Y = np.array([x.transmission for x in lsst_tcurves])

    ssp_flux_table = 10 ** (
        -0.4
        * photpop.precompute_ssp_restmags(ssp_data.ssp_wave, ssp_data.ssp_flux, X, Y)
    )
    ssp_flux_table_multimag = np.swapaxes(np.swapaxes(ssp_flux_table, 0, 2), 1, 2)

    t_obs = _age_at_z_kern(z_obs, *DEFAULT_COSMOLOGY)
    t0 = age_at_z0(*DEFAULT_COSMOLOGY)
    logt0 = np.log10(t0)

    n_t = 120
    tarr = np.linspace(T_TABLE_MIN, t0, n_t)

    _ZH = np.zeros(n_halos)
    hosts_logmh_at_z = np.linspace(10, 15, n_halos)

    ran_key, halo_key = jran.split(ran_key, 2)

    args = (
        DEFAULT_DIFFMAHPOP_PARAMS,
        tarr,
        hosts_logmh_at_z,
        t_obs + _ZH,
        halo_key,
        logt0,
    )

    halopop = mc_cenpop(*args)

    logmp0 = halopop.log_mah[:, -1]

    ran_key, sfh_key = jran.split(ran_key, 2)

    args = (
        DEFAULT_DIFFSTARPOP_PARAMS,
        halopop.mah_params,
        logmp0,
        logmp0,
        logmp0,
        _ZH + t0,
        ran_key,
        tarr,
    )
    _res = mc_diffstar_sfh_galpop(*args)
    diffstar_params_ms, diffstar_params_q, sfh_ms, sfh_q, frac_q, mc_is_q = _res

    sfh_galpop = np.where(mc_is_q.reshape((-1, 1)), sfh_q, sfh_ms)
    smh_galpop = cumulative_mstar_formed_galpop(tarr, sfh_galpop)

    args = (
        tw_photgrad.DEFAULT_SPSPOP_PARAMS,
        DEFAULT_COSMOLOGY,
        ssp_data,
        ssp_flux_table_multimag,
        wave_eff_arr,
        z_obs,
        tarr,
        sfh_galpop,
    )
    mags = kern(*args)

    alt_fburstpop_params = (
        tw_photgrad.DEFAULT_SPSPOP_PARAMS.burstpop_params.fburstpop_params._replace(
            lgfburst_logsm_ylo_q=-3.0,
            lgfburst_logsm_ylo_ms=-3.0,
            lgfburst_logsm_yhi_q=-3.0,
            lgfburst_logsm_yhi_ms=-3.0,
        )
    )

    alt_burstpop_params = tw_photgrad.DEFAULT_SPSPOP_PARAMS.burstpop_params._replace(
        fburstpop_params=alt_fburstpop_params
    )

    alt_spspop_params = tw_photgrad.DEFAULT_SPSPOP_PARAMS._replace(
        burstpop_params=alt_burstpop_params
    )

    args = (
        alt_spspop_params,
        DEFAULT_COSMOLOGY,
        ssp_data,
        ssp_flux_table_multimag,
        wave_eff_arr,
        z_obs,
        tarr,
        sfh_galpop,
    )
    alt_mags = kern(*args)

    return mags, alt_mags, halopop, sfh_galpop, smh_galpop, mc_is_q


def plot_delta_mag_lsst_vs_logsm(z_obs, n_halos=2_000, drn_ssp_data=DEFAULT_DSPS_DRN):
    if not HAS_MATPLOTLIB:
        raise ImportError("Must have matplotlib installed to use this function")

    _res = get_burstiness_delta_mag_quantities(
        z_obs, n_halos=n_halos, drn_ssp_data=drn_ssp_data
    )
    mags, alt_mags, halopop, sfh_galpop, smh_galpop, mc_is_q = _res

    fig, ax = plt.subplots(1, 1)
    ax.set_ylim(-2.5, 0.2)
    ax.set_xlim(6, 12)

    ax.scatter(
        np.log10(smh_galpop[:, -1]),
        alt_mags[:, 3] - mags[:, 3],
        s=1,
        color=morange,
        label=r"$M_{\rm i}$",
    )
    ax.scatter(
        np.log10(smh_galpop[:, -1]),
        alt_mags[:, 1] - mags[:, 1],
        s=1,
        color=mgreen,
        label=r"$M_{\rm g}$",
    )
    ax.scatter(
        np.log10(smh_galpop[:, -1]),
        alt_mags[:, 0] - mags[:, 0],
        s=1,
        color=mpurple,
        label=r"$M_{\rm u}$",
    )
    ax.legend(markerscale=4)

    xlabel = ax.set_xlabel(r"$\log_{10}M_{\star}$")
    ax.set_title(r"$z=0\ {\rm centrals}$")
    ylabel = ax.set_ylabel(r"$M_{\rm bursty}-M_{\rm smooth}$")

    fig.savefig(
        "delta_mag_burstiness.png",
        bbox_extra_artists=[xlabel, ylabel],
        bbox_inches="tight",
        dpi=200,
    )
    return fig


def plot_delta_mag_lsst_vs_ssfr(
    z_obs,
    figname="delta_mag_burstiness_ssfr.png",
    n_halos=2_000,
    drn_ssp_data=DEFAULT_DSPS_DRN,
):
    if not HAS_MATPLOTLIB:
        raise ImportError("Must have matplotlib installed to use this function")

    _res = get_burstiness_delta_mag_quantities(
        z_obs, n_halos=n_halos, drn_ssp_data=drn_ssp_data
    )
    mags, alt_mags, halopop, sfh_galpop, smh_galpop, mc_is_q = _res

    ssfr_z0 = np.log10(sfh_galpop[:, -1]) - np.log10(smh_galpop[:, -1])

    fig, ax = plt.subplots(1, 1)
    ax.set_ylim(-2.5, 0.2)
    ax.set_xlim(-15, -9)

    ax.scatter(
        ssfr_z0, alt_mags[:, 3] - mags[:, 3], s=1, color=morange, label=r"$M_{\rm i}$"
    )
    ax.scatter(
        ssfr_z0, alt_mags[:, 1] - mags[:, 1], s=1, color=mgreen, label=r"$M_{\rm g}$"
    )
    ax.scatter(
        ssfr_z0, alt_mags[:, 0] - mags[:, 0], s=1, color=mpurple, label=r"$M_{\rm u}$"
    )
    ax.legend(markerscale=4)

    xlabel = ax.set_xlabel(r"$\log_{10}{\rm sSFR}$")
    ax.set_title(r"$z=0\ {\rm centrals}$")
    ylabel = ax.set_ylabel(r"$M_{\rm bursty}-M_{\rm smooth}$")

    fig.savefig(
        figname, bbox_extra_artists=[xlabel, ylabel], bbox_inches="tight", dpi=200
    )
    return fig
