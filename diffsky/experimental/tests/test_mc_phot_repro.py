""""""

import numpy as np
import pytest
from diffstar import DEFAULT_DIFFSTAR_PARAMS
from dsps.cosmology import DEFAULT_COSMOLOGY
from dsps.sfh.diffburst import DEFAULT_BURST_PARAMS
from jax import random as jran

from ...dustpop.tw_dust import DEFAULT_DUST_PARAMS
from ...param_utils import diffsky_param_wrapper as dpw
from .. import mc_diffsky_seds as mcsed
from .. import mc_phot_repro
from ..kernels import ssp_weight_kernels_repro as sspwkr
from . import test_lc_phot_kern as tlcphk


@pytest.mark.skip
def test_mc_phot_kern_agrees_with_mc_diffsky_seds_phot_kern(num_halos=75):
    """Enforce agreement to 1e-4 for the photometry computed by these two functions:
    1. mcsed._mc_diffsky_phot_kern
    2. mc_phot_repro._mc_phot_kern

    """
    ran_key = jran.key(0)
    lc_data, tcurves = tlcphk._get_weighted_lc_data_for_unit_testing(
        num_halos=num_halos
    )

    fb = 0.156

    phot_info = mcsed._mc_diffsky_phot_kern(
        ran_key,
        lc_data.z_obs,
        lc_data.t_obs,
        lc_data.mah_params,
        lc_data.logmp0,
        lc_data.t_table,
        lc_data.ssp_data,
        lc_data.precomputed_ssp_mag_table,
        lc_data.z_phot_table,
        lc_data.wave_eff_table,
        *dpw.DEFAULT_PARAM_COLLECTION,
        DEFAULT_COSMOLOGY,
        fb,
    )

    phot_info2 = mc_phot_repro._mc_phot_kern(
        ran_key,
        lc_data.z_obs,
        lc_data.t_obs,
        lc_data.mah_params,
        lc_data.ssp_data,
        lc_data.precomputed_ssp_mag_table,
        lc_data.z_phot_table,
        lc_data.wave_eff_table,
        *dpw.DEFAULT_PARAM_COLLECTION,
        DEFAULT_COSMOLOGY,
        fb,
    )[0]._asdict()

    TOL = 1e-4
    for p, p2 in zip(phot_info["obs_mags"], phot_info2["obs_mags"]):
        assert np.allclose(p, p2, rtol=TOL)

    assert "av" in phot_info["dust_params"]._fields
    assert "av" in phot_info2.keys()
    for pname in DEFAULT_DUST_PARAMS._fields:
        assert np.allclose(
            getattr(phot_info["dust_params"], pname), phot_info2[pname], rtol=TOL
        )

    assert "lgfburst" in phot_info["burst_params"]._fields
    assert "lgfburst" in phot_info2.keys()
    for pname in DEFAULT_BURST_PARAMS._fields[1:]:
        assert np.allclose(
            getattr(phot_info["burst_params"], pname), phot_info2[pname], rtol=TOL
        )

    assert np.allclose(phot_info["uran_av"], phot_info2["uran_av"])
    assert np.allclose(phot_info["uran_delta"], phot_info2["uran_delta"])
    assert np.allclose(phot_info["uran_funo"], phot_info2["uran_funo"])


@pytest.mark.skip
def test_mc_dbk_kern(num_halos=75):
    ran_key = jran.key(0)
    lc_data, tcurves = tlcphk._get_weighted_lc_data_for_unit_testing(
        num_halos=num_halos
    )

    fb = 0.156
    ran_key, phot_key = jran.split(ran_key, 2)

    phot_kern_results, phot_randoms = mc_phot_repro._mc_phot_kern(
        phot_key,
        dpw.DEFAULT_PARAM_COLLECTION[0],
        lc_data.z_obs,
        lc_data.t_obs,
        lc_data.mah_params,
        lc_data.ssp_data,
        lc_data.precomputed_ssp_mag_table,
        lc_data.z_phot_table,
        lc_data.wave_eff_table,
        *dpw.DEFAULT_PARAM_COLLECTION[1:],
        DEFAULT_COSMOLOGY,
        fb,
    )
    assert np.all(np.isfinite(phot_kern_results.obs_mags))
    assert np.all(phot_kern_results.lgfburst[phot_kern_results.mc_sfh_type < 2] < -7)

    assert np.allclose(
        np.sum(phot_kern_results.ssp_weights, axis=(1, 2)), 1.0, rtol=1e-4
    )
    assert np.all(phot_kern_results.frac_ssp_errors > 0)
    assert np.all(phot_kern_results.frac_ssp_errors < 5)

    ran_key, dbk_key = jran.split(ran_key, 2)
    burst_params = DEFAULT_BURST_PARAMS._replace(
        lgfburst=phot_kern_results.lgfburst,
        lgyr_peak=phot_kern_results.lgyr_peak,
        lgyr_max=phot_kern_results.lgyr_max,
    )
    args = (
        lc_data.t_obs,
        lc_data.ssp_data,
        phot_kern_results.t_table,
        phot_kern_results.sfh_table,
        burst_params,
        phot_kern_results.lgmet_weights,
        dbk_key,
    )
    dbk_randoms, dbk_weights, disk_bulge_history = mc_phot_repro._mc_dbk_kern(*args)
    assert np.all(np.isfinite(dbk_weights.ssp_weights_bulge))
    assert np.all(np.isfinite(dbk_weights.ssp_weights_disk))
    assert np.all(np.isfinite(dbk_weights.ssp_weights_knots))

    assert np.allclose(
        np.sum(dbk_weights.ssp_weights_bulge, axis=(1, 2)), 1.0, rtol=1e-4
    )
    assert np.allclose(
        np.sum(dbk_weights.ssp_weights_disk, axis=(1, 2)), 1.0, rtol=1e-4
    )
    assert np.allclose(
        np.sum(dbk_weights.ssp_weights_knots, axis=(1, 2)), 1.0, rtol=1e-4
    )

    assert np.all(dbk_weights.mstar_bulge > 0)
    assert np.all(dbk_weights.mstar_disk > 0)
    assert np.all(dbk_weights.mstar_knots > 0)

    args = (
        phot_kern_results.ssp_photflux_table,
        dbk_weights,
        phot_kern_results.dust_frac_trans,
        phot_kern_results.wave_eff_galpop,
        phot_kern_results.frac_ssp_errors,
        phot_randoms.delta_mag_ssp_scatter,
    )
    _res = mc_phot_repro.get_dbk_phot(*args)
    obs_mags_bulge, obs_mags_disk, obs_mags_knots = _res

    # np.all(phot_info.logsm_obs > np.log10(dbk_weights.mstar_bulge.flatten()))
    # np.all(phot_info.logsm_obs > np.log10(dbk_weights.mstar_disk.flatten()))
    # np.all(phot_info.logsm_obs > np.log10(dbk_weights.mstar_knots.flatten()))

    assert np.all(np.isfinite(obs_mags_bulge))
    assert np.all(np.isfinite(obs_mags_disk))
    assert np.all(np.isfinite(obs_mags_knots))

    assert not np.allclose(phot_kern_results.obs_mags, obs_mags_bulge, rtol=1e-4)
    assert np.all(phot_kern_results.obs_mags <= obs_mags_bulge)

    assert not np.allclose(phot_kern_results.obs_mags, obs_mags_disk, rtol=1e-4)
    assert np.all(phot_kern_results.obs_mags <= obs_mags_disk)

    assert not np.allclose(phot_kern_results.obs_mags, obs_mags_knots, rtol=1e-4)
    assert np.all(phot_kern_results.obs_mags <= obs_mags_knots)

    a = 10 ** (-0.4 * obs_mags_bulge)
    b = 10 ** (-0.4 * obs_mags_disk)
    c = 10 ** (-0.4 * obs_mags_knots)
    mtot = -2.5 * np.log10(a + b + c)

    magdiff = mtot - phot_kern_results.obs_mags
    assert np.all(np.abs(magdiff) < 0.1)

    mean_magdiff = np.mean(magdiff, axis=0)  # shape = (n_bands,)
    assert np.allclose(mean_magdiff, 0.0, atol=0.01)

    std_magdiff = np.std(magdiff, axis=0)
    assert np.all(std_magdiff < 0.01)


def test_sed_kern(num_halos=75):
    n_gals = num_halos
    ran_key = jran.key(0)
    lc_data, tcurves = tlcphk._get_weighted_lc_data_for_unit_testing(
        num_halos=num_halos
    )

    fb = 0.156
    ran_key, phot_key = jran.split(ran_key, 2)

    phot_kern_results, phot_randoms = mc_phot_repro._mc_phot_kern(
        phot_key,
        dpw.DEFAULT_PARAM_COLLECTION[0],
        lc_data.z_obs,
        lc_data.t_obs,
        lc_data.mah_params,
        lc_data.ssp_data,
        lc_data.precomputed_ssp_mag_table,
        lc_data.z_phot_table,
        lc_data.wave_eff_table,
        *dpw.DEFAULT_PARAM_COLLECTION[1:],
        DEFAULT_COSMOLOGY,
        fb,
    )
    sfh_params = DEFAULT_DIFFSTAR_PARAMS._make(
        [getattr(phot_kern_results, key) for key in DEFAULT_DIFFSTAR_PARAMS._fields]
    )
    sed_kern_results = mc_phot_repro._sed_kern(
        phot_randoms,
        sfh_params,
        lc_data.z_obs,
        lc_data.t_obs,
        lc_data.mah_params,
        lc_data.ssp_data,
        *dpw.DEFAULT_PARAM_COLLECTION[1:],
        DEFAULT_COSMOLOGY,
        fb,
    )
    rest_sed = sed_kern_results[0]
    assert False, rest_sed.shape
    # dust_frac_trans = sed_kern_results[0].swapaxes(1, 2)  # (n_gals, n_age, n_wave)
    # frac_ssp_errors = sed_kern_results[1]  # (n_gals, n_wave)
    # ssp_weights = sed_kern_results[2]  # (n_gals, n_met, n_age)
    # ssp_sed = lc_data.ssp_data.ssp_flux  # (n_met, n_age, n_wave)

    # n_met, n_age, n_wave = lc_data.ssp_data.ssp_flux.shape
    # a = dust_frac_trans.reshape((n_gals, 1, n_age, n_wave))
    # b = frac_ssp_errors.reshape((n_gals, 1, 1, n_wave))
    # c = ssp_weights.reshape((n_gals, n_met, n_age, 1))
    # d = ssp_sed.reshape((1, n_met, n_age, n_wave))
    # e = phot_kern_results.logsm_obs.reshape((n_gals, 1))
    # rest_sed = np.sum(a * b * c * d, axis=(1, 2)) * e
    # assert False, rest_sed.shape


# def test_vmapped_dust():
#     n_gals = 250

#     lc_data, tcurves = tlcphk._get_weighted_lc_data_for_unit_testing(num_halos=n_gals)

#     ran_key = jran.key(0)
#     ran_key, av_key, delta_key, funo_key = jran.split(ran_key, 4)
#     uran_av = jran.uniform(av_key, shape=(n_gals,))
#     uran_delta = jran.uniform(delta_key, shape=(n_gals,))
#     uran_funo = jran.uniform(funo_key, shape=(n_gals,))

#     logsm_obs = np.linspace(7, 12, n_gals)
#     logssfr_obs = np.linspace(-13, -8, n_gals)

#     ran_key, wave_key = jran.split(ran_key, 2)

#     n_bands = 5
#     wave_eff_galpop = jran.uniform(
#         wave_key, minval=1_000, maxval=10_000, shape=(n_gals, n_bands)
#     )

#     frac_trans, dust_params = sspwkr.compute_dust_attenuation(
#         uran_av,
#         uran_delta,
#         uran_funo,
#         logsm_obs,
#         logssfr_obs,
#         lc_data.ssp_data,
#         lc_data.z_obs,
#         wave_eff_galpop,
#         dpw.DEFAULT_PARAM_COLLECTION[2].dustpop_params,
#         dpw.DEFAULT_PARAM_COLLECTION[3],
#     )
#     frac_trans2 = np.swapaxes(frac_trans, 1, 2)
#     assert False, (frac_trans.shape, frac_trans2.shape)
