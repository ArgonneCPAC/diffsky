"""
"""
import os

import numpy as np
from dsps.constants import T_TABLE_MIN
from dsps.cosmology import DEFAULT_COSMOLOGY
from dsps.data_loaders.retrieve_fake_fsps_data import load_fake_ssp_data

from .. import tw_photgrad

DEFAULT_NGALS = 5
DEFAULT_NFILTERS = 3

NGALS = os.environ.get("TASSO_NGALS", DEFAULT_NGALS)
NFILTERS = os.environ.get("TASSO_NFILTERS", DEFAULT_NFILTERS)


def test_approx_tw_photgrad_evaluates():
    ssp_data = load_fake_ssp_data()
    n_met, n_age, n_wave = ssp_data.ssp_flux.shape
    ssp_flux_table = np.ones((n_met, n_age))
    wave_eff_aa = 5000.0
    z_obs = 0.25
    t_table = np.linspace(T_TABLE_MIN, 13.8, 100)
    sfr_table = np.ones_like(t_table)

    args = (
        tw_photgrad.DEFAULT_SPSPOP_PARAMS,
        DEFAULT_COSMOLOGY,
        ssp_data,
        ssp_flux_table,
        wave_eff_aa,
        z_obs,
        t_table,
        sfr_table,
    )
    mag = tw_photgrad.calc_approx_singlemag_singlegal(*args)
    assert mag.shape == ()
    assert np.isfinite(mag)


def test_approx_tw_photgrad_deriv_evaluates():
    ssp_data = load_fake_ssp_data()
    n_met, n_age, n_wave = ssp_data.ssp_flux.shape
    ssp_flux_table = np.ones((n_met, n_age))
    wave_eff_aa = 5000.0
    z_obs = 0.25
    t_table = np.linspace(T_TABLE_MIN, 13.8, 100)
    sfr_table = np.ones_like(t_table)

    args = (
        tw_photgrad.DEFAULT_SPSPOP_PARAMS,
        DEFAULT_COSMOLOGY,
        ssp_data,
        ssp_flux_table,
        wave_eff_aa,
        z_obs,
        t_table,
        sfr_table,
    )
    mag, grads = tw_photgrad.calc_approx_singlemag_singlegal_grads(*args)
    assert mag.shape == ()
    assert np.isfinite(mag)

    assert np.all(np.isfinite(grads.burstpop_params.fburstpop_params))
    assert np.all(np.isfinite(grads.burstpop_params.tburstpop_params))

    assert np.all(np.isfinite(grads.dustpop_params.avpop_params))
    assert np.all(np.isfinite(grads.dustpop_params.deltapop_params))
    assert np.all(np.isfinite(grads.dustpop_params.funopop_params))


def test_approx_tw_photgrad_galpop_deriv_evaluates():
    ssp_data = load_fake_ssp_data()
    n_met, n_age, n_wave = ssp_data.ssp_flux.shape

    ssp_flux_table = np.ones((NGALS, n_met, n_age))
    wave_eff_aa = 5000.0
    z_obs = 0.25
    t_table = np.linspace(T_TABLE_MIN, 13.8, 100)
    sfr_table = np.ones((NGALS, t_table.size))

    args = (
        tw_photgrad.DEFAULT_SPSPOP_PARAMS,
        DEFAULT_COSMOLOGY,
        ssp_data,
        ssp_flux_table,
        wave_eff_aa,
        z_obs,
        t_table,
        sfr_table,
    )
    mag, grads = tw_photgrad.calc_approx_singlemag_galpop_grads(*args)
    assert mag.shape == (NGALS,)
    assert np.all(np.isfinite(mag))

    assert np.all(np.isfinite(grads.burstpop_params.fburstpop_params))
    assert np.all(np.isfinite(grads.burstpop_params.tburstpop_params))

    assert np.all(np.isfinite(grads.dustpop_params.avpop_params))
    assert np.all(np.isfinite(grads.dustpop_params.deltapop_params))
    assert np.all(np.isfinite(grads.dustpop_params.funopop_params))


def test_tw_photgrad_evaluates():
    ssp_data = load_fake_ssp_data()
    wave_eff_aa = 5000.0
    z_obs = 0.25
    t_table = np.linspace(T_TABLE_MIN, 13.8, 100)
    sfr_table = np.ones_like(t_table)

    args = (
        tw_photgrad.DEFAULT_SPSPOP_PARAMS,
        DEFAULT_COSMOLOGY,
        ssp_data,
        wave_eff_aa,
        z_obs,
        t_table,
        sfr_table,
    )

    mag = tw_photgrad.calc_singlemag_singlegal(*args)
    assert mag.shape == ()
    assert np.isfinite(mag)


def test_tw_photgrad_deriv_evaluates():
    ssp_data = load_fake_ssp_data()
    wave_eff_aa = 5000.0
    z_obs = 0.25
    t_table = np.linspace(T_TABLE_MIN, 13.8, 100)
    sfr_table = np.ones_like(t_table)

    args = (
        tw_photgrad.DEFAULT_SPSPOP_PARAMS,
        DEFAULT_COSMOLOGY,
        ssp_data,
        wave_eff_aa,
        z_obs,
        t_table,
        sfr_table,
    )
    grads = tw_photgrad.calc_singlemag_singlegal_grads(*args)

    assert np.all(np.isfinite(grads.burstpop_params.fburstpop_params))
    assert np.all(np.isfinite(grads.burstpop_params.tburstpop_params))

    assert np.all(np.isfinite(grads.dustpop_params.avpop_params))
    assert np.all(np.isfinite(grads.dustpop_params.deltapop_params))
    assert np.all(np.isfinite(grads.dustpop_params.funopop_params))


def test_tw_photgrad_galpop_deriv_evaluates():
    ssp_data = load_fake_ssp_data()

    wave_eff_aa = 5_000.0
    z_obs = np.linspace(0.1, 3.0, NGALS)
    t_table = np.linspace(T_TABLE_MIN, 13.8, 100)
    sfr_table = np.ones((NGALS, t_table.size))

    args = (
        tw_photgrad.DEFAULT_SPSPOP_PARAMS,
        DEFAULT_COSMOLOGY,
        ssp_data,
        wave_eff_aa,
        z_obs,
        t_table,
        sfr_table,
    )
    grads = tw_photgrad.calc_singlemag_galpop_grads(*args)

    assert np.all(np.isfinite(grads.burstpop_params.fburstpop_params))
    assert np.all(np.isfinite(grads.burstpop_params.tburstpop_params))

    assert np.all(np.isfinite(grads.dustpop_params.avpop_params))
    assert np.all(np.isfinite(grads.dustpop_params.deltapop_params))
    assert np.all(np.isfinite(grads.dustpop_params.funopop_params))


def test_calc_multimag_galpop_grads_evaluates():
    ssp_data = load_fake_ssp_data()

    wave_eff_aa = np.linspace(1_000, 10_000, NFILTERS)
    z_obs = np.linspace(0.1, 3.0, NGALS)
    t_table = np.linspace(T_TABLE_MIN, 13.8, 100)
    sfr_table = np.ones((NGALS, t_table.size))

    args = (
        tw_photgrad.DEFAULT_SPSPOP_PARAMS,
        DEFAULT_COSMOLOGY,
        ssp_data,
        wave_eff_aa,
        z_obs,
        t_table,
        sfr_table,
    )
    grads = tw_photgrad.calc_multimag_galpop_grads(*args)

    assert np.all(np.isfinite(grads.burstpop_params.fburstpop_params))
    assert np.all(np.isfinite(grads.burstpop_params.tburstpop_params))

    assert np.all(np.isfinite(grads.dustpop_params.avpop_params))
    assert np.all(np.isfinite(grads.dustpop_params.deltapop_params))
    assert np.all(np.isfinite(grads.dustpop_params.funopop_params))
