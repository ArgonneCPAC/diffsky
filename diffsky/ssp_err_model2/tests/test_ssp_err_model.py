""" """

import numpy as np
from jax import random as jran

from .. import defaults as ssp_err_defaults
from .. import ssp_err_model

TOL = 1e-2


def test_default_params_are_in_bounds():
    for key in ssp_err_model.DEFAULT_SSPERR_PARAMS._fields:
        bounds = getattr(ssp_err_model.SSPERR_PBOUNDS, key)
        val = getattr(ssp_err_model.DEFAULT_SSPERR_PARAMS, key)
        assert bounds[0] < val < bounds[1], key


def test_param_u_param_names_propagate_properly():
    gen = zip(
        ssp_err_model.DEFAULT_SSPERR_U_PARAMS._fields,
        ssp_err_model.DEFAULT_SSPERR_PARAMS._fields,
    )
    for u_key, key in gen:
        assert u_key[:2] == "u_"
        assert u_key[2:] == key

    inferred_default_params = ssp_err_model.get_bounded_ssperr_params(
        ssp_err_model.DEFAULT_SSPERR_U_PARAMS
    )
    assert set(inferred_default_params._fields) == set(
        ssp_err_model.DEFAULT_SSPERR_PARAMS._fields
    )

    inferred_default_u_params = ssp_err_model.get_unbounded_ssperr_params(
        ssp_err_model.DEFAULT_SSPERR_PARAMS
    )
    assert set(inferred_default_u_params._fields) == set(
        ssp_err_model.DEFAULT_SSPERR_U_PARAMS._fields
    )


def test_get_bounded_ssperr_params_fails_when_passing_params():
    try:
        ssp_err_model.get_bounded_ssperr_params(ssp_err_model.DEFAULT_SSPERR_PARAMS)
        raise NameError("get_bounded_ssperr_params should not accept u_params")
    except AttributeError:
        pass


def test_get_unbounded_ssperr_params_fails_when_passing_u_params():
    try:
        ssp_err_model.get_unbounded_ssperr_params(ssp_err_model.DEFAULT_SSPERR_U_PARAMS)
        raise NameError("get_unbounded_ssperr_params should not accept u_params")
    except AttributeError:
        pass


def test_compute_delta_mags_all_bands():

    logsm = 10.0

    fuv_orig = np.random.uniform(0, 1, 1)
    nuv_orig = np.random.uniform(0, 1, 1)
    u_orig = np.random.uniform(0, 1, 1)
    g_orig = np.random.uniform(0, 1, 1)
    r_orig = np.random.uniform(0, 1, 1)
    i_orig = np.random.uniform(0, 1, 1)

    mags_orig = (fuv_orig, nuv_orig, u_orig, g_orig, r_orig, i_orig)
    dmag_params = ssp_err_model.DEFAULT_SSPERR_PARAMS._replace(
        z0p0_dgr_yhi=0.4,
        z0p5_dgr_yhi=0.4,
        z1p1_dgr_yhi=0.4,
    )

    z_obs = 0.4

    delta_mags = ssp_err_model.compute_delta_mags_all_bands(logsm, z_obs, dmag_params)

    mags_orig = np.array(mags_orig)
    mags_new = mags_orig + delta_mags

    assert not np.allclose(mags_new, mags_orig)


def test_frac_ssp_err_at_z_obs_singlegal():
    logsm = 10.0
    z_obs = 1.0
    wave_obs = 2000.0
    frac_ssp_err_z_obs = ssp_err_model.frac_ssp_err_at_z_obs_singlegal(
        ssp_err_model.DEFAULT_SSPERR_PARAMS, logsm, z_obs, wave_obs
    )
    assert np.all(np.isfinite(frac_ssp_err_z_obs))


def test_frac_ssp_err_at_z_obs_galpop():
    ran_key = jran.key(0)
    n_gals = 10_000
    ZZ = np.zeros(n_gals)
    logsm = 10.0 + ZZ
    z_obs = 1.0 + ZZ
    wave_obs = np.array((2_000, 5_000, 7_000))
    wave_obs_galpop = np.tile(wave_obs, n_gals).reshape((n_gals, wave_obs.size))
    assert wave_obs_galpop.shape == (n_gals, wave_obs.size)

    frac_min = 10 ** (-0.4 * ssp_err_defaults.DMAG_BOUNDS[1])
    frac_max = 10 ** (-0.4 * ssp_err_defaults.DMAG_BOUNDS[0])

    n_tests = 100
    for __ in range(n_tests):
        ran_key, test_key = jran.split(ran_key, 2)
        uran = jran.uniform(test_key, shape=(len(ssp_err_model.DEFAULT_SSPERR_PARAMS),))
        gen = zip(uran, ssp_err_model.DEFAULT_SSPERR_U_PARAMS)
        u_params = ssp_err_model.DEFAULT_SSPERR_U_PARAMS._make([x + u for x, u in gen])
        params = ssp_err_model.get_bounded_ssperr_params(u_params)

        frac_ssp_err_z_obs = ssp_err_model.frac_ssp_err_at_z_obs_galpop(
            params, logsm, z_obs, wave_obs_galpop
        )
        assert frac_ssp_err_z_obs.shape == (n_gals, 3)
        assert np.all(np.isfinite(frac_ssp_err_z_obs))
        assert np.all(frac_ssp_err_z_obs >= frac_min)
        assert np.all(frac_ssp_err_z_obs <= frac_max)

        # Test that zero-scatter results are unchanged
        delta_rest_mags_scatter = np.zeros((n_gals, ssp_err_model.LAMBDA_REST.size))
        frac_ssp_err_z_obs2, frac_ssp_err_z_obs2_nonoise = (
            ssp_err_model.get_noisy_frac_ssp_err_galpop(
                params, logsm, z_obs, wave_obs_galpop, delta_rest_mags_scatter
            )
        )
        assert frac_ssp_err_z_obs2.shape == frac_ssp_err_z_obs.shape
        assert np.allclose(frac_ssp_err_z_obs2, frac_ssp_err_z_obs, rtol=1e-4)
        assert np.allclose(frac_ssp_err_z_obs2, frac_ssp_err_z_obs2_nonoise, rtol=1e-4)

        delta_rest_mags_scatter = (
            np.zeros((n_gals, ssp_err_model.LAMBDA_REST.size)) + 0.1
        )
        frac_ssp_err_z_obs2, frac_ssp_err_z_obs2_nonoise = (
            ssp_err_model.get_noisy_frac_ssp_err_galpop(
                params, logsm, z_obs, wave_obs_galpop, delta_rest_mags_scatter
            )
        )
        assert not np.allclose(
            frac_ssp_err_z_obs2, frac_ssp_err_z_obs2_nonoise, rtol=1e-4
        )

        ran_key, scatter_key = jran.split(ran_key, 2)
        _res = ssp_err_model.frac_ssp_err_lambda_scatter_galpop(
            params, logsm, z_obs, wave_obs_galpop, scatter_key
        )
        frac_ssp_err_z_obs3, frac_ssp_err_z_obs3_nonoise, delta_rest_mags_scatter = _res
        assert frac_ssp_err_z_obs3.shape == (n_gals, 3)
        assert delta_rest_mags_scatter.shape == (n_gals, 6)

        assert np.all(np.isfinite(frac_ssp_err_z_obs3))
        assert np.allclose(
            frac_ssp_err_z_obs3_nonoise, frac_ssp_err_z_obs2_nonoise, rtol=1e-4
        )
        assert not np.allclose(
            frac_ssp_err_z_obs3, frac_ssp_err_z_obs3_nonoise, atol=0.001
        )

        frac_ssp_err_z_obs3b, frac_ssp_err_z_obs3b_nonoise = (
            ssp_err_model.get_noisy_frac_ssp_err_galpop(
                params, logsm, z_obs, wave_obs_galpop, delta_rest_mags_scatter
            )
        )
        assert np.allclose(frac_ssp_err_z_obs3b, frac_ssp_err_z_obs3, rtol=1e-4)
        assert np.allclose(frac_ssp_err_z_obs3b_nonoise, frac_ssp_err_z_obs, rtol=1e-4)
