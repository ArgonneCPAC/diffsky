""" """

import numpy as np

from jax import random as jran

from ..ssp_err_model import (
    DEFAULT_SSPERR_PARAMS,
    DEFAULT_SSPERR_PDICT,
    DEFAULT_SSPERR_U_PARAMS,
    SSPERR_PBOUNDS_PDICT,
    F_sps_err_lambda_galpop,
    compute_delta_mags_all_bands,
    delta_mag_from_lambda_rest,
    get_bounded_ssperr_params,
    get_unbounded_ssperr_params,
    add_delta_mag_to_photometry,
    noisy_delta_mag,
    compute_delta_scatter,
)

TOL = 1e-2


def test_default_params_are_in_bounds():
    for key in DEFAULT_SSPERR_PDICT.keys():
        bounds = SSPERR_PBOUNDS_PDICT[key]
        val = DEFAULT_SSPERR_PDICT[key]
        assert bounds[0] < val < bounds[1], key


def test_param_u_param_names_propagate_properly():
    gen = zip(DEFAULT_SSPERR_U_PARAMS._fields, DEFAULT_SSPERR_PARAMS._fields)
    for u_key, key in gen:
        assert u_key[:2] == "u_"
        assert u_key[2:] == key

    inferred_default_params = get_bounded_ssperr_params(DEFAULT_SSPERR_U_PARAMS)
    assert set(inferred_default_params._fields) == set(DEFAULT_SSPERR_PARAMS._fields)

    inferred_default_u_params = get_unbounded_ssperr_params(DEFAULT_SSPERR_PARAMS)
    assert set(inferred_default_u_params._fields) == set(
        DEFAULT_SSPERR_U_PARAMS._fields
    )


def test_get_bounded_ssperr_params_fails_when_passing_params():
    try:
        get_bounded_ssperr_params(DEFAULT_SSPERR_PARAMS)
        raise NameError("get_bounded_ssperr_params should not accept u_params")
    except AttributeError:
        pass


def test_get_unbounded_ssperr_params_fails_when_passing_u_params():
    try:
        get_unbounded_ssperr_params(DEFAULT_SSPERR_U_PARAMS)
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
    dmag_params = DEFAULT_SSPERR_PARAMS._replace(
        z0p0_dgr_yhi=0.4,
        z0p5_dgr_yhi=0.4,
        z1p1_dgr_yhi=0.4,
    )

    z_obs = 0.4

    delta_mags = compute_delta_mags_all_bands(logsm, z_obs, dmag_params)

    mags_orig = np.array(mags_orig)
    mags_new = mags_orig + delta_mags

    assert not np.allclose(mags_new, mags_orig)


def test_F_ssp_err_lambda():

    n_gals = 2_000
    logsmarr = np.linspace(8, 12, n_gals)

    wavelength = np.array([5200.0, 5400.0, 5800.0, 6300.0])
    wave_eff_rest = np.array([3000.0, 4000.0, 5000.0, 5500.0, 6000.0, 6500.0])

    z_obs = 0.4

    F_sps_err = F_sps_err_lambda_galpop(
        DEFAULT_SSPERR_PARAMS, logsmarr, z_obs, wavelength, wave_eff_rest
    )

    assert len(F_sps_err) == n_gals
    assert np.all(~np.isnan(F_sps_err))


def test_delta_mag_from_lambda_rest():

    n_gals = 2_000
    logsmarr = np.linspace(8, 12, n_gals)

    wavelength = 5800.0
    wave_eff_rest = np.array([3000.0, 4000.0, 5000.0, 5500.0, 6000.0, 6500.0])

    z_obs = 0.4

    delta_mag = delta_mag_from_lambda_rest(
        DEFAULT_SSPERR_PARAMS, z_obs, logsmarr, wavelength, wave_eff_rest
    )

    assert np.all(~np.isnan(delta_mag))
    assert delta_mag.shape == (n_gals,)


def test_compute_delta_scatter():

    ran_key = jran.PRNGKey(0)

    n_gals = 2_000
    logsmarr = np.linspace(8, 12, n_gals)
    n_filters = 4

    z_obs = 0.4

    wavelength = np.array((3800.0, 4800.0, 5800.0, 6200.0))

    wave_eff_rest = np.array([3000.0, 4000.0, 5000.0, 5500.0, 6000.0, 6500.0])

    delta_mag = delta_mag_from_lambda_rest(
        DEFAULT_SSPERR_PARAMS, z_obs, logsmarr, wavelength, wave_eff_rest
    )

    delta_scatter = compute_delta_scatter(ran_key, delta_mag)

    assert np.all(~np.isnan(delta_scatter))
    assert delta_scatter.shape == (n_gals, n_filters)


def test_noisy_delta_mag():

    n_gals = 2_000
    logsmarr = np.linspace(8, 12, n_gals)

    n_filters = 4

    z_obs = 0.4

    wavelength = np.array((3800.0, 4800.0, 5800.0, 6200.0))

    wave_eff_rest = np.array([3000.0, 4000.0, 5000.0, 5500.0, 6000.0, 6500.0])

    ran_key = jran.PRNGKey(0)

    noisy_dmag = noisy_delta_mag(
        DEFAULT_SSPERR_PARAMS, z_obs, logsmarr, wavelength, wave_eff_rest, ran_key
    )

    assert np.all(~np.isnan(noisy_dmag))
    assert noisy_dmag.shape == (n_gals, n_filters)


def test_add_delta_mag_to_photometry():

    n_gals = 2_000
    logsmarr = np.linspace(8, 12, n_gals)
    n_filters = 4
    m = np.linspace(18, 24, n_filters)
    M = np.tile(m, n_gals)
    mags = M.reshape(n_gals, n_filters)

    wavelength = np.array((3800.0, 4800.0, 5800.0, 6200.0))

    wave_eff_rest = np.array([3000.0, 4000.0, 5000.0, 5500.0, 6000.0, 6500.0])

    z_obs = 0.4

    ran_key = jran.PRNGKey(0)

    q_key, ms_key = jran.split(ran_key, 2)

    res = add_delta_mag_to_photometry(
        DEFAULT_SSPERR_PARAMS,
        z_obs,
        logsmarr,
        logsmarr,
        wavelength,
        wave_eff_rest,
        q_key,
        ms_key,
        mags,
        mags,
        mags,
        mags
    )

    (new_mags_q_smooth, new_mags_q_bursty, new_mags_ms_smooth, new_mags_ms_bursty) = res

    assert np.all(~np.isnan(new_mags_q_smooth))
    assert new_mags_q_smooth.shape == (n_gals, n_filters)
