""" """

import numpy as np

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
    n_gals = 1_000
    ZZ = np.zeros(n_gals)
    logsm = 10.0 + ZZ
    z_obs = 1.0 + ZZ
    wave_obs = 2000.0 + ZZ
    frac_ssp_err_z_obs = ssp_err_model.frac_ssp_err_at_z_obs_singlegal(
        ssp_err_model.DEFAULT_SSPERR_PARAMS, logsm, z_obs, wave_obs
    )
    assert np.all(np.isfinite(frac_ssp_err_z_obs))
