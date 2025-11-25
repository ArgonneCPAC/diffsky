""" """

import numpy as np

from ..ssp_err_model import (
    DEFAULT_SSPERR_PARAMS,
    DEFAULT_SSPERR_U_PARAMS,
    SSPERR_PBOUNDS,
    compute_delta_mags_all_bands,
    get_bounded_ssperr_params,
    get_unbounded_ssperr_params,
)

TOL = 1e-2


def test_default_params_are_in_bounds():
    for key in DEFAULT_SSPERR_PARAMS._fields:
        bounds = getattr(SSPERR_PBOUNDS, key)
        val = getattr(DEFAULT_SSPERR_PARAMS, key)
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
