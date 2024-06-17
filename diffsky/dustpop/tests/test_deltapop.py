"""
"""
import numpy as np
from jax import random as jran

from ..deltapop import (
    DEFAULT_DELTAPOP_PARAMS,
    DEFAULT_DELTAPOP_U_PARAMS,
    get_bounded_deltapop_params,
    get_delta_from_deltapop_params,
    get_delta_from_deltapop_u_params,
    get_unbounded_deltapop_params,
)

TOL = 1e-2


def test_param_u_param_names_propagate_properly():
    gen = zip(DEFAULT_DELTAPOP_U_PARAMS._fields, DEFAULT_DELTAPOP_PARAMS._fields)
    for u_key, key in gen:
        assert u_key[:2] == "u_"
        assert u_key[2:] == key

    inferred_default_params = get_bounded_deltapop_params(DEFAULT_DELTAPOP_U_PARAMS)
    assert set(inferred_default_params._fields) == set(DEFAULT_DELTAPOP_PARAMS._fields)

    inferred_default_u_params = get_unbounded_deltapop_params(DEFAULT_DELTAPOP_PARAMS)
    assert set(inferred_default_u_params._fields) == set(
        DEFAULT_DELTAPOP_U_PARAMS._fields
    )


def test_get_bounded_deltapop_params_fails_when_passing_params():
    try:
        get_bounded_deltapop_params(DEFAULT_DELTAPOP_PARAMS)
        raise NameError("get_bounded_deltapop_params should not accept params")
    except AttributeError:
        pass


def test_get_unbounded_deltapop_params_fails_when_passing_u_params():
    try:
        get_unbounded_deltapop_params(DEFAULT_DELTAPOP_U_PARAMS)
        raise NameError("get_unbounded_deltapop_params should not accept u_params")
    except AttributeError:
        pass


def test_get_av_from_deltapop_params_fails_when_passing_u_params():
    logsm, logssfr = 10.0, -11.0

    try:
        get_delta_from_deltapop_params(DEFAULT_DELTAPOP_U_PARAMS, logsm, logssfr)
        raise NameError("get_delta_from_deltapop_params should not accept u_params")
    except AttributeError:
        pass


def test_get_av_from_deltapop_u_params_fails_when_passing_params():
    logsm, logssfr = 10.0, -11.0

    try:
        get_delta_from_deltapop_u_params(DEFAULT_DELTAPOP_PARAMS, logsm, logssfr)
        raise NameError("get_delta_from_deltapop_u_params should not accept params")
    except AttributeError:
        pass


def test_get_bursty_age_weights_pop_evaluates():
    n_gals = 500
    ran_key = jran.PRNGKey(0)
    logsm_key, logssfr_key = jran.split(ran_key, 2)
    gal_logsm = jran.uniform(logsm_key, minval=0, maxval=10, shape=(n_gals,))
    gal_logssfr = jran.uniform(logssfr_key, minval=-12, maxval=-8, shape=(n_gals,))

    gal_deltapop = get_delta_from_deltapop_params(
        DEFAULT_DELTAPOP_PARAMS, gal_logsm, gal_logssfr
    )
    assert gal_deltapop.shape == (n_gals,)


def test_u_param_inversion():
    ran_key = jran.PRNGKey(0)
    logsm_key, logssfr_key = jran.split(ran_key, 2)
    assert np.allclose(
        DEFAULT_DELTAPOP_PARAMS,
        get_bounded_deltapop_params(DEFAULT_DELTAPOP_U_PARAMS),
        rtol=TOL,
    )

    inferred_default_params = get_bounded_deltapop_params(
        get_unbounded_deltapop_params(DEFAULT_DELTAPOP_PARAMS)
    )
    assert np.allclose(DEFAULT_DELTAPOP_PARAMS, inferred_default_params, rtol=TOL)

    n_gals = 500
    gal_logsm = jran.uniform(logsm_key, minval=0, maxval=10, shape=(n_gals,))
    gal_logssfr = jran.uniform(logssfr_key, minval=-12, maxval=-8, shape=(n_gals,))

    gal_deltapop = get_delta_from_deltapop_params(
        DEFAULT_DELTAPOP_PARAMS, gal_logsm, gal_logssfr
    )
    assert gal_deltapop.shape == (n_gals,)
    assert np.all(np.isfinite(gal_deltapop))

    gal_deltapop_u = get_delta_from_deltapop_u_params(
        DEFAULT_DELTAPOP_U_PARAMS, gal_logsm, gal_logssfr
    )
    assert gal_deltapop_u.shape == (n_gals,)
    assert np.all(np.isfinite(gal_deltapop_u))

    assert np.allclose(gal_deltapop, gal_deltapop_u, rtol=1e-4)
