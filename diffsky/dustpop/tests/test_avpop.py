"""
"""
import numpy as np
from jax import random as jran

from ..avpop import (
    DEFAULT_AVPOP_PARAMS,
    DEFAULT_AVPOP_U_PARAMS,
    LGAV_BOUNDS,
    get_av_from_avpop_params,
    get_av_from_avpop_u_params,
    get_bounded_avpop_params,
    get_unbounded_avpop_params,
)

TOL = 1e-2


def test_param_u_param_names_propagate_properly():
    gen = zip(DEFAULT_AVPOP_U_PARAMS._fields, DEFAULT_AVPOP_PARAMS._fields)
    for u_key, key in gen:
        assert u_key[:2] == "u_"
        assert u_key[2:] == key

    inferred_default_params = get_bounded_avpop_params(DEFAULT_AVPOP_U_PARAMS)
    assert set(inferred_default_params._fields) == set(DEFAULT_AVPOP_PARAMS._fields)

    inferred_default_u_params = get_unbounded_avpop_params(DEFAULT_AVPOP_PARAMS)
    assert set(inferred_default_u_params._fields) == set(DEFAULT_AVPOP_U_PARAMS._fields)


def test_get_bounded_avpop_params_fails_when_passing_params():
    try:
        get_bounded_avpop_params(DEFAULT_AVPOP_PARAMS)
        raise NameError("get_bounded_avpop_params should not accept params")
    except AttributeError:
        pass


def test_get_unbounded_avpop_params_fails_when_passing_u_params():
    try:
        get_unbounded_avpop_params(DEFAULT_AVPOP_U_PARAMS)
        raise NameError("get_unbounded_avpop_params should not accept u_params")
    except AttributeError:
        pass


def test_get_av_from_avpop_params_fails_when_passing_u_params():
    logsm, logssfr = 10.0, -11.0

    try:
        get_av_from_avpop_params(DEFAULT_AVPOP_U_PARAMS, logsm, logssfr)
        raise NameError("get_av_from_avpop_params should not accept u_params")
    except AttributeError:
        pass


def test_get_av_from_avpop_u_params_fails_when_passing_params():
    logsm, logssfr = 10.0, -11.0

    try:
        get_av_from_avpop_u_params(DEFAULT_AVPOP_PARAMS, logsm, logssfr)
        raise NameError("get_av_from_avpop_u_params should not accept params")
    except AttributeError:
        pass


def test_get_bursty_age_weights_pop_evaluates():
    ran_key = jran.PRNGKey(0)
    logsm_key, logssfr_key = jran.split(ran_key, 2)
    n_gals = 500
    logsm = jran.uniform(logsm_key, minval=0, maxval=10, shape=(n_gals,))
    logssfr = jran.uniform(logssfr_key, minval=-12, maxval=-8, shape=(n_gals,))

    av = get_av_from_avpop_params(DEFAULT_AVPOP_PARAMS, logsm, logssfr)
    assert av.shape == (n_gals,)
    assert np.all(np.isfinite(av))


def test_avpop_u_param_inversion_default_params():
    assert np.allclose(
        DEFAULT_AVPOP_PARAMS,
        get_bounded_avpop_params(DEFAULT_AVPOP_U_PARAMS),
        rtol=TOL,
    )
    assert np.allclose(
        DEFAULT_AVPOP_U_PARAMS,
        get_unbounded_avpop_params(DEFAULT_AVPOP_PARAMS),
        rtol=TOL,
    )


def test_get_bursty_age_weights_pop_u_param_inversion():
    n_tests = 1_000
    ran_key = jran.PRNGKey(0)

    n_gals = 500

    for __ in range(n_tests):
        ran_key, logsm_key, logssfr_key = jran.split(ran_key, 3)
        logsm = jran.uniform(logsm_key, minval=0, maxval=10, shape=(n_gals,))
        logssfr = jran.uniform(logssfr_key, minval=-12, maxval=-8, shape=(n_gals,))

        av = get_av_from_avpop_params(DEFAULT_AVPOP_PARAMS, logsm, logssfr)
        assert av.shape == (n_gals,)
        assert np.all(np.isfinite(av))

        av_u = get_av_from_avpop_u_params(DEFAULT_AVPOP_U_PARAMS, logsm, logssfr)
        assert av_u.shape == (n_gals,)
        assert np.all(np.isfinite(av_u))

        assert np.allclose(av, av_u, rtol=1e-4)

        assert np.all(LGAV_BOUNDS[0] <= np.log10(av))
        assert np.all(np.log10(av) < LGAV_BOUNDS[1])
