"""
"""

import numpy as np
from jax import random as jran
from jax.nn import softplus

from ..avpop_mono import (
    AVPOP_PBOUNDS,
    DEFAULT_AVPOP_PARAMS,
    DEFAULT_AVPOP_U_PARAMS,
    DELTA_SUAV_AGE_BOUNDS,
    SUAV_BOUNDS,
    ZERODUST_AVPOP_PARAMS,
    AvPopUParams,
    get_av_from_avpop_params_galpop,
    get_av_from_avpop_params_singlegal,
    get_av_from_avpop_u_params_galpop,
    get_av_from_avpop_u_params_singlegal,
    get_bounded_avpop_params,
    get_unbounded_avpop_params,
)

TOL = 1e-2
N_AGE = 75
LGAGE_GYR = np.linspace(5, 10.25, N_AGE) - 9.0

N_GALS = 10
ZZ = np.zeros(N_GALS)

EPSILON = 1e-5


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
        raise NameError("get_bounded_avpop_params should not accept u_params")
    except AttributeError:
        pass


def test_get_unbounded_avpop_params_fails_when_passing_u_params():
    try:
        get_unbounded_avpop_params(DEFAULT_AVPOP_U_PARAMS)
        raise NameError("get_unbounded_avpop_params should not accept u_params")
    except AttributeError:
        pass


def test_default_params_are_in_bounds():

    gen = zip(DEFAULT_AVPOP_PARAMS, DEFAULT_AVPOP_PARAMS._fields)
    for val, key in gen:
        bound = getattr(AVPOP_PBOUNDS, key)
        assert bound[0] < val < bound[1]


def test_get_av_from_avpop_params_fails_when_passing_u_params():
    logsm, logssfr, redshift = 10.0 + ZZ, -11.0 + ZZ, 1.0 + ZZ

    try:
        get_av_from_avpop_params_galpop(
            DEFAULT_AVPOP_U_PARAMS, logsm, logssfr, redshift, LGAGE_GYR
        )
        raise NameError("get_av_from_avpop_params should not accept u_params")
    except AttributeError:
        pass


def test_get_av_from_avpop_u_params_fails_when_passing_params():
    logsm, logssfr, redshift = 10.0 + ZZ, -11.0 + ZZ, 1.0 + ZZ

    try:
        get_av_from_avpop_u_params_galpop(
            DEFAULT_AVPOP_PARAMS, logsm, logssfr, redshift, LGAGE_GYR
        )
        raise NameError("get_av_from_avpop_u_params should not accept u_params")
    except AttributeError:
        pass


def test_get_av_from_avpop_params_galpop_evaluates():
    ran_key = jran.PRNGKey(0)
    logsm_key, logssfr_key, z_key = jran.split(ran_key, 3)
    n_gals = 500
    logsm = jran.uniform(logsm_key, minval=0, maxval=10, shape=(n_gals,))
    logssfr = jran.uniform(logssfr_key, minval=-12, maxval=-8, shape=(n_gals,))
    redshift = jran.uniform(z_key, minval=0, maxval=10, shape=(n_gals,))

    av = get_av_from_avpop_params_galpop(
        DEFAULT_AVPOP_PARAMS, logsm, logssfr, redshift, LGAGE_GYR
    )
    assert av.shape == (n_gals, N_AGE)
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


def test_get_av_from_avpop_u_params_galpop_u_param_inversion_galpop():
    n_tests = 1_000
    ran_key = jran.PRNGKey(0)

    n_gals = 500

    for __ in range(n_tests):
        ran_key, logsm_key, logssfr_key, z_key = jran.split(ran_key, 4)
        logsm = jran.uniform(logsm_key, minval=0, maxval=10, shape=(n_gals,))
        logssfr = jran.uniform(logssfr_key, minval=-12, maxval=-8, shape=(n_gals,))
        redshift = jran.uniform(z_key, minval=0, maxval=10, shape=(n_gals,))

        av = get_av_from_avpop_params_galpop(
            DEFAULT_AVPOP_PARAMS, logsm, logssfr, redshift, LGAGE_GYR
        )
        assert av.shape == (n_gals, N_AGE)
        assert np.all(np.isfinite(av))

        av_u = get_av_from_avpop_u_params_galpop(
            DEFAULT_AVPOP_U_PARAMS, logsm, logssfr, redshift, LGAGE_GYR
        )
        assert av_u.shape == (n_gals, N_AGE)
        assert np.all(np.isfinite(av_u))

        assert np.allclose(av, av_u, rtol=1e-4)

        assert np.all(SUAV_BOUNDS[0] <= np.log10(av))
        assert np.all(np.log10(av) < SUAV_BOUNDS[1])


def test_get_av_from_avpop_u_params_galpop_pop_u_param_inversion_singlegal():
    n_tests = 1_000
    ran_key = jran.PRNGKey(0)

    for __ in range(n_tests):
        ran_key, logsm_key, logssfr_key, z_key = jran.split(ran_key, 4)
        logsm = jran.uniform(logsm_key, minval=0, maxval=10, shape=())
        logssfr = jran.uniform(logssfr_key, minval=-12, maxval=-8, shape=())
        redshift = jran.uniform(z_key, minval=0, maxval=10, shape=())

        av = get_av_from_avpop_params_singlegal(
            DEFAULT_AVPOP_PARAMS, logsm, logssfr, redshift, LGAGE_GYR
        )
        assert av.shape == (N_AGE,)
        assert np.all(np.isfinite(av))

        av_u = get_av_from_avpop_u_params_singlegal(
            DEFAULT_AVPOP_U_PARAMS, logsm, logssfr, redshift, LGAGE_GYR
        )
        assert av_u.shape == (N_AGE,)
        assert np.all(np.isfinite(av_u))

        assert np.allclose(av, av_u, rtol=1e-4)

        assert np.all(SUAV_BOUNDS[0] <= np.log10(av))
        assert np.all(np.log10(av) < SUAV_BOUNDS[1])


def test_get_av_from_avpop_is_monotonic_with_logsm_and_logssfr():
    n_tests = 100
    ran_key = jran.PRNGKey(0)

    n_gals = 50
    logsmarr = np.linspace(0, 20, n_gals)
    logssfrarr = np.linspace(-20, 20, n_gals)
    n_pars = len(DEFAULT_AVPOP_PARAMS)
    ZZ = np.zeros(n_gals)

    for __ in range(n_tests):
        ran_key, z_key, u_p_key = jran.split(ran_key, 3)

        ran_u_params = jran.uniform(u_p_key, minval=-10, maxval=10, shape=(n_pars,))
        avpop_params = get_bounded_avpop_params(AvPopUParams(*ran_u_params))

        redshift = jran.uniform(z_key, minval=0, maxval=10, shape=()) + ZZ

        suav_max = SUAV_BOUNDS[1] + DELTA_SUAV_AGE_BOUNDS[1]
        av_max = softplus(suav_max)
        for logssfr in logssfrarr:
            av = get_av_from_avpop_params_galpop(
                avpop_params, logsmarr, logssfr + ZZ, redshift, LGAGE_GYR
            )
            assert av.shape == (n_gals, N_AGE)
            assert np.all(av >= 0.0)
            assert np.all(av <= av_max)
            assert np.all(np.isfinite(av))
            assert np.all(np.diff(av, axis=0) >= -EPSILON)

        for logsm in logsmarr:
            av = get_av_from_avpop_params_galpop(
                avpop_params, logsm + ZZ, logssfrarr, redshift, LGAGE_GYR
            )
            assert av.shape == (n_gals, N_AGE)
            assert np.all(av >= 0.0)
            assert np.all(av <= av_max)
            assert np.all(np.isfinite(av))
            assert np.all(np.diff(av, axis=0) >= -EPSILON)


def test_get_av_is_always_within_bounds_for_all_u_params():
    """Walk around AvPopUParams space and compute Av for 5_000 random galaxies.
    Enforce that Av is always within bounds"""
    n_tests = 1_000
    n_gals = 5_000
    n_params = len(DEFAULT_AVPOP_PARAMS)
    ran_key = jran.PRNGKey(0)

    suav_min = SUAV_BOUNDS[0] + DELTA_SUAV_AGE_BOUNDS[0]
    av_min = softplus(suav_min)
    suav_max = SUAV_BOUNDS[1] + DELTA_SUAV_AGE_BOUNDS[1]
    av_max = softplus(suav_max)

    for __ in range(n_tests):
        ran_key, logsm_key, logssfr_key, z_key, u_p_key = jran.split(ran_key, 5)
        logsm = jran.uniform(logsm_key, minval=0, maxval=20, shape=(n_gals,))
        logssfr = jran.uniform(logssfr_key, minval=-20, maxval=0, shape=(n_gals,))
        redshift = jran.uniform(z_key, minval=0, maxval=10, shape=(n_gals,))
        u_params = jran.uniform(u_p_key, minval=-1_000, maxval=1_000, shape=(n_params,))
        avpop_u_params = AvPopUParams(*u_params)
        av = get_av_from_avpop_u_params_galpop(
            avpop_u_params, logsm, logssfr, redshift, LGAGE_GYR
        )

        assert np.all(av >= av_min), (av.min(), av_min)
        assert np.all(av <= av_max), (av.max(), av_max)


def test_zerodust_params_are_in_bounds():

    gen = zip(ZERODUST_AVPOP_PARAMS, ZERODUST_AVPOP_PARAMS._fields)
    for val, key in gen:
        bound = getattr(AVPOP_PBOUNDS, key)
        assert bound[0] < val < bound[1]


def test_zerodust_params_are_invertible():
    u_params = get_unbounded_avpop_params(ZERODUST_AVPOP_PARAMS)
    params = get_bounded_avpop_params(u_params)
    for p, p_orig in zip(ZERODUST_AVPOP_PARAMS, params):
        assert np.all(np.isfinite(p))
        assert np.allclose(p, p_orig, rtol=1e-4)


def test_av_is_finite_and_tiny_for_zerodust_params():
    ran_key = jran.PRNGKey(0)
    logsm_key, logssfr_key, z_key = jran.split(ran_key, 3)
    n_gals = 500
    logsm = jran.uniform(logsm_key, minval=5, maxval=13, shape=(n_gals,))
    logssfr = jran.uniform(logssfr_key, minval=-14, maxval=-6, shape=(n_gals,))
    redshift = jran.uniform(z_key, minval=0, maxval=10, shape=(n_gals,))

    av = get_av_from_avpop_params_galpop(
        ZERODUST_AVPOP_PARAMS, logsm, logssfr, redshift, LGAGE_GYR
    )
    assert av.shape == (n_gals, N_AGE)
    assert np.all(np.isfinite(av))
    assert np.all(av < 0.05)
