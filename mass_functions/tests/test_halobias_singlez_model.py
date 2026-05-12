""""""

import numpy as np
from jax import random as jran

from .. import halobias_singlez_model as hbm


def enforce_param_bounds(ytp, s0, s1, s2, s3, s4, s5):
    assert ytp > hbm.HB_YTP_PBOUNDS[0]
    assert ytp < hbm.HB_YTP_PBOUNDS[1]
    assert s1 > s0
    assert s2 > s1
    assert s3 > s2
    assert s4 > s3
    assert s5 > s4


def test_tw_quintuple_sigmoid():
    lgm = np.linspace(11, 15, 100)
    lgb = hbm.predict_lgbias_kern(hbm.HALOBIAS_PARAMS, lgm)
    assert np.all(np.isfinite(lgb))
    assert np.all(lgb > -2)
    assert np.all(lgb < 3)
    assert np.all(np.diff(lgb) > 0)


def test_default_params_are_in_bounds():
    enforce_param_bounds(*hbm.HALOBIAS_PARAMS)


def test_default_params_are_mutual_inverses():
    u_params = hbm.get_unbounded_halobias_params(hbm.HALOBIAS_PARAMS)
    assert np.allclose(u_params, hbm.HALOBIAS_U_PARAMS, rtol=1e-4)

    params = hbm.get_bounded_halobias_params(hbm.HALOBIAS_U_PARAMS)
    assert np.allclose(params, hbm.HALOBIAS_PARAMS, rtol=1e-4)


def test_halobias_bounding_kernels_are_invertible():
    ran_key = jran.key(0)

    n_tests = 100

    for itest in range(n_tests):
        ran_key, test_key = jran.split(ran_key, 2)
        u_params = jran.uniform(test_key, shape=(len(hbm.HALOBIAS_PARAMS) - 1))
        params = hbm._get_bounded_slope_params_kern(u_params)
        u_params2 = hbm._get_unbounded_slope_params_kern(params)
        assert np.allclose(u_params, u_params2, rtol=1e-4)

        # Enforce that this test is nontrivial to get right
        params_cfact = hbm._get_bounded_slope_params_kern(u_params * 1.01)
        u_params_cfact = hbm._get_unbounded_slope_params_kern(params_cfact)
        assert not np.allclose(u_params, u_params_cfact, rtol=1e-4)


def test_halobias_params_are_invertible():
    ran_key = jran.key(0)

    n_tests = 100

    for itest in range(n_tests):
        ran_key, test_key = jran.split(ran_key, 2)
        u_params = jran.uniform(test_key, shape=(len(hbm.HALOBIAS_PARAMS)))
        u_params = hbm.HALOBIAS_U_PARAMS._make(u_params)
        params = hbm.get_bounded_halobias_params(u_params)
        enforce_param_bounds(*params)

        u_params2 = hbm.get_unbounded_halobias_params(params)
        assert np.allclose(u_params, u_params2, rtol=1e-4)


def test_get_bounded_params_fails_when_passing_params():
    try:
        hbm.get_bounded_halobias_params(hbm.HALOBIAS_PARAMS)
        raise NameError("get_bounded_halobias_params should not accept u_params")
    except AttributeError:
        pass


def test_get_UNbounded_params_fails_when_passing_U_params():
    try:
        hbm.get_unbounded_halobias_params(hbm.HALOBIAS_U_PARAMS)
        raise NameError("get_unbounded_halobias_params should not accept params")
    except AttributeError:
        pass
