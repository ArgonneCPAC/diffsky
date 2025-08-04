""""""

import numpy as np
from jax import random as jran

from .. import halobias_model as hbm


def test_tw_quintuple_sigmoid():
    lgm = np.linspace(11, 15, 100)
    lgb = hbm.predict_lgbias_kern(hbm.HALOBIAS_PARAMS, lgm)
    assert np.all(np.isfinite(lgb))
    assert np.all(lgb > -2)
    assert np.all(lgb < 3)
    assert np.all(np.diff(lgb) > 0)


def test_default_params_are_in_bounds():
    assert hbm.HALOBIAS_PARAMS.hb_ytp > -3
    assert hbm.HALOBIAS_PARAMS.hb_s1 > hbm.HALOBIAS_PARAMS.hb_s0
    assert hbm.HALOBIAS_PARAMS.hb_s2 > hbm.HALOBIAS_PARAMS.hb_s1
    assert hbm.HALOBIAS_PARAMS.hb_s3 > hbm.HALOBIAS_PARAMS.hb_s2
    assert hbm.HALOBIAS_PARAMS.hb_s4 > hbm.HALOBIAS_PARAMS.hb_s3
    assert hbm.HALOBIAS_PARAMS.hb_s5 > hbm.HALOBIAS_PARAMS.hb_s4


def test_halobias_params_are_invertible():
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
