""""""

import numpy as np
from jax import random as jran

from .. import halobias_model as hbm
from .. import halobias_singlez_model as hbszm


def enforce_param_bounds(
    ytp,
    s0,
    s1,
    s2,
    s3,
    s4,
    s5,
    hb_ytp_z_x0,
    hb_ytp_z_k,
    hb_ytp_z_lo,
    hb_ytp_z_hi,
    hb_x0_lgm_us,
    hb_k_lgm_us,
    hb_ylo_lgm_us_zlo,
    hb_yhi_lgm_us_zlo,
    hb_ylo_lgm_us_zhi,
    hb_yhi_lgm_us_zhi,
    hb_dus_z_x0,
    hb_dus_z_k,
):
    assert ytp > hbszm.HB_YTP_PBOUNDS[0]
    assert ytp < hbszm.HB_YTP_PBOUNDS[1]
    assert s1 > s0
    assert s2 > s1
    assert s3 > s2
    assert s4 > s3
    assert s5 > s4

    assert hb_ytp_z_k > hbm.K_BOUNDS[0]
    assert hb_ytp_z_k < hbm.K_BOUNDS[1]
    assert hb_k_lgm_us > hbm.K_BOUNDS[0]
    assert hb_k_lgm_us < hbm.K_BOUNDS[1]
    assert hb_dus_z_k > hbm.K_BOUNDS[0]
    assert hb_dus_z_k < hbm.K_BOUNDS[1]


def test_tw_quintuple_sigmoid():
    lgm = np.linspace(11, 15, 100)
    z_list = np.linspace(0, 10, 20)
    for redshift in z_list:
        lgb = hbm.predict_lgbias_kern(hbm.HALOBIAS_PARAMS, lgm, redshift)
        assert np.all(np.isfinite(lgb))
        assert np.all(lgb > -2)
        assert np.all(lgb < 3.5)
        assert np.all(np.diff(lgb) > 0)


def test_default_params_are_in_bounds():
    enforce_param_bounds(*hbm.HALOBIAS_PARAMS)


def test_default_params_are_mutual_inverses():
    u_params = hbm.get_unbounded_halobias_params(hbm.HALOBIAS_PARAMS)
    assert np.allclose(u_params, hbm.HALOBIAS_U_PARAMS, rtol=1e-4)

    params = hbm.get_bounded_halobias_params(hbm.HALOBIAS_U_PARAMS)
    assert np.allclose(params, hbm.HALOBIAS_PARAMS, rtol=1e-4)


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


def test_get_unbounded_params_fails_when_passing_u_params():
    try:
        hbm.get_unbounded_halobias_params(hbm.HALOBIAS_U_PARAMS)
        raise NameError("get_unbounded_halobias_params should not accept params")
    except AttributeError:
        pass
