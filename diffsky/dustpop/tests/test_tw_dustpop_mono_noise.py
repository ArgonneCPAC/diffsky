""" """

import numpy as np
from jax import random as jran

from .. import tw_dustpop_mono as twd
from .. import tw_dustpop_mono_noise as twdn

TOL = 1e-4


def test_dustpop_noise_u_param_inversion_default_params():
    u_params = twdn.get_unbounded_dustpop_scatter_params(
        twdn.DEFAULT_DUSTPOP_SCATTER_PARAMS
    )
    assert np.all(np.isfinite(u_params))
    params = twdn.get_bounded_dustpop_scatter_params(u_params)
    assert np.all(np.isfinite(params))
    assert np.allclose(params, twdn.DEFAULT_DUSTPOP_SCATTER_PARAMS)


def test_param_u_param_names_propagate_properly():
    gen = zip(
        twdn.DEFAULT_DUSTPOP_SCATTER_U_PARAMS._fields,
        twdn.DEFAULT_DUSTPOP_SCATTER_PARAMS._fields,
    )
    for u_key, key in gen:
        assert u_key[:2] == "u_"
        assert u_key[2:] == key

    inferred_default_params = twdn.get_bounded_dustpop_scatter_params(
        twdn.DEFAULT_DUSTPOP_SCATTER_U_PARAMS
    )
    assert set(inferred_default_params._fields) == set(
        twdn.DEFAULT_DUSTPOP_SCATTER_PARAMS._fields
    )

    inferred_default_u_params = twdn.get_unbounded_dustpop_scatter_params(
        twdn.DEFAULT_DUSTPOP_SCATTER_PARAMS
    )
    assert set(inferred_default_u_params._fields) == set(
        twdn.DEFAULT_DUSTPOP_SCATTER_U_PARAMS._fields
    )


def test_get_bounded_params_fails_when_passing_params():
    try:
        twdn.get_bounded_dustpop_scatter_params(twdn.DEFAULT_DUSTPOP_SCATTER_PARAMS)
        raise NameError("get_bounded_dustpop_scatter_params should not accept params")
    except AttributeError:
        pass


def test_get_unbounded_params_fails_when_passing_params():
    try:
        twdn.get_unbounded_dustpop_scatter_params(twdn.DEFAULT_DUSTPOP_SCATTER_U_PARAMS)
        raise NameError(
            "get_unbounded_dustpop_scatter_params should not accept u_params"
        )
    except AttributeError:
        pass


def test_default_params_are_in_bounds():

    gen = zip(
        twdn.DEFAULT_DUSTPOP_SCATTER_PARAMS, twdn.DEFAULT_DUSTPOP_SCATTER_PARAMS._fields
    )
    for val, key in gen:
        bound = getattr(twdn.DUSTPOP_SCATTER_PBOUNDS, key)
        assert bound[0] < val < bound[1]


def test_random_params_are_always_invertible():
    ran_key = jran.key(0)
    n_params = len(twdn.DEFAULT_DUSTPOP_SCATTER_PARAMS)
    n_tests = 100
    for __ in range(n_tests):
        ran_key, test_key = jran.split(ran_key, 2)
        u_params = jran.uniform(test_key, minval=-100, maxval=100, shape=(n_params,))
        u_params = twdn.DEFAULT_DUSTPOP_SCATTER_U_PARAMS._make(u_params)
        params = twdn.get_bounded_dustpop_scatter_params(u_params)
        assert np.all(np.isfinite(params))
        u_params2 = twdn.get_unbounded_dustpop_scatter_params(params)
        assert np.all(np.isfinite(u_params2))
        assert np.allclose(u_params, u_params2, rtol=TOL)


def test_calc_ftrans_singlegal_singlewave_from_dustpop_params():
    wave_aa = 5_000.0
    logsm = 10.0
    logssfr = -10.0
    redshift = 0.5

    n_age = 37
    ssp_lg_age_gyr = np.linspace(5, 10.1, n_age) - 9.0

    random_draw_av = 0.5
    random_draw_delta = 0.5
    random_draw_funo = 0.5
    args = (
        twd.DEFAULT_DUSTPOP_PARAMS,
        wave_aa,
        logsm,
        logssfr,
        redshift,
        ssp_lg_age_gyr,
        random_draw_av,
        random_draw_delta,
        random_draw_funo,
        twdn.DEFAULT_DUSTPOP_SCATTER_PARAMS,
    )
    frac_trans = twdn.calc_ftrans_singlegal_singlewave_from_dustpop_params(*args)
    assert np.all(np.isfinite(frac_trans))
    assert frac_trans.shape == (n_age,)
    assert np.all(frac_trans >= 0)
    assert np.all(frac_trans <= 1)
