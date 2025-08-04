""" """

import numpy as np
from jax import random as jran

from ... import halobias_singlez_model as hbm
from ...halobias_singlez_model import HALOBIAS_U_PARAMS, HaloBiasParams
from ..fit_halobias_singlez_model import (
    _loss_func_single_redshift,
    halobias_singlez_fitter,
)


def get_random_params(key):
    noise = jran.uniform(key, minval=0.1, maxval=0.1, shape=(len(HALOBIAS_U_PARAMS),))
    ran_params = HaloBiasParams(*noise)
    return ran_params


def test_get_random_params():
    ran_key = jran.PRNGKey(0)
    ran_hmf_params = get_random_params(ran_key)

    nhalos = 100
    logmparr = np.linspace(10, 15, nhalos)
    lgbias = hbm.predict_lgbias_kern(ran_hmf_params, logmparr)
    assert lgbias.shape == (nhalos,)
    assert np.all(np.isfinite(lgbias))


def test_hmf_fitter_correctly_minimizes_loss():
    ran_key = jran.PRNGKey(0)
    ran_params = get_random_params(ran_key)

    target_lgmp = np.linspace(11, 15.0, 20)
    lgbias_target = hbm.predict_lgbias_kern(ran_params, target_lgmp)
    loss_data = target_lgmp, lgbias_target

    loss_init = _loss_func_single_redshift(HALOBIAS_U_PARAMS, loss_data)
    assert loss_init > 0

    res = halobias_singlez_fitter(loss_data)
    p_best, loss_best, loss_hist, params_hist, fit_terminates = res
    assert loss_best < loss_init
    assert np.all(np.isfinite(loss_hist))
    for p in p_best:
        assert np.all(np.isfinite(p))
    assert loss_best == np.min(loss_hist)
    assert fit_terminates == 1

    u_p_best = hbm.get_unbounded_halobias_params(p_best)
    loss_best_inferred = _loss_func_single_redshift(u_p_best, loss_data)
    assert np.allclose(loss_best_inferred, loss_best, rtol=1e-3)

    assert np.log10(loss_best) < -2.0
