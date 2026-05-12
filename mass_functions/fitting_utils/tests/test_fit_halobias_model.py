""" """

import numpy as np
from jax import random as jran

from ... import halobias_model as hbm
from ...halobias_model import HALOBIAS_U_PARAMS, HaloBiasParams
from ..fit_halobias_model import _loss_func_multi_z, halobias_model_fitter


def get_random_params(key):
    noise = jran.uniform(key, minval=0.1, maxval=0.1, shape=(len(HALOBIAS_U_PARAMS),))
    ran_params = HaloBiasParams(*noise)
    return ran_params


def test_get_random_params():
    ran_key = jran.PRNGKey(0)
    ran_hmf_params = get_random_params(ran_key)

    nhalos = 100
    logmparr = np.linspace(10, 15, nhalos)
    redshift = 1.0
    lgbias = hbm.predict_lgbias_kern(ran_hmf_params, logmparr, redshift)
    assert lgbias.shape == (nhalos,)
    assert np.all(np.isfinite(lgbias))


def test_hmf_fitter_correctly_minimizes_loss():
    ran_key = jran.PRNGKey(0)
    ran_params = get_random_params(ran_key)

    target_lgmp = np.linspace(11, 15.0, 20)

    target_redshifts = np.array((0.0, 2.0, 5.0))
    loss_data_collector = []
    for z in target_redshifts:
        lgbias_target = hbm.predict_lgbias_kern(ran_params, target_lgmp, z)
        loss_data = target_lgmp, lgbias_target, z
        loss_data_collector.append(loss_data)

    loss_init = _loss_func_multi_z(HALOBIAS_U_PARAMS, loss_data_collector)
    assert loss_init > 0

    res = halobias_model_fitter(loss_data_collector)
    p_best, loss_best, loss_hist, params_hist, fit_terminates = res
    assert loss_best < loss_init
    assert np.all(np.isfinite(loss_hist))
    for p in p_best:
        assert np.all(np.isfinite(p))
    assert loss_best == np.min(loss_hist)
    assert fit_terminates == 1

    u_p_best = hbm.get_unbounded_halobias_params(p_best)
    loss_best_inferred = _loss_func_multi_z(u_p_best, loss_data_collector)
    assert np.allclose(loss_best_inferred, loss_best, rtol=1e-3)

    assert np.log10(loss_best) < -1.0
