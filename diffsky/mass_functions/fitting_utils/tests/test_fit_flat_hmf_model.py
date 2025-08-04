""" """

import numpy as np
from jax import random as jran

from ...flat_hmf_model import FLAT_HMF_PARAMS, FlatHMFParams, predict_cuml_hmf
from ..fit_flat_hmf_model import _loss_func_multi_z, hmf_fitter


def get_random_params(key):
    noise = jran.uniform(key, minval=0.1, maxval=0.1, shape=(len(FLAT_HMF_PARAMS),))
    ran_params = FlatHMFParams(*noise)
    return ran_params


def test_get_random_params():
    ran_key = jran.PRNGKey(0)
    ran_hmf_params = get_random_params(ran_key)

    nhalos = 100
    logmparr = np.linspace(10, 15, nhalos)
    redshift = 1.0
    hmf_pred = predict_cuml_hmf(ran_hmf_params, logmparr, redshift)
    assert hmf_pred.shape == (nhalos,)
    assert np.all(np.isfinite(hmf_pred))


def test_hmf_fitter_correctly_minimizes_loss():
    ran_key = jran.PRNGKey(0)
    ran_params = get_random_params(ran_key)

    target_lgmp = np.linspace(11, 15.0, 20)
    target_redshift = 0.1
    target_hmf = predict_cuml_hmf(ran_params, target_lgmp, target_redshift)
    loss_data_0 = target_redshift, target_lgmp, target_hmf

    target_lgmp = np.linspace(11, 14.0, 10)
    target_redshift = 1.1
    target_hmf = predict_cuml_hmf(ran_params, target_lgmp, target_redshift)
    loss_data_1 = target_redshift, target_lgmp, target_hmf

    loss_data = [loss_data_0, loss_data_1]

    loss_init = _loss_func_multi_z(FLAT_HMF_PARAMS, loss_data)
    assert loss_init > 0

    res = hmf_fitter(loss_data)
    p_best, loss_best, loss_hist, params_hist, fit_terminates = res
    assert loss_best < loss_init
    assert np.all(np.isfinite(loss_hist))
    for p in p_best:
        assert np.all(np.isfinite(p))
    assert loss_best == np.min(loss_hist)
    assert fit_terminates == 1

    loss_best_inferred = _loss_func_multi_z(p_best, loss_data)
    assert np.allclose(loss_best_inferred, loss_best, rtol=1e-3)

    indx = np.argmin(loss_hist)
    p_best_from_hist = params_hist[indx]
    for p in p_best_from_hist:
        assert np.all(np.isfinite(p))

    for a, b in zip(p_best, p_best_from_hist):
        assert np.allclose(a, b)
