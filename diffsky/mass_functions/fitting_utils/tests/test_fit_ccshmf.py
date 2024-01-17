"""
"""
import numpy as np
from jax import random as jran

from ...ccshmf_model import (
    DEFAULT_CCSHMF_PARAMS,
    CCSHMF_Params,
    YLO_Params,
    YTP_Params,
    predict_ccshmf,
)
from ..fit_ccshmf import _loss_func_multi_mhost, ccshmf_fitter


def get_random_params(ran_key, p0=DEFAULT_CCSHMF_PARAMS):
    ytp_params, ylo_params = DEFAULT_CCSHMF_PARAMS
    ytp_key, ylo_key = jran.split(ran_key, 2)
    ytp_noise = jran.uniform(ytp_key, minval=0.1, maxval=0.1, shape=(len(ytp_params),))
    ylo_noise = jran.uniform(ylo_key, minval=0.1, maxval=0.1, shape=(len(ylo_params),))

    ran_ytp_params = YLO_Params(*[x + y for x, y in zip(ytp_params, ytp_noise)])
    ran_ylo_params = YTP_Params(*[x + y for x, y in zip(ytp_params, ylo_noise)])
    ran_params = CCSHMF_Params(ran_ytp_params, ran_ylo_params)
    return ran_params


def test_ccshmf_fitter_correctly_minimizes_loss():
    ran_key = jran.PRNGKey(0)
    ran_params = get_random_params(ran_key)

    target_lgmu_bins = np.linspace(-3, 0.0, 20)

    target_lgmhost = 12.5
    target_lg_ccshmf = predict_ccshmf(ran_params, target_lgmhost, target_lgmu_bins)
    loss_data_0 = target_lgmhost, target_lgmu_bins, target_lg_ccshmf

    target_lgmhost = 13.5
    target_lg_ccshmf = predict_ccshmf(ran_params, target_lgmhost, target_lgmu_bins)
    loss_data_1 = target_lgmhost, target_lgmu_bins, target_lg_ccshmf

    loss_data = [loss_data_0, loss_data_1]

    loss_init = _loss_func_multi_mhost(DEFAULT_CCSHMF_PARAMS, loss_data)

    res = ccshmf_fitter(loss_data)
    p_best, loss_best, loss_hist, params_hist, fit_terminates = res
    assert loss_best < loss_init
    assert np.all(np.isfinite(loss_hist))
    assert loss_best == np.min(loss_hist)
    indx = np.argmin(loss_hist)
    p_best_from_hist = params_hist[indx]
    assert np.allclose(p_best, p_best_from_hist)
    assert fit_terminates == 1

    loss_best_inferred = _loss_func_multi_mhost(p_best, loss_data)
    assert np.allclose(loss_best_inferred, loss_best, rtol=1e-3)
