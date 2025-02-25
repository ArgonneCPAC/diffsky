"""
"""

import numpy as np
from jax import random as jran

from .. import cosmos_mstar_model as cmm


def get_fake_cosmos_data(ran_key, n=2_000):
    collector = []
    for field in cmm.PhotData._fields:
        ran_key, field_key = jran.split(ran_key, 2)
        arr = jran.uniform(field_key, minval=0, maxval=2, shape=(n,))
        collector.append(arr)
    photdata = cmm.PhotData(*collector)
    return photdata


def test_fit_model():
    ran_key = jran.key(0)

    try:
        photdata = cmm.load_cosmos20_tdata()
    except (ImportError, KeyError):
        photdata = get_fake_cosmos_data(ran_key)

    logsm = cmm.predict_logsm(cmm.DEFAULT_PARAMS, photdata)
    loss_init = cmm._loss_kern(cmm.DEFAULT_PARAMS, photdata)
    assert np.all(np.isfinite(logsm))
    assert np.all(np.isfinite(loss_init))

    p_best, loss_arr = cmm.fit_model(100, photdata, step_size=0.01)

    loss_best = cmm._loss_kern(p_best, photdata)
    assert loss_best < loss_init
