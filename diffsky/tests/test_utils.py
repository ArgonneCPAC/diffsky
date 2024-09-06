"""
"""

import numpy as np

from .. import utils as ut


def test_sigmoid_inversion():
    xarr = np.linspace(-10, 10, 500)

    x0, k, ylo, yhi = 0, 0.1, -5, 5
    y = ut._sigmoid(xarr, x0, k, ylo, yhi)
    x2 = ut._inverse_sigmoid(y, x0, k, ylo, yhi)
    assert np.allclose(xarr, x2, rtol=1e-4)


def test_smhm_loss_penalty():
    logsm_target = np.linspace(8, 12, 100)
    logsm_pred = np.random.uniform(-0.5, 0.5, 100) + logsm_target
    penalty = 10.0
    loss = ut.smhm_loss_penalty(logsm_pred, logsm_target, penalty)
    assert np.all(loss >= 0)
    assert np.all(loss <= penalty)
    assert np.any(loss > 0)
    assert np.any(loss < penalty)
