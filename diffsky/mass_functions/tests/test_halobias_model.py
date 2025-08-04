""""""

import numpy as np

from .. import halobias_model as hbm


def test_tw_quintuple_sigmoid():
    lgm = np.linspace(11, 15, 100)
    lgb = hbm.predict_lgbias_kern(hbm.HALOBIAS_PARAMS, lgm)
    assert np.all(np.isfinite(lgb))
    assert np.all(lgb > -2)
    assert np.all(lgb < 3)
    assert np.all(np.diff(lgb) > 0)
