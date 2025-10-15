""""""

import numpy as np

from .. import load_diffsky_sfh_model_calibrations as ldsfh


def test_load_diffsky_u_params_for_sfh_model():

    sfh_model_nicknames = ["tng"]
    for sfh_model_nickname in sfh_model_nicknames:
        u_params = ldsfh.load_diffsky_u_params_for_sfh_model(sfh_model_nickname)
        assert np.all(np.isfinite(u_params))
