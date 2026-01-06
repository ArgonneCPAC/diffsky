""""""

import numpy as np

from .. import load_diffsky_sfh_model_calibrations as ldsfh


def test_load_diffsky_u_params_for_sfh_model_in_spsop_calibrations():

    sfh_model_nicknames = ldsfh.SPSOP_CALIBRATIONS
    for sfh_model_nickname in sfh_model_nicknames:
        u_params = ldsfh.load_diffsky_u_params_for_sfh_model(sfh_model_nickname)
        assert np.all(np.isfinite(u_params))


def test_load_diffsky_u_params_for_sfh_model_in_diffstarpop_calibrations():

    sfh_model_nicknames = ldsfh.DIFFSTARPOP_CALIBRATIONS
    sizes = []
    for sfh_model_nickname in sfh_model_nicknames:
        u_params = ldsfh.load_diffsky_u_params_for_sfh_model(sfh_model_nickname)
        assert np.all(np.isfinite(u_params))
        sizes.append(len(u_params))
    assert len(np.unique(sizes)) == 1
    assert sizes[0] > 50
