""""""

import numpy as np
import jax
from jax import random as jran

from .. import lc_sfh_centrals
from . import test_lc_data_sfh


def test_mc_lc_sfh():
    ran_key = jran.key(0)
    lc_data = test_lc_data_sfh._get_mc_lc_data_sfh_centrals_for_unit_testing()

    sfh_lightcone = lc_sfh_centrals.mc_lc_sfh(ran_key, lc_data)

    for x in sfh_lightcone:
        assert np.all(np.isfinite(x))
