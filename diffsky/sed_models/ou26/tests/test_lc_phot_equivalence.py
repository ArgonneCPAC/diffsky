""""""

import numpy as np
import pytest
from diffstar.diffstarpop.kernels.params import (
    DiffstarPop_Params_Diffstarpopfits_mgash as sfh_models,
)
from jax import random as jran

from ....experimental import mc_phot
from .. import diffsky_param_wrapper_merging as dpwm
from .. import lc_phot
from . import test_lc_data_phot

NUM_HALOS = 20
PC1 = dpwm.DEFAULT_PARAM_COLLECTION._replace(diffstarpop_params=sfh_models["tng"])
PC2 = dpwm.DEFAULT_PARAM_COLLECTION._replace(diffstarpop_params=sfh_models["smdpl_dr1"])


@pytest.mark.parametrize("param_collection", [PC1, PC2])
def test_lc_phot_is_equivalent_to_mc_phot(param_collection):
    """Test that the refactored kernels in lc_phot exactly agree with the old code"""
    ran_key = jran.key(0)
    lc_data, tcurves = test_lc_data_phot._get_weighted_lc_data_phot_for_unit_testing(
        num_halos=NUM_HALOS, ran_key=ran_key
    )

    MC_MERGE = 0
    pknew = lc_phot.mc_lc_phot(
        ran_key, lc_data, MC_MERGE, param_collection=param_collection
    )[0]
    pkold = mc_phot.mc_lc_phot(
        ran_key, lc_data, MC_MERGE, param_collection=param_collection
    )[0]

    cols_to_check = ("obs_mags", "obs_mags_q", "obs_mags_ms", "obs_mags_bursty")
    for col in cols_to_check:
        x_new = getattr(pknew, col)
        x_old = getattr(pkold, col)
        assert np.allclose(x_new, x_old, rtol=1e-3)
