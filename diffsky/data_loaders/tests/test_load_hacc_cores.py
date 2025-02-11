"""
"""

import os

import numpy as np
import pytest
from jax import random as jran

from .. import load_hacc_cores as lhc

NO_HACC_MSG = "Must have haccytrees installed to run this test"
POBOY_MSG = "This test only runs on poboy machine"

DRN_LJ_POBOY = "/Users/aphearin/work/DATA/LastJourney/coretrees"
DRN_LJ_DMAH_POBOY = "/Users/aphearin/work/DATA/LastJourney/diffmah_fits"

BNPAT_CORE_DATA = "m000p.coreforest.{}.hdf5"

try:
    assert os.path.isdir(DRN_LJ_POBOY)
    assert lhc.HAS_HACCYTREES
    CAN_RUN_HACC_DATA_TESTS = True
except AssertionError:
    CAN_RUN_HACC_DATA_TESTS = False


@pytest.mark.skipif(not CAN_RUN_HACC_DATA_TESTS, reason=POBOY_MSG)
def test_load_last_journey_data():
    sim_name = "LastJourney"
    subvol = 0
    chunknum = 49
    nchunks = 50
    iz_obs = 100
    ran_key = jran.key(0)
    drn_cores = DRN_LJ_POBOY
    drn_diffmah = DRN_LJ_DMAH_POBOY
    diffsky_data = lhc.load_diffsky_data(
        sim_name, subvol, chunknum, nchunks, iz_obs, ran_key, drn_cores, drn_diffmah
    )
    for x in diffsky_data["subcat"].mah_params:
        assert np.all(np.isfinite(x))

    for x in diffsky_data["subcat"][1:]:
        assert np.all(np.isfinite(x))

    n_diffmah_fits = diffsky_data["subcat"].mah_params.logm0.size
    n_forest = diffsky_data["subcat"].logmp0.size
    assert n_forest == n_diffmah_fits, "mismatch between forest and diffmah fits"
