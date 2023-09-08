"""
"""
import numpy as np
from diffstar.defaults import SFR_MIN as DIFFSTAR_SFR_MIN
from dsps.constants import SFR_MIN as DSPS_SFR_MIN


def test_diffstar_dsps_sfr_min_consistency():
    assert np.allclose(DIFFSTAR_SFR_MIN, DSPS_SFR_MIN)
