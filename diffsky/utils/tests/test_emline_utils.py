""""""

import numpy as np
import pytest
from dsps.data_loaders.retrieve_fake_fsps_data import load_fake_ssp_data

from .. import emline_utils

try:
    from astropy import units as u

    HAS_ASTROPY = True
except ImportError:
    HAS_ASTROPY = False
ASTROPY_MSG = "Must have astropy installed to run this unit test"


@pytest.mark.skipif(not HAS_ASTROPY, reason=ASTROPY_MSG)
def test_lsun_cgs_units():
    assert np.allclose(emline_utils.L_SUN_CGS, u.Lsun.to(u.erg / u.s), rtol=1e-4)


def test_fake_lineflux_table():
    ssp_data = load_fake_ssp_data()
    lineflux_table = emline_utils.fake_lineflux_table(
        ssp_data.ssp_lgmet, ssp_data.ssp_lg_age_gyr
    )
    assert np.all(np.isfinite(lineflux_table))

    # The youngest stars should have non-zero line flux
    assert np.all(lineflux_table[:, 0] > 0)

    # The oldest stars should have zero line flux
    assert np.allclose(lineflux_table[:, -1], 0.0)
