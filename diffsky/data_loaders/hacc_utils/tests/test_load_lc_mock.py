""""""

import os

import pytest

from .. import load_lc_mock as llcm

_THIS_DRNAME = os.path.dirname(os.path.abspath(__file__))
TESTING_DATA = os.path.join(_THIS_DRNAME, "testing_data")

BN_MOCK = "lc_cores-453.0.diffsky_gals.hdf5"

URL_ROOT = "https://portal.nersc.gov/project/hacc/aphearin/diffsky_data"
URL_SUBDRN = "ci_test_output/synthetic_cores/smdpl_dr1"
MOCK_URL_DRN = os.path.join(URL_ROOT, URL_SUBDRN)


@pytest.mark.skip
def test_load_diffsky_lc_patch():
    diffsky_lc_patch = llcm.load_diffsky_lc_patch(TESTING_DATA, BN_MOCK)  # noqa


@pytest.mark.skip
def test_compute_dbk_phot_from_diffsky_mock():
    diffsky_lc_patch = llcm.load_diffsky_lc_patch(TESTING_DATA, BN_MOCK)
    phot_info = llcm.compute_dbk_phot_from_diffsky_mock(**diffsky_lc_patch)  # noqa
