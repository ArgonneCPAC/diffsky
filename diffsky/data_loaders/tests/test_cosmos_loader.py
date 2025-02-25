"""
"""

import os
import subprocess

import pytest

from .. import cosmos20_loader as c20

_THIS_DRNAME = os.path.dirname(os.path.abspath(__file__))
TESTING_DRNAME = os.path.join(_THIS_DRNAME, "testing_data")
SHASUM_FN = os.path.join(TESTING_DRNAME, "cosmos20_shasum.dat")
UPDATE_MSG = "File stored in {0} not consistent with cosmos20_colors package version"
ENV_VAR_MSG = "load_cosmos20 can only be tested if COSMOS20_DRN is in the env"
FILE_DNE_MSG = "Environment variable COSMOS20_DRN={0} does not contain {1}"
ASTROPY_MSG = "Must have astropy installed to run tests of cosmos20_loader"


def check_if_cosmos20_file_exists():
    drn = os.environ.get("COSMOS20_DRN", None)
    if drn is None:
        has_file = False
        fn = ""
        msg = "Environment variable COSMOS20_DRN is not set"
    else:
        fn = os.path.join(drn, c20.COSMOS20_BASENAME)
        has_file = os.path.isfile(fn)
        msg = FILE_DNE_MSG.format(drn, c20.COSMOS20_BASENAME)
    return has_file, fn, msg


@pytest.mark.skipif(not c20.HAS_ASTROPY, reason=ASTROPY_MSG)
@pytest.mark.skipif(os.environ.get("COSMOS20_DRN", None) is None, reason=ENV_VAR_MSG)
def test_cosmos20_dataset_exists_when_env_var_is_set():
    """If the COSMOS20_DRN env variable is set, enforce that data exists on disk"""
    data_exists, fn, msg = check_if_cosmos20_file_exists()
    assert data_exists, msg


HAS_COSMOS20, COSMOS20_FN, HAS_COSMOS20_MSG = check_if_cosmos20_file_exists()


@pytest.mark.skipif(not c20.HAS_ASTROPY, reason=ASTROPY_MSG)
@pytest.mark.skipif(not HAS_COSMOS20, reason=HAS_COSMOS20_MSG)
def test_cosmos20_dataset_is_up_to_date():
    """Enforce that dataset has the correct shasum"""
    command = "shasum {0}".format(COSMOS20_FN)
    raw_result = subprocess.check_output(command, shell=True)
    actual_shasum = raw_result.strip().split()[0].decode()
    with open(SHASUM_FN, "r") as f:
        correct_shasum = f.readline().strip()
    assert actual_shasum == correct_shasum, UPDATE_MSG.format(COSMOS20_FN)


@pytest.mark.skipif(not c20.HAS_ASTROPY, reason=ASTROPY_MSG)
@pytest.mark.skipif(not HAS_COSMOS20, reason=HAS_COSMOS20_MSG)
def test_cosmos20_dataset_loads():
    """Enforce that dataset loads when it exists"""
    cat = c20.load_cosmos20()
    assert cat is not None


@pytest.mark.skipif(not c20.HAS_ASTROPY, reason=ASTROPY_MSG)
@pytest.mark.skipif(not HAS_COSMOS20, reason=HAS_COSMOS20_MSG)
def test_cosmos20_quality_cuts_do_not_change():
    """Enforce that the data-loader quality cuts are frozen.
    The purpose of this unit test is to standardize the quality cuts made on the
    catalog that define the summary statistics. If the quality cuts change,
    this unit test will need to be updated.
    """
    cat = c20.load_cosmos20(apply_cuts=True)
    assert len(cat) == 706601


@pytest.mark.skipif(not c20.HAS_ASTROPY, reason=ASTROPY_MSG)
@pytest.mark.skipif(not HAS_COSMOS20, reason=HAS_COSMOS20_MSG)
def test_cosmos20_sky_area_is_frozen():
    """Enforce that the sky area of COSMOS-20 is frozen."""
    assert c20.SKY_AREA == 1.21
