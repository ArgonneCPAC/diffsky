""""""

import pytest

from .. import metadata_mock

try:
    from astropy import units as u
    from astropy.cosmology import units as cu

    u.add_enabled_units(cu)

    HAS_ASTROPY = True
except ImportError:
    HAS_ASTROPY = False
ASTROPY_MSG = "Must have astropy installed to run tests of metadata_mock"


@pytest.mark.skipif(not HAS_ASTROPY, reason=ASTROPY_MSG)
def test_get_column_metadata():

    column_metadata = metadata_mock.get_column_metadata()
    assert len(column_metadata) > 0
    for colname, metadata in column_metadata.items():
        unit_string, description = metadata
        assert len(description) > 0
        try:
            u.Unit(unit_string)
        except ValueError:
            raise ValueError(
                f"`{colname}` has unrecognized unit string = `{unit_string}`"
            )
