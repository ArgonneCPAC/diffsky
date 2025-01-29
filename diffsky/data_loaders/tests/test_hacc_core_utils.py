"""
"""

import pytest

from .. import hacc_core_utils as hcu

NO_HACC_MSG = "Must have haccytrees installed to run this test"


@pytest.mark.skipif(not hcu.HAS_HACCYTREES, reason=NO_HACC_MSG)
def test_get_diffstar_cosmo_quantities():
    sim_name = "LastJourney"
    fb, lgt0 = hcu.get_diffstar_cosmo_quantities(sim_name)
    assert 0.1 < fb < 0.2
    assert 1.1 < lgt0 < 1.2
