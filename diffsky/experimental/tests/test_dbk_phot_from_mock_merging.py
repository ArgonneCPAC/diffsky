""""""

import pytest

from .. import dbk_phot_from_mock_merging as dbkpmm


@pytest.mark.xfail
def test_reproduce_mock_dbk_merging_kern():
    assert hasattr(dbkpmm, "_reproduce_mock_phot_kern")
    raise NotImplementedError()
