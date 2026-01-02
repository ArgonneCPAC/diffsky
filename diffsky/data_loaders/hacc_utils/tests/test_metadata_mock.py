""""""

from .. import metadata_mock


def test_get_column_metadata():

    column_metadata = metadata_mock.get_column_metadata()
    assert len(column_metadata) > 0
