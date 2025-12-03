""""""


def test_metadata_sfh_mock_imports():
    from .. import metadata_sfh_mock

    assert hasattr(metadata_sfh_mock, "HEADER_COMMENT")
