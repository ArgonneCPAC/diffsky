import os
from pathlib import Path

import pytest


@pytest.fixture
def opencosmo_data_path():
    path = os.environ.get("OPENCOSMO_DIFFSKY_DATA_PATH")
    if path is None:
        raise ValueError(
            "To run opencosmo tests, set the OPENCOSMO_DIFFSKY_DATA_PATH environment variable"
        )
    return Path(path)


def pytest_collection_modifyitems(session, config, items):
    # Define your condition here. For example, skip if an environment variable is not set.
    # The condition can be anything, including checking file paths or system platform.
    skip_condition = not os.environ.get("RUN_OPENCOSMO_TESTS") == "True"

    if skip_condition:
        skip_marker = pytest.mark.skip(
            reason="Tests in this directory are skipped due to a specific condition"
        )
        for item in items:
            # Check if the test item belongs to the current conftest.py's directory or a subdirectory
            # This is a general approach, you might need to adjust path logic based on your structure.
            item.add_marker(skip_marker)
