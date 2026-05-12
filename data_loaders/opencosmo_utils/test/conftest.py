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


@pytest.fixture
def version_checking():
    if os.getenv("GITHUB_ACTIONS") == "true":
        return "strict"
    return None
