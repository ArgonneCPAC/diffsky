from pathlib import Path

import opencosmo as oc
import pytest

from diffsky.data_loaders.opencosmo_utils.compute import compute_phot_from_diffsky_mocks
from diffsky.data_loaders.opencosmo_utils.load import load_diffsky_mock


@pytest.fixture
def test_data_dir():
    return Path(__file__).parent / "test_data"


def test_load(test_data_dir):
    catalog, aux_data = load_diffsky_mock(test_data_dir)
    catalog_synths, aux_data_synths = load_diffsky_mock(test_data_dir, synth_cores=True)

    assert isinstance(catalog, oc.Lightcone)
    assert isinstance(catalog_synths, oc.Lightcone)
    assert len(catalog_synths) > len(catalog)

    assert isinstance(aux_data, dict)
    assert isinstance(aux_data_synths, dict)


def test_compute(test_data_dir):
    catalog, aux_data = load_diffsky_mock(test_data_dir)
    results = compute_phot_from_diffsky_mocks(catalog, aux_data, insert=False)
