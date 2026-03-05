import opencosmo as oc

from diffsky.data_loaders.opencosmo_utils import (
    load_diffsky_mock,
)


def test_load(opencosmo_data_path):
    catalog, aux_data = load_diffsky_mock(opencosmo_data_path)
    catalog_synths, aux_data_synths = load_diffsky_mock(
        opencosmo_data_path, synth_cores=True
    )

    assert isinstance(catalog, oc.Lightcone)
    assert isinstance(catalog_synths, oc.Lightcone)
    assert len(catalog_synths) > len(catalog)

    assert isinstance(aux_data, dict)
    assert isinstance(aux_data_synths, dict)
