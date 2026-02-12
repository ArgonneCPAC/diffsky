from pathlib import Path

import h5py
import numpy as np
import opencosmo as oc
import pytest
from jax.scipy.stats import norm as jnorm

from diffsky.data_loaders.opencosmo_utils import (
    add_transmission_curves,
    compute_dbk_phot_from_diffsky_mock,
    compute_dbk_seds_from_diffsky_mock,
    compute_phot_from_diffsky_mock,
    compute_seds_from_diffsky_mock,
    load_diffsky_mock,
)


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


def test_compute_photometry(test_data_dir):
    with h5py.File(test_data_dir / "lc_cores-487.diffsky_gals.hdf5") as f:
        z_phots = f["header"]["catalog_info"]["z_phot_table"][:]

    bands = ("lsst_u", "lsst_g", "lsst_r", "lsst_i", "lsst_z", "lsst_y")
    catalog, aux_data = load_diffsky_mock(test_data_dir)
    results = compute_phot_from_diffsky_mock(catalog, aux_data, bands, insert=False)

    original_data = catalog.select(bands).get_data("numpy")
    for band in bands:
        assert np.all(
            np.isclose(results[f"{band}_new"], original_data[band], atol=1e-2)
        )


def test_compute_photometry_with_batch(test_data_dir):
    bands = ("lsst_u", "lsst_g", "lsst_r", "lsst_i", "lsst_z", "lsst_y")
    catalog, aux_data = load_diffsky_mock(test_data_dir)
    results = compute_phot_from_diffsky_mock(
        catalog, aux_data, bands, insert=False, batch_size=1000
    )

    original_data = catalog.select(bands).get_data("numpy")
    for band in bands:
        assert np.all(
            np.isclose(results[f"{band}_new"], original_data[band], atol=1e-2)
        )


def test_compute_photometry_custom_bands(test_data_dir):
    with h5py.File(test_data_dir / "lc_cores-487.diffsky_gals.hdf5") as f:
        z_phots = f["header"]["catalog_info"]["z_phot_table"][:]

    wave = np.linspace(200, 8_000, 500)
    fake_tcurve1 = jnorm.pdf(wave, loc=3_000.0, scale=500.0)
    fake_tcurve2 = jnorm.pdf(wave, loc=5_000.0, scale=500.0)

    catalog, aux_data = load_diffsky_mock(test_data_dir)

    aux_data = add_transmission_curves(
        aux_data, fake_tcurve_1=(wave, fake_tcurve1), fake_tcurve_2=(wave, fake_tcurve2)
    )

    results = compute_phot_from_diffsky_mock(
        catalog, aux_data, ["fake_tcurve_1", "fake_tcurve_2"], insert=False
    )
    raise NotImplementedError(
        "Need to figure out how to test that this actually worked"
    )


def test_compute_photometry_custom_bands_insert(test_data_dir):
    with h5py.File(test_data_dir / "lc_cores-487.diffsky_gals.hdf5") as f:
        z_phots = f["header"]["catalog_info"]["z_phot_table"][:]

    wave = np.linspace(200, 8_000, 500)
    fake_tcurve1 = jnorm.pdf(wave, loc=3_000.0, scale=500.0)
    fake_tcurve2 = jnorm.pdf(wave, loc=5_000.0, scale=500.0)

    catalog, aux_data = load_diffsky_mock(test_data_dir)

    aux_data = add_transmission_curves(
        aux_data, fake_tcurve_1=(wave, fake_tcurve1), fake_tcurve_2=(wave, fake_tcurve2)
    )

    results = compute_phot_from_diffsky_mock(
        catalog, aux_data, ["fake_tcurve_1", "fake_tcurve_2"], insert=True
    )
    raise NotImplementedError(
        "Need to figure out how to test that this actually worked"
    )


def test_compute_dbk_photometry(test_data_dir):
    with h5py.File(test_data_dir / "lc_cores-487.diffsky_gals.hdf5") as f:
        z_phots = f["header"]["catalog_info"]["z_phot_table"][:]

    catalog, aux_data = load_diffsky_mock(test_data_dir)
    results = compute_dbk_phot_from_diffsky_mock(catalog, aux_data, insert=False)
    original_data = catalog.select(results.keys()).get_data("numpy")
    for name, computed_values in results.items():
        assert np.allclose(computed_values, original_data[name], atol=1e-2)


def test_compute_seds(test_data_dir):
    with h5py.File(test_data_dir / "lc_cores-487.diffsky_gals.hdf5") as f:
        z_phots = f["header"]["catalog_info"]["z_phot_table"][:]

    bands = ("lsst_u", "lsst_g", "lsst_r", "lsst_i", "lsst_z", "lsst_y")
    catalog, aux_data = load_diffsky_mock(test_data_dir)
    results = compute_seds_from_diffsky_mock(catalog, aux_data, bands, insert=False)
    raise NotImplementedError


def test_compute_seds_with_batching(test_data_dir):
    with h5py.File(test_data_dir / "lc_cores-487.diffsky_gals.hdf5") as f:
        z_phots = f["header"]["catalog_info"]["z_phot_table"][:]

    bands = ("lsst_u", "lsst_g", "lsst_r", "lsst_i", "lsst_z", "lsst_y")
    catalog, aux_data = load_diffsky_mock(test_data_dir)
    results = compute_seds_from_diffsky_mock(
        catalog, aux_data, bands, insert=False, batch_size=100
    )
    raise NotImplementedError


def test_compute_seds_insert(test_data_dir):
    with h5py.File(test_data_dir / "lc_cores-487.diffsky_gals.hdf5") as f:
        z_phots = f["header"]["catalog_info"]["z_phot_table"][:]

    bands = ("lsst_u", "lsst_g", "lsst_r", "lsst_i", "lsst_z", "lsst_y")
    catalog, aux_data = load_diffsky_mock(test_data_dir)
    results = compute_seds_from_diffsky_mock(catalog, aux_data, bands, insert=True)
    assert "rest_sed" in results.columns
    raise NotImplementedError


def test_compute_dbk_seds(test_data_dir):
    with h5py.File(test_data_dir / "lc_cores-487.diffsky_gals.hdf5") as f:
        z_phots = f["header"]["catalog_info"]["z_phot_table"][:]

    bands = ("lsst_u", "lsst_g", "lsst_r", "lsst_i", "lsst_z", "lsst_y")
    catalog, aux_data = load_diffsky_mock(test_data_dir)
    results = compute_dbk_seds_from_diffsky_mock(catalog, aux_data, bands, insert=False)
    raise NotImplementedError
