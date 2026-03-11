import os
from importlib.util import find_spec

import numpy as np
import pytest
from jax.scipy.stats import norm as jnorm

oc_ = find_spec("opencosmo")

if oc_ is None and os.getenv("RUN_OPENCOSMO_TESTS") == "true":
    raise ImportError(
        "You asked to run opencosmo tests, but opencosmo is not installed"
    )


elif oc_ is not None:
    from diffsky.data_loaders.opencosmo_utils import (
        add_transmission_curves,
        compute_dbk_phot_from_diffsky_mock,
        compute_dbk_seds_from_diffsky_mock,
        compute_phot_from_diffsky_mock,
        compute_seds_from_diffsky_mock,
        load_diffsky_mock,
    )


@pytest.fixture(params=[True, False], ids=["with_synth_cores", "without_synth_cores"])
def synth_cores(request):
    """A fixture that yields True and False."""
    param_value = request.param
    # Perform setup with the parameter value if needed
    yield param_value


def test_get_z_phot_tables(opencosmo_data_path, synth_cores):
    catalog, aux_data = load_diffsky_mock(opencosmo_data_path, synth_cores=synth_cores)
    for slice_catalog in catalog.values():
        __verify_z_phot_table(slice_catalog)


def __verify_z_phot_table(ds):
    z_phot_table = ds.header.catalog_info["zphot_table"]
    redshifts = ds.select("redshift_true").get_data("numpy")
    z_min, z_max = np.min(redshifts), np.max(redshifts)
    assert np.all(np.sort(z_phot_table) == z_phot_table)
    assert z_phot_table[0] <= z_min and z_phot_table[-1] >= z_max


def test_compute_photometry(opencosmo_data_path, version_checking, synth_cores):
    bands = ("lsst_u", "lsst_g", "lsst_r", "lsst_i", "lsst_z", "lsst_y")
    catalog, aux_data = load_diffsky_mock(
        opencosmo_data_path, version_check=version_checking, synth_cores=synth_cores
    )
    results = compute_phot_from_diffsky_mock(catalog, aux_data, bands, insert=False)

    original_data = catalog.select(bands).get_data("numpy")
    for band in bands:
        assert np.all(
            np.isclose(results[f"{band}_new"], original_data[band], atol=1e-2)
        )


def test_compute_photometry_with_batch(
    opencosmo_data_path, version_checking, synth_cores
):
    bands = ("lsst_u", "lsst_g", "lsst_r", "lsst_i", "lsst_z", "lsst_y")
    catalog, aux_data = load_diffsky_mock(
        opencosmo_data_path, version_check=version_checking, synth_cores=synth_cores
    )
    results = compute_phot_from_diffsky_mock(
        catalog, aux_data, bands, insert=False, batch_size=1000
    )

    original_data = catalog.select(bands).get_data("numpy")
    for band in bands:
        assert np.all(
            np.isclose(results[f"{band}_new"], original_data[band], atol=1e-2)
        )
    results = compute_phot_from_diffsky_mock(catalog, aux_data, bands, insert=False)

    original_data = catalog.select(bands).get_data("numpy")
    for band in bands:
        assert np.all(
            np.isclose(results[f"{band}_new"], original_data[band], atol=1e-2)
        )


def test_compute_photometry_custom_bands(
    opencosmo_data_path, version_checking, synth_cores
):
    wave = np.linspace(200, 8_000, 500)
    fake_tcurve1 = jnorm.pdf(wave, loc=3_000.0, scale=500.0)
    fake_tcurve2 = jnorm.pdf(wave, loc=5_000.0, scale=500.0)

    catalog, aux_data = load_diffsky_mock(
        opencosmo_data_path, version_check=version_checking, synth_cores=synth_cores
    )

    aux_data = add_transmission_curves(
        aux_data, fake_tcurve_1=(wave, fake_tcurve1), fake_tcurve_2=(wave, fake_tcurve2)
    )

    results = compute_phot_from_diffsky_mock(
        catalog, aux_data, ["fake_tcurve_1", "fake_tcurve_2"], insert=False
    )
    for magnitudes in results.values():
        assert np.all((magnitudes > 10) & (magnitudes < 30))


def test_compute_photometry_custom_bands_insert(
    opencosmo_data_path, version_checking, synth_cores
):
    wave = np.linspace(200, 8_000, 500)
    fake_tcurve1 = jnorm.pdf(wave, loc=3_000.0, scale=500.0)
    fake_tcurve2 = jnorm.pdf(wave, loc=5_000.0, scale=500.0)

    catalog, aux_data = load_diffsky_mock(
        opencosmo_data_path, version_check=version_checking, synth_cores=synth_cores
    )

    aux_data = add_transmission_curves(
        aux_data, fake_tcurve_1=(wave, fake_tcurve1), fake_tcurve_2=(wave, fake_tcurve2)
    )

    catalog = compute_phot_from_diffsky_mock(
        catalog, aux_data, ["fake_tcurve_1", "fake_tcurve_2"], insert=True
    )

    results = catalog.select(("fake_tcurve_1", "fake_tcurve_2")).get_data("numpy")
    for magnitudes in results.values():
        assert np.all((magnitudes > 10) & (magnitudes < 30))


def test_compute_dbk_photometry(opencosmo_data_path, version_checking, synth_cores):
    bands = ("lsst_u", "lsst_g", "lsst_r", "lsst_i", "lsst_z", "lsst_y")
    catalog, aux_data = load_diffsky_mock(
        opencosmo_data_path, version_check=version_checking, synth_cores=synth_cores
    )
    results = compute_dbk_phot_from_diffsky_mock(catalog, aux_data, bands, insert=False)
    columns = [col.removesuffix("_new") for col in results.keys()]

    original_data = catalog.select(columns).get_data("numpy")
    for name in columns:
        assert np.allclose(results[f"{name}_new"], original_data[name], atol=1e-2)


def test_compute_seds(opencosmo_data_path, version_checking, synth_cores):
    catalog, aux_data = load_diffsky_mock(
        opencosmo_data_path, version_check=version_checking, synth_cores=synth_cores
    )
    results = compute_seds_from_diffsky_mock(catalog, aux_data, insert=False)
    seds = results["rest_sed"]
    expected_shape = (len(catalog), len(aux_data["ssp_data"].ssp_wave))
    assert seds.shape == expected_shape
    assert np.all(~np.isnan(seds))


def test_compute_seds_with_batching(opencosmo_data_path, version_checking, synth_cores):
    catalog, aux_data = load_diffsky_mock(
        opencosmo_data_path, version_check=version_checking, synth_cores=synth_cores
    )
    results_batched = compute_seds_from_diffsky_mock(
        catalog, aux_data, insert=False, batch_size=100
    )
    results_nobatch = compute_seds_from_diffsky_mock(catalog, aux_data, insert=False)
    seds = results_batched["rest_sed"]

    expected_shape = (len(catalog), len(aux_data["ssp_data"].ssp_wave))
    assert seds.shape == expected_shape
    assert np.all(~np.isnan(seds))
    assert np.allclose(
        results_batched["rest_sed"], results_nobatch["rest_sed"], rtol=1e-6
    )


def test_compute_seds_insert(opencosmo_data_path, version_checking, synth_cores):
    catalog, aux_data = load_diffsky_mock(
        opencosmo_data_path, version_check=version_checking, synth_cores=synth_cores
    )
    catalog = compute_seds_from_diffsky_mock(catalog, aux_data, insert=True)
    assert "rest_sed" in catalog.columns

    seds = catalog.select("rest_sed").get_data("numpy")

    expected_shape = (len(catalog), len(aux_data["ssp_data"].ssp_wave))
    assert seds.shape == expected_shape
    assert np.all(~np.isnan(seds))


def test_compute_dbk_seds(opencosmo_data_path, version_checking, synth_cores):
    catalog, aux_data = load_diffsky_mock(
        opencosmo_data_path, version_check=version_checking, synth_cores=synth_cores
    )
    results = compute_dbk_seds_from_diffsky_mock(catalog, aux_data, insert=False)

    expected_shape = (len(catalog), len(aux_data["ssp_data"].ssp_wave))
    for seds in results.values():
        assert seds.shape == expected_shape
        assert np.all(~np.isnan(seds))
