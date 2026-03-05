import numpy as np
from jax.scipy.stats import norm as jnorm

from diffsky.data_loaders.opencosmo_utils import (
    add_transmission_curves,
    compute_dbk_phot_from_diffsky_mock,
    compute_dbk_seds_from_diffsky_mock,
    compute_phot_from_diffsky_mock,
    compute_seds_from_diffsky_mock,
    load_diffsky_mock,
)


def test_compute_photometry(opencosmo_data_path):
    bands = ("lsst_u", "lsst_g", "lsst_r", "lsst_i", "lsst_z", "lsst_y")
    catalog, aux_data = load_diffsky_mock(opencosmo_data_path)
    results = compute_phot_from_diffsky_mock(catalog, aux_data, bands, insert=False)

    original_data = catalog.select(bands).get_data("numpy")
    for band in bands:
        assert np.all(
            np.isclose(results[f"{band}_new"], original_data[band], atol=1e-2)
        )


def test_compute_photometry_with_batch(opencosmo_data_path):
    bands = ("lsst_u", "lsst_g", "lsst_r", "lsst_i", "lsst_z", "lsst_y")
    catalog, aux_data = load_diffsky_mock(opencosmo_data_path)
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


def test_compute_photometry_custom_bands(opencosmo_data_path):
    wave = np.linspace(200, 8_000, 500)
    fake_tcurve1 = jnorm.pdf(wave, loc=3_000.0, scale=500.0)
    fake_tcurve2 = jnorm.pdf(wave, loc=5_000.0, scale=500.0)

    catalog, aux_data = load_diffsky_mock(opencosmo_data_path)

    aux_data = add_transmission_curves(
        aux_data, fake_tcurve_1=(wave, fake_tcurve1), fake_tcurve_2=(wave, fake_tcurve2)
    )

    results = compute_phot_from_diffsky_mock(
        catalog, aux_data, ["fake_tcurve_1", "fake_tcurve_2"], insert=False
    )
    for magnitudes in results.values():
        assert np.all((magnitudes > 10) & (magnitudes < 30))


def test_compute_photometry_custom_bands_insert(opencosmo_data_path):
    wave = np.linspace(200, 8_000, 500)
    fake_tcurve1 = jnorm.pdf(wave, loc=3_000.0, scale=500.0)
    fake_tcurve2 = jnorm.pdf(wave, loc=5_000.0, scale=500.0)

    catalog, aux_data = load_diffsky_mock(opencosmo_data_path)

    aux_data = add_transmission_curves(
        aux_data, fake_tcurve_1=(wave, fake_tcurve1), fake_tcurve_2=(wave, fake_tcurve2)
    )

    catalog = compute_phot_from_diffsky_mock(
        catalog, aux_data, ["fake_tcurve_1", "fake_tcurve_2"], insert=True
    )

    results = catalog.select(("fake_tcurve_1", "fake_tcurve_2")).get_data("numpy")
    for magnitudes in results.values():
        assert np.all((magnitudes > 10) & (magnitudes < 30))


def test_compute_dbk_photometry(opencosmo_data_path):
    bands = ("lsst_u", "lsst_g", "lsst_r", "lsst_i", "lsst_z", "lsst_y")
    catalog, aux_data = load_diffsky_mock(opencosmo_data_path)
    results = compute_dbk_phot_from_diffsky_mock(catalog, aux_data, bands, insert=False)
    columns = [col.removesuffix("_new") for col in results.keys()]

    original_data = catalog.select(columns).get_data("numpy")
    for name in columns:
        assert np.allclose(results[f"{name}_new"], original_data[name], atol=1e-2)


def test_compute_seds(opencosmo_data_path):
    bands = ("lsst_u", "lsst_g", "lsst_r", "lsst_i", "lsst_z", "lsst_y")
    catalog, aux_data = load_diffsky_mock(opencosmo_data_path)
    results = compute_seds_from_diffsky_mock(catalog, aux_data, bands, insert=False)
    seds = results["rest_sed"]
    expected_shape = (len(catalog), len(aux_data["ssp_data"].ssp_wave))
    assert seds.shape == expected_shape
    assert np.all(~np.isnan(seds))


def test_compute_seds_with_batching(opencosmo_data_path):
    bands = ("lsst_u", "lsst_g", "lsst_r", "lsst_i", "lsst_z", "lsst_y")
    catalog, aux_data = load_diffsky_mock(opencosmo_data_path)
    results_batched = compute_seds_from_diffsky_mock(
        catalog, aux_data, bands, insert=False, batch_size=100
    )
    results_nobatch = compute_seds_from_diffsky_mock(
        catalog, aux_data, bands, insert=False
    )
    seds = results_batched["rest_sed"]
    expected_shape = (len(catalog), len(aux_data["ssp_data"].ssp_wave))
    assert seds.shape == expected_shape
    assert np.all(~np.isnan(seds))
    assert np.all(results_batched["rest_sed"] == results_nobatch["rest_sed"])


def test_compute_seds_insert(opencosmo_data_path):
    bands = ("lsst_u", "lsst_g", "lsst_r", "lsst_i", "lsst_z", "lsst_y")
    catalog, aux_data = load_diffsky_mock(opencosmo_data_path)
    catalog = compute_seds_from_diffsky_mock(catalog, aux_data, bands, insert=True)
    assert "rest_sed" in catalog.columns

    seds = catalog.select("rest_sed").get_data("numpy")

    expected_shape = (len(catalog), len(aux_data["ssp_data"].ssp_wave))
    assert seds.shape == expected_shape
    assert np.all(~np.isnan(seds))


def test_compute_dbk_seds(opencosmo_data_path):
    bands = ("lsst_u", "lsst_g", "lsst_r", "lsst_i", "lsst_z", "lsst_y")
    catalog, aux_data = load_diffsky_mock(opencosmo_data_path)
    results = compute_dbk_seds_from_diffsky_mock(catalog, aux_data, bands, insert=False)

    expected_shape = (len(catalog), len(aux_data["ssp_data"].ssp_wave))
    for seds in results.values():
        assert seds.shape == expected_shape
        assert np.all(~np.isnan(seds))
