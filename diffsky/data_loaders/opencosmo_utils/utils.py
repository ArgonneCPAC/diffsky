from collections import namedtuple

import numpy as np
import opencosmo as oc
from dsps.cosmology import age_at_z


def get_z_phot_tables(catalog: oc.Lightcone):
    """
    Retrieve z_phot_tables from the underlying catalogs, or construct
    them if not found.
    """
    z_phot_tables = {}
    for slice_name, dataset in catalog.items():
        z_phot_tables[slice_name] = _get_z_phot_table_from_dataset(dataset)
    return z_phot_tables


def _get_z_phot_table_from_dataset(dataset: oc.Dataset):
    z_phot_table = dataset.header.catalog_info.get("zphot_table")
    if z_phot_table is None:
        return _estimate_z_phot_table(dataset)
    return np.array(z_phot_table)


def _estimate_z_phot_table(dataset: oc.Dataset):
    min_z, max_z = dataset.header.lightcone["z_range"]
    min_z = 0.95 * min_z
    max_z = 1.05 * max_z
    return np.linspace(min_z, max_z, 15)


def unpack_photometry(data, band_names, suffix, gal_id, *args):
    result = unpack_photometry_array(data.obs_mags, band_names, suffix)
    result["gal_id"] = gal_id
    return result


def unpack_dbk_photometry(data, band_names, suffix, gal_id, include_extras):
    phot_info, dbk_weights = data
    obs_mags_disk = phot_info.obs_mags_disk
    obs_mags_bulge = phot_info.obs_mags_bulge
    obs_mags_knot = phot_info.obs_mags_knots

    bulge_bands = [f"{bn}_bulge" for bn in band_names]
    disk_bands = [f"{bn}_disk" for bn in band_names]
    knot_bands = [f"{bn}_knots" for bn in band_names]

    output = unpack_photometry_array(obs_mags_bulge, bulge_bands, suffix)
    output |= unpack_photometry_array(obs_mags_disk, disk_bands, suffix)
    output |= unpack_photometry_array(obs_mags_knot, knot_bands, suffix)
    output["gal_id"] = gal_id
    if include_extras is not None:
        phot_info = phot_info._asdict()
        output |= {name: np.array(phot_info[name]) for name in include_extras}

    return output


def unpack_photometry_array(data, band_names, suffix):
    to_unpack = np.array(data).T
    return {f"{name}{suffix}": to_unpack[i] for i, name in enumerate(band_names)}


def validate_batch_size(batch_size):
    if batch_size == -1:
        return
    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError(
            f"batch_size must be -1 (no batching) or a positive integer, got {batch_size!r}"
        )


def split_central_indices(catalog: oc.Lightcone, batch_size: int):
    central_indices = np.where(catalog.select("central").get_data("numpy"))[0]
    if batch_size == -1:
        return [central_indices]
    split_indices = np.arange(batch_size, len(central_indices), batch_size)
    return np.split(central_indices, split_indices)


def prep_cosmology_parameters(cosmology):
    try:
        w0 = cosmology.wo
        wa = cosmology.wa
    except AttributeError:  # Why astropy... Why...
        w0 = -1
        wa = 0
    Cosmology_t = namedtuple("Cosmology_t", ("Om0", "w0", "wa", "h"))
    return Cosmology_t(cosmology.Om0, w0, wa, cosmology.h)


def age_at_z_(redshift_true, cosmology):
    result = {
        "t_obs": np.array(
            age_at_z(
                redshift_true, cosmology.Om0, cosmology.w0, cosmology.wa, cosmology.h
            )
        )
    }
    return result
