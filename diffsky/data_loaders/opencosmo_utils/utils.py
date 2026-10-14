import functools
import inspect
from collections import namedtuple

import numpy as np
import opencosmo as oc
from dsps.cosmology import age_at_z
from jax import numpy as jnp


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


def determine_split_points(catalog, batch_size):
    """
    Split the catalog into batches of complete host+satellite groups, each
    containing roughly batch_size galaxies. Returns the top host indices for
    each batch, and the size of the largest batch (which all batches should be
    padded to so the kernels only compile once).
    """
    top_host_idx = catalog.select("top_host_idx").get_data("numpy")
    unique_top_host, counts = np.unique(top_host_idx, return_counts=True)

    if batch_size == -1 or len(catalog) <= batch_size:
        return [unique_top_host], len(catalog)

    cs = np.cumsum(counts)
    n_chunks = max(1, round(cs[-1] / batch_size))
    boundaries = cs[-1] / n_chunks * np.arange(1, n_chunks)
    split_points = np.unique(np.searchsorted(cs, boundaries) + 1)
    split_points = split_points[split_points < len(unique_top_host)]

    top_host_idx_splits = np.array_split(unique_top_host, split_points)
    split_counts = np.add.reduceat(counts, np.insert(split_points, 0, 0))
    pad_size = int(np.max(split_counts))
    return top_host_idx_splits, pad_size


def pad_galaxy_inputs(func, static_kwargs, pad_size: int):
    """
    Wrap func so that every per-galaxy column it receives is padded to pad_size
    before the call, and every output is trimmed back afterwards. Anything in
    static_kwargs (lookup tables, parameters, etc.) is passed through untouched.

    Padded galaxies are copies of the last galaxy, but are made into isolated
    centrals that host themselves so they do not merge with any real galaxy.
    """

    accepted = inspect.signature(func).parameters

    @functools.wraps(func)
    def wrapper(**kwargs):
        n_gal = len(kwargs["gal_id"])
        padded = {
            name: value if name in static_kwargs else _pad_first_axis(value, pad_size)
            for name, value in kwargs.items()
            if name in accepted
        }
        padded["central"] = padded["central"].at[n_gal:].set(1)
        padded["top_host_idx"] = (
            padded["top_host_idx"].at[n_gal:].set(jnp.arange(n_gal, pad_size))
        )
        result = func(**padded)
        return {name: value[:n_gal] for name, value in result.items()}

    return wrapper


def _pad_first_axis(arr, pad_size):
    arr = jnp.asarray(arr)
    pad_width = [(0, pad_size - arr.shape[0])] + [(0, 0)] * (arr.ndim - 1)
    return jnp.pad(arr, pad_width, mode="edge")


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
