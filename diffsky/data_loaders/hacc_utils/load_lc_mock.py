""""""

import os
from glob import glob

import h5py
import numpy as np

from .. import load_flat_hdf5
from . import hacc_core_utils as hcu
from . import lc_mock as lcmp
from . import lightcone_utils
from . import load_lc_cores as llcc


def load_diffsky_lightcone(drn, sim_name, z_min, z_max, patch_list, keys=None):
    fn_list_all = glob(os.path.join(drn, lcmp.LC_MOCK_BNPAT.format("*", "*")))
    bn_list_all = [os.path.basename(fn) for fn in fn_list_all]
    patch_info_all = [lcmp.get_patch_info_from_mock_basename(bn) for bn in bn_list_all]

    _res = hcu.get_timestep_range_from_z_range(sim_name, z_min, z_max)
    timestep_min, timestep_max = _res[2:]

    fn_collector = []
    for i, patch_info in enumerate(patch_info_all):
        stepnum, patchnum = patch_info
        keep_patch = patchnum in patch_list
        keep_snap = (stepnum >= timestep_min) & (stepnum <= timestep_max)
        keep = keep_snap & keep_patch
        if keep:
            fn_collector.append(fn_list_all[i])

    data_collector = []
    for fn in fn_collector:
        lc_diffsky_data = load_flat_hdf5(fn, dataset="data", keys=keys)
        data_collector.append(lc_diffsky_data)

    diffsky_data = dict()
    for key in data_collector[0].keys():
        diffsky_data[key] = np.concatenate([x[key] for x in data_collector])

    return diffsky_data


def load_mock_metadata(fn_mock):
    metadata_dict = dict()

    hacc_cosmology = dict()
    nbody_info = dict()
    software_info = dict()
    index_data = dict()

    with h5py.File(fn_mock, "r") as hdf:
        for key, val in hdf["metadata"].attrs.items():
            metadata_dict[key] = val

            for cosmo_pname, cosmo_pval in hdf["metadata"]["cosmology"].attrs.items():
                hacc_cosmology[cosmo_pname] = cosmo_pval
            metadata_dict["cosmology"] = hacc_cosmology

            for nbody_key, nbody_val in hdf["metadata"]["nbody_info"].attrs.items():
                nbody_info[nbody_key] = nbody_val
            metadata_dict["nbody_info"] = nbody_info

            for soft_key, soft_val in hdf["metadata"][
                "software_version_info"
            ].attrs.items():
                software_info[soft_key] = soft_val
            metadata_dict["software_version_info"] = software_info

            z_phot_table = hdf["metadata"]["z_phot_table"][...]
            metadata_dict["z_phot_table"] = z_phot_table

            if "index" in hdf["metadata"].keys():
                index_data["offset"] = hdf["metadata"]["index"]["offset"][...]
                index_data["count"] = hdf["metadata"]["index"]["count"][...]
                index_data["unique_id"] = hdf["metadata"]["index"]["unique_id"][...]
                metadata_dict["index"] = index_data

    bn_mock = os.path.basename(fn_mock)
    stepnum, lc_patch = lcmp.get_patch_info_from_mock_basename(bn_mock)
    metadata_dict["stepnum"] = stepnum
    metadata_dict["lc_patch"] = lc_patch

    drn_mock = os.path.dirname(fn_mock)
    fn_lc_patch_decomp = os.path.join(drn_mock, "lc_cores-decomposition.txt")
    if os.path.isfile(fn_lc_patch_decomp):
        _res = lightcone_utils.read_lc_ra_dec_patch_decomposition(fn_lc_patch_decomp)
        patch_decomposition, __, solid_angles = _res
        metadata_dict["sky_area_degsq"] = solid_angles[lc_patch]

    ssp_data = lcmp.load_diffsky_ssp_data(drn_mock, metadata_dict["mock_version_name"])
    metadata_dict["ssp_data"] = ssp_data

    tcurves = lcmp.load_diffsky_tcurves(drn_mock, metadata_dict["mock_version_name"])
    metadata_dict["tcurves"] = tcurves

    sim_info = lcmp.load_diffsky_sim_info(fn_mock)
    metadata_dict["sim_info"] = sim_info

    param_collection = lcmp.load_diffsky_param_collection_merging(
        drn_mock, metadata_dict["mock_version_name"]
    )
    metadata_dict["param_collection"] = param_collection

    return metadata_dict


def load_lc_patch_collection(fn_list, keys):
    """Concatenate a collection of mock data from an input list of files"""
    mock_collector = []
    for fn in fn_list:
        mock_bn = load_flat_hdf5(fn, dataset="data", keys=keys)
        mock_collector.append(mock_bn)
    mock = dict()
    for key in keys:
        mock[key] = np.concatenate([mock_bn[key] for mock_bn in mock_collector])
    return mock


def estimate_nchunks(fn_lc_mock, batch_size):
    """Estimate number of chunks to approximately divide mock into batches of input size"""
    arr = load_flat_hdf5(fn_lc_mock, dataset="data", keys=["central"])["central"]
    n = arr.size
    nchunks = max(1, n // batch_size)
    return nchunks


def load_lc_mock_chunk(fn_lc_mock, *, nchunks, chunknum, lc_mock_keys=None):
    with h5py.File(fn_lc_mock, "r") as hdf:
        if lc_mock_keys is None:
            lc_mock_keys = list(hdf["data"].keys())

        keys_to_throw_out = ["unlensed_magnitudes", "unlensed_fluxes"]
        for key in keys_to_throw_out:
            if key in lc_mock_keys:
                lc_mock_keys.pop(lc_mock_keys.index(key))

        lc_mock, (istart, iend) = llcc._read_lc_cores_chunk(
            hdf, nchunks, chunknum, lc_mock_keys, index_dataset="metadata"
        )

    return lc_mock, (istart, iend)


def get_lc_mock_chunk(lc_mock, metadata, *, nchunks, chunknum, lc_mock_keys=None):
    if lc_mock_keys is None:
        lc_mock_keys = list(lc_mock.keys())

    offset = metadata["index"]["offset"]
    count = metadata["index"]["count"]
    read_start, read_end = llcc.compute_read_start_end_for_chunk(
        nchunks, chunknum, offset, count
    )
    lc_mock_chunk = {key: lc_mock[key][read_start:read_end] for key in lc_mock_keys}

    lc_mock_chunk["top_host_idx_chunk"] = lc_mock_chunk["top_host_idx"] - read_start
    return lc_mock_chunk


def load_diffsky_lc_patch(fn_mock, keys=None):
    diffsky_data = load_flat_hdf5(fn_mock, dataset="data", keys=keys)
    metadata = load_mock_metadata(fn_mock)
    return diffsky_data, metadata
