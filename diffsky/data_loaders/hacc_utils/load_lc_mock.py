""""""

import os
from glob import glob

import h5py
import numpy as np

from ...data_loaders import load_flat_hdf5
from . import hacc_core_utils as hcu
from . import lc_mock_production as lcmp


def load_lc_mock_info(fn_mock_data):
    drn_mock = os.path.dirname(fn_mock_data)

    with h5py.File(fn_mock_data, "r") as hdf:
        mock_version_name = hdf["metadata"].attrs["mock_version_name"]

    z_phot_table = lcmp.load_diffsky_z_phot_table(fn_mock_data)
    t_table = lcmp.load_diffsky_t_table(drn_mock, mock_version_name)
    ssp_data = lcmp.load_diffsky_ssp_data(drn_mock, mock_version_name)
    sim_info = lcmp.load_diffsky_sim_info(fn_mock_data)
    param_collection = lcmp.load_diffsky_param_collection(drn_mock, mock_version_name)

    mock_info = dict()
    # mock_info["mock_version_name"] = mock_version_name
    mock_info["sim_info"] = sim_info
    mock_info["ssp_data"] = ssp_data
    mock_info["param_collection"] = param_collection
    mock_info["t_table"] = t_table
    mock_info["z_phot_table"] = z_phot_table
    mock_info["tcurves"] = lcmp.load_diffsky_tcurves(drn_mock, mock_version_name)

    return mock_info


def load_diffsky_lightcone(drn, sim_name, z_min, z_max, patch_list):
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
        lc_diffsky_data = load_flat_hdf5(fn, dataset="data")
        data_collector.append(lc_diffsky_data)

    diffsky_data = dict()
    for key in data_collector[0].keys():
        diffsky_data[key] = np.concatenate([x[key] for x in data_collector])

    return diffsky_data
