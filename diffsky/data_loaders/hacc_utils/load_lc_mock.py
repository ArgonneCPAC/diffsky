""""""

import os
from glob import glob

import h5py
import numpy as np
from diffmah import DEFAULT_MAH_PARAMS
from diffstar import DEFAULT_DIFFSTAR_PARAMS
from dsps.cosmology.flat_wcdm import age_at_z

from ... import phot_utils
from ...data_loaders import load_flat_hdf5
from ...experimental import dbk_from_mock2
from ...experimental import precompute_ssp_phot as psspp
from . import hacc_core_utils as hcu
from . import lc_mock_production as lcmp


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


def load_diffsky_lc_patch(drn_mock, bn_mock):
    fn_mock = os.path.join(drn_mock, bn_mock)
    diffsky_data = load_flat_hdf5(fn_mock, dataset="data")

    with h5py.File(fn_mock, "r") as hdf:
        mock_version_name = hdf["metadata"].attrs["mock_version_name"]

    tcurves = lcmp.load_diffsky_tcurves(drn_mock, mock_version_name)
    ssp_data = lcmp.load_diffsky_ssp_data(drn_mock, mock_version_name)
    sim_info = lcmp.load_diffsky_sim_info(fn_mock)

    param_collection = lcmp.load_diffsky_param_collection(drn_mock, mock_version_name)

    z_phot_table = lcmp.load_diffsky_z_phot_table(fn_mock)

    diffsky_lc_patch = dict(
        diffsky_data=diffsky_data,
        ssp_data=ssp_data,
        param_collection=param_collection,
        z_phot_table=z_phot_table,
        tcurves=tcurves,
        sim_info=sim_info,
    )
    return diffsky_lc_patch


def compute_phot_from_diffsky_mock(
    *, diffsky_data, ssp_data, param_collection, sim_info, z_phot_table, tcurves
):
    mah_params = DEFAULT_MAH_PARAMS._make(
        [diffsky_data[key] for key in DEFAULT_MAH_PARAMS._fields]
    )
    sfh_params = DEFAULT_DIFFSTAR_PARAMS._make(
        [diffsky_data[key] for key in DEFAULT_DIFFSTAR_PARAMS._fields]
    )
    t_obs = age_at_z(diffsky_data["redshift_true"], *sim_info.cosmo_params)

    wave_eff_table = phot_utils.get_wave_eff_table(z_phot_table, tcurves)

    precomputed_ssp_mag_table = psspp.get_precompute_ssp_mag_redshift_table(
        tcurves, ssp_data, z_phot_table, sim_info.cosmo_params
    )

    mc_is_q = np.where(diffsky_data["mc_sfh_type"] == 0, True, False)
    args = (
        mc_is_q,
        diffsky_data["uran_av"],
        diffsky_data["uran_delta"],
        diffsky_data["uran_funo"],
        diffsky_data["uran_pburst"],
        diffsky_data["delta_mag_ssp_scatter"],
        sfh_params,
        diffsky_data["redshift_true"],
        t_obs,
        mah_params,
        diffsky_data["fknot"],
        ssp_data,
        precomputed_ssp_mag_table,
        z_phot_table,
        wave_eff_table,
        param_collection.mzr_params,
        param_collection.spspop_params,
        param_collection.scatter_params,
        param_collection.ssperr_params,
        sim_info.cosmo_params,
        sim_info.fb,
    )
    _res = dbk_from_mock2._reproduce_mock_dbk_kern(*args)
    (
        phot_info,
        phot_randoms,
        disk_bulge_history,
        obs_mags_bulge,
        obs_mags_disk,
        obs_mags_knots,
    ) = _res
    phot_info = phot_info._asdict()
    phot_info["obs_mags_bulge"] = obs_mags_bulge
    phot_info["obs_mags_disk"] = obs_mags_disk
    phot_info["obs_mags_knots"] = obs_mags_knots
    return phot_info
