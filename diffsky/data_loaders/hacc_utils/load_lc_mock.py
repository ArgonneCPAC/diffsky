""""""

import os
from glob import glob

import h5py
import numpy as np
from diffmah import DEFAULT_MAH_PARAMS
from diffstar import DEFAULT_DIFFSTAR_PARAMS
from dsps.cosmology.flat_wcdm import age_at_z
from jax import jit as jjit
from jax import numpy as jnp
from jax import vmap

from diffsky.ssp_err_model import ssp_err_model

from ... import phot_utils
from ...data_loaders import load_flat_hdf5
from ...experimental import precompute_ssp_phot as psspp
from ...experimental.dbk_from_mock import _disk_bulge_knot_phot_from_mock
from . import hacc_core_utils as hcu
from . import lc_mock_production as lcmp

interp_vmap = jjit(vmap(jnp.interp, in_axes=(None, None, 0)))


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


def get_ssp_phot_tables(tcurves, z_phot_table, ssp_data, sim_info):

    wave_eff_table = phot_utils.get_wave_eff_table(z_phot_table, tcurves)
    ssp_mag_table = psspp.get_precompute_ssp_mag_redshift_table(
        tcurves, ssp_data, z_phot_table, sim_info.cosmo_params
    )
    return ssp_mag_table, wave_eff_table


def compute_mock_photometry(
    mock_data, mock_info, tcurves, ssp_mag_table, wave_eff_table
):
    mah_params = DEFAULT_MAH_PARAMS._make(
        [mock_data[key] for key in DEFAULT_MAH_PARAMS._fields]
    )
    sfh_params = DEFAULT_DIFFSTAR_PARAMS._make(
        [mock_data[key] for key in DEFAULT_DIFFSTAR_PARAMS._fields]
    )
    t_obs = age_at_z(mock_data["redshift_true"], *mock_info["sim_info"].cosmo_params)

    delta_scatter_rest_table = jnp.where(
        mock_data["mc_sfh_type"].reshape((-1, 1)) == 0,
        mock_data["delta_scatter_q"],
        mock_data["delta_scatter_ms"],
    )
    rest_wave_eff = phot_utils.get_wave_eff_from_tcurves(tcurves, 0.0)
    delta_scatter_obs_table = interp_vmap(
        rest_wave_eff, ssp_err_model.LAMBDA_REST, delta_scatter_rest_table
    )

    args = (
        mock_data["redshift_true"],
        t_obs,
        mah_params,
        mock_data["logmp0"],
        mock_info["t_table"],
        mock_info["ssp_data"],
        ssp_mag_table,
        mock_info["z_phot_table"],
        wave_eff_table,
        sfh_params,
        mock_info["param_collection"].mzr_params,
        mock_info["param_collection"].spspop_params,
        mock_info["param_collection"].scatter_params,
        mock_info["param_collection"].ssperr_params,
        mock_info["sim_info"].cosmo_params,
        mock_info["sim_info"].fb,
        mock_data["uran_av"],
        mock_data["uran_delta"],
        mock_data["uran_funo"],
        delta_scatter_obs_table,
        mock_data["mc_sfh_type"],
        mock_data["fknot"],
    )
    phot_data = _disk_bulge_knot_phot_from_mock(*args)

    return phot_data
