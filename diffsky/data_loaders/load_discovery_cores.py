"""
"""

import os
import typing

import h5py
import numpy as np
from diffmah.data_loaders.load_hacc_mahs import _load_forest
from diffmah.diffmah_kernels import DEFAULT_MAH_PARAMS, MAH_PBOUNDS, _log_mah_kern
from diffmah.diffmahpop_kernels.bimod_censat_params import DEFAULT_DIFFMAHPOP_PARAMS
from diffmah.diffmahpop_kernels.mc_bimod_cens import mc_diffmah_cenpop
from diffmah.diffmahpop_kernels.mc_bimod_sats import mc_diffmah_satpop
from diffmah.fitting_helpers import compute_indx_t_peak_halopop
from dsps.cosmology import flat_wcdm
from jax import jit as jjit
from jax import random as jran
from jax import vmap

_H = (0, 0, None)
log_mah_kern_vmap = jjit(vmap(_log_mah_kern, in_axes=_H))

BNPAT_DIFFMAH = "subvol_{0}_chunk_{1}.hdf5"
BNPAT_CORES = "m000p.coreforest.{0}.hdf5"
N_DISCOVERY_SUBVOL = 96

DRNPAT_CORES_LCRC = "/lcrc/group/cosmodata/simulations/DESI_W0WA/{0}/coreforest/forest"
DRNPAT_CORES_TASSO = "/Users/aphearin/work/DATA/DESI_W0WA/{0}"

# MASS_COLNAME should be consistent with the column used in the diffmah fits
MASS_COLNAME = "infall_tree_node_mass"

# Simulated MAHs with fewer points than N_MIN_MAH_PTS will get a synthetic MAH
N_MIN_MAH_PTS = 4


class SubhaloCatalog(typing.NamedTuple):
    mah_params: np.ndarray
    logmp0: np.ndarray
    logmp_pen_inf: np.ndarray
    logmp_ult_inf: np.ndarray
    logmhost_pen_inf: np.ndarray
    logmhost_ult_inf: np.ndarray
    t_obs: np.ndarray
    t_pen_inf: np.ndarray
    t_ult_inf: np.ndarray
    upids: np.ndarray
    t_pen_indx: np.ndarray
    t_ult_indx: np.ndarray
    fake_mah: np.ndarray


def _infer_drn_cores(sim_name, machine="LCRC"):
    sim_name = sim_name.upper()
    if "LCDM" in sim_name:
        sim_pat = "LCDM"
    elif "W0WA" in sim_name:
        sim_pat = "W0WA"
    else:
        raise ValueError(f"Unrecognized sim_name = {sim_name}")

    sim_name = "Discovery" + sim_pat

    machine = machine.upper()
    if machine == "LCRC":
        drn_cores = DRNPAT_CORES_LCRC.format(sim_pat)
    elif machine == "TASSO":
        drn_cores = DRNPAT_CORES_TASSO.format(sim_pat)
    else:
        raise ValueError(f"Unrecognized machine = {machine}")

    return sim_name, drn_cores


def load_forest_chunk(
    subvol, chunknum, nchunks, sim_name=None, machine=None, drn_cores=None
):
    if drn_cores is None:
        sim_name, drn_cores = _infer_drn_cores(sim_name, machine)
    bn_cores = BNPAT_CORES.format(subvol)
    fn_cores = os.path.join(drn_cores, bn_cores)

    sim, forest_matrices, zarr = _load_forest(fn_cores, sim_name, chunknum, nchunks)

    cosmo_dsps = flat_wcdm.CosmoParams(
        *(sim.cosmo.Omega_m, sim.cosmo.w0, sim.cosmo.wa, sim.cosmo.h)
    )

    tarr = flat_wcdm.age_at_z(zarr, *cosmo_dsps)

    return sim, cosmo_dsps, forest_matrices, zarr, tarr


def get_infall_time_indices(
    host_row, is_central, top_host_row, secondary_top_host_row, iz
):
    """Timestep of first infall into penultimate and ultimate hosts, -1 for centrals"""
    _X = top_host_row
    M_ult_host = host_row == _X[:, iz].reshape((-1, 1))
    indx_ult_inf_case2 = np.argmax(M_ult_host[:, : iz + 1], axis=1)

    core_only_minus1 = np.all(host_row == -1, axis=1)
    is_central_whole_life = ~np.any((host_row > -1) & (is_central < 1), axis=1)
    msk_case1 = core_only_minus1 | is_central_whole_life

    indx_t_ult_inf = np.where(msk_case1, -1, indx_ult_inf_case2)

    _Y = secondary_top_host_row
    M_pen_host = _X == _Y[:, iz].reshape((-1, 1))
    indx_pen_inf_case3 = np.argmax(M_pen_host, axis=1)

    msk_case3 = ~msk_case1 & (secondary_top_host_row[:, iz] != -1)

    indx_t_pen_inf = np.where(msk_case3, indx_pen_inf_case3, -1)

    return indx_t_ult_inf, indx_t_pen_inf


def load_discovery_core_data(
    subvol,
    chunknum,
    nchunks,
    iz_obs,
    drn_diffmah,
    sim_name=None,
    machine=None,
    drn_cores=None,
):
    _res = _load_forest_t_indices(
        subvol,
        chunknum,
        nchunks,
        iz_obs,
        sim_name=sim_name,
        machine=machine,
        drn_cores=drn_cores,
    )
    diffmah_data = _load_discovery_diffmah_data(
        drn_diffmah, subvol, chunknum, nchunks, N_DISCOVERY_SUBVOL
    )
    mah_params = DEFAULT_MAH_PARAMS._make(
        [diffmah_data[key] for key in DEFAULT_MAH_PARAMS._fields]
    )
    ret = (*_res, diffmah_data, mah_params)
    return ret


def load_discovery_diffsky_data(
    subvol,
    chunknum,
    nchunks,
    iz_obs,
    drn_diffmah,
    ran_key,
    sim_name=None,
    machine=None,
    drn_cores=None,
    mass_colname=MASS_COLNAME,
):
    _res = load_discovery_core_data(
        subvol,
        chunknum,
        nchunks,
        iz_obs,
        drn_diffmah,
        sim_name=sim_name,
        machine=machine,
        drn_cores=drn_cores,
    )

    sim, cosmo_dsps, forest, zarr, tarr, logt0, t_obs = _res[:7]
    indx_t_ult_inf, indx_t_pen_inf, diffmah_data, mah_params_raw = _res[7:]

    core_key, pen_key, ult_key, ran_key = jran.split(ran_key, 4)

    mah_sim = forest[mass_colname]
    is_central_sim = forest["central"][:, iz_obs]

    args = (
        mah_params_raw,
        mah_sim,
        is_central_sim,
        diffmah_data,
        tarr,
        core_key,
        t_obs,
        logt0,
    )
    mah_params_cores, msk_impute_cores = impute_mah_params(*args)

    indx_pen = forest["secondary_top_host_row"][:, iz_obs]
    indx_top = forest["top_host_row"][:, iz_obs]

    mah_params_pen_hosts = mah_params_cores._make(
        [getattr(mah_params_cores, key)[indx_pen] for key in mah_params_cores._fields]
    )
    mah_params_top_hosts = mah_params_cores._make(
        [getattr(mah_params_cores, key)[indx_top] for key in mah_params_cores._fields]
    )

    mah_sim_pen_hosts = mah_sim[indx_pen]
    is_central_pen_hosts = forest["central"][indx_pen, iz_obs]
    keys = diffmah_data.keys()
    diffmah_data_pen_hosts = dict([(key, diffmah_data[key][indx_pen]) for key in keys])

    mah_sim_top_hosts = mah_sim[indx_top]
    is_central_top_hosts = forest["central"][indx_top, iz_obs]
    diffmah_data_top_hosts = dict([(key, diffmah_data[key][indx_top]) for key in keys])

    args = (
        mah_params_pen_hosts,
        mah_sim_pen_hosts,
        is_central_pen_hosts,
        diffmah_data_pen_hosts,
        tarr,
        pen_key,
        t_obs,
        logt0,
    )
    mah_params_pen_hosts, msk_impute_pen_hosts = impute_mah_params(*args)

    args = (
        mah_params_top_hosts,
        mah_sim_top_hosts,
        is_central_top_hosts,
        diffmah_data_top_hosts,
        tarr,
        ult_key,
        t_obs,
        logt0,
    )
    mah_params_top_hosts, msk_impute_top_hosts = impute_mah_params(*args)

    logmp_t_pen_inf = log_mah_kern_vmap(mah_params_cores, tarr[indx_t_pen_inf], logt0)
    logmp_t_ult_inf = log_mah_kern_vmap(mah_params_cores, tarr[indx_t_ult_inf], logt0)

    logmp_host_t_pen_inf = log_mah_kern_vmap(
        mah_params_pen_hosts, tarr[indx_t_pen_inf], logt0
    )
    logmp_host_t_ult_inf = log_mah_kern_vmap(
        mah_params_top_hosts, tarr[indx_t_ult_inf], logt0
    )

    t_pen_inf = tarr[indx_t_pen_inf]
    t_ult_inf = tarr[indx_t_ult_inf]

    pen_indx = forest["secondary_top_host_row"][:, iz_obs]
    ult_indx = forest["top_host_row"][:, iz_obs]

    n_halos = forest["top_host_row"].shape[0]
    t_obs_arr = np.zeros(n_halos) + t_obs

    upids = np.where(is_central_sim, -1, forest["top_host_row"][:, iz_obs])

    logmp0 = np.log10(mah_sim[:, -1])

    subcat = SubhaloCatalog(
        mah_params_cores,
        logmp0,
        logmp_t_pen_inf,
        logmp_t_ult_inf,
        logmp_host_t_pen_inf,
        logmp_host_t_ult_inf,
        t_obs_arr,
        t_pen_inf,
        t_ult_inf,
        upids,
        pen_indx,
        ult_indx,
        msk_impute_cores,
    )

    return subcat, forest, tarr, sim, zarr, cosmo_dsps


def impute_mask(mah_params, diffmah_data, n_min_mah=N_MIN_MAH_PTS):
    mah_keys = [key for key in DEFAULT_MAH_PARAMS._fields if key != "t_peak"]
    msk_nofit = np.ones(mah_params.logm0.size).astype(bool)
    for key in mah_keys:
        p = diffmah_data[key]
        msk_nofit = msk_nofit * (p == -99)
    msk_badfit = diffmah_data["fit_algo"] == -1  # fitter did not run
    msk_badloss = diffmah_data["loss"] < -0.01  # loss should be positive
    msk_badmah = diffmah_data["n_points_per_fit"] < n_min_mah

    msk_impute = msk_nofit | msk_badfit | msk_badloss | msk_badmah
    return msk_impute


def impute_mah_params(
    mah_params, mah_sim, is_central, diffmah_data, tarr, ran_key, t_obs, lgt0
):
    ran_key, cen_key, sat_key = jran.split(ran_key, 3)

    mah_sim = np.maximum.accumulate(mah_sim, axis=1)
    indx_t_peak = compute_indx_t_peak_halopop(mah_sim)
    t_peak_sim = tarr[indx_t_peak]

    msg_tp = "t_peak_sim must exceed diffmah.diffmah_kernels.MAH_PBOUNDS.t_peak[0]"
    assert np.all(t_peak_sim > MAH_PBOUNDS.t_peak[0]), msg_tp

    mah_params = mah_params._replace(t_peak=t_peak_sim)

    lgm_obs = np.log10(mah_sim[:, -1])
    t_obs = tarr[-1] + np.zeros_like(t_peak_sim)
    cenpop = mc_diffmah_cenpop(DEFAULT_DIFFMAHPOP_PARAMS, lgm_obs, t_obs, cen_key, lgt0)
    satpop = mc_diffmah_satpop(DEFAULT_DIFFMAHPOP_PARAMS, lgm_obs, t_obs, sat_key, lgt0)

    indx_t_peak_p1 = indx_t_peak + 1
    indx_max = tarr.size - 1
    indx_t_peak_p1 = np.where(indx_t_peak_p1 > indx_max, indx_max, indx_t_peak_p1)

    gen = zip(cenpop.mah_params, satpop.mah_params)
    _mah_params = [np.where(is_central, x, y) for x, y in gen]
    mah_params_fake = DEFAULT_MAH_PARAMS._make(_mah_params)

    msk_impute = impute_mask(mah_params, diffmah_data)
    gen = zip(mah_params_fake, mah_params)
    _mah_params_imputed = [np.where(msk_impute, x, y) for x, y in gen]
    mah_params_imputed = DEFAULT_MAH_PARAMS._make(_mah_params_imputed)

    return mah_params_imputed, msk_impute


def _load_forest_t_indices(
    subvol, chunknum, nchunks, iz_obs, sim_name=None, machine=None, drn_cores=None
):
    _res = load_forest_chunk(
        subvol,
        chunknum,
        nchunks,
        sim_name=sim_name,
        machine=machine,
        drn_cores=drn_cores,
    )
    sim, cosmo_dsps, forest, zarr, tarr = _res

    logt0 = float(np.log10(tarr[-1]))
    _z_obs = np.zeros(1) + zarr[iz_obs]
    t_obs = float(flat_wcdm.age_at_z(_z_obs, *cosmo_dsps)[0])

    host_row = forest["host_row"]
    is_central = forest["central"]
    top_host_row = forest["top_host_row"]
    secondary_top_host_row = forest["secondary_top_host_row"]
    args = host_row, is_central, top_host_row, secondary_top_host_row, iz_obs
    indx_t_ult_inf, indx_t_pen_inf = get_infall_time_indices(*args)

    ret = (
        sim,
        cosmo_dsps,
        forest,
        zarr,
        tarr,
        logt0,
        t_obs,
        indx_t_ult_inf,
        indx_t_pen_inf,
    )
    return ret


def _load_discovery_diffmah_data(drn, subvol, chunknum, nchunks, n_subvol_tot):
    nchar_subvol = len(str(n_subvol_tot))
    nchar_chunks = len(str(nchunks))
    subvol_str = f"{subvol:0{nchar_subvol}d}"
    chunknum_str = f"{chunknum:0{nchar_chunks}d}"

    bname = BNPAT_DIFFMAH.format(subvol_str, chunknum_str)
    fn_diffmah = os.path.join(drn, bname)

    diffmah_data = _load_flat_hdf5(fn_diffmah)
    return diffmah_data


def _load_flat_hdf5(fn):
    data = dict()
    with h5py.File(fn, "r") as hdf:
        for key in hdf.keys():
            data[key] = hdf[key][...]
    return data
