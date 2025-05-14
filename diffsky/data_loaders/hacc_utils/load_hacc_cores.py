"""General purpose data-loading utilities for HACC cores"""

import os
from collections import namedtuple

import h5py
import numpy as np
from diffmah.diffmah_kernels import DEFAULT_MAH_PARAMS, MAH_PBOUNDS, _log_mah_kern
from diffmah.diffmahpop_kernels.bimod_censat_params import DEFAULT_DIFFMAHPOP_PARAMS
from diffmah.diffmahpop_kernels.mc_bimod_cens import mc_diffmah_cenpop
from diffmah.diffmahpop_kernels.mc_bimod_sats import mc_diffmah_satpop
from diffmah.fitting_helpers.diffmah_fitter_helpers import compute_indx_t_peak_halopop
from dsps.cosmology import flat_wcdm
from jax import jit as jjit
from jax import random as jran
from jax import vmap

from .. import load_flat_hdf5
from . import hacc_core_utils as hcu
from .defaults import DIFFMAH_MASS_COLNAME, N_MIN_MAH_PTS

try:
    from mpi4py import MPI

    COMM = MPI.COMM_WORLD
except ImportError:
    MPI = COMM = None

try:
    from haccytrees import Simulation as HACCSim
    from haccytrees import coretrees

    HAS_HACCYTREES = True
except ImportError:
    HAS_HACCYTREES = False

_H = (0, 0, None)
log_mah_kern_vmap = jjit(vmap(_log_mah_kern, in_axes=_H))

BNPAT_DIFFMAH = "subvol_{0}_diffmah_fits.hdf5"
BNPAT_CORES = "m000p.coreforest.{0}.hdf5"


_SUBCAT_COLNAMES = (
    "mah_params",
    "logmp_t_obs",
    "logmp0",
    "logmp_pen_inf",
    "logmp_ult_inf",
    "logmhost_pen_inf",
    "logmhost_ult_inf",
    "t_obs",
    "t_pen_inf",
    "t_ult_inf",
    "upids",
    "pen_host_indx",
    "ult_host_indx",
    "fake_mah",
)
SubhaloCatalog = namedtuple("SubhaloCatalog", _SUBCAT_COLNAMES)

DIFFSKY_DATA_DICT_KEYS = ("subcat", "sim", "tarr", "zarr")

EMPTY_MAH_PARAMS = DEFAULT_MAH_PARAMS._make([None] * len(DEFAULT_MAH_PARAMS))
EMPTY_SUBCAT_DATA = [EMPTY_MAH_PARAMS, *[None] * len(_SUBCAT_COLNAMES[1:])]
EMPTY_SUBCAT = SubhaloCatalog._make(EMPTY_SUBCAT_DATA)


def _get_all_avail_basenames(drn, pat, subvolumes):
    fname_list = [os.path.join(drn, pat.format(i)) for i in subvolumes]
    for fn in fname_list:
        assert os.path.isfile(fn), fn
    return fname_list


def load_diffsky_data_per_rank(
    sim_name,
    subvol,
    chunknum,
    nchunks,
    iz_obs,
    ran_key,
    drn_cores,
    drn_diffmah,
    mass_colname=DIFFMAH_MASS_COLNAME,
    comm=None,
):
    if comm is None:
        comm = MPI.COMM_WORLD

    if comm.rank == 0:
        diffsky_data = load_diffsky_data(
            sim_name,
            subvol,
            chunknum,
            nchunks,
            iz_obs,
            ran_key,
            drn_cores,
            drn_diffmah,
            mass_colname=mass_colname,
        )
    else:
        diffsky_data = dict()
        diffsky_data["subcat"] = EMPTY_SUBCAT
        diffsky_data["sim"] = None
        diffsky_data["tarr"] = None
        diffsky_data["zarr"] = None

    diffsky_data["tarr"] = comm.bcast(diffsky_data["tarr"], root=0)
    diffsky_data["zarr"] = comm.bcast(diffsky_data["zarr"], root=0)
    diffsky_data["sim"] = HACCSim.simulations[sim_name]
    diffsky_data["subcat"] = hcu.scatter_subcat(diffsky_data["subcat"], comm)

    return diffsky_data


def load_diffsky_data(
    sim_name,
    subvol,
    chunknum,
    nchunks,
    iz_obs,
    ran_key,
    drn_cores,
    drn_diffmah,
    mass_colname=DIFFMAH_MASS_COLNAME,
    include_fields=(),
):
    fn_cores = os.path.join(drn_cores, BNPAT_CORES.format(subvol))

    # At a minimum we need `central` and mass
    include_fields = ("central", mass_colname, *include_fields)
    include_fields = list(set(include_fields))  # discard possible redundancies

    _res = load_coreforest_and_metadata(
        fn_cores, sim_name, chunknum, nchunks, include_fields=include_fields
    )
    sim, cosmo_dsps, forest, zarr, tarr, logt0 = _res
    t_obs = flat_wcdm._age_at_z_kern(zarr[iz_obs], *cosmo_dsps)

    diffmah_fit_data, mah_params_raw = load_diffmah_data_for_forest(
        drn_diffmah, subvol, forest
    )
    indx_t_ult_inf, indx_t_pen_inf = get_infall_time_indices(
        forest["host_row"],
        forest["central"],
        forest["top_host_row"],
        forest["secondary_top_host_row"],
        iz_obs,
    )

    core_key, pen_key, ult_key, ran_key = jran.split(ran_key, 4)

    mah_sim = forest[mass_colname]
    is_central_sim = forest["central"][:, iz_obs]

    args = (
        mah_params_raw,
        mah_sim,
        is_central_sim,
        diffmah_fit_data,
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
    keys = diffmah_fit_data.keys()
    diffmah_data_pen_hosts = dict(
        [(key, diffmah_fit_data[key][indx_pen]) for key in keys]
    )

    mah_sim_top_hosts = mah_sim[indx_top]
    is_central_top_hosts = forest["central"][indx_top, iz_obs]
    diffmah_data_top_hosts = dict(
        [(key, diffmah_fit_data[key][indx_top]) for key in keys]
    )

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

    logmp_t_obs = _log_mah_kern(mah_params_cores, tarr[iz_obs], logt0)
    logmp0 = _log_mah_kern(mah_params_cores, tarr[-1], logt0)

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

    pen_host_indx = forest["secondary_top_host_row"][:, iz_obs]
    ult_host_indx = forest["top_host_row"][:, iz_obs]

    n_halos = forest["top_host_row"].shape[0]
    t_obs_arr = np.zeros(n_halos) + t_obs

    upids = np.where(is_central_sim, -1, forest["top_host_row"][:, iz_obs])

    subcat = SubhaloCatalog(
        mah_params_cores,
        logmp_t_obs,
        logmp0,
        logmp_t_pen_inf,
        logmp_t_ult_inf,
        logmp_host_t_pen_inf,
        logmp_host_t_ult_inf,
        t_obs_arr,
        t_pen_inf,
        t_ult_inf,
        upids,
        pen_host_indx,
        ult_host_indx,
        msk_impute_cores,
    )

    _ret = (subcat, sim, tarr, zarr)
    diffsky_data = dict([(key, val) for key, val in zip(DIFFSKY_DATA_DICT_KEYS, _ret)])
    return diffsky_data


def load_coreforest_and_metadata(
    fn_cores, sim_name, chunknum, nchunks, include_fields=()
):
    sim = HACCSim.simulations[sim_name]
    zarr = sim.step2z(np.array(sim.cosmotools_steps))

    forest_matrices = coretrees.corematrix_reader(
        fn_cores,
        calculate_secondary_host_row=True,
        nchunks=nchunks,
        chunknum=chunknum,
        simulation=sim,
        include_fields=list(include_fields),
    )

    cosmo_dsps = flat_wcdm.CosmoParams(
        *(sim.cosmo.Omega_m, sim.cosmo.w0, sim.cosmo.wa, sim.cosmo.h)
    )

    tarr = flat_wcdm.age_at_z(zarr, *cosmo_dsps)
    lgt0 = np.log10(flat_wcdm.age_at_z0(*cosmo_dsps))

    return sim, cosmo_dsps, forest_matrices, zarr, tarr, lgt0


def get_infall_time_indices(
    host_row, is_central, top_host_row, secondary_top_host_row, iz
):
    """Timestep of first infall into penultimate and ultimate hosts, -1 for centrals"""
    _X = top_host_row
    M_ult_host = host_row == _X[:, iz].reshape((-1, 1))
    indx_t_ult_inf_case2 = np.argmax(M_ult_host[:, : iz + 1], axis=1)

    # Was the core identified prior to iz?
    core_only_minus1 = np.all(host_row[:, : iz + 1] == -1, axis=1)

    # Was the core always a central prior to iz?
    is_central_whole_life = ~np.any(
        (host_row[:, : iz + 1] > -1) & (is_central[:, : iz + 1] < 1), axis=1
    )

    # msk_case1: non-satellites
    msk_case1 = core_only_minus1 | is_central_whole_life

    # indx_t_ult_inf = indx_t_ult_inf_case2 for satellites, -1 otherwise
    indx_t_ult_inf = np.where(msk_case1, -1, indx_t_ult_inf_case2)

    _Y = secondary_top_host_row
    M_pen_host = _X == _Y[:, iz].reshape((-1, 1))
    indx_t_pen_inf_case3 = np.argmax(M_pen_host, axis=1)

    # msk_case3: satellites with an existing secondary host at iz
    msk_case3 = ~msk_case1 & (secondary_top_host_row[:, iz] != -1)

    # indx_t_ult_inf = indx_t_ult_inf_case3 for sats-of-sats, -1 otherwise
    indx_t_pen_inf = np.where(msk_case3, indx_t_pen_inf_case3, -1)

    return indx_t_ult_inf, indx_t_pen_inf


def load_diffmah_data_for_forest(drn, subvol, forest):
    """"""
    bname = BNPAT_DIFFMAH.format(subvol)
    fn_diffmah = os.path.join(drn, bname)

    cf_first_row = forest["absolute_row_idx"][0]
    cf_last_row = forest["absolute_row_idx"][-1]

    diffmah_fit_data = load_flat_hdf5(
        fn_diffmah, istart=cf_first_row, iend=cf_last_row + 1
    )
    mah_params = DEFAULT_MAH_PARAMS._make(
        [diffmah_fit_data[key] for key in DEFAULT_MAH_PARAMS._fields]
    )

    return diffmah_fit_data, mah_params


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


def write_sfh_mock_to_disk(diffsky_data, sfh_params, rank_outname):
    with h5py.File(rank_outname, "w") as hdf_out:
        for key in DEFAULT_MAH_PARAMS._fields:
            hdf_out[key] = getattr(diffsky_data["subcat"].mah_params, key)
        for key in sfh_params.ms_params._fields:
            hdf_out[key] = getattr(sfh_params.ms_params, key)
        for key in sfh_params.q_params._fields:
            hdf_out[key] = getattr(sfh_params.q_params, key)


def collate_hdf5_file_collection(fname_collection, fnout):
    fn = fname_collection[0]
    with h5py.File(fn, "r") as hdf:
        mock_keys = list(hdf.keys())

    for key in mock_keys:
        col_data_collector = []
        for fn_in in fname_collection:
            with h5py.File(fn_in, "r") as hdf_in:
                arr = hdf_in[key][...]
                col_data_collector.append(arr)
        complete_arr = np.concatenate(col_data_collector)

        with h5py.File(fnout, "w") as hdf_out:
            hdf_out[key] = complete_arr


def concatenate_diffsky_subcats(subcats):
    """Concatenate a collection of subhalo catalogs"""
    mah_params_collector = []
    for mah_pname in subcats[0].mah_params._fields:
        seq = [getattr(cat.mah_params, mah_pname) for cat in subcats]
        arr = np.concatenate(seq)
        mah_params_collector.append(arr)
    mah_params = subcats[0].mah_params._make(*[mah_params_collector])

    data = dict()
    data["mah_params"] = mah_params

    for pname in subcats[0]._fields[1:]:
        seq = [getattr(cat, pname) for cat in subcats]
        if seq[0].shape == ():
            assert np.allclose(seq, seq[0])
            data[pname] = seq[0]
        else:
            arr = np.concatenate(seq)
            data[pname] = arr

    subcat = subcats[0]._make([*data.values()])

    counts = [len(cat.logmp0) for cat in subcats]
    nskip_per_chunk = np.insert(np.cumsum(counts), 0, 0)[:-1]
    indx_skip = np.repeat(nskip_per_chunk, counts)
    subcat = subcat._replace(ult_host_indx=subcat.ult_host_indx + indx_skip)
    subcat = subcat._replace(pen_host_indx=subcat.pen_host_indx + indx_skip)

    msk_cen = subcat.upids == -1
    upids_correct = subcat.upids + indx_skip
    upids_correct = np.where(msk_cen, -1, upids_correct)
    subcat = subcat._replace(upids=upids_correct)

    return subcat


def write_subcat_to_disk(subcat, fname):
    with h5py.File(fname, "w") as hdf:
        hdf["mah_params"] = np.array([*subcat.mah_params]).T
        for pname, arr in zip(subcat._fields[1:], subcat[1:]):
            hdf[pname] = arr


def load_subcat_from_disk(fname):
    data = dict()
    with h5py.File(fname, "r") as hdf:
        for key in hdf.keys():
            data[key] = hdf[key][...]
    data["mah_params"] = DEFAULT_MAH_PARAMS._make(
        [data["mah_params"][:, i] for i in range(data["mah_params"].shape[1])]
    )
    subcat = SubhaloCatalog._make([data[key] for key in SubhaloCatalog._fields])
    return subcat
