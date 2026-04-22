""""""

import os
from collections import namedtuple

import h5py
import numpy as np
from diffmah import DEFAULT_MAH_PARAMS, logmh_at_t_obs
from diffmah.diffmahpop_kernels.bimod_censat_params import DEFAULT_DIFFMAHPOP_PARAMS
from diffmah.diffmahpop_kernels.mc_bimod_cens import mc_diffmah_cenpop
from diffmah.diffmahpop_kernels.mc_bimod_sats import mc_diffmah_satpop
from dsps.cosmology import flat_wcdm
from jax import random as jran

from ...data_loaders import load_flat_hdf5
from . import defaults as hacc_defaults
from . import haccsims

SIM_INFO_KEYS = ("sim", "cosmo_params", "z_sim", "t_sim", "lgt0", "fb", "num_subvols")
DiffskySimInfo = namedtuple("DiffskySimInfo", SIM_INFO_KEYS)

try:
    import haccytrees  # noqa

    HAS_HACCYTREES = True
except ImportError:
    HAS_HACCYTREES = False


def get_diffsky_info_from_hacc_sim(sim_name):
    sim = haccsims.simulations[sim_name]

    cosmo_params = flat_wcdm.CosmoParams(
        *(sim.cosmo.Omega_m, sim.cosmo.w0, sim.cosmo.wa, sim.cosmo.h)
    )
    z_sim = sim.redshifts
    t_sim = flat_wcdm.age_at_z(z_sim, *cosmo_params)

    t0 = flat_wcdm.age_at_z0(*cosmo_params)
    lgt0 = np.log10(t0)
    fb = sim.cosmo.Omega_b / sim.cosmo.Omega_m

    if sim_name == "LastJourney":
        num_subvols = 256
    elif sim_name in ("DiscoveryLCDM", "DiscoveryW0WA"):
        num_subvols = 96
    else:
        num_subvols = np.nan

    diffsky_info = DiffskySimInfo(
        sim, cosmo_params, z_sim, t_sim, lgt0, fb, num_subvols
    )

    return diffsky_info


def load_lc_diffsky_patch_data(fn_lc_diffsky, indir_lc_data, *, istart=0, iend=None):
    diffsky_data = load_flat_hdf5(fn_lc_diffsky, istart=istart, iend=iend)

    bn_in = os.path.basename(fn_lc_diffsky)
    bn_lc = os.path.basename(bn_in).replace(".diffsky_data.hdf5", ".hdf5")
    fn_lc = os.path.join(indir_lc_data, bn_lc)
    lc_data = load_flat_hdf5(fn_lc, istart=istart, iend=iend, dataset="data")

    lc_data["redshift_true"] = 1 / lc_data["scale_factor"] - 1

    assert lc_data["redshift_true"].shape[0] == diffsky_data["logm0"].shape[0]

    return lc_data, diffsky_data


def collect_lc_diffsky_data(fn_list, drn_lc_data=None):
    drn_diffsky_data = os.path.dirname(fn_list[0])
    if drn_lc_data is None:
        drn_lc_data = drn_diffsky_data

    diffsky_data_collector = []
    lc_data_collector = []
    for fn in fn_list:
        lc_diffsky_data = load_flat_hdf5(fn)
        diffsky_data_collector.append(lc_diffsky_data)

        bn_lc = os.path.basename(fn).replace(".diffsky_data.hdf5", ".hdf5")
        fn_lc = os.path.join(drn_lc_data, bn_lc)
        lc_data = load_flat_hdf5(fn_lc)
        lc_data_collector.append(lc_data)

    diffsky_data = dict()
    for key in diffsky_data_collector[0].keys():
        diffsky_data[key] = np.concatenate([x[key] for x in diffsky_data_collector])

    lc_data = dict()
    for key in lc_data_collector[0].keys():
        lc_data[key] = np.concatenate([x[key] for x in lc_data_collector])

    lc_data["redshift_true"] = 1 / lc_data["scale_factor"] - 1

    assert lc_data["redshift_true"].shape[0] == diffsky_data["logm0"].shape[0]

    return lc_data, diffsky_data


def get_imputed_mah_params(ran_key, diffsky_data, lgt0):
    msk_has_diffmah_fit = get_diffmah_has_fit_mask(diffsky_data)
    msk_nofit = ~msk_has_diffmah_fit

    num_nofit = msk_nofit.sum()
    if num_nofit == 0:
        mah_params = [diffsky_data[key] for key in DEFAULT_MAH_PARAMS._fields]
        mah_params = DEFAULT_MAH_PARAMS._make(mah_params)
    else:
        t_obs_nofit = diffsky_data["t_obs"][msk_nofit]
        lgmp_obs_nofit = np.log10(diffsky_data["mp0"][msk_nofit])
        is_central = np.ones(msk_nofit.sum()).astype(int)

        fake_mah_params = generate_fake_mah_params(
            ran_key, t_obs_nofit, lgmp_obs_nofit, is_central, lgt0
        )

        mah_params = []
        for pname in DEFAULT_MAH_PARAMS._fields:
            arr = np.copy(diffsky_data[pname])
            arr[msk_nofit] = getattr(fake_mah_params, pname)
            mah_params.append(arr)
        mah_params = DEFAULT_MAH_PARAMS._make(mah_params)

    return mah_params, msk_has_diffmah_fit


def get_diffmah_has_fit_mask(diffsky_data, npts_mah_min=hacc_defaults.N_MIN_MAH_PTS):
    msk_npts = diffsky_data["n_points_per_fit"] > npts_mah_min
    msk_loss = diffsky_data["loss"] > 0
    msk_loss &= diffsky_data["loss"] < 1
    msk_logm0 = diffsky_data["logm0"] > 0
    msk_has_diffmah_fit = msk_npts & msk_loss & msk_logm0
    return msk_has_diffmah_fit


def generate_fake_mah_params(ran_key, t_obs, lgmp_obs, is_central, lgt0):
    cen_key, sat_key = jran.split(ran_key, 2)

    cenpop = mc_diffmah_cenpop(
        DEFAULT_DIFFMAHPOP_PARAMS, lgmp_obs, t_obs, cen_key, lgt0
    )
    satpop = mc_diffmah_satpop(
        DEFAULT_DIFFMAHPOP_PARAMS, lgmp_obs, t_obs, sat_key, lgt0
    )

    gen = zip(cenpop.mah_params, satpop.mah_params)
    _mah_params = [np.where(is_central, x, y) for x, y in gen]
    fake_mah_params = DEFAULT_MAH_PARAMS._make(_mah_params)

    return fake_mah_params


def _read_lc_cores_chunk(fobj, nchunks, chunknum, keys_to_read, index_dataset=None):
    """Read a forest-complete chunk of data from lc_cores"""

    if index_dataset is None:
        index_dataset = fobj
    else:
        index_dataset = fobj[index_dataset]

    nindex = len(index_dataset["index"]["offset"])
    nstart = (nindex // nchunks) * chunknum
    nend = (nindex // nchunks) * (chunknum + 1)

    read_start = index_dataset["index"]["offset"][nstart]
    if chunknum == nchunks - 1:
        read_end = (
            index_dataset["index"]["offset"][-1] + index_dataset["index"]["count"][-1]
        )
    else:
        read_end = index_dataset["index"]["offset"][nend]

    lc_cores_chunk = {}
    for key in keys_to_read:
        lc_cores_chunk[key] = fobj["data"][key][read_start:read_end]

    # shift look-up-indices for the chunk
    lc_cores_chunk["top_host_idx_chunk"] = lc_cores_chunk["top_host_idx"] - read_start
    lc_cores_chunk["secondary_top_host_idx_chunk"] = (
        lc_cores_chunk["secondary_top_host_idx"] - read_start
    )

    return lc_cores_chunk, (read_start, read_end)


def load_lc_cf_chunk(fn_lc_cf, drn_lc_cores, *, nchunks, chunknum, lc_cores_keys=None):
    bn_lc_cf = os.path.basename(fn_lc_cf)
    bn_lc_cores = os.path.basename(bn_lc_cf).replace(".diffsky_data.hdf5", ".hdf5")
    fn_lc_cores = os.path.join(drn_lc_cores, bn_lc_cores)

    with h5py.File(fn_lc_cores, "r") as hdf:
        if lc_cores_keys is None:
            lc_cores_keys = list(hdf["data"].keys())

        lc_data, (istart, iend) = _read_lc_cores_chunk(
            hdf, nchunks, chunknum, lc_cores_keys
        )

    diffsky_data = load_flat_hdf5(fn_lc_cf, istart=istart, iend=iend)

    lc_data["redshift_true"] = 1.0 / lc_data["scale_factor"] - 1.0

    assert lc_data["redshift_true"].shape[0] == diffsky_data["logm0"].shape[0]

    return lc_data, diffsky_data


def load_lc_mock_chunk(fn_lc_mock, *, nchunks, chunknum, lc_mock_keys=None):
    with h5py.File(fn_lc_mock, "r") as hdf:
        if lc_mock_keys is None:
            lc_mock_keys = list(hdf["data"].keys())

        lc_mock, (istart, iend) = _read_lc_cores_chunk(
            hdf, nchunks, chunknum, lc_mock_keys, index_dataset="metadata"
        )

    return lc_mock, (istart, iend)


def compute_additional_haloprops(
    diffsky_data, sim_info, halo_indx=None, sec_halo_indx=None
):
    """"""
    n_gals = len(diffsky_data["t_peak"])
    additional_haloprops = dict()
    additional_haloprops["nhalos_weights"] = np.ones(n_gals).astype("float")
    additional_haloprops["t_infall"] = diffsky_data["t_peak"]

    if halo_indx is None:
        additional_haloprops["halo_indx"] = diffsky_data["top_host_idx_chunk"]
    else:
        additional_haloprops["halo_indx"] = halo_indx

    if sec_halo_indx is None:
        additional_haloprops["sec_halo_indx"] = diffsky_data[
            "secondary_top_host_idx_chunk"
        ]
    else:
        additional_haloprops["sec_halo_indx"] = sec_halo_indx

    mah_params = DEFAULT_MAH_PARAMS._make(
        [diffsky_data[key] for key in DEFAULT_MAH_PARAMS._fields]
    )
    mah_params_host = DEFAULT_MAH_PARAMS._make(
        [
            diffsky_data[key][additional_haloprops["halo_indx"]]
            for key in DEFAULT_MAH_PARAMS._fields
        ]
    )
    additional_haloprops["logmp_infall"] = logmh_at_t_obs(
        mah_params, diffsky_data["t_peak"], sim_info.lgt0
    )
    additional_haloprops["logmhost_infall"] = logmh_at_t_obs(
        mah_params_host, diffsky_data["t_peak"], sim_info.lgt0
    )

    diffsky_data.update(additional_haloprops)
    return diffsky_data
