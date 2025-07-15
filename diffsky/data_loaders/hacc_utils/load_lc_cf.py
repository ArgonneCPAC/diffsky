""""""

import os
from collections import namedtuple

import numpy as np
from diffmah import DEFAULT_MAH_PARAMS
from diffmah.diffmahpop_kernels.bimod_censat_params import DEFAULT_DIFFMAHPOP_PARAMS
from diffmah.diffmahpop_kernels.mc_bimod_cens import mc_diffmah_cenpop
from diffmah.diffmahpop_kernels.mc_bimod_sats import mc_diffmah_satpop
from dsps.cosmology import flat_wcdm
from jax import random as jran

from ...data_loaders import load_flat_hdf5
from . import defaults as hacc_defaults

try:
    from haccytrees import Simulation as HACCSim

    HAS_HACCYTREES = True
except ImportError:
    HAS_HACCYTREES = False

SIM_INFO_KEYS = ("sim", "cosmo_params", "z_sim", "t_sim", "lgt0", "fb", "num_subvols")
DiffskySimInfo = namedtuple("DiffskySimInfo", SIM_INFO_KEYS)


def get_diffsky_info_from_hacc_sim(sim_name):
    sim = HACCSim.simulations[sim_name]

    cosmo_params = flat_wcdm.CosmoParams(
        *(sim.cosmo.Omega_m, sim.cosmo.w0, sim.cosmo.wa, sim.cosmo.h)
    )
    z_sim = np.array(sim.step2z(np.array(sim.cosmotools_steps)))
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


def load_lc_diffsky_patch_data(fn_lc_diffsky, indir_lc_data):
    diffsky_data = load_flat_hdf5(fn_lc_diffsky)

    bn_in = os.path.basename(fn_lc_diffsky)
    bn_lc = os.path.basename(bn_in).replace(".diffsky_data.hdf5", ".hdf5")
    fn_lc = os.path.join(indir_lc_data, bn_lc)
    lc_data = load_flat_hdf5(fn_lc)

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


def get_imputed_mah_params(ran_key, diffsky_data, lc_data, lgt0):
    msk_has_diffmah_fit = get_diffmah_has_fit_mask(diffsky_data)
    msk_nofit = ~msk_has_diffmah_fit

    num_nofit = msk_nofit.sum()
    if num_nofit == 0:
        mah_params = [diffsky_data[key] for key in DEFAULT_MAH_PARAMS._fields]
        mah_params = DEFAULT_MAH_PARAMS._make(mah_params)
    else:
        t_obs_nofit = lc_data["t_obs"][msk_nofit]
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
