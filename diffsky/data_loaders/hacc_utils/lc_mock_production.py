""" """

import h5py
import numpy as np
from diffmah import DEFAULT_MAH_PARAMS, logmh_at_t_obs
from diffstar import DEFAULT_DIFFSTAR_PARAMS
from diffstarpop import mc_diffstar_sfh_galpop
from diffstarpop.defaults import DEFAULT_DIFFSTARPOP_PARAMS
from diffstarpop.param_utils import mc_select_diffstar_params
from dsps.cosmology import flat_wcdm

from ...utils.sfh_utils import get_logsm_logssfr_at_t_obs
from . import load_lc_cf

LC_CF_BNPAT = "lc_cores-{0}.{1}.diffsky_data.hdf5"
LC_MOCK_BNPAT = LC_CF_BNPAT.replace("diffsky_data", "diffsky_gals")


def write_lc_sfh_mock_to_disk(fnout, lc_data, diffsky_data):
    with h5py.File(fnout, "w") as hdf_out:

        # Write diffmah params
        for key in DEFAULT_MAH_PARAMS._fields:
            hdf_out[key] = diffsky_data[key]
        hdf_out["has_diffmah_fit"] = diffsky_data["has_diffmah_fit"]
        hdf_out["logmp0"] = diffsky_data["logmp0"]
        hdf_out["logmp_obs"] = diffsky_data["logmp_obs"]

        # Write diffstar params
        for key in DEFAULT_DIFFSTAR_PARAMS.ms_params._fields:
            hdf_out[key] = diffsky_data[key]
        for key in DEFAULT_DIFFSTAR_PARAMS.q_params._fields:
            hdf_out[key] = diffsky_data[key]
        hdf_out["logsm_obs"] = diffsky_data["logsm_obs"]
        hdf_out["logssfr_obs"] = diffsky_data["logssfr_obs"]

        hdf_out["z_obs"] = lc_data["z_obs"]
        hdf_out["ra"] = lc_data["phi"]
        hdf_out["dec"] = np.pi / 2.0 - lc_data["theta"]
        hdf_out["snapnum"] = lc_data["snapnum"]


def add_sfh_quantities_to_mock(sim_info, lc_data, diffsky_data, ran_key):
    lc_data["t_obs"] = flat_wcdm.age_at_z(lc_data["z_obs"], *sim_info.cosmo_params)

    mah_params, msk_has_diffmah_fit = load_lc_cf.get_imputed_mah_params(
        ran_key, diffsky_data, lc_data, sim_info.lgt0
    )
    for pname, pval in zip(mah_params._fields, mah_params):
        diffsky_data[pname] = pval
    diffsky_data["has_diffmah_fit"] = msk_has_diffmah_fit

    logmp0 = logmh_at_t_obs(
        mah_params, np.zeros(mah_params.logm0.size) + 10**sim_info.lgt0, sim_info.lgt0
    )
    diffsky_data["logmp0"] = logmp0

    logmp_obs = logmh_at_t_obs(
        mah_params, np.zeros(mah_params.logm0.size) + lc_data["t_obs"], sim_info.lgt0
    )
    diffsky_data["logmp_obs"] = logmp_obs

    lgmu_infall = np.zeros_like(logmp0)
    logmhost_infall = np.copy(logmp0)
    gyr_since_infall = np.zeros_like(logmp0)
    upids = np.where(lc_data["central"] == 1, -1, 0)

    t_table = np.linspace(0.1, 10**sim_info.lgt0, 100)

    args = (
        DEFAULT_DIFFSTARPOP_PARAMS,
        mah_params,
        logmp0,
        upids,
        lgmu_infall,
        logmhost_infall,
        gyr_since_infall,
        ran_key,
        t_table,
    )

    _res = mc_diffstar_sfh_galpop(*args)
    sfh_params_ms, sfh_params_q, sfh_ms, sfh_q, frac_q, mc_is_q = _res

    sfh = np.where(mc_is_q.reshape((-1, 1)), sfh_q, sfh_ms)
    sfh_params = mc_select_diffstar_params(sfh_params_q, sfh_params_ms, mc_is_q)

    for key in sfh_params.ms_params._fields:
        diffsky_data[key] = getattr(sfh_params.ms_params, key)
    for key in sfh_params.q_params._fields:
        diffsky_data[key] = getattr(sfh_params.q_params, key)

    logsm_obs, logssfr_obs = get_logsm_logssfr_at_t_obs(lc_data["t_obs"], t_table, sfh)
    diffsky_data["logsm_obs"] = logsm_obs
    diffsky_data["logssfr_obs"] = logssfr_obs

    return lc_data, diffsky_data
