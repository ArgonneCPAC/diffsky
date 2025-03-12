""" """

import numpy as np
from dsps.data_loaders import retrieve_fake_fsps_data as rffd
from jax import random as jran

from .. import mc_diffsky_phot as mcdp


def test_mc_diffsky_galpop_lsst_phot():
    ran_key = jran.key(0)
    lgmp_min = 12.0
    z_obs = 0.2
    Lbox = 20.0
    volume_com = Lbox**3
    ssp_data = rffd.load_fake_ssp_data()

    args = (ran_key, z_obs, lgmp_min, volume_com, ssp_data)
    diffsky_data = mcdp.mc_diffsky_galpop_lsst_phot(
        *args, drn_ssp_data=None, return_internal_quantities=True
    )
    for key in diffsky_data.keys():
        if "rest_ugrizy" in key:
            assert np.all(np.isfinite(diffsky_data[key]))

    assert np.all(np.isfinite(diffsky_data["frac_trans_nonoise_rest"]))
    assert np.all(diffsky_data["frac_trans_nonoise_rest"] >= 0)
    assert np.all(diffsky_data["frac_trans_nonoise_rest"] <= 1)
    assert np.all(np.isfinite(diffsky_data["frac_trans_noisy_rest"]))
    assert np.all(diffsky_data["frac_trans_noisy_rest"] >= 0)
    assert np.all(diffsky_data["frac_trans_noisy_rest"] <= 1)


def test_mc_diffsky_cenpop_lsst_phot():
    ran_key = jran.key(0)
    z_obs = 0.2
    hosts_logmh_at_z = np.linspace(10, 15, 300)
    ssp_data = rffd.load_fake_ssp_data()

    args = (ran_key, z_obs, hosts_logmh_at_z, ssp_data)
    diffsky_data = mcdp.mc_diffsky_cenpop_lsst_phot(
        *args, drn_ssp_data=None, return_internal_quantities=True
    )
    for key in diffsky_data.keys():
        if "ugrizy" in key:
            assert np.all(np.isfinite(diffsky_data[key]))

    assert np.all(np.isfinite(diffsky_data["frac_trans_nonoise_rest"]))
    assert np.all(diffsky_data["frac_trans_nonoise_rest"] >= 0)
    assert np.all(diffsky_data["frac_trans_nonoise_rest"] <= 1)
    assert np.all(np.isfinite(diffsky_data["frac_trans_noisy_rest"]))
    assert np.all(diffsky_data["frac_trans_noisy_rest"] >= 0)
    assert np.all(diffsky_data["frac_trans_noisy_rest"] <= 1)

    n_gals = diffsky_data["sfh"].shape[0]
    assert n_gals == hosts_logmh_at_z.size

    rest_keys = [key for key in diffsky_data.keys() if "rest_ugrizy" in key]
    for key in rest_keys:
        obs_key = key.replace("rest_", "obs_")
        assert obs_key in diffsky_data.keys()
