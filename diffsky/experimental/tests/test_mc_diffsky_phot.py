"""
"""

import numpy as np
from dsps.data_loaders import retrieve_fake_fsps_data as rffd
from jax import random as jran

from .. import mc_diffsky_phot as mcdp


def test_mc_diffsky_galpop_lsst_phot():
    ran_key = jran.key(0)
    lgmp_min = 12.0
    z_obs = 0.2
    Lbox = 20.0
    ssp_data = rffd.load_fake_ssp_data()

    args = (ran_key, z_obs, lgmp_min, ssp_data)
    diffsky_data = mcdp.mc_diffsky_galpop_lsst_phot(
        *args, volume_com=Lbox**3, drn_ssp_data=None, return_internal_quantities=True
    )
    for key in diffsky_data.keys():
        if "rest_ugrizy" in key:
            assert np.all(np.isfinite(diffsky_data[key]))

    assert np.all(np.isfinite(diffsky_data["frac_trans_nonoise"]))
    assert np.all(diffsky_data["frac_trans_nonoise"] >= 0)
    assert np.all(diffsky_data["frac_trans_nonoise"] <= 1)
    assert np.all(np.isfinite(diffsky_data["frac_trans_noisy"]))
    assert np.all(diffsky_data["frac_trans_noisy"] >= 0)
    assert np.all(diffsky_data["frac_trans_noisy"] <= 1)
