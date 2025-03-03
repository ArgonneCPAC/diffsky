"""
"""

import numpy as np
from dsps.data_loaders import retrieve_fake_fsps_data as rffd
from jax import random as jran

from .. import mc_diffsky_phot as mcd


def test_mc_diffstar_galhalo_pop():
    ran_key = jran.key(0)
    lgmp_min = 11.0
    z_obs = 0.2
    Lbox = 50.0
    args = (ran_key, lgmp_min, z_obs, Lbox)

    diffstar_data = mcd.mc_diffstar_galhalo_pop(*args)

    for p in diffstar_data["subcat"].mah_params:
        assert np.all(np.isfinite(p))


def test_mc_diffsky_galhalo_pop():
    ran_key = jran.key(0)
    lgmp_min = 11.0
    z_obs = 0.2
    Lbox = 50.0
    ssp_data = rffd.load_fake_ssp_data()
    args = (ran_key, lgmp_min, z_obs, Lbox, ssp_data)

    diffsky_data = mcd.mc_diffsky_galhalo_pop(*args)

    n_gals, n_age = diffsky_data["bursty_age_weights"].shape
    assert diffsky_data["subcat"].logmp0.shape == (n_gals,)
    assert ssp_data.ssp_lg_age_gyr.shape == (n_age,)
    assert np.all(np.isfinite(diffsky_data["bursty_age_weights"]))
    n_gals, n_met = diffsky_data["lgmet_weights"].shape

    assert np.all(np.isfinite(diffsky_data["bursty_ssp_weights"]))
    assert diffsky_data["bursty_ssp_weights"].shape == (n_gals, n_met, n_age)
