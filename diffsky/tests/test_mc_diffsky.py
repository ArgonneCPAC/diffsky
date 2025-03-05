"""
"""

import numpy as np
from dsps.data_loaders import retrieve_fake_fsps_data as rffd
from jax import random as jran

from .. import mc_diffsky as mcd


def test_mc_diffstar_galpop():
    ran_key = jran.key(0)
    n_cens_input = 200
    hosts_logmh_at_z = np.linspace(10, 15, n_cens_input)
    lgmp_min = 11.0
    z_obs = 0.01
    args = (ran_key, z_obs, lgmp_min)
    diffsky_data = mcd.mc_diffstar_galpop(
        *args, hosts_logmh_at_z=hosts_logmh_at_z, return_internal_quantities=True
    )

    for p in diffsky_data["subcat"].mah_params:
        assert np.all(np.isfinite(p))
    assert np.all(np.isfinite(diffsky_data["smh"]))

    n_cens_subcat = np.sum(diffsky_data["subcat"].upids == -1)
    assert n_cens_subcat == n_cens_input
    n_sats_subcat = np.sum(diffsky_data["subcat"].upids != -1)
    assert n_sats_subcat > 0

    assert diffsky_data["t_obs"] > 13.5

    # Enforce return_internal_quantities supplies additional info
    for key in ("sfh_params_q", "sfh_ms", "frac_q"):
        assert key in diffsky_data.keys()


def test_mc_diffstar_cenpop():
    ran_key = jran.key(0)
    n_cens = 200
    hosts_logmh_at_z = np.linspace(10, 15, n_cens)
    z_obs = 0.1
    args = (ran_key, z_obs)
    diffsky_data = mcd.mc_diffstar_cenpop(
        *args, hosts_logmh_at_z=hosts_logmh_at_z, return_internal_quantities=True
    )

    for p in diffsky_data["subcat"].mah_params:
        assert np.all(np.isfinite(p))
    assert np.all(np.isfinite(diffsky_data["smh"]))

    assert diffsky_data["subcat"].logmp0.size == n_cens

    # Enforce return_internal_quantities supplies additional info
    for key in ("sfh_params_q", "sfh_ms", "frac_q"):
        assert key in diffsky_data.keys()


def test_mc_diffsky_galhalo_pop():
    ran_key = jran.key(0)
    lgmp_min = 12.0
    z_obs = 0.2
    Lbox = 20.0
    ssp_data = rffd.load_fake_ssp_data()

    args = (ran_key, z_obs, lgmp_min, ssp_data)
    diffsky_data, gal_flux_table_nodust, frac_trans = mcd.mc_diffsky_lsst_photpop(
        *args, volume_com=Lbox**3, drn_ssp_data=None
    )
    for key in (
        "rest_ugrizy_smooth_nodust",
        "rest_ugrizy_bursty_nodust",
        "rest_ugrizy_smooth_dust",
        "rest_ugrizy_bursty_dust",
    ):
        assert np.all(np.isfinite(diffsky_data[key]))
