""" """

import numpy as np
from jax import random as jran

from .. import mc_diffmah_tpeak as mcd

_EXPECTED_SUBCAT_COLNAMES = (
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
)


def test_mc_subhalo_catalog_singlez():

    ran_key = jran.PRNGKey(0)
    lgmp_min = 11.0
    redshift = 0.5
    Lbox = 25.0
    volume_com = Lbox**3
    args = ran_key, redshift, lgmp_min, volume_com

    subcat = mcd.mc_subhalos(*args)
    for x in subcat:
        assert np.all(np.isfinite(x))

    n_gals = subcat.logmp_pen_inf.size
    assert subcat.logmp_pen_inf.shape == (n_gals,)
    for mah_p in subcat.mah_params:
        assert mah_p.shape == (n_gals,)

    logmp_ult_inf_sats = subcat.logmp_ult_inf[subcat.upids != -1]
    logmhost_ult_inf_sats = subcat.logmhost_ult_inf[subcat.upids != -1]
    logmu_ult_sats = logmp_ult_inf_sats - logmhost_ult_inf_sats
    assert np.mean(logmu_ult_sats) < -0.5

    sat_bigger_than_host = (
        subcat.logmp_ult_inf[subcat.upids != -1]
        > subcat.logmhost_ult_inf[subcat.upids != -1]
    )
    assert np.mean(sat_bigger_than_host) < 0.2


def test_mc_subhalo_catalog_colnames_are_stable():
    """Changing colnames of SubhaloCatalog may break something downstream"""
    for key in _EXPECTED_SUBCAT_COLNAMES:
        msg = f"`{key}` missing from mc_diffmah_tpeak.SubhaloCatalog"
        assert key in mcd.SubhaloCatalog._fields, msg


def test_mc_subhalo_catalog_input_logmh_grid():
    ran_key = jran.PRNGKey(0)
    lgmp_min = 11.0
    redshift = 0.5

    n_hosts = 250
    hosts_logmh_at_z = np.linspace(lgmp_min, 15, n_hosts)
    args = ran_key, redshift, lgmp_min
    subcat = mcd.mc_subhalos(*args, hosts_logmh_at_z=hosts_logmh_at_z)
    for x in subcat:
        assert np.all(np.isfinite(x))


def test_mc_host_halos():

    ran_key = jran.PRNGKey(0)
    redshift = 0.5
    Lbox = 25.0
    args = ran_key, redshift

    subcat = mcd.mc_host_halos(*args, lgmp_min=11, volume_com=Lbox**3)
    for x in subcat:
        assert np.all(np.isfinite(x))

    n_gals = subcat.logmp_pen_inf.size
    assert subcat.logmp_pen_inf.shape == (n_gals,)
    for mah_p in subcat.mah_params:
        assert mah_p.shape == (n_gals,)

    n_cens = 200
    hosts_logmh_at_z = np.linspace(10, 15, n_cens)
    subcat = mcd.mc_host_halos(*args, hosts_logmh_at_z=hosts_logmh_at_z)
    assert subcat.logmp0.size == n_cens
