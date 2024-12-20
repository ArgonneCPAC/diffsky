"""
"""

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
    args = ran_key, lgmp_min, redshift, volume_com

    subcat = mcd.mc_subhalos(*args)
    for x in subcat:
        assert np.all(np.isfinite(x))

    n_gals = subcat.logmp_pen_inf.size
    assert subcat.logmp_pen_inf.shape == (n_gals,)
    for mah_p in subcat.mah_params:
        assert mah_p.shape == (n_gals,)


def test_mc_subhalo_catalog_colnames_are_stable():
    """Changing colnames of SubhaloCatalog may break something downstream"""
    for key in _EXPECTED_SUBCAT_COLNAMES:
        msg = f"`{key}` missing from mc_diffmah_tpeak.SubhaloCatalog"
        assert key in mcd.SubhaloCatalog._fields, msg
