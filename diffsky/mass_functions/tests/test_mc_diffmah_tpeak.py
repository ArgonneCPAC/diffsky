"""
"""

import numpy as np
from jax import random as jran

from ..mc_diffmah_tpeak import mc_subhalos


def test_mc_subhalo_catalog_singlez():

    ran_key = jran.PRNGKey(0)
    lgmp_min = 11.0
    redshift = 0.5
    Lbox = 25.0
    volume_com = Lbox**3
    args = ran_key, lgmp_min, redshift, volume_com

    subcat = mc_subhalos(*args)
    for x in subcat:
        assert np.all(np.isfinite(x))

    n_gals = subcat.lgmp_pen_inf.size
    assert subcat.lgmp_pen_inf.shape == (n_gals,)
    for mah_p in subcat.mah_params:
        assert mah_p.shape == (n_gals,)
