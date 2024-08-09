"""
"""

import numpy as np
from jax import random as jran

from ..mc_diffmah_tpeak import mc_subhalo_catalog_singlez


def test_mc_subhalo_catalog_singlez():

    ran_key = jran.PRNGKey(0)
    lgmp_min = 11.0
    redshift = 0.5
    Lbox = 25.0
    volume_com = Lbox**3
    args = ran_key, lgmp_min, redshift, volume_com

    subhalo_catalog = mc_subhalo_catalog_singlez(*args)
    for x in subhalo_catalog:
        assert np.all(np.isfinite(x))
