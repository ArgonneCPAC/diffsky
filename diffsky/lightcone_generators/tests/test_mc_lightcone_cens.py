""""""

import numpy as np
from jax import random as jran

from .. import mc_lightcone_cens as mclcc


def test_weighted_lc_halos_sfh():
    ran_key = jran.key(0)
    n_host_halos = 100
    z_min, z_max = 0.05, 3.0
    lgmp_min, lgmp_max = 10.0, 15.0
    sky_area_degsq = 100.0
    args = (
        ran_key,
        n_host_halos,
        z_min,
        z_max,
        lgmp_min,
        lgmp_max,
        sky_area_degsq,
    )
    sfh_lightcone = mclcc.weighted_lc_halos_sfh(*args)
    for x in sfh_lightcone:
        assert np.all(np.isfinite(x))
