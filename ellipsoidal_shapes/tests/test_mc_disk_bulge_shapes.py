""""""

import numpy as np
from jax import random as jran

from .. import mc_disk_bulge_shapes as mcdbs


def test_mc_disk_bulge_ellipsoids():
    ran_key = jran.key(0)
    n = 10_000

    size_key, shape_key = jran.split(ran_key, 2)
    r50 = jran.uniform(size_key, minval=0.5, maxval=3.0, shape=(2 * n,))
    r50_disk = r50[:n]
    r50_bulge = r50[n:]
    disk_ellipse, bulge_ellipse = mcdbs.mc_disk_bulge_ellipsoids(
        shape_key, r50_disk, r50_bulge
    )
    for ellipse in (disk_ellipse, bulge_ellipse):
        assert np.all(ellipse.psi >= -np.pi)
        assert np.all(ellipse.psi <= np.pi)
