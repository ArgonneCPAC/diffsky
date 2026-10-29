""""""

import numpy as np
from jax import random as jran

from .. import mc_disk_bulge_shapes as mcdbs


def test_mc_disk_bulge_ellipsoids():
    ran_key = jran.key(0)
    n = 10_000

    size_key, shape_key, x_key, y_key, z_key = jran.split(ran_key, 5)
    r50 = jran.uniform(size_key, minval=0.5, maxval=3.0, shape=(2 * n,))
    r50_disk = r50[:n]
    r50_bulge = r50[n:]
    pos_x = jran.uniform(x_key, minval=-100, maxval=100, shape=(n,))
    pos_y = jran.uniform(y_key, minval=-100, maxval=100, shape=(n,))
    pos_z = jran.uniform(z_key, minval=100, maxval=200, shape=(n,))
    disk_ellipse, bulge_ellipse = mcdbs.mc_disk_bulge_ellipsoids(
        shape_key,
        r50_disk,
        r50_bulge,
        pos_x,
        pos_y,
        pos_z,
        envelop=True,
        ellipticity_type=0,
    )

    for ellipse in (disk_ellipse, bulge_ellipse):
        assert np.all(ellipse.psi >= -np.pi / 2.)
        assert np.all(ellipse.psi <= np.pi / 2.)
        assert np.all(ellipse.ellipticity <= 1.0)
        assert np.all(ellipse.ellipticity >= 0.0)
        assert np.allclose(np.sum(ellipse.e_alpha * np.array([1, 0])[np.newaxis, :], axis=-1), np.cos(ellipse.psi))
        assert np.allclose(np.sum(ellipse.e_beta * np.array([1, 0])[np.newaxis, :], axis=-1), np.cos(ellipse.psi + np.pi / 2.0))
        assert np.all(ellipse.mu_inclination >= 0)
        assert np.all(ellipse.mu_inclination <= 1)
        assert np.all(ellipse.b_over_a <= 1) & np.all(ellipse.c_over_a > 0)
        assert np.all(ellipse.c_over_a <= ellipse.b_over_a)

        # without intrinsic alignment the orientations are isotropic, so
        # μ = cos(i) is uniform in [0, 1] (Kolmogorov-Smirnov test, 99.9%
        # confidence with a Bonferroni correction for the two components)
        mu = np.sort(np.asarray(ellipse.mu_inclination))
        ecdf = np.arange(1, n + 1) / n
        ks_distance = max(np.max(ecdf - mu), np.max(mu - (ecdf - 1 / n)))
        assert ks_distance < np.sqrt(-np.log(1e-3 / 2 / 2) / 2) / np.sqrt(n)
