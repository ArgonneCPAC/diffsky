""""""

from collections import namedtuple

import numpy as np
from jax import numpy as jnp
from jax import random as jran
from jax import vmap

from .. import ellipse_proj_kernels as eproj

EllipsoidParams = namedtuple("EllipsoidParams", ("a", "b", "c", "mu", "phi"))


def generate_test_ellipse_params(ran_key, n):

    a = np.ones(n)

    ran_key, b_key = jran.split(ran_key, 2)
    b = a / jran.uniform(b_key, minval=1.0, maxval=100.0, shape=(n,))

    ran_key, c_key = jran.split(ran_key, 2)
    c = b / jran.uniform(c_key, minval=1.0, maxval=100.0, shape=(n,))

    ran_key, mu_key = jran.split(ran_key, 2)
    mu = jran.uniform(mu_key, minval=-1, maxval=1, shape=(n,))

    ran_key, phi_key = jran.split(ran_key, 2)
    phi = jran.uniform(phi_key, minval=0, maxval=2 * np.pi, shape=(n,))

    return EllipsoidParams(a, b, c, mu, phi)


def test_3d_ellipse_params():
    ran_key = jran.key(0)
    npts = 500_000
    a, b, c, mu, phi = generate_test_ellipse_params(ran_key, npts)
    _2d_params = eproj._compute_2d_ellipse_params(a, b, c, mu, phi)
    for p in _2d_params:
        assert np.all(np.isfinite(p))

    A, B, C, Delta = _2d_params
    assert np.all(Delta > 0)

    trace = A + C
    det = A * C - 0.25 * B**2
    discriminant = trace**2 - 4 * det

    _EPS = 1e-7
    assert np.all(trace > -_EPS), trace.min()
    assert np.all(det > -_EPS), det.min()
    assert np.all(discriminant > -_EPS), discriminant.min()

    # Enforce agreement with manually recomputed params above
    # Also enforces existence of fields for A, B, C, Delta
    ellipse2d = eproj.compute_ellipse2d(a, b, c, mu, phi)
    assert np.allclose(ellipse2d.A, A)
    assert np.allclose(ellipse2d.B, B)
    assert np.allclose(ellipse2d.C, C)
    assert np.allclose(ellipse2d.Delta, Delta)


def test_projected_ellipticity_is_correctly_bounded():
    ran_key = jran.key(0)
    npts = 500_000
    a, b, c, mu, phi = generate_test_ellipse_params(ran_key, npts)
    ellipse2d = eproj.compute_ellipse2d(a, b, c, mu, phi)
    assert np.all(np.isfinite(ellipse2d.ellipticity))

    # Enforce beta is finite
    msk_beta_nan = ~np.isfinite(ellipse2d.beta)
    num_beta_nan = msk_beta_nan.sum()
    bad_vals = ellipse2d._make([x[msk_beta_nan] for x in ellipse2d])
    assert num_beta_nan == 0, bad_vals

    # Enforce all returned ellipse params are finite
    for p, pname in zip(ellipse2d, ellipse2d._fields):
        assert np.all(np.isfinite(p)), pname

    assert np.all(ellipse2d.ellipticity >= 0)
    assert np.all(ellipse2d.ellipticity < 1)

    assert np.all(ellipse2d.alpha > 0)
    assert np.all(ellipse2d.beta > 0)

    assert np.all(ellipse2d.alpha <= ellipse2d.beta)
    assert np.any(ellipse2d.alpha < ellipse2d.beta)


def test_calculate_ellipse2d_omega():
    ran_key = jran.key(0)
    npts = 500_000
    a, b, c, mu, phi = generate_test_ellipse_params(ran_key, npts)
    ellipse2d = eproj.compute_ellipse2d(a, b, c, mu, phi)
    assert np.all(np.isfinite(ellipse2d.psi))
    assert np.all(ellipse2d.psi >= -np.pi)
    assert np.all(ellipse2d.psi <= np.pi)

    omega = eproj._calculate_ellipse2d_omega(ellipse2d.A, ellipse2d.B, ellipse2d.C)
    assert np.all(np.isfinite(omega))
    assert np.all(omega >= -np.pi / 2)
    assert np.all(omega <= np.pi / 2)

    assert np.allclose(2 * omega, ellipse2d.psi)


def test_projected_ellipse_axis_vectors():
    ran_key = jran.key(0)
    npts = 50_000
    a, b, c, mu, phi = generate_test_ellipse_params(ran_key, npts)
    ellipse2d = eproj.compute_ellipse2d(a, b, c, mu, phi)

    assert np.allclose(np.sum(ellipse2d.e_beta**2, axis=1), 1.0)
    assert np.allclose(np.sum(ellipse2d.e_alpha**2, axis=1), 1.0)

    psi = np.arctan2(ellipse2d.e_beta[:, 1], ellipse2d.e_beta[:, 0])
    assert np.allclose(psi, ellipse2d.psi)

    dot_vmap = vmap(jnp.dot, in_axes=(0, 0))
    angle = dot_vmap(ellipse2d.e_beta, ellipse2d.e_alpha)
    assert np.allclose(angle, 0.0, atol=1e-4)
