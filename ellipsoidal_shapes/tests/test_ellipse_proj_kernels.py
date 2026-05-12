""""""

from collections import namedtuple

import numpy as np
from jax import numpy as jnp
from jax import random as jran
from jax import vmap

from .. import ellipse_proj_kernels as eproj

EllipsoidParams = namedtuple("EllipsoidParams", ("a", "b", "c", "mu", "phi"))


def generate_test_ellipse_params(ran_key, n, ba_ratio_max=100.0, cb_ratio_max=100.0):

    a = np.ones(n)

    ran_key, b_key = jran.split(ran_key, 2)
    b = a / jran.uniform(b_key, minval=1.1, maxval=ba_ratio_max, shape=(n,))

    ran_key, c_key = jran.split(ran_key, 2)
    c = b / jran.uniform(c_key, minval=1.1, maxval=cb_ratio_max, shape=(n,))

    ran_key, mu_key = jran.split(ran_key, 2)
    mu = jran.uniform(mu_key, minval=-1, maxval=1, shape=(n,))

    ran_key, phi_key = jran.split(ran_key, 2)
    phi = jran.uniform(phi_key, minval=0, maxval=2 * np.pi, shape=(n,))

    return EllipsoidParams(a, b, c, mu, phi)


def test_3d_ellipse_params():
    """Reimplement _calculate_ellipse2d_axes and manually enforce agreement"""
    ran_key = jran.key(0)
    npts = 500_000
    a, b, c, mu, phi = generate_test_ellipse_params(ran_key, npts)
    _2d_params = eproj._compute_2d_ellipse_params(a, b, c, mu, phi)
    for p in _2d_params:
        assert np.all(np.isfinite(p))

    A, B, C = _2d_params

    trace = A + C
    det = A * C - 0.25 * B**2
    discriminant = trace**2 - 4 * det

    _EPS = 1e-7
    assert np.all(trace > -_EPS), trace.min()
    assert np.all(det > -_EPS), det.min()
    assert np.all(discriminant > -_EPS), discriminant.min()

    # Enforce agreement with manually recomputed params above
    # Also enforces existence of fields for A, B, C
    ellipse2d = eproj.compute_ellipse2d(a, b, c, mu, phi)
    assert np.allclose(ellipse2d.A, A)
    assert np.allclose(ellipse2d.B, B)
    assert np.allclose(ellipse2d.C, C)


def test_projected_beta_alpha_respect_mathematical_bounds():
    """Enforce beta and alpha respect mathematical bounds"""
    ran_key = jran.key(0)
    npts = 500_000
    a, b, c, mu, phi = generate_test_ellipse_params(ran_key, npts)
    ellipse2d = eproj.compute_ellipse2d(a, b, c, mu, phi)
    assert np.all(np.isfinite(ellipse2d.ellipticity))

    # Enforce alpha is finite
    msk_alpha_nan = ~np.isfinite(ellipse2d.alpha)
    num_alpha_nan = msk_alpha_nan.sum()
    assert num_alpha_nan == 0

    # Enforce all returned ellipse params are finite
    for p, pname in zip(ellipse2d, ellipse2d._fields):
        assert np.all(np.isfinite(p)), pname

    assert np.all(ellipse2d.ellipticity >= 0)
    assert np.all(ellipse2d.ellipticity < 1)

    assert np.all(ellipse2d.beta > 0)
    assert np.all(ellipse2d.alpha > 0)

    assert np.all(ellipse2d.beta <= ellipse2d.alpha)
    assert np.any(ellipse2d.beta < ellipse2d.alpha)


def test_calculate_ellipse2d_omega_respect_mathematical_bounds():
    """Enforce angle omega in xy-plane respects mathematical bounds"""
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


def test_projected_ellipse_axis_vectors_respect_mathematical_bounds():
    """Enforce ellipse axis vectors respect mathematical bounds"""
    ran_key = jran.key(0)
    npts = 50_000
    a, b, c, mu, phi = generate_test_ellipse_params(ran_key, npts)
    ellipse2d = eproj.compute_ellipse2d(a, b, c, mu, phi)

    assert np.allclose(np.sum(ellipse2d.e_alpha**2, axis=1), 1.0)
    assert np.allclose(np.sum(ellipse2d.e_beta**2, axis=1), 1.0)

    psi = np.arctan2(ellipse2d.e_alpha[:, 1], ellipse2d.e_alpha[:, 0])
    assert np.allclose(psi, ellipse2d.psi)

    dot_vmap = vmap(jnp.dot, in_axes=(0, 0))
    angle = dot_vmap(ellipse2d.e_alpha, ellipse2d.e_beta)
    assert np.allclose(angle, 0.0, atol=1e-4)


def test_beta_alpha_for_zaxis_projection():
    """When projecting along z-axis, α==a and β==b"""
    n = 100
    ZZ = np.zeros(n)
    ran_key = jran.key(0)
    ellipse3d = generate_test_ellipse_params(
        ran_key, n, ba_ratio_max=5.0, cb_ratio_max=5.0
    )
    # Define line-of-sight axis to be the z-axis
    ellipse3d = ellipse3d._replace(mu=ZZ + 1)
    ellipse2d = eproj.compute_ellipse2d(*ellipse3d)

    assert np.allclose(ellipse3d.a, ellipse2d.alpha, rtol=0.1)
    assert np.allclose(ellipse3d.b, ellipse2d.beta, rtol=0.1)


def test_beta_alpha_for_xaxis_projection():
    """When projecting along x-axis, α==b and β==c"""
    n = 1
    ZZ = np.zeros(n)
    ran_key = jran.key(0)
    ellipse3d = generate_test_ellipse_params(
        ran_key, n, ba_ratio_max=5.0, cb_ratio_max=5.0
    )
    # Define line-of-sight axis to be the x-axis
    ellipse3d = ellipse3d._replace(mu=ZZ, phi=ZZ)
    ellipse2d = eproj.compute_ellipse2d(*ellipse3d)

    assert np.allclose(ellipse3d.b, ellipse2d.alpha, rtol=0.1)
    assert np.allclose(ellipse3d.c, ellipse2d.beta, rtol=0.1)


def test_beta_alpha_for_yaxis_projection():
    """When projecting along y-axis, α==a and β==c"""
    n = 100
    ZZ = np.zeros(n)
    ran_key = jran.key(0)
    ellipse3d = generate_test_ellipse_params(
        ran_key, n, ba_ratio_max=5.0, cb_ratio_max=5.0
    )
    # Define line-of-sight axis to be the y-axis
    ellipse3d = ellipse3d._replace(mu=ZZ, phi=ZZ + np.pi / 2)
    ellipse2d = eproj.compute_ellipse2d(*ellipse3d)

    assert np.allclose(ellipse3d.a, ellipse2d.alpha, rtol=0.1)
    assert np.allclose(ellipse3d.c, ellipse2d.beta, rtol=0.1)
