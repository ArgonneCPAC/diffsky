""""""

from collections import namedtuple

import numpy as np
from jax import numpy as jnp
from jax import random as jran
from jax import vmap

from .. import ellipse_proj_kernels as eproj

EllipsoidParams = namedtuple("EllipsoidParams", ("a", "b", "c", "mu", "phi", "omega"))


def generate_test_ellipse_params(ran_key, n, ba_ratio_max=50.0, cb_ratio_max=50.0):

    a = np.ones(n)

    ran_key, b_key = jran.split(ran_key, 2)
    b = a / jran.uniform(b_key, minval=1.1, maxval=ba_ratio_max, shape=(n,))

    ran_key, c_key = jran.split(ran_key, 2)
    c = b / jran.uniform(c_key, minval=1.1, maxval=cb_ratio_max, shape=(n,))

    ran_key, mu_key = jran.split(ran_key, 2)
    mu = jran.uniform(mu_key, minval=-1, maxval=1, shape=(n,))

    ran_key, phi_key = jran.split(ran_key, 2)
    phi = jran.uniform(phi_key, minval=0, maxval=2 * np.pi, shape=(n,))

    ran_key, omega_key = jran.split(ran_key, 2)
    omega = jran.uniform(omega_key, minval=0, maxval=2 * np.pi, shape=(n,))

    return EllipsoidParams(a, b, c, mu, phi, omega)


def test_mc_ellipse2d_respect_mathematical_bounds():
    """Enforce mc_ellipse2d respect mathematical bounds"""
    ran_key = jran.key(0)
    npts = 500_000
    a, b, c, _, _, _ = generate_test_ellipse_params(ran_key, npts)
    ellipse2d = eproj.mc_ellipse2d(a, b / a, c / a, ran_key, sample_omega=True)
    assert np.all(ellipse2d.ellipticity >= 0.0)
    assert np.all(ellipse2d.ellipticity <= 1.0)
    assert np.all(ellipse2d.psi >= -np.pi / 2.)
    assert np.all(ellipse2d.psi <= np.pi / 2.)
    assert np.allclose(np.sum(ellipse2d.e_alpha * np.array([1, 0])[np.newaxis, :], axis=-1), np.cos(ellipse2d.psi))
    assert np.allclose(np.sum(ellipse2d.e_beta * np.array([1, 0])[np.newaxis, :], axis=-1), np.cos(ellipse2d.psi + np.pi / 2.0))


def test_mc_orthonormal_vectors():
    """Enforce mc_orthonormal_vectors return ortho-normal vectors"""
    ran_key = jran.key(0)
    npts = 500_000
    atol = 1e-6
    A, B, C = eproj.mc_orthonormal_vectors(ran_key, npts)
    assert np.allclose(np.sum(A * B, axis=-1), 0.0, atol=atol)
    assert np.allclose(np.sum(A * C, axis=-1), 0.0, atol=atol)
    assert np.allclose(np.sum(B * C, axis=-1), 0.0, atol=atol)
    assert np.allclose(np.sum(A * A, axis=-1), 1.0, atol=atol)
    assert np.allclose(np.sum(B * B, axis=-1), 1.0, atol=atol)
    assert np.allclose(np.sum(C * C, axis=-1), 1.0, atol=atol)
    assert np.allclose(
        np.linalg.norm(np.cross(A, B) - C, axis=-1), 0.0, atol=atol
    )


def test_3d_ellipse_params():
    """Reimplement _calculate_ellipse2d_axes and manually enforce agreement"""
    ran_key = jran.key(0)
    npts = 500_000
    a, b, c, mu, phi, omega = generate_test_ellipse_params(ran_key, npts)
    _2d_params = eproj._compute_2d_ellipse_params(a, b, c, mu, phi, omega)
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
    ellipse2d = eproj.compute_ellipse2d(a, b, c, mu, phi, omega)
    assert np.allclose(ellipse2d.A, A)
    assert np.allclose(ellipse2d.B, B)
    assert np.allclose(ellipse2d.C, C)


def test_projected_beta_alpha_respect_mathematical_bounds():
    """Enforce beta and alpha respect mathematical bounds"""
    ran_key = jran.key(0)
    npts = 500_000
    a, b, c, mu, phi, omega = generate_test_ellipse_params(ran_key, npts)
    ellipse2d = eproj.compute_ellipse2d(a, b, c, mu, phi, omega)
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

    # Enforce 2d axis ratio is bounded by the min of 3d axis ratios
    axis_ratio_2d = ellipse2d.beta / ellipse2d.alpha
    ba_axis_ratio_3d = b / a
    ca_axis_ratio_3d = c / a
    cb_axis_ratio_3d = c / b

    axis_ratio_2d_min = np.minimum(
        np.minimum(ba_axis_ratio_3d, ca_axis_ratio_3d), cb_axis_ratio_3d
    )

    assert np.all(axis_ratio_2d >= axis_ratio_2d_min)
    assert np.all(axis_ratio_2d <= 1)


def test_calculate_ellipse2d_psi_respect_mathematical_bounds():
    """Enforce angle psi in the uv-plane respects mathematical bounds"""
    ran_key = jran.key(0)
    npts = 500_000
    a, b, c, mu, phi, omega = generate_test_ellipse_params(ran_key, npts)
    ellipse2d = eproj.compute_ellipse2d(a, b, c, mu, phi, omega)
    assert np.all(np.isfinite(ellipse2d.psi))
    assert np.all(ellipse2d.psi >= -np.pi / 2)
    assert np.all(ellipse2d.psi < np.pi / 2)

    psi = eproj._calculate_ellipse2d_psi(ellipse2d.A, ellipse2d.B, ellipse2d.C)
    assert np.allclose(psi, ellipse2d.psi)


def test_projected_ellipse_axis_vectors_respect_mathematical_bounds():
    """Enforce ellipse axis vectors respect mathematical bounds"""
    ran_key = jran.key(0)
    npts = 50_000
    a, b, c, mu, phi, omega = generate_test_ellipse_params(ran_key, npts)
    ellipse2d = eproj.compute_ellipse2d(a, b, c, mu, phi, omega)

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


# ---------------------------------------------------------------------------
# Helpers for the tests of the sim-frame projection and random-vector functions
# ---------------------------------------------------------------------------


def _ks_distance(samples, cdf):
    """Kolmogorov-Smirnov distance between the empirical CDF of samples and cdf"""
    x = np.sort(np.asarray(samples))
    n = x.size
    model = cdf(x)
    i = np.arange(1, n + 1)
    return max(np.max(i / n - model), np.max(model - (i - 1) / n))


def _ks_bound(n, n_comparisons=1):
    """Kolmogorov-Smirnov critical distance at 99.9% confidence for n samples,
    Bonferroni-corrected so that n_comparisons tests together keep a 0.1% false
    failure rate (coefficient sqrt(-ln(α/2)/2) = 1.95 for a single test)"""
    alpha = 1e-3 / n_comparisons
    return np.sqrt(-np.log(alpha / 2) / 2) / np.sqrt(n)


def _uniform_cdf(lo, hi):
    return lambda x: np.clip((x - lo) / (hi - lo), 0.0, 1.0)


def _axial_diff(psi1, psi2):
    """Difference of two axial angles (period π), wrapped into [-π/2, π/2)"""
    return np.mod(psi1 - psi2 + np.pi / 2, np.pi) - np.pi / 2


def _azimuth_about(vecs, axes):
    """Azimuthal angle of vecs around axes, in a basis built independently of eproj"""
    axes = axes / np.linalg.norm(axes, axis=-1, keepdims=True)
    ref = np.where(np.abs(axes[:, 2:3]) < 0.5, [[0.0, 0.0, 1.0]], [[0.0, 1.0, 0.0]])
    e1 = np.cross(axes, ref)
    e1 /= np.linalg.norm(e1, axis=-1, keepdims=True)
    e2 = np.cross(axes, e1)
    return np.arctan2(np.sum(vecs * e2, axis=-1), np.sum(vecs * e1, axis=-1))


def _independent_projection(A_v, B_v, C_v, a, b, c, obs_x, obs_y, envelop=True):
    """Project the ellipsoid with plain numpy linear algebra, independently of eproj

    envelop=True: the silhouette is the marginal of the covariance R diag(a²,b²,c²) Rᵀ
    on the observed plane, so its semi-axes² are the eigenvalues of Pᵀ Σ P.
    envelop=False: the z=0 cross-section is the restriction of the shape matrix
    M = R diag(1/a²,1/b²,1/c²) Rᵀ to the observed plane, Q = Pᵀ M P.
    Returns alpha, beta, psi, and the uv quadratic form Q (Au² + Buv + Cv² = 1).
    """
    R = np.stack([A_v, B_v, C_v], axis=-1)  # columns are the body axes
    P = np.stack([obs_x, obs_y], axis=-1)  # columns are the observed u, v axes
    if envelop:
        sig = np.stack([a**2, b**2, c**2], axis=-1)
        Sigma = np.einsum("nij,nj,nkj->nik", R, sig, R)
        S2 = np.einsum("nji,njk,nkl->nil", P, Sigma, P)
        evals, evecs = np.linalg.eigh(S2)
        alpha, beta = np.sqrt(evals[:, 1]), np.sqrt(evals[:, 0])
        major = evecs[:, :, 1]
        Q = np.linalg.inv(S2)
    else:
        inv_sig = np.stack([a**-2, b**-2, c**-2], axis=-1)
        M = np.einsum("nij,nj,nkj->nik", R, inv_sig, R)
        Q = np.einsum("nji,njk,nkl->nil", P, M, P)
        evals, evecs = np.linalg.eigh(Q)
        alpha, beta = 1 / np.sqrt(evals[:, 0]), 1 / np.sqrt(evals[:, 1])
        major = evecs[:, :, 0]
    psi = _axial_diff(np.arctan2(major[:, 1], major[:, 0]), 0.0)
    return alpha, beta, psi, Q


def _random_ellipsoids_and_frames(ran_key, n):
    """Random a>=b>=c, random body axes, and random observer axes"""
    shape_key, body_key, obs_key = jran.split(ran_key, 3)
    a, b, c, _, _, _ = generate_test_ellipse_params(
        shape_key, n, ba_ratio_max=5.0, cb_ratio_max=5.0
    )
    A_v, B_v, C_v = (np.asarray(x) for x in eproj.mc_orthonormal_vectors(body_key, n))
    obs_x, obs_y, _ = (np.asarray(x) for x in eproj.mc_orthonormal_vectors(obs_key, n))
    return (A_v, B_v, C_v), (np.asarray(a), np.asarray(b), np.asarray(c)), (obs_x, obs_y)


# ---------------------------------------------------------------------------
# _unitize, random_unit_vectors_3d, random_perpendicular_directions,
# mc_orthonormal_vectors
# ---------------------------------------------------------------------------


def test_unitize_returns_unit_vectors_and_preserves_direction():
    """_unitize returns exact unit vectors along the input, for any non-zero length"""
    ran_key = jran.key(0)
    n = 10_000
    dir_key, scale_key = jran.split(ran_key, 2)
    v = np.asarray(jran.normal(dir_key, shape=(n, 3)))
    scale = 10 ** np.asarray(jran.uniform(scale_key, minval=-9, maxval=9, shape=(n, 1)))
    vhat = np.asarray(eproj._unitize(v * scale))
    assert np.all(np.isfinite(vhat))
    assert np.allclose(np.linalg.norm(vhat, axis=-1), 1.0, rtol=0, atol=1e-12)
    assert np.allclose(vhat, v / np.linalg.norm(v, axis=-1, keepdims=True), atol=1e-12)

    # the zero vector has no direction: it is mapped to zero rather than to nan,
    # with a zero (not nan) gradient
    import jax

    assert np.all(np.asarray(eproj._unitize(np.zeros((3, 3)))) == 0.0)
    grad0 = np.asarray(jax.grad(lambda u: jnp.sum(eproj._unitize(u)))(jnp.zeros(3)))
    assert np.all(grad0 == 0.0)

    # away from zero the Jacobian is the analytic (I - v̂v̂ᵀ)/|v|
    u = jnp.array([0.3, -1.2, 2.0])
    uhat = np.asarray(u) / np.linalg.norm(u)
    jac = np.asarray(jax.jacobian(eproj._unitize)(u))
    assert np.allclose(jac, (np.eye(3) - np.outer(uhat, uhat)) / np.linalg.norm(u), atol=1e-12)


def test_random_unit_vectors_3d_are_isotropic():
    """random_unit_vectors_3d is uniform on the sphere: every component and
    azimuth has the analytic (uniform) distribution, and the mean vector vanishes"""
    ran_key = jran.key(0)
    n = 200_000
    # generate random 3D vectors
    v = np.asarray(eproj.random_unit_vectors_3d(ran_key, n))
    assert v.shape == (n, 3)
    assert np.all(np.isfinite(v))
    assert np.allclose(np.linalg.norm(v, axis=-1), 1.0, atol=1e-12)

    # Archimedes: the projection onto any fixed axis is uniform in [-1, 1]
    for axis in (np.array([1.0, 0, 0]), np.array([0, 0, 1.0]), np.ones(3) / np.sqrt(3)):
        assert _ks_distance(v @ axis, _uniform_cdf(-1, 1)) < _ks_bound(n)
    assert _ks_distance(np.arctan2(v[:, 1], v[:, 0]), _uniform_cdf(-np.pi, np.pi)) < _ks_bound(n)
    assert np.all(np.abs(v.mean(axis=0)) < 5 / np.sqrt(3 * n))

    # reproducible for the same key, different for different keys
    assert np.array_equal(v, eproj.random_unit_vectors_3d(ran_key, n))
    assert not np.allclose(v, eproj.random_unit_vectors_3d(jran.key(1), n))


def test_random_perpendicular_directions_are_uniform_around_input():
    """Output is a unit vector perpendicular to the (unnormalized) input, with a
    uniformly distributed azimuth around it -- also near the coordinate axes,
    where the function switches its internal reference vector"""
    ran_key = jran.key(0)
    n = 100_000
    # test under various input vector norms and orientations
    for v_key, scale, axis in (
        (jran.key(1), 1.0, None),
        (jran.key(2), 1e4, None),
        (jran.key(3), 1e-4, None),
        (jran.key(4), 1.0, np.array([1.0, 0.0, 0.0])),
        (jran.key(5), 1.0, np.array([0.0, 1.0, 0.0])),
    ):
        v = np.asarray(jran.normal(v_key, shape=(n, 3)))
        if axis is not None:  # cluster the inputs tightly around a coordinate axis
            v = axis + 0.05 * v
        v = scale * v
        p = np.asarray(eproj.random_perpendicular_directions(v, ran_key))
        vhat = v / np.linalg.norm(v, axis=-1, keepdims=True)
        assert np.all(np.isfinite(p))
        assert np.allclose(np.linalg.norm(p, axis=-1), 1.0, atol=1e-12)
        assert np.allclose(np.sum(p * vhat, axis=-1), 0.0, atol=1e-12)
        phi = _azimuth_about(p, vhat)
        assert _ks_distance(phi, _uniform_cdf(-np.pi, np.pi)) < _ks_bound(n)


def test_random_perpendicular_directions_corner_cases():
    """Exactly axis-aligned inputs, a single (3,) input, and the zero vector"""
    ran_key = jran.key(0)
    axes = np.array(
        [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0], [-1.0, 0, 0], [0, 0, -1.0], [0.9, 0.1, 0]]
    )
    p = np.asarray(eproj.random_perpendicular_directions(axes, ran_key))
    assert np.allclose(np.linalg.norm(p, axis=-1), 1.0, atol=1e-12)
    assert np.allclose(np.sum(p * axes / np.linalg.norm(axes, axis=-1, keepdims=True), axis=-1), 0.0, atol=1e-12)

    p1 = np.asarray(eproj.random_perpendicular_directions(np.array([0.0, 0.0, 2.0]), ran_key))
    assert p1.shape == (1, 3)
    assert np.isclose(p1[0, 2], 0.0, atol=1e-12)

    # no perpendicular direction exists for the zero vector: zeros, not nan
    p0 = np.asarray(eproj.random_perpendicular_directions(np.zeros((4, 3)), ran_key))
    assert np.all(p0 == 0.0)


def test_mc_orthonormal_vectors_are_haar_distributed():
    """[A, B, C] is a uniformly random rotation matrix (Haar measure): each axis is
    isotropic, and the rotation angle θ has CDF (θ - sin θ)/π"""
    ran_key = jran.key(0)
    n = 200_000
    A, B, C = (np.asarray(x) for x in eproj.mc_orthonormal_vectors(ran_key, n))
    R = np.stack([A, B, C], axis=-1)
    assert np.allclose(np.linalg.det(R), 1.0, atol=1e-12)
    assert np.allclose(np.einsum("nji,njk->nik", R, R), np.eye(3), atol=1e-12)

    for vec in (A, B, C):
        assert _ks_distance(vec[:, 2], _uniform_cdf(-1, 1)) < _ks_bound(n)
        assert _ks_distance(vec[:, 0], _uniform_cdf(-1, 1)) < _ks_bound(n)

    cos_theta = np.clip((np.trace(R, axis1=1, axis2=2) - 1) / 2, -1, 1)
    theta = np.arccos(cos_theta)
    assert _ks_distance(theta, lambda t: (t - np.sin(t)) / np.pi) < _ks_bound(n)


# ---------------------------------------------------------------------------
# compute_ellipse2d_in_sim_frame
# ---------------------------------------------------------------------------


def test_compute_ellipse2d_in_sim_frame_matches_independent_projection():
    """Reproduce α, β, ψ and the quadratic-form coefficients with independent
    numpy linear algebra, for both the silhouette and the cross-section"""
    n = 20_000
    (A_v, B_v, C_v), (a, b, c), (obs_x, obs_y) = _random_ellipsoids_and_frames(jran.key(0), n)
    for envelop in (True, False):
        e2d = eproj.compute_ellipse2d_in_sim_frame(
            A_v, B_v, C_v, a, b, c, obs_x, obs_y, envelop=envelop
        )
        for p, pname in zip(e2d, e2d._fields):
            assert np.all(np.isfinite(p)), pname
        alpha, beta, psi, Q = _independent_projection(
            A_v, B_v, C_v, a, b, c, obs_x, obs_y, envelop=envelop
        )
        assert np.allclose(e2d.alpha, alpha, rtol=1e-8)
        assert np.allclose(e2d.beta, beta, rtol=1e-8)
        assert np.allclose(e2d.A, Q[:, 0, 0], rtol=1e-8)
        assert np.allclose(e2d.B, 2 * Q[:, 0, 1], rtol=1e-8, atol=1e-10)
        assert np.allclose(e2d.C, Q[:, 1, 1], rtol=1e-8)
        # psi is ill-defined for nearly circular images; compare where it is not
        msk = beta / alpha < 0.999
        assert np.allclose(_axial_diff(e2d.psi, psi)[msk], 0.0, atol=1e-6)

        assert np.all(e2d.psi >= -np.pi / 2) & np.all(e2d.psi < np.pi / 2)
        assert np.all(e2d.beta <= e2d.alpha)
        assert np.all(e2d.alpha <= a * (1 + 1e-12))
        assert np.all(e2d.beta >= c * (1 - 1e-12))
        assert np.allclose(e2d.e_alpha, np.stack([np.cos(e2d.psi), np.sin(e2d.psi)], -1))
        assert np.allclose(np.sum(e2d.e_alpha * e2d.e_beta, axis=-1), 0.0, atol=1e-12)


def test_compute_ellipse2d_in_sim_frame_agrees_with_compute_ellipse2d():
    """Both projection routines agree when the observer axes are built from the
    (μ, φ, ω) Euler angles that compute_ellipse2d uses"""
    n = 20_000
    a, b, c, mu, phi, omega = generate_test_ellipse_params(jran.key(0), n)
    R = np.asarray(eproj._get_eulerxzx_matrix_from_angles(phi + np.pi / 2, np.arccos(mu), omega))
    obs_x, obs_y = R[:, :, 0], R[:, :, 1]
    body = [np.tile(e, (n, 1)) for e in np.eye(3)]
    for ellipticity_type in (0, 1, 2, 3):
        ref = eproj.compute_ellipse2d(a, b, c, mu, phi, omega, ellipticity_type=ellipticity_type)
        new = eproj.compute_ellipse2d_in_sim_frame(
            *body, a, b, c, obs_x, obs_y, ellipticity_type=ellipticity_type
        )
        for field in ("alpha", "beta", "ellipticity", "A", "B", "C"):
            assert np.allclose(getattr(new, field), getattr(ref, field), rtol=1e-8, atol=1e-10), field
        msk = ref.beta / ref.alpha < 0.999
        assert np.allclose(_axial_diff(new.psi, ref.psi)[msk], 0.0, atol=1e-6)


def test_compute_ellipse2d_in_sim_frame_symmetries():
    """Physical symmetries: a common rotation of ellipsoid and observer, reversing
    body axes (including a left-handed triad), rotating the observer about the
    line of sight, and viewing the mirror image from behind"""
    n = 20_000
    (A_v, B_v, C_v), (a, b, c), (obs_x, obs_y) = _random_ellipsoids_and_frames(jran.key(0), n)
    ref = eproj.compute_ellipse2d_in_sim_frame(A_v, B_v, C_v, a, b, c, obs_x, obs_y)
    good = ref.beta / ref.alpha < 0.999

    def _assert_same_ellipse(new, dpsi=0.0, mirror=False):
        assert np.allclose(new.alpha, ref.alpha, rtol=1e-8)
        assert np.allclose(new.beta, ref.beta, rtol=1e-8)
        expected_psi = -ref.psi + np.pi / 2 if mirror else ref.psi + dpsi
        assert np.allclose(_axial_diff(new.psi, expected_psi)[good], 0.0, atol=1e-6)

    # common rotation G of everything leaves the image unchanged
    G = np.stack(eproj.mc_orthonormal_vectors(jran.key(9), n), axis=-1)
    rot = lambda v: np.einsum("nij,nj->ni", G, v)  # noqa: E731
    _assert_same_ellipse(eproj.compute_ellipse2d_in_sim_frame(
        rot(A_v), rot(B_v), rot(C_v), a, b, c, rot(obs_x), rot(obs_y)))

    # an ellipsoid is symmetric under reversing any of its axes
    _assert_same_ellipse(eproj.compute_ellipse2d_in_sim_frame(
        -A_v, -B_v, C_v, a, b, c, obs_x, obs_y))
    _assert_same_ellipse(eproj.compute_ellipse2d_in_sim_frame(
        A_v, B_v, -C_v, a, b, c, obs_x, obs_y))

    # rotating the observer's u-v axes by γ about the LoS rotates psi by -γ
    gamma = np.asarray(jran.uniform(jran.key(8), minval=-np.pi, maxval=np.pi, shape=(n, 1)))
    obs_x_rot = np.cos(gamma) * obs_x + np.sin(gamma) * obs_y
    obs_y_rot = -np.sin(gamma) * obs_x + np.cos(gamma) * obs_y
    _assert_same_ellipse(eproj.compute_ellipse2d_in_sim_frame(
        A_v, B_v, C_v, a, b, c, obs_x_rot, obs_y_rot), dpsi=-gamma[:, 0])

    # swapping u and v views the image from behind, mirrored about the u=v line
    _assert_same_ellipse(eproj.compute_ellipse2d_in_sim_frame(
        A_v, B_v, C_v, a, b, c, obs_y, obs_x), mirror=True)


def test_compute_ellipse2d_in_sim_frame_axis_aligned_views():
    """Corner case of Euler gimbal lock: viewing exactly along a principal axis
    shows the other two axes at their full lengths and orientations. Many
    rotations are used because whether rounding lands exactly on z'_z = +-1
    varies from row to row and from machine to machine"""
    n = 20_000
    a, b, c, _, _, _ = generate_test_ellipse_params(jran.key(0), n, 5.0, 5.0)
    gamma = np.asarray(jran.uniform(jran.key(1), minval=-np.pi, maxval=np.pi, shape=(n, 1)))
    xhat, yhat, zhat = (np.tile(e, (n, 1)) for e in np.eye(3))

    # viewing along C (minor axis): u rotated by γ in the A-B plane, so A sits at ψ=-γ
    u = np.cos(gamma) * xhat + np.sin(gamma) * yhat
    v = -np.sin(gamma) * xhat + np.cos(gamma) * yhat
    for los_sign in (1, -1):
        e2d = eproj.compute_ellipse2d_in_sim_frame(xhat, yhat, zhat, a, b, c, u, los_sign * v)
        assert np.allclose(e2d.alpha, a, rtol=1e-10)
        assert np.allclose(e2d.beta, b, rtol=1e-10)
        assert np.allclose(_axial_diff(e2d.psi, -los_sign * gamma[:, 0]), 0.0, atol=1e-8)

    # viewing along A: image is b x c with B along u
    e2d = eproj.compute_ellipse2d_in_sim_frame(xhat, yhat, zhat, a, b, c, yhat, zhat)
    assert np.allclose(e2d.alpha, b, rtol=1e-10)
    assert np.allclose(e2d.beta, c, rtol=1e-10)
    assert np.allclose(_axial_diff(e2d.psi, 0.0), 0.0, atol=1e-8)

    # viewing along B: image is a x c with A along v
    e2d = eproj.compute_ellipse2d_in_sim_frame(xhat, yhat, zhat, a, b, c, zhat, xhat)
    assert np.allclose(e2d.alpha, a, rtol=1e-10)
    assert np.allclose(e2d.beta, c, rtol=1e-10)
    assert np.allclose(_axial_diff(e2d.psi, np.pi / 2), 0.0, atol=1e-8)


def test_compute_ellipse2d_in_sim_frame_sphere_and_ellipticity_types():
    """A sphere projects to a circle of the same radius; the four ellipticity
    definitions follow from q = β/α; an invalid definition raises"""
    n = 2_000
    (A_v, B_v, C_v), _, (obs_x, obs_y) = _random_ellipsoids_and_frames(jran.key(0), n)
    r = np.full(n, 2.5)
    for ellipticity_type in (0, 1, 2, 3):
        e2d = eproj.compute_ellipse2d_in_sim_frame(
            A_v, B_v, C_v, r, r, r, obs_x, obs_y, ellipticity_type=ellipticity_type
        )
        assert np.allclose(e2d.alpha, r, rtol=1e-12)
        assert np.allclose(e2d.beta, r, rtol=1e-12)
        assert np.allclose(e2d.ellipticity, 0.0, atol=1e-12)
        assert np.all(np.isfinite(e2d.psi))

    (A_v, B_v, C_v), (a, b, c), (obs_x, obs_y) = _random_ellipsoids_and_frames(jran.key(1), n)
    expected = {
        0: lambda q: 1 - q,
        1: lambda q: (1 - q) / (1 + q),
        2: lambda q: (1 - q**2) / (1 + q**2),
        3: lambda q: -np.log(q),
    }
    for ellipticity_type, func in expected.items():
        e2d = eproj.compute_ellipse2d_in_sim_frame(
            A_v, B_v, C_v, a, b, c, obs_x, obs_y, ellipticity_type=ellipticity_type
        )
        q = e2d.beta / e2d.alpha
        assert np.allclose(e2d.ellipticity, func(q))
        assert np.all(e2d.ellipticity >= 0)
        if ellipticity_type < 3:
            assert np.all(e2d.ellipticity < 1)

    try:
        eproj.compute_ellipse2d_in_sim_frame(
            A_v, B_v, C_v, a, b, c, obs_x, obs_y, ellipticity_type=4
        )
    except ValueError:
        pass
    else:
        raise AssertionError("ellipticity_type=4 should raise ValueError")


def test_calculate_ellipse2d_axes_precision_from_elongated_to_round():
    """The semi-axes recovered from the quadratic-form coefficients keep full
    precision from elongated to almost perfectly round ellipses, so that a tiny
    ellipticity 1 - q is still resolved (a trace**2 - 4*det discriminant
    cancels for round shapes and loses it)"""
    n = 5_000
    theta = np.asarray(jran.uniform(jran.key(0), minval=0, maxval=np.pi, shape=(n,)))
    cs, sn = np.cos(theta), np.sin(theta)
    for q in (1e-2, 0.3, 0.9, 1 - 1e-4, 1 - 1e-6, 1 - 1e-8, 1 - 1e-12):
        alpha, beta = 1.3, 1.3 * q
        lam_a, lam_b = alpha**-2, beta**-2
        A = cs * cs * lam_a + sn * sn * lam_b
        C = sn * sn * lam_a + cs * cs * lam_b
        B = 2 * cs * sn * (lam_a - lam_b)
        al, be = (np.asarray(x) for x in eproj._calculate_ellipse2d_axes(A, B, C))
        assert np.allclose(al, alpha, rtol=1e-11, atol=0), q
        assert np.allclose(be, beta, rtol=1e-11, atol=0), q
        assert np.allclose(1 - be / al, 1 - q, rtol=0, atol=1e-13), q

    # the semi-axes are not differentiable at an exactly round image (the two
    # eigenvalues cross), but their gradient must stay finite rather than nan
    import jax

    grad = jax.grad(lambda x: jnp.sum(jnp.stack(eproj._calculate_ellipse2d_axes(*x))))(jnp.array([0.7, 0.0, 0.7]))
    assert np.all(np.isfinite(np.asarray(grad)))


def test_compute_ellipse2d_in_sim_frame_gradients_near_alignment():
    """Gradients with respect to the viewing direction are finite and match
    central finite differences for every orientation, including exactly along a
    principal axis (Euler-angle gimbal lock), from either side"""
    import jax

    a, b, c = 2.0, 1.0, 0.5
    eye = jnp.eye(3)

    def outputs(p, los_sign):
        tx, ty, g = p
        los = jnp.array([tx, ty, los_sign]) / jnp.sqrt(tx**2 + ty**2 + 1.0)
        e1 = jnp.cross(los, jnp.array([jnp.cos(g), jnp.sin(g), 0.0]))
        e1 = e1 / jnp.linalg.norm(e1)
        e2d = eproj.compute_ellipse2d_in_sim_frame(
            eye[0:1], eye[1:2], eye[2:3], jnp.array([a]), jnp.array([b]), jnp.array([c]),
            e1[None], jnp.cross(los, e1)[None],
        )
        # cos/sin of 2 psi are smooth where psi itself wraps
        return jnp.stack([e2d.alpha[0], e2d.beta[0], jnp.cos(2 * e2d.psi[0]), jnp.sin(2 * e2d.psi[0])])

    h = 1e-6
    for tilt in (0.0, 1e-12, 1e-9, 1e-6, 1e-3):
        for los_sign in (1.0, -1.0):
            p = jnp.array([tilt, 0.7 * tilt, 0.4])
            jac_fwd = np.asarray(jax.jacfwd(lambda q: outputs(q, los_sign))(p))
            jac_rev = np.asarray(jax.jacrev(lambda q: outputs(q, los_sign))(p))
            assert np.all(np.isfinite(jac_fwd)) & np.all(np.isfinite(jac_rev)), tilt
            finite_diff = np.stack([
                (np.asarray(outputs(p.at[i].add(h), los_sign)) - np.asarray(outputs(p.at[i].add(-h), los_sign))) / (2 * h)
                for i in range(3)
            ], axis=-1)
            assert np.allclose(jac_fwd, finite_diff, atol=1e-8), tilt
            assert np.allclose(jac_rev, jac_fwd, atol=1e-12), tilt


def test_silhouette_encloses_cross_section():
    """The silhouette (envelop=True) contains the central cross-section
    (envelop=False), so both of its semi-axes are at least as long"""
    n = 20_000
    (A_v, B_v, C_v), (a, b, c), (obs_x, obs_y) = _random_ellipsoids_and_frames(jran.key(0), n)
    env = eproj.compute_ellipse2d_in_sim_frame(A_v, B_v, C_v, a, b, c, obs_x, obs_y, envelop=True)
    cut = eproj.compute_ellipse2d_in_sim_frame(A_v, B_v, C_v, a, b, c, obs_x, obs_y, envelop=False)
    assert np.all(env.alpha >= cut.alpha * (1 - 1e-12))
    assert np.all(env.beta >= cut.beta * (1 - 1e-12))
    assert np.any(env.alpha > cut.alpha * 1.01)


# ---------------------------------------------------------------------------
# observer frames and assert_orthonormality
# ---------------------------------------------------------------------------


def _r_hat(ra_deg, dec_deg):
    ra, dec = np.deg2rad(ra_deg), np.deg2rad(dec_deg)
    return np.stack([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)], -1)


def _assert_right_handed_orthonormal(x, y, z, atol=1e-12):
    for u in (x, y, z):
        assert np.allclose(np.linalg.norm(u, axis=-1), 1.0, atol=atol)
    assert np.allclose(np.sum(x * y, axis=-1), 0.0, atol=atol)
    assert np.allclose(np.sum(x * z, axis=-1), 0.0, atol=atol)
    assert np.allclose(np.sum(y * z, axis=-1), 0.0, atol=atol)
    assert np.allclose(np.cross(x, y), z, atol=atol)


def test_observer_frame_axes_from_ra_dec_geometry():
    """(West, North, -LoS): right-handed, -LoS toward the observer, West along
    decreasing RA and North along increasing Dec (checked by finite differences)"""
    n = 20_000
    ra_key, dec_key = jran.split(jran.key(0), 2)
    ra = np.asarray(jran.uniform(ra_key, minval=0, maxval=360, shape=(n,)))
    dec = np.rad2deg(np.arcsin(np.asarray(jran.uniform(dec_key, minval=-0.999, maxval=0.999, shape=(n,)))))
    x, y, z = (np.asarray(v) for v in eproj.observer_frame_axes_from_ra_dec(ra, dec))
    _assert_right_handed_orthonormal(x, y, z)
    assert np.allclose(z, -_r_hat(ra, dec), atol=1e-12)
    assert np.allclose(x[:, 2], 0.0, atol=1e-12)
    assert np.all(y[:, 2] >= 0)

    eps = 1e-6
    west = _r_hat(ra - eps, dec) - _r_hat(ra, dec)
    north = _r_hat(ra, dec + eps) - _r_hat(ra, dec)
    assert np.allclose(west / np.linalg.norm(west, axis=-1, keepdims=True), x, atol=1e-5)
    assert np.allclose(north / np.linalg.norm(north, axis=-1, keepdims=True), y, atol=1e-5)

    # RA is periodic, and the ascending node only shifts the RA zero-point
    for v1, v2 in zip(eproj.observer_frame_axes_from_ra_dec(ra + 360, dec), (x, y, z)):
        assert np.allclose(v1, v2, atol=1e-12)
    shifted = eproj.observer_frame_axes_from_ra_dec(ra, dec, ascending_ra_deg=30.0)
    for v1, v2 in zip(shifted, eproj.observer_frame_axes_from_ra_dec(ra - 30.0, dec)):
        assert np.allclose(v1, v2, atol=1e-12)


def test_observer_frame_axes_from_pos_3D_agrees_with_ra_dec():
    """The xyz-based frame equals the RA/Dec-based frame when RA/Dec are computed
    with diffsky's own get_ra_dec, and does not depend on the distance"""
    from ...data_loaders.hacc_utils import lightcone_utils as hlu

    n = 20_000
    pos = np.asarray(jran.normal(jran.key(0), shape=(n, 3))) * 500.0
    ra, dec = (np.asarray(v) for v in hlu.get_ra_dec(pos[:, 0], pos[:, 1], pos[:, 2]))
    from_pos = [np.asarray(v) for v in eproj.observer_frame_axes_from_pos_3D(*pos.T)]
    from_radec = [np.asarray(v) for v in eproj.observer_frame_axes_from_ra_dec(ra, dec)]
    _assert_right_handed_orthonormal(*from_pos)
    for v1, v2 in zip(from_pos, from_radec):
        assert np.allclose(v1, v2, atol=1e-10)
    for scale in (1e-3, 1e4):
        for v1, v2 in zip(eproj.observer_frame_axes_from_pos_3D(*(scale * pos).T), from_pos):
            assert np.allclose(v1, v2, atol=1e-12)


def test_observer_frame_axes_at_and_near_the_poles():
    """At the celestial poles West is undefined: the xyz-based frame uses the
    RA = 0 that get_ra_dec assigns there, so it equals the RA/Dec-based frame
    of the same position, and stays orthonormal; arbitrarily close to the pole
    it follows the RA/Dec-based frame of each galaxy's own RA"""
    from ...data_loaders.hacc_utils import lightcone_utils as hlu

    pos = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -3.0]])
    from_pos = [np.asarray(v) for v in eproj.observer_frame_axes_from_pos_3D(*pos.T)]
    _assert_right_handed_orthonormal(*from_pos)
    assert np.allclose(from_pos[2], -pos / np.linalg.norm(pos, axis=-1, keepdims=True))
    ra, dec = (np.asarray(v) for v in hlu.get_ra_dec(*pos.T))
    for v1, v2 in zip(from_pos, eproj.observer_frame_axes_from_ra_dec(ra, dec)):
        assert np.allclose(v1, v2, atol=1e-12)

    # just off the pole, at RA = 0, 90, 200 degrees
    for offset in (1e-13, 1e-11, 1e-9, 1e-6, 1e-3):
        ra = np.array([0.0, 90.0, 200.0])
        dec = np.full(3, 90.0 - np.rad2deg(offset))
        pos = _r_hat(ra, dec)
        from_pos = [np.asarray(v) for v in eproj.observer_frame_axes_from_pos_3D(*pos.T)]
        _assert_right_handed_orthonormal(*from_pos, atol=1e-12)
        from_radec = [np.asarray(v) for v in eproj.observer_frame_axes_from_ra_dec(ra, dec)]
        for v1, v2 in zip(from_pos, from_radec):
            assert np.allclose(v1, v2, atol=1e-12)

    # the RA/Dec frame is well defined exactly at the poles, for any RA
    ra = np.array([0.0, 45.0, 300.0])
    for dec in (90.0, -90.0):
        x, y, z = (np.asarray(v) for v in eproj.observer_frame_axes_from_ra_dec(ra, np.full(3, dec)))
        _assert_right_handed_orthonormal(x, y, z)


def test_assert_orthonormality_warns_only_for_bad_axes(capfd):
    """assert_orthonormality prints a warning (without raising) only when an axis
    set violates orthonormality by more than atol"""
    import jax

    n = 1_000
    x, y, z = (np.asarray(v) for v in eproj.mc_orthonormal_vectors(jran.key(0), n))
    eproj.assert_orthonormality(x, y, z)
    jax.effects_barrier()
    assert "assert_orthonormality" not in capfd.readouterr().out

    small = x.copy()
    small[0] += 1e-9
    eproj.assert_orthonormality(small, y, z)
    jax.effects_barrier()
    assert "assert_orthonormality" not in capfd.readouterr().out

    for args in (
        (x * 1.01, y, z),
        (x, y, -z),  # left-handed
        (x, x, z),  # not orthogonal
    ):
        eproj.assert_orthonormality(*args)
        jax.effects_barrier()
        assert "assert_orthonormality" in capfd.readouterr().out


# ---------------------------------------------------------------------------
# Analytic expectations for random projections
# ---------------------------------------------------------------------------


def _oblate_q_cdf(gamma):
    """P(q <= Q) for the silhouette of a randomly oriented oblate spheroid
    a = b > c = γa: q² = cos²θ + γ² sin²θ with cos θ uniform in [0, 1]"""
    return lambda q: np.sqrt(np.clip((q**2 - gamma**2) / (1 - gamma**2), 0.0, 1.0))


def _prolate_q_cdf(gamma):
    """P(q <= Q) for the silhouette of a randomly oriented prolate spheroid
    a > b = c = γa (also the central cross-section of an oblate spheroid):
    q² = γ² / (sin²θ + γ² cos²θ) with cos θ uniform in [0, 1]"""
    return lambda q: np.sqrt(np.clip((1 - gamma**2 / q**2) / (1 - gamma**2), 0.0, 1.0))


def _ellipsoid_surface_area(a, b, c):
    """Closed-form surface area of an ellipsoid with a > b > c"""
    from scipy.special import ellipeinc, ellipkinc

    phi = np.arccos(c / a)
    m = a**2 * (b**2 - c**2) / (b**2 * (a**2 - c**2))
    elliptic = ellipeinc(phi, m) * np.sin(phi) ** 2 + ellipkinc(phi, m) * np.cos(phi) ** 2
    return 2 * np.pi * c**2 + 2 * np.pi * a * b / np.sin(phi) * elliptic


def test_random_projections_of_spheroids_match_analytic_distributions():
    """Randomly oriented spheroids have exact projected axis-ratio distributions:
    the oblate silhouette always has α = a, the prolate silhouette always has
    β = b, and the oblate central cross-section always has α = a; the
    distributions of q = β/α are given by _oblate_q_cdf and _prolate_q_cdf.
    Both projection routines are checked: compute_ellipse2d_in_sim_frame with
    random body axes, and mc_ellipse2d with random (μ, φ, ω)"""
    n = 100_000
    gammas = (0.1, 0.4, 0.8)
    n_ks = 5 * len(gammas)
    body = eproj.mc_orthonormal_vectors(jran.key(0), n)
    obs_x, obs_y, _ = (np.tile(v, (n, 1)) for v in eproj.mc_orthonormal_vectors(jran.key(1), 1))
    one = np.ones(n)

    def _check(e2d, cdf, label):
        q = np.asarray(e2d.beta / e2d.alpha)
        assert np.all(np.isfinite(q)) & np.all(q <= 1.0), label
        dist = _ks_distance(q, cdf)
        assert dist < _ks_bound(n, n_ks), (label, dist)

    for i, gamma in enumerate(gammas):
        g = gamma * one

        oblate = eproj.compute_ellipse2d_in_sim_frame(*body, one, one, g, obs_x, obs_y)
        assert np.allclose(oblate.alpha, 1.0, rtol=1e-7)
        _check(oblate, _oblate_q_cdf(gamma), ("oblate", gamma))

        cut = eproj.compute_ellipse2d_in_sim_frame(*body, one, one, g, obs_x, obs_y, envelop=False)
        assert np.allclose(cut.alpha, 1.0, rtol=1e-7)
        _check(cut, _prolate_q_cdf(gamma), ("oblate cross-section", gamma))

        prolate = eproj.compute_ellipse2d_in_sim_frame(*body, one, g, g, obs_x, obs_y)
        assert np.allclose(prolate.beta, gamma, rtol=1e-7)
        _check(prolate, _prolate_q_cdf(gamma), ("prolate", gamma))

        key = jran.key(10 + i)
        _check(eproj.mc_ellipse2d(one, one, g, key), _oblate_q_cdf(gamma), ("mc_ellipse2d oblate", gamma))
        _check(eproj.mc_ellipse2d(one, g, g, key), _prolate_q_cdf(gamma), ("mc_ellipse2d prolate", gamma))


def test_projected_area_of_triaxial_ellipsoids():
    """The silhouette area πab of a triaxial ellipsoid along a line of sight with
    body-frame components n is π sqrt(b²c² n_A² + a²c² n_B² + a²b² n_C²) for
    every object; and by Cauchy's theorem the mean projected area over random
    orientations equals a quarter of the surface area"""
    n = 50_000
    (A_v, B_v, C_v), (a, b, c), (obs_x, obs_y) = _random_ellipsoids_and_frames(jran.key(0), n)
    e2d = eproj.compute_ellipse2d_in_sim_frame(A_v, B_v, C_v, a, b, c, obs_x, obs_y)
    los = np.cross(obs_x, obs_y)
    nA, nB, nC = (np.sum(los * v, axis=-1) for v in (A_v, B_v, C_v))
    area = np.sqrt((b * c * nA) ** 2 + (a * c * nB) ** 2 + (a * b * nC) ** 2)
    assert np.allclose(e2d.alpha * e2d.beta, area, rtol=1e-8)

    n = 400_000
    body = eproj.mc_orthonormal_vectors(jran.key(2), n)
    obs_x, obs_y, _ = (np.tile(v, (n, 1)) for v in eproj.mc_orthonormal_vectors(jran.key(3), 1))
    for abc in ((3.0, 2.0, 0.5), (1.0, 0.9, 0.2), (2.0, 1.0, 0.99)):
        a, b, c = (np.full(n, x) for x in abc)
        e2d = eproj.compute_ellipse2d_in_sim_frame(*body, a, b, c, obs_x, obs_y)
        proj_area = np.pi * np.asarray(e2d.alpha * e2d.beta)
        expected = _ellipsoid_surface_area(*abc) / 4
        sigma = np.std(proj_area) / np.sqrt(n)
        assert abs(np.mean(proj_area) - expected) < 4 * sigma, (abc, np.mean(proj_area), expected, sigma)
