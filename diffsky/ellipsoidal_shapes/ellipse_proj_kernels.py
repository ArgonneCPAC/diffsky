"""Calculate 2d ellipse from projecting a 3d triaxial ellipsoid

Use Euler ZXZ rotation formula to align the ellipsoid body frame x-y-z with the line-of-sight
(LoS) frame u-v-LoS, and finally project onto the uv-plane.

Two projection methods are available via the `envelop` flag:
  envelop=True  (default) — correct 2D silhouette via Schur complement of S33
  envelop=False           — z=0 cross-section (sub-matrix method, faster but approximate)
Also see https://arxiv.org/abs/1203.6833 (Joachimi+13), and
https://ui.adsabs.harvard.edu/abs/2016ApJ...830..123C/abstract (Chen+16)

Four types of ellipticity definitions are available via the `ellipticity_type` flag,
see https://arxiv.org/abs/astro-ph/0107431 Section 2.2.1 for the definitions.

Credits:
    - Claude Code: implementation of the JAX projection, debugging, refactoring
    - Jiachuan Xu: planning of the Euler-angle based silhouette projection, debugging, testing
    - Iris Reed: Independent implementation of the Joachimi+13 projection, debugging, code comparison
    - Carter Williams: Independent implementation of the Joachimi+13 projection, code comparison
    - Jonathan Blazek: Discussions on the projection intuition
"""

from collections import namedtuple
from functools import partial

from jax import jit as jjit
from jax import numpy as jnp
from jax import random as jran

Ellipse2DParams = namedtuple(
    "Ellipse2DParams",
    ("alpha", "beta", "psi", "ellipticity", "e_alpha", "e_beta", "A", "B", "C"),
)


def mc_mu_phi(n, ran_key):
    """Monte Carlo realization of random projection angles mu, phi"""
    mu_key, phi_key = jran.split(ran_key, 2)
    mu_ran = jran.uniform(mu_key, minval=-1, maxval=1, shape=(n,))
    phi_ran = jran.uniform(phi_key, minval=0, maxval=2 * jnp.pi, shape=(n,))
    return mu_ran, phi_ran


def mc_mu_phi_omega(n, ran_key):
    """Monte Carlo realization of random projection angles mu, phi, omega"""
    mu_key, phi_key, omega_key = jran.split(ran_key, 3)
    mu_ran = jran.uniform(mu_key, minval=-1, maxval=1, shape=(n,))
    phi_ran = jran.uniform(phi_key, minval=0, maxval=2 * jnp.pi, shape=(n,))
    omega_ran = jran.uniform(omega_key, minval=0, maxval=2 * jnp.pi, shape=(n,))
    return mu_ran, phi_ran, omega_ran


def mc_ellipsoid_params(r50, b_over_a, c_over_a, ran_key, sample_omega=True):
    """Monte Carlo realization of 2d ellipse with random projection angles mu, phi, omega"""
    if sample_omega:
        mu_ran, phi_ran, omega_ran = mc_mu_phi_omega(r50.size, ran_key)
    else:
        mu_ran, phi_ran = mc_mu_phi(r50.size, ran_key)
        omega_ran = jnp.zeros_like(mu_ran)
    a = r50
    b = b_over_a * a
    c = c_over_a * a
    return compute_ellipse2d(a, b, c, mu_ran, phi_ran, omega_ran)


@partial(jjit, static_argnums=(6, 7))
def compute_ellipse2d(a, b, c, mu, phi, omega, envelop=True, ellipticity_type=0):
    """Compute 2d ellipse parameters defined line-of-sight projection of 3d ellipsoid

    Parameters
    ----------
    a, b, c : arrays, shape (n, )
        Length of the major, intermediate, and minor axes, respectively
            x-axis is the major axis (length a)
            y-axis is the intermediate axis (length b)
            z-axis is the minor axis (length c)

    mu, phi, omega : arrays, shape (n, )
        Angles defining the line-of-sight projection
        mu = cos(θ) pertains to the angle between the z-axis and line-of-sight direction
        phi is the azimuthal angle around the z-axis, in radians
        omega is the angle between the u-axis and the pivot axis (axis perpendicular to
        both the z-axis and the line-of-sight direction), in radians

    envelop : bool, static
        True  — correct 2D projection envelope via Schur complement of S33:
                  S_env = S_2d - (1/S33) s_col s_col^T
                where s_col = S_rotated[:2, 2].
                Derived by requiring the discriminant of the quadratic in z
                (S33 z² + 2(S13 x + S23 y) z + (quad in x,y - 1) = 0) to vanish.
        False — z=0 cross-section: S_2d = S_rotated[:2, :2] (approximate).

    ellipticity_type: int, optional
        Definition of the ellipticity returned in the Ellipse2DParams. See
            https://arxiv.org/abs/astro-ph/0107431
        Define the axis ratio q = beta/alpha of the projected 2D ellipse, then
        - 0 (Default): ellipticity = e = 1-q
        - 1: ellipticity = g = (1-q)/(1+q) (reduced shear)
        - 2: ellipticity = delta = (1-q^2)/(1+q^2) (distortion)
        - 3: ellipticity = eta = -ln(q) (conformal shear)

    Returns
    -------
    Ellipse2DParams : namedtuple of arrays of shape (n, )
        fields: (alpha, beta, psi, e1, e2, ellipticity, A, B, C)
            alpha : semi-major axis (β<α)
            beta : semi-minor axis (β<α)
            psi : angle in radians between semi-major axis α and u-axis
            e_alpha, e_beta : xy-coordinates of semi-major and semi-minor axes
            ellipticity : depends on ellipticity_type, 1 - beta/alpha by default
            A, B, C : Coefficients of ellipse equation Au² + Buv + Cv² = 1

    Notes
    -----
    a >= b >= c are defined by Eq 4 (Chen+16):
        (x/a)^2 + (y/b)^2 + (z/c)^2 = 1

        x-axis is the major axis (length a)
        y-axis is the intermediate axis (length b)
        z-axis is the minor axis (length c)

    θ is the angle between the z-axis and line-of-sight direction
    φ is the azimuthal angle around the z-axis, in radians
    ω is the angle between the u-axis in the 2D projected plane, and the pivot
        axis (axis perpendicular to both the z-axis and the LoS direction), in radians

    β and α are defined by Eq 11 (Chen+16):
        (U/α)^2 + (V/β)^2 = 1

    A, B, and C are defined by Eq 8 (Chen+16):
        Au^2 + Buv + Cv^2 = 1

    For projection angles {θ, Φ}, with μ ≡ cosθ, Eq 9 (Chen+16) defines the relationship:
        {A, B, C, θ, Φ} ==> {α, β}

    First compute eigenvalues {λ_1, λ_2} from the trace and discriminant of Eq 12 (Chen+16)
        λ_1 = 1 / β^2
        λ_2 = 1 / α^2

    Since λ_1 >= λ_2, then β<=α, so that β is semi-minor and α is semi-major

    e_alpha : uv coordinates of α
    e_beta : uv coordinates of β

    psi is the angle between the semi-major axis α and the u-axis in the 2D projected plane
        -π/2 < psi < π/2
        psi = 0 ==> semi-major axis α is aligned with u-axis,
            i.e., e_alpha = (1, 0), e_2=0
        psi = π/2 ==> semi-major axis α is aligned with v-axis,
            i.e., e_alpha = (0, 1), e_2=0
        psi = π/4 ==> semi-major axis α is aligned with (+u)-(+v) diagonal,
            i.e., e_alpha = (sqrt(2), sqrt(2)), e_1=0
        psi = -π/4 ==> semi-major axis α is aligned with (+u)-(-v) diagonal,
            i.e., e_alpha = (sqrt(2), -sqrt(2)), e_1=0
        psi > 0 ==> Counter-clockwise rotation of u-axis needed to align with α
        psi < 0 ==> Clockwise rotation of u-axis needed to align with α
    """
    A, B, C = _compute_2d_ellipse_params(a, b, c, mu, phi, omega, envelop=envelop)
    # get semi-major and semi-minor axes
    alpha, beta = _calculate_ellipse2d_axes(A, B, C)
    # get ellipticity
    q = beta / alpha
    if ellipticity_type == 0:
        ellipticity = 1.0 - q
    elif ellipticity_type == 1:
        ellipticity = (1.0 - q) / (1.0 + q)
    elif ellipticity_type == 2:
        ellipticity = (1.0 - q**2) / (1.0 + q**2)
    elif ellipticity_type == 3:
        ellipticity = -jnp.log(q)
    else:
        raise ValueError(
            f"Invalid ellipticity_type: {ellipticity_type}. Must be 0, 1, 2, or 3."
        )
    # get angle psi between semi-major axis and u-axis
    psi = _calculate_ellipse2d_psi(A, B, C)
    # get the coordinates of the semi-major and semi-minor unit-axes in the uv-plane
    e_alpha, e_beta = _get_xy_coords_of_projected_semi_axes(psi)
    ellipse2d = Ellipse2DParams(alpha, beta, psi, ellipticity, e_alpha, e_beta, A, B, C)
    return ellipse2d


@partial(jjit, static_argnums=(6,))
def _compute_2d_ellipse_params(a, b, c, mu, phi, omega, envelop=True):
    """Calculate the coefficients A, B, C through 3D-to-2D projection of a triaxial ellipsoid
    The projection works in the ellipsoid's principal axes frame, where the ellipsoid is defined by:
        (x/a)^2 + (y/b)^2 + (z/c)^2 = 1.
    The LoS is defined by the polar angle theta (mu = cos(theta)) and azimuthal angle phi (CCW from x).
    The third angle omega is the spin around the LoS after aligning it with z. Therefore, the combination
    of (mu, phi, omega) forms an Euler angle triplet in the ZXZ convention that rotates the x-y-z into
    the u-v-LoS. The Euler rotations are applied as follows:
    1. Rotate around the z-axis by phi + pi/2
    2. Rotate around the new x-axis by theta = arccos(mu)
    3. Rotate around the new z-axis (LoS) by omega.
    """
    # ── Shape matrix diagonal ───────────────────────────────────────────────
    d = jnp.stack([1.0 / a**2, 1.0 / b**2, 1.0 / c**2], axis=-1)  # (..., 3)

    # -- Rotation matrix that maps x-y-z to u-v-LoS (Euler ZXZ convention) ---
    theta = jnp.arccos(mu)  # polar angle from z-axis to LoS
    # passive rotation: R @ [x,y,z] = [u,v,LoS]
    R = _get_eulerxzx_matrix_from_angles(
        phi + jnp.pi / 2.0, theta, omega
    )  # (..., 3, 3)

    # -- Quadratic form in the u-v-LoS frame: S_rotated = R^T @ S @ R --------
    S_rotated = jnp.swapaxes(R, -1, -2) @ (d[..., :, None] * R)

    # ── 2D projection ────────────────────────────────────────────────────────
    if envelop:
        # Schur complement of S33: envelope of the full 3D projection
        # Derived from discriminant = 0 of the quadratic in z
        s_col = S_rotated[..., :2, 2]  # [S13, S23]
        S33 = S_rotated[..., 2, 2]
        S_2d = (
            S_rotated[..., :2, :2]
            - (s_col[..., :, None] * s_col[..., None, :]) / S33[..., None, None]
        )
    else:
        # z=0 cross-section (sub-matrix method)
        S_2d = S_rotated[..., :2, :2]

    # close-form eigen decomposition for 2x2 matrix
    A = S_2d[..., 0, 0]
    B = S_2d[..., 0, 1] + S_2d[..., 1, 0]
    C = S_2d[..., 1, 1]

    return A, B, C


@jjit
def _get_eulerxzx_matrix_from_angles(a, b, c):
    """Intrinsic ZXZ rotation: rotate by a about z, then by b about the
    new x, then by c about the newest z. First-applied rotation is
    leftmost: R = Rz(a) @ Rx(b) @ Rz(c). All angles are in radians.
    a, b, c may be scalars or arrays of shape (n,); the returned rotation
    matrices have shape (3, 3) or (n, 3, 3), respectively.
    """
    ca, sa = jnp.cos(a), jnp.sin(a)
    cb, sb = jnp.cos(b), jnp.sin(b)
    cc, sc = jnp.cos(c), jnp.sin(c)
    zero, one = jnp.zeros_like(ca), jnp.ones_like(ca)
    Rz_a = jnp.stack(
        [
            jnp.stack([ca, -sa, zero], axis=-1),
            jnp.stack([sa, ca, zero], axis=-1),
            jnp.stack([zero, zero, one], axis=-1),
        ],
        axis=-2,
    )
    Rx_b = jnp.stack(
        [
            jnp.stack([one, zero, zero], axis=-1),
            jnp.stack([zero, cb, -sb], axis=-1),
            jnp.stack([zero, sb, cb], axis=-1),
        ],
        axis=-2,
    )
    Rz_c = jnp.stack(
        [
            jnp.stack([cc, -sc, zero], axis=-1),
            jnp.stack([sc, cc, zero], axis=-1),
            jnp.stack([zero, zero, one], axis=-1),
        ],
        axis=-2,
    )
    return Rz_a @ Rx_b @ Rz_c


@jjit
def _get_eulerzxz_angle_from_basis(x_prime, y_prime):
    """
    Recover ZXZ intrinsic Euler angles (alpha, beta, gamma) from the rotated
    x'-y'-z' basis axes. (z' is calculated as x' cross y')

    Parameters
    ----------
    x_prime, y_prime : array_like, shape (..., 3)
        Vectors of the new x' and y' axes. The function will normalize them to unit
        vectors.

    Returns
    -------
    alpha, beta, gamma : ndarray
        Euler angles in radians.
    """
    x_ = jnp.asarray(x_prime, dtype=float)
    y_ = jnp.asarray(y_prime, dtype=float)

    # Normalize inputs to unit vectors
    x_ = x_ / jnp.linalg.norm(x_, axis=-1, keepdims=True)
    y_ = y_ / jnp.linalg.norm(y_, axis=-1, keepdims=True)
    z_ = jnp.cross(x_, y_)  # Ensure z' is orthogonal to x' and y'

    cos_beta = jnp.clip(z_[..., 2], -1.0, 1.0)
    beta = jnp.arccos(cos_beta)
    sin_beta = jnp.sin(beta)

    no_lock = jnp.abs(sin_beta) > 1e-10

    # General case: invert the analytic expressions above.
    # Gimbal lock fallback: alpha absorbs the free angle, gamma = 0.
    alpha = jnp.where(
        no_lock,
        jnp.arctan2(z_[..., 0], -z_[..., 1]),
        jnp.arctan2(x_[..., 1], x_[..., 0]),
    )
    gamma = jnp.where(no_lock, jnp.arctan2(x_[..., 2], y_[..., 2]), 0.0)

    # Wrap to required ranges: alpha, gamma in [0, 2pi]; beta already in [0, pi]
    alpha = alpha % (2.0 * jnp.pi)
    gamma = gamma % (2.0 * jnp.pi)

    return alpha, beta, gamma


@jjit
def _calculate_ellipse2d_axes(A, B, C):
    trace = A + C
    trace = jnp.maximum(trace, 0.0)

    det = A * C - 0.25 * B**2
    det = jnp.maximum(det, 0.0)

    discriminant = trace**2 - 4 * det
    discriminant = jnp.maximum(discriminant, 0.0)

    sqrt_discriminant = jnp.sqrt(discriminant)

    # Eigenvalues {λ_1, λ_2} (λ = 1/semi-axis²)
    lambda1 = 0.5 * (trace + sqrt_discriminant)  # Larger eigenvalue
    lambda2 = 0.5 * (trace - sqrt_discriminant)  # Smaller eigenvalue

    msk_neg = lambda2 <= 0  # edge case where trace=sqrt_discriminant (β=α)
    # set edge case to circle with unit length
    lambda1 = jnp.where(msk_neg, 1.0, lambda1)
    lambda2 = jnp.where(msk_neg, 1.0, lambda2)

    # Calculate semi-axes β and α
    alpha = jnp.sqrt(1.0 / lambda2)  # Semi-major axis (from smaller eigenvalue)
    beta = jnp.sqrt(1.0 / lambda1)  # Semi-minor axis (from larger eigenvalue)

    return alpha, beta


@jjit
def _calculate_ellipse2d_psi(A, B, C):
    """Get angle psi between semi-major axis and u-axis [-π/2, π/2)"""
    psi = 0.5 * jnp.arctan2(B, A - C) + jnp.pi / 2.0
    psi = jnp.mod(psi + jnp.pi / 2, jnp.pi) - jnp.pi / 2
    return psi


@jjit
def _get_xy_coords_of_projected_semi_axes(psi):
    x_alpha = jnp.cos(psi)
    y_alpha = jnp.sin(psi)

    x_beta = jnp.cos(psi + jnp.pi / 2)
    y_beta = jnp.sin(psi + jnp.pi / 2)

    e_alpha = jnp.array((x_alpha, y_alpha)).T
    e_beta = jnp.array((x_beta, y_beta)).T

    return e_alpha, e_beta


@jjit
def _transform_axes_to_frame(A, x_prime, y_prime, z_prime):
    """
    Re-express vector A in the frame (x', y', z').
    All inputs shape (N, 3). Returns A', shape (N, 3).
    """

    def _dot(V, frame_ax):
        return jnp.sum(V * frame_ax, axis=-1)

    def _to_frame(V):
        return jnp.stack(
            [_dot(V, x_prime), _dot(V, y_prime), _dot(V, z_prime)], axis=-1
        )

    return _to_frame(A)
