"""Calculate 2d ellipse from projecting a 3d triaxial ellipsoid

Adapt Chen+16 to JAX
https://ui.adsabs.harvard.edu/abs/2016ApJ...830..123C/abstract

"""

from collections import namedtuple

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


def mc_ellipsoid_params(r50, b_over_a, c_over_a, ran_key):
    """Monte Carlo realization of 2d ellipse with random projection angles mu, phi"""
    los_key, psi_key = jran.split(ran_key, 2)
    mu_ran, phi_ran = mc_mu_phi(r50.size, los_key)
    a = r50
    b = b_over_a * a
    c = c_over_a * a
    ellipse2d = compute_ellipse2d(a, b, c, mu_ran, phi_ran)
    psi = jran.uniform(psi_key, minval=-jnp.pi, maxval=jnp.pi, shape=r50.shape)
    ellipse2d = ellipse2d._replace(psi=psi)
    return ellipse2d


@jjit
def compute_ellipse2d(a, b, c, mu, phi):
    """Compute 2d ellipse parameters defined line-of-sight projection of 3d ellipsoid

    Parameters
    ----------
    a, b, c : arrays, shape (n, )
        Length of the major, intermediate, and minor axes, respectively
            x-axis is the major axis (length a)
            y-axis is the intermediate axis (length b)
            z-axis is the minor axis (length c)

    mu, phi : arrays, shape (n, )
        Angles defining the line-of-sight projection
        mu = cos(θ) pertains to the angle between the z-axis and line-of-sight direction
        phi is the azimuthal angle around the z-axis

    Returns
    -------
    Ellipse2DParams : namedtuple of arrays of shape (n, )
        fields: (alpha, beta, psi, e1, e2, ellipticity, A, B, C)
            alpha : semi-major axis (β<α)
            beta : semi-minor axis (β<α)
            psi : angle in radians between semi-major axis α and x-axis
            e_alpha, e_beta : xy-coordinates of semi-major and semi-minor axes
            ellipticity : ε ≡ 1 - beta/alpha
            A, B, C : Coefficients of ellipse equation Ax² + Bxy + Cy² = 1

    Notes
    -----
    a >= b >= c are defined by Eq 4:
        (x/a)^2 + (y/b)^2 + (z/c)^2 = 1

        x-axis is the major axis (length a)
        y-axis is the intermediate axis (length b)
        z-axis is the minor axis (length c)

    θ is the angle between the z-axis and line-of-sight direction
    φ is the azimuthal angle around the z-axis

    β and α are defined by Eq 11:
        (X/α)^2 + (Y/β)^2 = 1

    A, B, and C are defined by Eq 8:
        Ax^2 + Bxy + Cy^2 = 1

    For projection angles {θ, Φ}, with μ ≡ cosθ, Eq 9 defines the relationship:
        {A, B, C, θ, Φ} ==> {α, β}

    First compute eigenvalues {λ_1, λ_2} from the trace and discriminant of Eq 12
        λ_1 = 1 / β^2
        λ_2 = 1 / α^2

    Since λ_1 >= λ_2, then β<=α, so that β is semi-minor and α is semi-major

    e_alpha : xy coordinates of α
    e_beta : xy coordinates of β

    psi = 2ω is defined by the last line in Eq 12
        -π < psi < π
        psi = 0 ==> semi-major axis α is aligned with x-axis, i.e., e_alpha = (1, 0)
        psi > 0 ==> Counter-clockwise rotation of x-axis needed to align with α
        psi < 0 ==> Clockwise rotation of x-axis needed to align with α

        psi = π/2 ==> semi-major axis α is aligned with y-axis, i.e., e_alpha = (0, 1)
        psi = -π/2 ==> semi-major axis α anti-aligned with y-axis, e_alpha = (0, -1)
        psi = +/-π ==> semi-major axis α is aligned with -x-axis, i.e., e_alpha = (-1, 0)

    """
    A, B, C = _compute_2d_ellipse_params(a, b, c, mu, phi)
    alpha, beta = _calculate_ellipse2d_axes(A, B, C)
    ellipticity = 1.0 - beta / alpha
    omega = _calculate_ellipse2d_omega(A, B, C)
    psi = 2.0 * omega
    e_alpha, e_beta = _get_xy_coords_of_projected_semi_axes(psi)
    ellipse2d = Ellipse2DParams(alpha, beta, psi, ellipticity, e_alpha, e_beta, A, B, C)
    return ellipse2d


@jjit
def _compute_2d_ellipse_params(a, b, c, mu, phi):
    """See Equation 9 (fixes error in paper - simplified derivation uses z=0)"""
    cos_phi = jnp.cos(phi)
    sin_phi = jnp.sin(phi)
    cos2_phi = cos_phi**2
    sin2_phi = sin_phi**2
    mu2 = mu**2
    one_minus_mu2 = 1.0 - mu2
    a2 = a * a
    b2 = b * b
    c2 = c * c

    A = (mu2 * cos2_phi / a2) + (mu2 * sin2_phi / b2) + (one_minus_mu2 / c2)
    C = (sin2_phi / a2) + (cos2_phi / b2)
    B = 2 * mu * sin_phi * cos_phi * (1 / b2 - 1 / a2)

    return A, B, C


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
def _calculate_ellipse2d_omega(A, B, C):
    """Variable ω defined in Eq 12"""
    omega = 0.5 * jnp.arctan2(B, A - C)
    return omega


@jjit
def _get_xy_coords_of_projected_semi_axes(psi):
    x_alpha = jnp.cos(psi)
    y_alpha = jnp.sin(psi)

    x_beta = jnp.cos(psi + jnp.pi / 2)
    y_beta = jnp.sin(psi + jnp.pi / 2)

    e_alpha = jnp.array((x_alpha, y_alpha)).T
    e_beta = jnp.array((x_beta, y_beta)).T

    return e_alpha, e_beta
