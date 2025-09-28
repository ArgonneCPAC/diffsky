"""Calculate 2d ellipse from projecting a 3d triaxial ellipsoid

Adapt Chen+16 to JAX
https://ui.adsabs.harvard.edu/abs/2016ApJ...830..123C/abstract

"""

from collections import namedtuple

from jax import jit as jjit
from jax import numpy as jnp

Ellipse2DParams = namedtuple(
    "Ellipse2DParams",
    (
        "beta",
        "alpha",
        "psi",
        "ellipticity",
        "e_beta",
        "e_alpha",
        "A",
        "B",
        "C",
        "Delta",
    ),
)


@jjit
def compute_ellipse2d(a, b, c, mu, phi):
    """Compute 2d ellipse parameters defined line-of-sight projection of 3d ellipsoid

    Parameters
    ----------
    A, B, C : arrays, shape (n, )

    Returns
    -------
    Ellipse2DParams : namedtuple of arrays of shape (n, )
        fields: (beta, alpha, psi, e1, e2, ellipticity, A, B, C, Delta)
            beta : semi-major axis (α<β)
            alpha : semi-minor axis (α<β)
            psi : angle in radians between semi-major axis β and x-axis
            e_beta, e_alpha : xy-coordinates of semi-major and semi-minor axes
            ellipticity : ε ≡ 1 - alpha/beta
            A, B, C : Coefficients of ellipse equation Ax² + Bxy + Cy² = 1
            Delta : denominator of equations defining A, B, C

    Notes
    -----
    a >= b >= c are defined by Eq 4:
        (x/a)^2 + (y/b)^2 + (z/c)^2 = 1

        x-axis is the major axis (length a)
        y-axis is the intermediate axis (length b)
        z-axis is the minor axis (length c)

    θ is the angle between the z-axis and line-of-sight direction
    φ is the azimuthal angle around the z-axis

    α and β are defined by Eq 11:
        (X/α)^2 + (Y/β)^2 = 1

    A, B, and C are defined by Eq 8:
        Ax^2 + Bxy + Cy^2 = 1

    For projection angles {θ, Φ}, with μ ≡ cosθ, Eq 9 defines the relationship:
        {A, B, C, θ, Φ} ==> {α, β}

    First compute eigenvalues {λ_1, λ_2} from the trace and discriminant of Eq 12
        λ_1 = 1 / α^2
        λ_2 = 1 / β^2

    Since λ_1 >= λ_2, then α<=β, so that α is semi-minor and β is semi-major

    e_beta : xy coordinates of β
    e_alpha : xy coordinates of α

    psi = 2ω is defined by the last line in Eq 12
        -π < psi < π
        psi = 0 ==> semi-major axis β is aligned with x-axis, i.e., e_beta = (1, 0)
        psi > 0 ==> Counter-clockwise rotation of x-axis needed to align with β
        psi < 0 ==> Clockwise rotation of x-axis needed to align with β

        psi = π/2 ==> semi-major axis β is aligned with y-axis, i.e., e_beta = (0, 1)
        psi = -π/2 ==> semi-major axis β anti-aligned with y-axis, e_beta = (0, -1)
        psi = +/-π ==> semi-major axis β is aligned with -x-axis, i.e., e_beta = (-1, 0)

    """
    A, B, C, Delta = _compute_2d_ellipse_params(a, b, c, mu, phi)
    beta, alpha = _calculate_ellipse2d_axes(A, B, C)
    ellipticity = 1.0 - alpha / beta
    omega = _calculate_ellipse2d_omega(A, B, C)
    psi = 2.0 * omega
    e_beta, e_alpha = _get_xy_coords_of_projected_semi_axes(psi)
    ellipse2d = Ellipse2DParams(
        beta, alpha, psi, ellipticity, e_beta, e_alpha, A, B, C, Delta
    )
    return ellipse2d


@jjit
def _compute_2d_ellipse_params(a, b, c, mu, phi):
    """
    Compute ellipse coefficients {A, B, C, Δ} defined by Eq. 8:
        Ax² + Bxy + Cy² = 1

    """
    # Precompute trig and define convenient notation
    cos_phi = jnp.cos(phi)
    sin_phi = jnp.sin(phi)
    cos2_phi = cos_phi**2
    sin2_phi = sin_phi**2
    mu2 = mu**2
    one_minus_mu2 = 1.0 - mu2

    # Calculate Δ after some regrouping of terms in Eq 9
    term1 = a**2 * (cos2_phi + sin2_phi * one_minus_mu2)
    term2 = b**2 * mu2
    term3 = c**2 * (sin2_phi + cos2_phi * one_minus_mu2)
    Delta = term1 + term2 + term3

    A = (c**2 * cos2_phi + a**2 * sin2_phi + b**2 * mu2) / Delta

    B = 2 * (a**2 - c**2) * cos_phi * sin_phi * one_minus_mu2 / Delta

    C = (c**2 * sin2_phi + a**2 * cos2_phi + b**2 * mu2) / Delta

    return A, B, C, Delta


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

    msk_neg = lambda2 <= 0  # edge case where trace=sqrt_discriminant (α=β)
    # set edge case to circle with unit length
    lambda1 = jnp.where(msk_neg, 1.0, lambda1)
    lambda2 = jnp.where(msk_neg, 1.0, lambda2)

    # Calculate semi-axes α and β
    beta = jnp.sqrt(1.0 / lambda2)  # Semi-major axis (from smaller eigenvalue)
    alpha = jnp.sqrt(1.0 / lambda1)  # Semi-minor axis (from larger eigenvalue)

    return beta, alpha


@jjit
def _calculate_ellipse2d_omega(A, B, C):
    """Variable ω defined in Eq 12"""
    omega = 0.5 * jnp.arctan2(B, A - C)
    return omega


@jjit
def _get_xy_coords_of_projected_semi_axes(psi):
    x_beta = jnp.cos(psi)
    y_beta = jnp.sin(psi)

    x_alpha = jnp.cos(psi + jnp.pi / 2)
    y_alpha = jnp.sin(psi + jnp.pi / 2)

    e_beta = jnp.array((x_beta, y_beta)).T
    e_alpha = jnp.array((x_alpha, y_alpha)).T

    return e_beta, e_alpha
