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


@partial(jjit, static_argnums=(6,7))
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
        raise ValueError(f"Invalid ellipticity_type: {ellipticity_type}. Must be 0, 1, 2, or 3.")
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
    d = jnp.stack([1.0/a**2, 1.0/b**2, 1.0/c**2], axis=-1)  # (..., 3)

    # -- Rotation matrix that maps x-y-z to u-v-LoS (Euler ZXZ convention) ---
    theta = jnp.arccos(mu)  # polar angle from z-axis to LoS
    # passive rotation: R @ [x,y,z] = [u,v,LoS]
    R = _Euler_ZXZ_(phi+jnp.pi/2., theta, omega)  # (..., 3, 3)

    # -- Quadratic form in the u-v-LoS frame: S_rotated = R^T @ S @ R --------
    S_rotated = jnp.swapaxes(R, -1, -2) @ (d[..., :, None] * R)

    # ── 2D projection ────────────────────────────────────────────────────────
    if envelop:
        # Schur complement of S33: envelope of the full 3D projection
        # Derived from discriminant = 0 of the quadratic in z
        s_col = S_rotated[..., :2, 2]                        # [S13, S23]
        S33   = S_rotated[..., 2, 2]
        S_2d  = S_rotated[..., :2, :2] - (
            s_col[..., :, None] * s_col[..., None, :]
        ) / S33[..., None, None]
    else:
        # z=0 cross-section (sub-matrix method)
        S_2d = S_rotated[..., :2, :2]

    # close-form eigen decomposition for 2x2 matrix
    A = S_2d[..., 0, 0]
    B = S_2d[..., 0, 1] + S_2d[..., 1, 0]
    C = S_2d[..., 1, 1]

    return A, B, C

@jjit
def _Euler_ZXZ_(a, b, c):
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

# ── Unit test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    """Spinning-top GIF: independent-construction visual unit test.

    Goal: confirm that compute_ellipse2d's analytic (alpha, beta, psi)
    prediction matches the true silhouette of an explicitly-rotated 3D
    ellipsoid wireframe -- where the wireframe's rotation is built WITHOUT
    using any of this module's own rotation/projection code (_Euler_ZXZ_,
    _project_ellipsoid_scalar, compute_ellipse2d), only elementary vector
    geometry (cross products). This is the function under test being checked
    against an independently-derived reference, not against itself.

    Reference frame: the ellipsoid's own principal-axis body frame (x, y, z).
    Observation frame: (u, v, w), with w along the line of sight.

    The static ellipsoid surface is parametrized directly in body coordinates:
        (a sin(alpha) cos(beta), b sin(alpha) sin(beta), c cos(alpha))
    with alpha the polar angle from body z and beta the azimuthal angle;
    lines of constant alpha or beta trace out the wireframe.

    For each (mu, phi, omega) along the trajectory, (u, v, w) are built directly:
        w     = (sqrt(1-mu^2) cos(phi), sqrt(1-mu^2) sin(phi), mu)
        pivot = normalize(z x w)
        aux   = w x pivot                     (already unit: w, pivot, aux orthonormal)
        u     =  cos(omega) pivot + sin(omega) aux
        v     = -sin(omega) pivot + cos(omega) aux
    A self-check below confirms this equals _Euler_ZXZ_(phi+pi/2, arccos(mu),
    omega)'s columns exactly (verified analytically too -- see the function
    docstring) -- but the wireframe itself only ever uses (u, v, w) via plain
    dot products, never that matrix.

    Each body-frame wireframe point is expressed in the (u, v, w) frame by
    dotting with (u, v, w); dropping the w-component gives its orthogonal
    projection along the line of sight. The GIF overlays this independently
    projected wireframe against compute_ellipse2d's analytic ellipse (the
    function under test) for the same (mu, phi, omega). Agreement verifies
    the envelope math in compute_ellipse2d.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from mpl_toolkits.mplot3d import Axes3D   # noqa: F401

    # ── Configuration ─────────────────────────────────────────────────────
    SEMI_A, SEMI_B, SEMI_C = 2.0, 1.5, 0.5   # ellipsoid semi-axes (a >= b >= c)
    N_FRAMES      = 96                        # total animation frames
    N_PHI_TURNS   = 2.0                       # precession turns over the animation
    N_OMEGA_TURNS = 7.0                       # spin turns over the animation
    FPS           = 12
    DPI           = 90
    SAVE_PATH     = "ellipsoid_projection.gif"

    # ── Trajectory of (mu, phi, omega) ──────────────────────────────────────
    # mu kept strictly inside (-1, 1): pivot = z x w is undefined exactly at
    # mu = +/-1 (a feature of this elementary cross-product construction --
    # _Euler_ZXZ_ itself has no such singularity, but the whole point here is
    # to avoid depending on it).
    _t         = np.linspace(0.0, 1.0, N_FRAMES)
    mu_traj    = 0.995 * np.cos(_t * np.pi)          # +0.995 -> -0.995 (face-on -> edge-on -> face-on)
    phi_traj   = _t * N_PHI_TURNS * 2 * np.pi
    omega_traj = _t * N_OMEGA_TURNS * 2 * np.pi

    # ── Independent (u, v, w) construction -- no diffsky rotation code used ──
    def _uvw_from_mu_phi_omega(mu, phi, omega):
        s = np.sqrt(max(1.0 - mu**2, 0.0))
        w = np.array([s * np.cos(phi), s * np.sin(phi), mu])
        pivot = np.cross([0.0, 0.0, 1.0], w)
        pivot /= np.linalg.norm(pivot)
        aux = np.cross(w, pivot)
        u = np.cos(omega) * pivot + np.sin(omega) * aux
        v = -np.sin(omega) * pivot + np.cos(omega) * aux
        return u, v, w

    # Self-check (not part of the wireframe pipeline below): confirm the
    # elementary construction above agrees with _Euler_ZXZ_, as derived
    # analytically in the docstring.
    _rng = np.random.default_rng(0)
    _max_err = 0.0
    for _ in range(200):
        _mu, _phi, _om = (_rng.uniform(-0.999, 0.999), _rng.uniform(-np.pi, np.pi),
                          _rng.uniform(-np.pi, np.pi))
        _u, _v, _w = _uvw_from_mu_phi_omega(_mu, _phi, _om)
        _R = np.array(_Euler_ZXZ_(_phi + np.pi / 2.0, np.arccos(_mu), _om))
        _max_err = max(_max_err, np.max(np.abs(np.stack([_u, _v, _w], axis=1) - _R)))
    print(f"[self-check] (u,v,w) construction vs _Euler_ZXZ_: max error = {_max_err:.2e}")
    assert _max_err < 1e-6, "independent (u,v,w) construction disagrees with _Euler_ZXZ_"

    # ── Static ellipsoid wireframe & surface, in BODY coordinates ────────────
    def _wireframe_lines_body(n_lat=7, n_lon=10):
        """Lines of constant polar angle (latitude) and azimuth (longitude)."""
        segs = []
        for alpha0 in np.linspace(0.1 * np.pi, 0.9 * np.pi, n_lat):
            beta = np.linspace(0.0, 2 * np.pi, 150)
            segs.append(np.stack([
                SEMI_A * np.sin(alpha0) * np.cos(beta),
                SEMI_B * np.sin(alpha0) * np.sin(beta),
                SEMI_C * np.full_like(beta, np.cos(alpha0)),
            ], axis=-1))
        for beta0 in np.linspace(0.0, 2 * np.pi, n_lon + 1)[:-1]:
            alpha = np.linspace(0.0, np.pi, 80)
            segs.append(np.stack([
                SEMI_A * np.sin(alpha) * np.cos(beta0),
                SEMI_B * np.sin(alpha) * np.sin(beta0),
                SEMI_C * np.cos(alpha),
            ], axis=-1))
        return segs

    _wire_body = _wireframe_lines_body()

    _beta_grid, _alpha_grid = np.meshgrid(np.linspace(0, 2 * np.pi, 48),
                                           np.linspace(0, np.pi, 24))
    _surf_body = np.stack([
        SEMI_A * np.sin(_alpha_grid) * np.cos(_beta_grid),
        SEMI_B * np.sin(_alpha_grid) * np.sin(_beta_grid),
        SEMI_C * np.cos(_alpha_grid),
    ], axis=-1)                                    # (n_lat, n_lon, 3)

    _body_zaxis = np.array([0.0, 0.0, SEMI_C])     # apex of the alpha=0 line: the body's own symmetry axis

    def _to_uvw(points_body, u, v, w):
        """points_body: (..., 3) body coords -> (..., 3) coords along (u, v, w)."""
        return np.stack([points_body @ u, points_body @ v, points_body @ w], axis=-1)

    # ── Prediction from compute_ellipse2d, over the whole trajectory ────────
    # This is the function under test -- called once, vectorised over frames.
    _ones = jnp.ones(N_FRAMES)
    res = compute_ellipse2d(
        _ones * SEMI_A, _ones * SEMI_B, _ones * SEMI_C,
        jnp.array(mu_traj), jnp.array(phi_traj), jnp.array(omega_traj),
        envelop=True,
    )
    pred = {k: np.array(getattr(res, k))
            for k in ("alpha", "beta", "psi", "e_alpha", "e_beta")}

    _COORD_CLR = {"u": "#d62728", "v": "#2ca02c", "w": "#1f77b4"}
    LIM = max(SEMI_A, SEMI_B, SEMI_C) * 1.5

    def _ellipse_curve_uv(pa, pb, ea, eb, n=100):
        """Parametric boundary of the predicted ellipse in the (u,v) plane."""
        t = np.linspace(0.0, 2 * np.pi, n)
        return (pa * np.cos(t)[:, None] * ea) + (pb * np.sin(t)[:, None] * eb)

    # ── Per-frame drawing (single combined 3D panel) ─────────────────────────

    def _draw(ax, i, u, v, w):
        """3D ellipsoid wireframe/surface in the (u,v,w) observation frame
        (ellipsoid body-z axis annotated in black), with the independently
        projected wireframe shadow and compute_ellipse2d's analytic ellipse
        (dark red) both drawn on the floor plane w = -LIM."""
        ax.cla()

        surf_uvw = _to_uvw(_surf_body, u, v, w)
        ax.plot_surface(surf_uvw[..., 0], surf_uvw[..., 1], surf_uvw[..., 2],
                         alpha=0.22, color="steelblue", rstride=1, cstride=1,
                         linewidth=0.3, edgecolor="steelblue")
        for seg in _wire_body:
            seg_uvw = _to_uvw(seg, u, v, w)
            ax.plot(seg_uvw[:, 0], seg_uvw[:, 1], seg_uvw[:, 2],
                    color="steelblue", alpha=0.35, lw=0.6)

        # Fixed (u, v, w) axes of the observation frame
        for lbl, vec in zip(["u", "v", "w"], np.eye(3)):
            ax.quiver(0, 0, 0, *(vec * LIM * 0.60),
                      color=_COORD_CLR[lbl], lw=1.0,
                      arrow_length_ratio=0.14, alpha=0.6)

        # Ellipsoid's own symmetry (body z) axis, expressed in (u, v, w)
        z_uvw = _to_uvw(_body_zaxis, u, v, w)
        ax.quiver(0, 0, 0, *z_uvw, color="k", lw=2.5,
                  arrow_length_ratio=0.15, label="body z-axis")

        # ── Floor (w = -LIM): independently-built wireframe shadow versus
        #    compute_ellipse2d's analytic ellipse -- the actual check ──────
        floor = -LIM
        surf_uv = surf_uvw[..., :2]
        ax.scatter(surf_uv[..., 0].ravel(), surf_uv[..., 1].ravel(),
                   np.full(surf_uv[..., 0].size, floor),
                   s=1, color="steelblue", alpha=0.15)
        for seg in _wire_body:
            seg_uv = _to_uvw(seg, u, v, w)[:, :2]
            ax.plot(seg_uv[:, 0], seg_uv[:, 1], np.full(len(seg_uv), floor),
                    color="steelblue", alpha=0.35, lw=0.7)
        _px, _py = np.meshgrid([-LIM, LIM], [-LIM, LIM])
        ax.plot_surface(_px, _py, np.full_like(_px, floor),
                        alpha=0.06, color="gray", linewidth=0)

        # compute_ellipse2d's analytic prediction (function under test)
        pa, pb = pred["alpha"][i], pred["beta"][i]
        ea, eb = pred["e_alpha"][i], pred["e_beta"][i]
        curve = _ellipse_curve_uv(pa, pb, ea, eb)
        ax.plot(curve[:, 0], curve[:, 1], np.full(len(curve), floor),
                color="darkred", lw=2.2,
                label=rf"compute_ellipse2d  $\alpha$={pa:.2f}  $\beta$={pb:.2f}")
        ax.quiver(0, 0, floor, *(pa * ea), 0, color="darkred", lw=1.8,
                  arrow_length_ratio=0.15)
        ax.quiver(0, 0, floor, *(pb * eb), 0, color="green", lw=1.8,
                  arrow_length_ratio=0.15)

        ax.set_xlim(-LIM, LIM); ax.set_ylim(-LIM, LIM); ax.set_zlim(-LIM, LIM)
        ax.set_xlabel("u", labelpad=1); ax.set_ylabel("v", labelpad=1)
        ax.set_zlabel("w", labelpad=1)
        ax.set_title(
            rf"$\mu$={mu_traj[i]:+.2f}  $\phi$={np.degrees(phi_traj[i]) % 360:.0f}°"
            rf"  $\omega$={np.degrees(omega_traj[i]) % 360:.0f}°"
            rf"  |  frame {i + 1}/{N_FRAMES}",
            fontsize=9, pad=4,
        )
        ax.legend(fontsize=8, loc="upper left")

    def _update(i):
        """FuncAnimation callback: redraw the combined panel for frame i."""
        u, v, w = _uvw_from_mu_phi_omega(mu_traj[i], phi_traj[i], omega_traj[i])
        _draw(ax3d, i, u, v, w)
        fig.suptitle(rf"Ellipsoid  $a$={SEMI_A}  $b$={SEMI_B}  $c$={SEMI_C}",
                     fontsize=10, y=0.98)

    # ── Render and save ───────────────────────────────────────────────────
    SLOWDOWN   = 5.0                 # play the saved GIF 5x slower
    SAVE_FPS   = FPS / SLOWDOWN

    fig  = plt.figure(figsize=(8, 7))
    ax3d = fig.add_subplot(1, 1, 1, projection="3d")

    ani = animation.FuncAnimation(
        fig, _update, frames=N_FRAMES,
        interval=SLOWDOWN * 1000 // FPS, blit=False,
    )
    ani.save(SAVE_PATH, writer="pillow", dpi=DPI, fps=SAVE_FPS)
    plt.close(fig)
    print(f"Saved → {SAVE_PATH}")
