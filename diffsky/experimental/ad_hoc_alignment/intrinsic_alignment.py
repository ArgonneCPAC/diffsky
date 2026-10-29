""" Standalone script for alignment functions.
This script collects functions needed for a central + radial alignment strategy for IA injection from halotools.
Original functions developed by Nick van Alfen, altered by Jiachuan Xu to be compatible with DiffSky Mock Catalog.
"""
import math
from functools import partial
import jax
import jax.numpy as jnp
from jax import jit as jjit
from jax import random as jran
from jax import lax
from jax.scipy.special import dawsn
from diffsky.ellipsoidal_shapes import ellipse_proj_kernels as eproj


def align_to_halo(key, halo_axisA_x, halo_axisA_y, halo_axisA_z,
                  alignment_strength, prim_gal_axis='A'):
    """ Aligns the galaxy axes to the halo axes with a given alignment strength.
    Parameters
    ----------
    halo_axisA_x, halo_axisA_y, halo_axisA_z : array_like, shape (N,)
        The x, y, and z components of the halo's major axis (A-axis)
    alignment_strength : float or array_like, shape (N,)
        The strength of the alignment, defined for the DW distribution. If a float, it is applied to all galaxies.
        If an array, it must have the same length as the number of galaxies.
    prim_gal_axis : str, optional
        The primary axis of the galaxy to align. Can be 'A', 'B', or 'C'. Default is 'A'.
    key : jax.random.PRNGKey
        Random key for JAX random number generation.
    Returns
    -------
    aligned_galaxy_axes : (array_like, array_like, array_like), shape (N, 3) each
        The aligned galaxy axes after applying the alignment transformation.
    """
    alignment_strength = _broadcast_alignment_strength(
        alignment_strength, len(halo_axisA_x)
    )

    # set prim_gal_axis orientation
    major_input_vectors = jnp.stack((halo_axisA_x, halo_axisA_y, halo_axisA_z), axis=-1)

    return align_to_axis(key, major_input_vectors, alignment_strength, prim_gal_axis)


def align_radially(key, halo_x, halo_y, halo_z,
                   x, y, z, alignment_strength, Lbox, prim_gal_axis='A'):
    """ Aligns the galaxy axes radially to the halo center with a given alignment strength.
    Parameters
    ----------
    halo_x, halo_y, halo_z : array_like, shape (N,)
        The x, y, and z coordinates of the halo center.
    x, y, z : array_like, shape (N,)
        The x, y, and z coordinates of the galaxies.
    Lbox : array_like, shape (3,)
        The size of the simulation box (for periodic boundary conditions).
    alignment_strength : float or array_like, shape (N,)
        The strength of the alignment, defined for the DW distribution. If a float, it is applied to all galaxies.
        If an array, it must have the same length as the number of galaxies.
    prim_gal_axis : str, optional
        The primary axis of the galaxy to align. Can be 'A', 'B', or 'C'. Default is 'A'.
    key : jax.random.PRNGKey
        Random key for JAX random number generation.
    Returns
    -------
    aligned_galaxy_axes : (array_like, array_like, array_like), shape (N, 3) each
        The aligned galaxy axes after applying the radial alignment transformation.
    """
    alignment_strength = _broadcast_alignment_strength(alignment_strength, len(halo_x))

    # set prim_gal_axis orientation. Galaxies at the host center have a zero
    # radial vector; align_to_axis gives those rows an isotropic orientation with
    # a fixed-shape, per-row selection, so the random draws never depend on how
    # many such galaxies there are (keeps this jit-compatible, and every other
    # galaxy's realization independent of them)
    major_input_vectors, _ = get_radial_vector(halo_x, halo_y, halo_z, x, y, z, Lbox)

    return align_to_axis(key, major_input_vectors, alignment_strength, prim_gal_axis)


def _broadcast_alignment_strength(alignment_strength, n):
    """ Broadcast a scalar or per-galaxy alignment strength to shape (n,).
    Accepts Python/numpy scalars and numpy/jax arrays; only the shape is checked,
    so this also works on traced values inside jitted functions.
    """
    alignment_strength = jnp.asarray(alignment_strength)
    if alignment_strength.ndim == 0:
        return jnp.full((n,), alignment_strength)
    if alignment_strength.shape != (n,):
        raise ValueError(
            f"alignment_strength must be a scalar or have shape ({n},), "
            f"got shape {alignment_strength.shape}"
        )
    return alignment_strength


@partial(jjit, static_argnums=(3,))
def align_to_axis(key, major_input_vectors, alignment_strength, prim_gal_axis="A"):
    """ Aligns the galaxy axes to a given vector with a specified alignment strength.
    Parameters
    ----------
    major_input_vectors : array_like, shape (N, 3)
        The input vectors to which the galaxy axes will be aligned.
    alignment_strength : float or array_like, shape (N,)
        The strength of the alignment, defined for the DW distribution. If a float, it is applied to all galaxies.
        If an array, it must have the same length as the number of galaxies.
    prim_gal_axis : str, optional
        The primary axis of the galaxy to align. Can be 'A', 'B', or 'C'. Default is 'A'.
    key : jax.random.PRNGKey
        Random key for JAX random number generation.
    Returns
    -------
    major_v, inter_v, minor_v : array_like, shape (N, 3)
        The aligned galaxy axes after applying the alignment transformation. The axes are returned in the order of major, intermediate, and minor axes.
    """
    # prim_gal_axis is a static argument, so it is a plain Python str at trace
    # time and can be validated with an ordinary raise (once per compilation)
    if prim_gal_axis not in ("A", "B", "C"):
        raise ValueError(
            f"Invalid prim_gal_axis: {prim_gal_axis!r}. Must be 'A', 'B', or 'C'."
        )
    alignment_strength = _broadcast_alignment_strength(
        alignment_strength, major_input_vectors.shape[0]
    )

    # Degenerate input vectors (zero, nan or inf, e.g. an unmeasured halo shape)
    # get a random orientation: mu = 0 about a placeholder axis. Flagging them from
    # the input keeps every other row's random draws unchanged.
    ref_norm = jnp.linalg.norm(major_input_vectors, axis=-1)
    is_bad = ~(jnp.isfinite(ref_norm) & (ref_norm > 0))
    ref = jnp.where(is_bad[:, None], jnp.array([1.0, 0.0, 0.0]), major_input_vectors)
    alignment_strength = jnp.where(is_bad, 0.0, alignment_strength)
    n_bad = jnp.sum(is_bad)
    lax.cond(
        n_bad > 0,
        lambda _: jax.debug.print(
            "{n} input alignment axis(axes) are degenerate (zero or not finite); "
            "those galaxies are given random orientations.", n=n_bad),
        lambda _: None,
        operand=None,
    )

    # B is spun about A by a fixed random angle within the smooth Watson frame,
    # so the whole triad varies smoothly with mu (needed for gradient-based fits)
    key_frame, key_spin = jran.split(key)
    A_v, B0, W = _sample_watson_frame(key_frame, ref, alignment_strength)
    spin = 2.0 * jnp.pi * jran.uniform(key_spin, (A_v.shape[0], 1))
    B_v = jnp.cos(spin) * B0 + jnp.sin(spin) * W
    C_v = jnp.cross(A_v, B_v)

    # depending on the prim_gal_axis, assign correlated axes
    if prim_gal_axis == 'A':
        major_v = A_v
        inter_v = B_v
        minor_v = C_v
    elif prim_gal_axis == 'B':
        major_v = C_v
        inter_v = A_v
        minor_v = B_v
    else:  # prim_gal_axis == 'C', validated above
        major_v = B_v
        inter_v = C_v
        minor_v = A_v

    return major_v, inter_v, minor_v


@jjit
def get_radial_vector(halo_x, halo_y, halo_z, x, y, z, Lbox):
    """
    caclulate the radial vector for satellite galaxies
    Parameters
    ----------
    x, y, z : array_like, shape (N,)
        galaxy positions
    halo_x, halo_y, halo_z : array_like, shape (N,)
        host halo positions
    Lbox : array_like
        array len(3) giving the simulation box size along each dimension
    Returns
    -------
    r_vec : array_like, shape (N, 3)
        array of radial vectors of shape (N, 3) between host haloes and satellites
    r : array_like, shape (N,)
        radial distance
    """
    # define halo-center - satellite vector
    # accounting for PBCs
    r_vec = jnp.stack((x - halo_x, y - halo_y, z - halo_z), axis=-1)
    r_vec = modular_vector(r_vec, Lbox)
    r = jnp.sqrt(jnp.sum(r_vec * r_vec, axis=-1))

    return r_vec, r


@jjit
def modular_vector(v, Lbox):
    """
    Apply periodic boundary conditions to a vector v.
    Parameters
    ----------
    v : array_like, shape (N,3)
        Input separation to be wrapped into the periodic box.
    Lbox : array_like, shape (3,)
        Value giving the simulation box size along each dimension.
    Returns
    -------
    v_mod : array_like, shape (N,3)
        Vector after applying periodic boundary conditions.
    """
    # whether to wrap is data-dependent (on Lbox), so select with jnp.where
    # rather than a Python-level if, which would require Lbox to be static
    is_periodic = jnp.all(jnp.isfinite(Lbox))
    v_wrapped = jnp.mod(v + 0.5 * Lbox, Lbox) - 0.5 * Lbox
    return jnp.where(is_periodic, v_wrapped, v)


# ===========================================================================
# Dimroth-Watson Distribution Sampling (editted based on diffHOD-IA)
# Convention: sample the polar angle theta and azimuthal angle phi from D-W
# P(theta, phi) = B(kappa)/(2pi) * exp(-kappa*cos^2(theta))sin(theta),
# where the norm factor (Nick's eqn 17 has typo)
# B(kappa) = 2pi / (\int_{-1}^{1} exp(-kappa*t^2) * dt)
# The alignment strength is usually defined as
# mu = -2/pi * tan^-1(kappa), which is in [-1, 1].
# mu = 0 means random orientation
# mu = -1 means perfect anti-alignment (orthogonal to ref axis)
# mu = 1 means perfect alignment (parallel to ref axis)
# See https://arxiv.org/abs/2311.07374 and https://arxiv.org/abs/2602.04977
# ===========================================================================

@jjit
def _erfi_real(x: jnp.ndarray) -> jnp.ndarray:
    """
    Real-valued imaginary error function, erfi(x) = (2/sqrt(pi)) exp(x^2) F(x),
    where F is Dawson's integral. Accurate to ~1e-14 relative wherever exp(x^2)
    is finite (|x| < ~26.6), and 100x faster than summing erfi's power/asymptotic
    series, since jax's dawsn is a short rational approximation.
    Parameters
    ----------
    x : jnp.ndarray
        Input array.
    Returns
    -------
    y: jnp.ndarray
        The computed erfi values for the input array.
    """
    y = (2.0 / math.sqrt(math.pi)) * jnp.exp(x * x) * dawsn(x)
    return y.astype(x.dtype)


@partial(jjit, static_argnums=(2,))
def _sample_t_watson(key, kappa, n_newton=10):
    """
    Sample t = cos(theta) from the Dimroth-Watson distribution using Newton's method.
    (Newton's method is used to solve the inverse CDF of the kappa<0 parallel-alignment branch)
    Parameters
    ----------
    key : jax.random.PRNGKey
        Random key for JAX random number generation, to generate CDF samples.
    kappa : jnp.ndarray
        Concentration parameter for the Dimroth-Watson distribution.
    n_newton : int, optional
        Number of Newton iterations for solving the inverse CDF. Default is 10:
        every draw is converged to machine precision for mu <= 0.9 (kappa >= -6.3),
        and the sampled distribution stays correct up to mu = 0.99, though a small
        fraction of individual draws there is not fully converged. Use ~20 for
        per-draw convergence at any mu.
    Returns
    -------
    t : jnp.ndarray
        Sampled values of t = cos(theta) from the Dimroth-Watson distribution
    """
    def one(k_i, u_i):
        """
        Sample t = cos(theta) for a single kappa and uniform random number u.
        Depending on the sign of kappa, different branches of the inverse CDF are used.
        1. For kappa > 0 (anti-alignment), use
            CDF(x) = 0.5*[1 + erf(sqrt(kappa)*x)/erf(sqrt(kappa)) ]
        2. For |kappa| < KAPPA_SMALL (nearly random), use the first-order expansion
            of the inverse of CDF(x) = 0.5*(1 + x) + kappa*(x - x^3)/6, i.e.
            t = t0 - kappa*t0*(1 - t0^2)/3 with t0 = 2u - 1
        3. For kappa < 0 (alignment), use Newton's method to solve
            CDF(x) = 0.5*[1 + erfi(sqrt(-kappa)*x)/erfi(sqrt(-kappa)) ]
        """
        # Below KAPPA_SMALL the expansion is used: it gives the correct nonzero
        # dt/dkappa at kappa = 0, and 1e-7 balances its error against the closed
        # forms' derivative error (~eps/|kappa|)
        KAPPA_SMALL = 1e-7

        def pos_branch(_):
            # km >= 3e-4 here, so no epsilon; the safe argument only keeps the
            # discarded (kappa <= 0) evaluation finite
            km = jnp.sqrt(jnp.where(k_i > 0, k_i, 1.0))
            den = jax.scipy.special.erf(km)
            arg = jnp.clip((2.0 * u_i - 1.0) * den, -1.0 + 1e-7, 1.0 - 1e-7)
            return jax.scipy.special.erfinv(arg) / km

        def zer_branch(_):
            t0 = 2.0 * u_i - 1.0
            return t0 - k_i * t0 * (1.0 - t0 * t0) / 3.0

        def neg_branch(_):
            sp = jnp.sqrt(jnp.where(k_i < 0, -k_i, 1.0))
            sp2 = sp * sp
            tp0 = 2.0 * u_i - 1.0
            den = _erfi_real(sp) + 1e-30

            # erfi(sp) and erfi(sp*t) individually overflow in _erfi_real's
            # asymptotic branch once sp gets large (near-total alignment,
            # mu -> 1), even though their ratio (and the pdf, which shares
            # the same exp(sp^2) factor) stays perfectly well-behaved. Once
            # BOTH sp and sp*t are large, use erfi's own asymptotic form
            # erfi(x) ~ exp(x^2)*S(x)/(sqrt(pi)*x) to cancel exp(sp^2)
            # analytically, leaving only exp(sp^2*(t^2-1)) <= 1 for |t|<=1
            # -- this never overflows, however large sp is. 80 is a
            # conservative cutoff (float32 exp overflows past x^2~88.7),
            # chosen so it doesn't engage until past where the direct path
            # is already known to work (validated up to mu=0.99, sp^2~64).
            OVERFLOW_THRESH = 80.0

            def body(_, tp_curr):
                spt = sp * tp_curr
                spt2 = spt * spt
                both_large = (sp2 > OVERFLOW_THRESH) & (spt2 > OVERFLOW_THRESH)

                # direct path: correct whenever neither term overflows, and
                # also correct (giving ~0) when only the denominator (sp
                # alone large) overflows -- the ratio of a finite numerator
                # over an overflowed-to-inf denominator is genuinely ~0.
                ratio_direct = _erfi_real(spt) / den
                pdf_direct = (
                    (sp / math.sqrt(math.pi))
                    * jnp.exp(-1 * k_i * tp_curr * tp_curr) / den
                )

                # stable path: only valid/needed when both sp and sp*t are
                # large, so guard the reciprocals against the masked-out
                # (small) case to avoid a spurious large value there (still
                # harmless since jnp.where discards it, but keeps this branch
                # itself finite).
                safe_t = jnp.where(both_large, tp_curr, 1.0)
                safe_spt2 = jnp.where(both_large, spt2, 1.0)
                inv2_sp = 1.0 / sp2
                inv2_spt = 1.0 / safe_spt2
                S_sp = 1.0 + 0.5 * inv2_sp + 0.75 * inv2_sp**2 + 1.875 * inv2_sp**3
                S_spt = 1.0 + 0.5 * inv2_spt + 0.75 * inv2_spt**2 + 1.875 * inv2_spt**3
                E = jnp.exp(sp2 * (tp_curr * tp_curr - 1.0))
                ratio_stable = E * (S_spt / S_sp) / safe_t
                pdf_stable = sp2 * E / S_sp

                ratio = jnp.where(both_large, ratio_stable, ratio_direct)
                pdf = jnp.where(both_large, pdf_stable, pdf_direct)

                F = 0.5 * (ratio + 1.0)
                # 2e-7 (not 1e-6): near mu=1, the true CDF root can sit
                # closer to +-1 than 1e-6 (the distribution's own width
                # there is ~1/(2*sp^2), which drops below 1e-6 once sp^2
                # exceeds ~5e5 -- reachable at mu=1 exactly, sp^2~6.4e5).
                # A too-loose clip silently saturates Newton at the
                # boundary instead of the true root. 2e-7 stays a couple
                # of float32 ULPs above 1.0 (~1.19e-7 there).
                tp_next = jnp.clip(
                    tp_curr - (F - u_i) / (pdf + 1e-30), -1.0 + 2e-7, 1.0 - 2e-7
                )
                return tp_next

            return lax.fori_loop(0, n_newton, body, tp0)

        return lax.cond(
            k_i < -KAPPA_SMALL,
            neg_branch,
            lambda _: lax.cond(k_i > KAPPA_SMALL, pos_branch, zer_branch, operand=None),
            operand=None,
        )

    N = kappa.shape[0]
    u_uni = jnp.clip(
        jran.uniform(key, (N,), minval=0.0, maxval=1.0), 1e-7, 1 - 1e-7
    )
    t = jax.vmap(one)(kappa, u_uni)
    return jnp.clip(t, -1.0 + 2e-7, 1.0 - 2e-7)


@partial(jjit, static_argnums=(3,))  # n_newton is the 4th argument (index 3)
def sample_watson_orientations(key, ref_dirs, mu, n_newton=10):
    """
    Dimroth–Watson axial samples about ref_dirs (unit vectors).
    Parameters
    ----------
    key : jax.random.PRNGKey
        Random key for JAX random number generation.
    ref_dirs : jnp.ndarray, shape (N, 3)
        Array of reference unit vectors (directions) about which the samples will be generated.
    mu : jnp.ndarray of shape (N,), or float
        Alignment strength parameter.
    n_newton : int, optional
        Number of Newton iterations for solving the inverse CDF. Default is 10:
        every draw is converged to machine precision for mu <= 0.9 (kappa >= -6.3),
        and the sampled distribution stays correct up to mu = 0.99, though a small
        fraction of individual draws there is not fully converged. Use ~20 for
        per-draw convergence at any mu.
    Returns
    -------
    n : jnp.ndarray, shape (N, 3)
        Array of sampled unit vectors (orientations) from the Dimroth–Watson distribution
        about the reference directions `ref_dirs`, with the specified alignment strength `mu`.
    """
    return _sample_watson_frame(key, ref_dirs, mu, n_newton)[0]


@partial(jjit, static_argnums=(3,))
def _sample_watson_frame(key, ref_dirs, mu, n_newton=10):
    """Dimroth-Watson axis A about ref_dirs, plus unit vectors B0 and W that
    complete a right-handed orthonormal frame (A, B0, W).
    For fixed key and ref_dirs, all three vary smoothly with mu: B0 is A rotated
    by 90 degrees towards the reference axis and W = ref x perp is fixed. (A
    perpendicular chosen from A alone cannot be smooth over the whole sphere.)
    """
    u_axis = eproj._unitize(ref_dirs)
    N = u_axis.shape[0]

    # turn mu into kappa
    mu_arr = jnp.asarray(mu, dtype=u_axis.dtype)
    if mu_arr.ndim == 0:
        mu = jnp.full((N,), mu_arr, dtype=u_axis.dtype)
    else:
        mu = mu_arr.reshape(-1)
        if mu.shape[0] == 1:
            mu = jnp.repeat(mu, N, axis=0)
    mu = jnp.clip(mu, -1.0 + 1e-6, 1.0 - 1e-6)
    kappa = jnp.tan(-0.5 * math.pi * mu)
    kappa = jnp.clip(kappa, -1e6, 1e6)

    # sample theta, and an azimuthally-random direction perpendicular to u_axis
    key_u, key_phi = jran.split(key)
    t = _sample_t_watson(key_u, kappa, n_newton=n_newton)
    sinth = jnp.sqrt(jnp.clip(1.0 - t * t, 0.0)).reshape(N, 1)
    costh = t.reshape(N, 1)
    perp = eproj.random_perpendicular_directions(u_axis, key_phi)

    # rotate the fixed frame (u_axis, perp, u_axis x perp) by theta about its third axis
    axis_a = eproj._unitize(costh * u_axis + sinth * perp)
    axis_b0 = -sinth * u_axis + costh * perp
    axis_w = jnp.cross(u_axis, perp)
    return axis_a, axis_b0, axis_w
