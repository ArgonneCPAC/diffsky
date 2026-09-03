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
    # type check for alignment_strength
    assert( ( isinstance(alignment_strength, float) ) or ( isinstance(alignment_strength, jnp.ndarray) ) )
    if isinstance( alignment_strength, float ):
        alignment_strength = alignment_strength * jnp.ones( len(halo_axisA_x) )
    
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
    # type check for alignment_strength
    assert( ( isinstance(alignment_strength, float) ) or ( isinstance(alignment_strength, jnp.ndarray) ) )
    if isinstance( alignment_strength, float ):
        alignment_strength = alignment_strength * jnp.ones( len(halo_x) )
    
    # set prim_gal_axis orientation
    major_input_vectors, r = get_radial_vector(halo_x, halo_y, halo_z, x, y, z, Lbox)
    # check for length 0 radial vectors
    mask = (r <= 0.0) | (~jnp.isfinite(r))
    N_bad_axes = int(jnp.sum(mask))
    if N_bad_axes > 0:
        key, subkey = jran.split(key)
        major_input_vectors = major_input_vectors.at[mask, :].set(
            eproj.random_unit_vectors_3d(subkey, N_bad_axes)
        )
    
    return align_to_axis(key, major_input_vectors, alignment_strength, prim_gal_axis)

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
    # type check for alignment_strength
    assert( ( isinstance(alignment_strength, float) ) or ( isinstance(alignment_strength, jnp.ndarray) ) )
    if isinstance( alignment_strength, float ):
        alignment_strength = alignment_strength * jnp.ones( len(major_input_vectors) )

    key_A, key_B = jran.split(key)

    #A_v = axes_correlated_with_input_vector(major_input_vectors, p=alignment_strength)
    A_v = sample_watson_orientations(key_A, major_input_vectors, alignment_strength)

    # check for bad vectors: not finite (nan/inf), or not unit-length -- the
    # latter also catches a degenerate all-zero reference axis (e.g. an
    # unmeasured/unresolved halo shape), which _unitize maps to 0 (finite,
    # so it would slip past a finite-only check) rather than nan.
    # N_bad_axes is a traced value here (align_to_axis is jitted), so we
    # can't branch on it or assign into A_v by count; instead always draw N
    # replacements and select per-row with jnp.where.
    a_norm = jnp.linalg.norm(A_v, axis=-1)
    mask = (~jnp.isfinite(a_norm)) | (jnp.abs(a_norm - 1.0) > 1e-3)
    n_bad_axes = jnp.sum(mask)
    A_v_random = eproj.random_unit_vectors_3d(key_A, A_v.shape[0])
    A_v = jnp.where(mask[:, None], A_v_random, A_v)
    lax.cond(
        n_bad_axes > 0,
        lambda _: jax.debug.print(
            '{n} correlated alignment axis(axes) were found to be degenerate '
            '(not finite or not unit-length). These were re-assigned random '
            'vectors.', n=n_bad_axes),
        lambda _: None,
        operand=None,
    )

    # randomly set secondary axis orientation
    B_v = eproj.random_perpendicular_directions(A_v, key_B)

    # the tertiary axis is determined
    C_v = jnp.cross(A_v, B_v)

    eproj.assert_orthonormality(A_v, B_v, C_v)

    # depending on the prim_gal_axis, assign correlated axes
    if prim_gal_axis == 'A':
        major_v = A_v
        inter_v = B_v
        minor_v = C_v
    elif prim_gal_axis == 'B':
        major_v = C_v
        inter_v = A_v
        minor_v = B_v
    elif prim_gal_axis == 'C':
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
    r_vec = jnp.stack((x-halo_x, y-halo_y, z-halo_z), axis=-1)
    r_vec = modular_vector(r_vec, Lbox)
    r = jnp.sqrt(jnp.sum(r_vec*r_vec, axis=-1))

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



### ===========================================================================
### Dimroth-Watson Distribution Sampling (editted based on diffHOD-IA)
### Convention: sample the polar angle theta and azimuthal angle phi from D-W
### P(theta, phi) = B(kappa)/(2pi) * exp(-kappa*cos^2(theta))sin(theta),
### where the norm factor (Nick's eqn 17 has typo)
###   B(kappa) = 2pi / (\int_{-1}^{1} exp(-kappa*t^2) * dt)
### The alignment strength is usually defined as
### mu = -2/pi * tan^-1(kappa), which is in [-1, 1].
###   mu = 0 means random orientation
###   mu = -1 means perfect anti-alignment (orthogonal to ref axis)
###   mu = 1 means perfect alignment (parallel to ref axis)
### See https://arxiv.org/abs/2311.07374 and https://arxiv.org/abs/2602.04977
### ===========================================================================

@jjit
def _erfi_real(x: jnp.ndarray) -> jnp.ndarray:
    """
    Stable real-valued erfi for JAX arrays.
    Matches your piecewise series/asymptotic form, with bounded iteration counts.
    Parameters
    ----------
    x : jnp.ndarray
        Input array.
    Returns
    -------
    y: jnp.ndarray
        The computed erfi values for the input array.
    """

    def small_branch(xs: jnp.ndarray) -> jnp.ndarray:
        # erfi(x) = 2/sqrt(pi) * sum_{n>=0} x^{2n+1} / (n! (2n+1))
        def body_fun(n, carry):
            term, s = carry
            term = term * (xs * xs) / n  # multiply by x^2 / n
            s = s + term / (2 * n + 1)
            return term, s

        term0 = xs
        s0 = xs
        termN, sN = lax.fori_loop(1, 20, body_fun, (term0, s0))
        return (2.0 / math.sqrt(math.pi)) * sN

    def large_branch(xl: jnp.ndarray) -> jnp.ndarray:
        inv = 1.0 / xl
        inv2 = inv * inv
        series = 1.0 + 0.5 * inv2 + 0.75 * inv2 * inv2 + 1.875 * inv2 * inv2 * inv2
        return jnp.exp(xl * xl) * series / (math.sqrt(math.pi) * xl)

    y_small = small_branch(x)
    y_large = large_branch(x)
    y = jnp.where(jnp.abs(x) <= 3.0, y_small, y_large)
    return y.astype(x.dtype)

@partial(jjit, static_argnums=(2,))
def _sample_t_watson(key, kappa, n_newton = 6):
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
        Number of Newton iterations for solving the inverse CDF. Default is 6.
    Returns
    -------
    t : jnp.ndarray
        Sampled values of t = cos(theta) from the Dimroth-Watson distribution
    """
    eps = 1e-12

    def one(k_i, u_i):
        """
        Sample t = cos(theta) for a single kappa and uniform random number u.
        Depending on the sign of kappa, different branches of the inverse CDF are used.
        1. For kappa > 0 (anti-alignment), use
            CDF(x) = 0.5*[1 + erf(sqrt(kappa)*x)/erf(sqrt(kappa)) ]
        2. For kappa = 0 (random), use
            CDF(x) = 0.5*(1 + x)
        3. For kappa < 0 (alignment), use Newton's method to solve
            CDF(x) = 0.5*[1 + erfi(sqrt(-kappa)*x)/erfi(sqrt(-kappa)) ]
        """
        def pos_branch(_):
            km = jnp.sqrt(k_i)
            den = jax.scipy.special.erf(km)
            arg = jnp.clip((2.0 * u_i - 1.0) * den, -1.0 + 1e-7, 1.0 - 1e-7)
            return jax.scipy.special.erfinv(arg) / (km + eps)

        def zer_branch(_):
            return 2.0 * u_i - 1.0

        def neg_branch(_):
            sp = jnp.sqrt(-k_i)
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
            k_i < -1e-12,
            neg_branch,
            lambda _: lax.cond(k_i > 1e-12, pos_branch, zer_branch, operand=None),
            operand=None,
        )
    
    N = kappa.shape[0]
    u_uni = jnp.clip(
        jran.uniform(key, (N,), minval=0.0, maxval=1.0), 1e-7, 1 - 1e-7
    )
    t = jax.vmap(one)(kappa, u_uni)
    return jnp.clip(t, -1.0 + 2e-7, 1.0 - 2e-7)

@partial(jjit, static_argnums=(3,))  # n_newton is the 4th argument (index 3)
def sample_watson_orientations(key, ref_dirs, mu, n_newton=6):
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
        Number of Newton iterations for solving the inverse CDF. Default is 6.
    Returns
    -------
    n : jnp.ndarray, shape (N, 3)
        Array of sampled unit vectors (orientations) from the Dimroth–Watson distribution
        about the reference directions `ref_dirs`, with the specified alignment strength `mu`.
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

    # compute the sampled orientations in the orthonormal basis
    n = eproj._unitize(costh * u_axis + sinth * perp)
    return n
