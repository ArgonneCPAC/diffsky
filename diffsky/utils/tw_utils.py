"""Triweight kernels in JAX"""

from jax import jit as jjit
from jax import lax
from jax import numpy as jnp


@jjit
def _tw_gauss_scalar(x, m, h):
    """Scalar-valued triweight kernel approximation to a gaussian PDF

    Parameters
    ----------
    x : float
        scalar value at which to evaluate the kernel

    m : float
        mean of the kernel

    h : float
        approximate 1-sigma width of the kernel

    Returns
    -------
    kern : float
        value of the kernel

    """
    y = (x - m) / h
    return lax.cond(
        y < -3,
        lambda x: 0.0,
        lambda x: lax.cond(
            x > 3,
            lambda xx: 0.0,
            lambda xx: 35 / 96 * (1 - (xx / 3) ** 2) ** 3 / h,
            x,
        ),
        y,
    )


@jjit
def _tw_erf_scalar(x, m, h):
    """Scalar-valued triweight kernel approximation to the erf function

    Parameters
    ----------
    x : float
        scalar value at which to evaluate the kernel

    m : float
        mean of the kernel

    h : float
        approximate 1-sigma width of the kernel

    Returns
    -------
    kern_cdf : float
        value of the kernel CDF

    """
    y = (x - m) / h
    return lax.cond(
        y < -3,
        lambda x: 0.0,
        lambda x: lax.cond(
            x > 3,
            lambda xx: 1.0,
            lambda xx: (
                -5 * xx**7 / 69984
                + 7 * xx**5 / 2592
                - 35 * xx**3 / 864
                + 35 * xx / 96
                + 1 / 2
            ),
            x,
        ),
        y,
    )


@jjit
def _tw_erf(x, m, h):
    """Triweight kernel version of an err function."""
    z = (x - m) / h
    val = -5 * z**7 / 69984 + 7 * z**5 / 2592 - 35 * z**3 / 864 + 35 * z / 96 + 1 / 2
    val = jnp.where(z < -3, 0, val)
    val = jnp.where(z > 3, 1, val)
    return val


@jjit
def _tw_gauss(x, m, h):
    z = (x - m) / h
    val = 35 / 96 * (1 - (z / 3) ** 2) ** 3 / h
    msk = (z < -3) | (z > 3)
    return jnp.where(msk, 0, val)


@jjit
def _tw_bin_jax_kern(m, h, L, H):
    """Integrated bin weight for the triweight kernel

    Parameters
    ----------
    m : array-like or scalar
        The value at which to evaluate the kernel

    h : array-like or scalar
        The approximate 1-sigma width of the kernel

    L : array-like or scalar
        The lower bin limit

    H : array-like or scalar
        The upper bin limit

    Returns
    -------
    bin_prob : array-like or scalar
        The value of the kernel integrated over the bin

    """
    return _tw_erf(H, m, h) - _tw_erf(L, m, h)


@jjit
def _tw_sigmoid(x, x0, tw_h, ymin, ymax):
    height_diff = ymax - ymin
    body = _tw_erf(x, x0, tw_h)
    return ymin + height_diff * body


@jjit
def _tw_sig_slope(x, xtp, ytp, x0, tw_h, lo, hi):
    slope = _tw_sigmoid(x, x0, tw_h, lo, hi)
    return ytp + slope * (x - xtp)


@jjit
def _tw_interp_kern(xarr, x0, x1, x2, y0, y1, y2):
    """Smooth version of np.interp(xarr, (x0, x1, x2), (y0, y1, y2))"""
    xa = 0.5 * (x0 + x1)
    xb = 0.5 * (x1 + x2)

    dx01 = (x1 - x0) / 3
    dx12 = (x2 - x1) / 3

    w01 = _tw_sigmoid(xarr, xa, dx01, y0, y1)
    w12 = _tw_sigmoid(xarr, xb, dx12, y1, y2)

    dxab = (xb - xa) / 3
    xab = x1
    w02 = _tw_sigmoid(xarr, xab, dxab, w01, w12)

    return w02
