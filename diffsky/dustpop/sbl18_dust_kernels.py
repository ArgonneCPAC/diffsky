""" """

from dsps.dust.att_curves import sbl18_k_lambda
from jax import jit as jjit
from jax import numpy as jnp

from ..utils.tw_utils import _tw_sig_slope

RV_C00 = 4.05
UV_BUMP_W0_MICRON = 0.2175  # Center of UV bump in micron
UV_BUMP_DW_MICRON = 0.0350  # Width of UV bump in micron


@jjit
def sbl18_dust_transmission(wave_micron, av, delta, funo, uv_bump_ampl):
    """Salim+18-based dust attention model

    Parameters
    ----------
    wave_micron : ndarray of shape (n, )
        Wavelength in micron

    av : float or ndarray of shape (n, )
        Normalization of the attenuation curve

    Returns
    -------
    F_trans : ndarray of shape (n, )
        Fraction of light transmitted through dust

    Notes
    -----
    k(λ) is computed from the triweight_k_lambda function

    A(λ) = k(λ) * (av/4.05)
    F_trans(λ) = 10^(-0.4*A(λ))
    F_trans(λ) = F_uno + (1-F_uno)*F_trans(λ)

    """
    k_lambda = sbl18_k_lambda(wave_micron, uv_bump_ampl, delta)

    # Compute the transmission fraction
    A_lambda = _att_curve_from_k_lambda(k_lambda, av)
    ftrans = 10.0 ** (-0.4 * A_lambda)

    # Modify ftrans according to the unobscured fraction
    ftrans = funo + (1.0 - funo) * ftrans

    return ftrans


@jjit
def triweight_k_lambda(
    wave_micron, xtp=-1.0, ytp=1.15, x0=0.5, tw_h=0.5, lo=-0.65, hi=-1.95
):
    """Smooth approximation to Noll+09 k(λ)

    Parameters
    ----------
    wave_micron : ndarray of shape (n, )
        Wavelength in micron

    Returns
    -------
    k_lambda : ndarray of shape (n, )
        The reddening curve k(λ) is related to the attenuation curve A(λ) as follows:

        k(λ) = A(λ) * (4.05/av)

    """
    lgx = jnp.log10(wave_micron)
    lgk_lambda = _tw_sig_slope(lgx, xtp, ytp, x0, tw_h, lo, hi)
    k_lambda = 10**lgk_lambda
    return k_lambda


@jjit
def _att_curve_from_k_lambda(k_lambda, av):
    """Normalize k(λ) according to (av/4.05) and clip at zero

    A(λ) = k(λ) * (av/4.05)

    """
    att_curve = av * k_lambda / RV_C00
    att_curve = jnp.where(att_curve < 0, 0.0, att_curve)
    return att_curve


@jjit
def power_law_vband_norm(wave_micron, delta, vband_micron=0.55):
    """Power law normalized at V-band wavelength λ_V=0.55 micron.

    Used to modify a baseline reddening curve model k_0(λ) as follows:

    k(λ) = k_0(λ) * (λ/λ_V)**δ

    Parameters
    ----------
    wave_micron : ndarray of shape (n, )
        Wavelength in micron

    delta : float
        Slope δ of the power-law modification (λ/λ_V)**δ

    Returns
    -------
    res : ndarray of shape (n, )

    """
    x = wave_micron / vband_micron
    return x**delta


@jjit
def _drude_bump(x, x0, gamma, ampl):
    """Drude profile of a bump feature seen in reddening curves

    The UV bump is typically located at λ=0.2175 micron,
    but _drude_bump can be used to introduce a generic bump in a power-law type model"""
    bump = x**2 * gamma**2 / ((x**2 - x0**2) ** 2 + x**2 * gamma**2)
    return ampl * bump
