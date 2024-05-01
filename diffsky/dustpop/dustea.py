"""Attenuation curve model from Faucher & Blanton 2023,
    https://arxiv.org/abs/2401.11007

"""

from jax import jit as jjit
from jax import numpy as jnp
from jax.scipy.special import erf

VBAND_AA = 5500.0
UV_BUMP_AA = 2050.0

AvBOUNDS = (0.0, 10.0)
DeltaBOUNDS = (-5.0, 0.0)
BumpBOUNDS = (0.0, 10.0)


@jjit
def squiggle(lgwave, lgc_squiggle, width_squiggle):
    """
    - this function is called from within the TEA model
    - smaller width parameter is more wide
    """
    z = width_squiggle * (lgwave - lgc_squiggle)
    zsq = z * z
    z4 = zsq * zsq
    num = z4 * z
    denom = z4 + 1
    term1 = num / denom
    term2 = z
    return -term1 + term2


@jjit
def skew_bump(lgwave, lgc_bump, uv_bump_width, bumpSkew):
    """
    - this function is called from within the TEA model
    - larger width parameter is steeper
    """
    bump = (1 + (uv_bump_width * (lgwave - lgc_bump)) ** 2) ** (-3 / 2)
    skew = 1 + erf((lgwave - lgc_bump) * bumpSkew)
    return bump * skew


@jjit
def att_curve(
    wave,
    Av,
    delta,
    bumpStrength,
    uv_bump_width=6.0,
    lgc_bump=jnp.log10(UV_BUMP_AA),
    bumpSkew=6.95,
    lgc_squiggle=3.285,
    width_squiggle=16.0,
):
    """TEA attenuation curve, as implemented in Faucher & Blanton 2023

    Function returns att, related to the transmission fraction as:
        F = 10 ** (-0.4 * att)

    """
    lgwave = jnp.log10(wave)
    a = Av * (wave / VBAND_AA) ** delta
    b = bumpStrength * skew_bump(lgwave, lgc_bump, uv_bump_width, bumpSkew)
    c = bumpStrength / 4 * squiggle(lgwave, lgc_squiggle, width_squiggle)
    att = a + b + c
    return att
