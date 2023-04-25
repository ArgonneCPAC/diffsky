""""""
from collections import OrderedDict
import typing
from jax import jit as jjit
from jax import numpy as jnp
from functools import partial
from dsps.utils import _sigmoid, _inverse_sigmoid, powerlaw_rvs


class FunoBBounds(typing.NamedTuple):
    b_lo: jnp.float32
    b_hi: jnp.float32


FUNO_B_BOUNDS = FunoBBounds(0.05, 1.0)
BOUNDING_K = 0.1
DEFAULT_B = 0.3
BPOP_DEFAULT_PDICT = OrderedDict(
    funo_u_b_logsm_x0_x0=10.0,
    funo_u_b_logsm_x0_q=0,
    funo_u_b_logsm_x0_ms=0,
    funo_u_b_logsm_ylo_x0=-11.0,
    funo_u_b_logsm_ylo_q=0,
    funo_u_b_logsm_ylo_ms=0,
    funo_u_b_logsm_yhi_x0=-10.0,
    funo_u_b_logsm_yhi_q=0,
    funo_u_b_logsm_yhi_ms=0,
)


BPOP_BOUNDS_DICT = OrderedDict(
    funo_u_b_logsm_x0_q=(9.0, 12.0),
    funo_u_b_logsm_x0_ms=(9.0, 12.0),
    funo_u_b_logsm_ylo_q=(-4.0, 0.0),
    funo_u_b_logsm_ylo_ms=(-4.0, 0.0),
    funo_u_b_logsm_yhi_q=(-4.0, 0.0),
    funo_u_b_logsm_yhi_ms=(-4.0, 0.0),
)
LGFRACUNO_LGSM_K = 5.0
LGFRACUNO_SSFR_K = 3.0
FUNO_PLAW_SLOPE = 3.0

BPOP_DEFAULT_PARMS = jnp.array(list(BPOP_DEFAULT_PDICT.values()))


@jjit
def _get_funo_u_b_from_galpop(logsm, logssfr, fracuno_pop_params):
    (
        funo_u_b_logsm_x0_x0,
        funo_u_b_logsm_x0_q,
        funo_u_b_logsm_x0_ms,
        funo_u_b_logsm_ylo_x0,
        funo_u_b_logsm_ylo_q,
        funo_u_b_logsm_ylo_ms,
        funo_u_b_logsm_yhi_x0,
        funo_u_b_logsm_yhi_q,
        funo_u_b_logsm_yhi_ms,
    ) = fracuno_pop_params

    funo_u_b_ssfr_x0 = _sigmoid(
        logsm,
        funo_u_b_logsm_x0_x0,
        LGFRACUNO_LGSM_K,
        funo_u_b_logsm_ylo_x0,
        funo_u_b_logsm_yhi_x0,
    )

    funo_u_b_ssfr_q = _sigmoid(
        logsm,
        funo_u_b_logsm_x0_q,
        LGFRACUNO_LGSM_K,
        funo_u_b_logsm_ylo_q,
        funo_u_b_logsm_yhi_q,
    )
    funo_u_b_ssfr_ms = _sigmoid(
        logsm,
        funo_u_b_logsm_x0_ms,
        LGFRACUNO_LGSM_K,
        funo_u_b_logsm_ylo_ms,
        funo_u_b_logsm_yhi_ms,
    )

    funo_u_b = _sigmoid(
        logssfr,
        funo_u_b_ssfr_x0,
        LGFRACUNO_SSFR_K,
        funo_u_b_ssfr_q,
        funo_u_b_ssfr_ms,
    )
    return funo_u_b


@jjit
def _get_b_from_u_b(u_b):
    lo, hi = FUNO_B_BOUNDS
    mid = DEFAULT_B
    b = _sigmoid(u_b, mid, BOUNDING_K, lo, hi)
    return b


@jjit
def _get_u_b_from_b(b):
    lo, hi = FUNO_B_BOUNDS
    mid = DEFAULT_B
    u_b = _inverse_sigmoid(b, mid, BOUNDING_K, lo, hi)
    return u_b


DEFAULT_U_B = _get_u_b_from_b(DEFAULT_B)


@partial(jjit, static_argnames=["npts"])
def monte_carlo_frac_unobs(ran_key, u_b, npts):
    a = 0.0
    g = 3.0
    b = _get_b_from_u_b(u_b)
    frac_unobs = b - powerlaw_rvs(ran_key, a, b, g, npts)
    return frac_unobs


@jjit
def _get_b_from_u_params(logsm, logssfr, lgfuno_pop_params):
    funo_u_b = _get_funo_u_b_from_galpop(logsm, logssfr, lgfuno_pop_params)
    b = _get_b_from_u_b(funo_u_b)
    return b


@jjit
def mc_funobs(ran_key, logsm, logssfr, lgfuno_pop_params):
    n_gals = logsm.shape[0]
    a = jnp.zeros(n_gals)
    g = jnp.zeros(n_gals) + FUNO_PLAW_SLOPE
    b = _get_b_from_u_params(logsm, logssfr, lgfuno_pop_params)
    frac_unobs = b - powerlaw_rvs(ran_key, a, b, g)
    return frac_unobs
