""""""
from collections import OrderedDict
import typing
from jax import jit as jjit
from jax import numpy as jnp
from dsps.utils import _sigmoid, _inverse_sigmoid, powerlaw_rvs


class FunoBBounds(typing.NamedTuple):
    b_lo: jnp.float32
    b_hi: jnp.float32


DEFAULT_LGFB_B = 0.3
LGFB_B_BOUNDS = FunoBBounds(0.05, 1.0)
BOUNDING_K = 0.1

BPOP_DEFAULT_PDICT = OrderedDict(
    funo_u_b_logsm_x0_x0=10.0,
    funo_u_b_logsm_x0_q=10.0,
    funo_u_b_logsm_x0_burst=10.0,
    funo_u_b_logsm_ylo_x0=-1.5,
    funo_u_b_logsm_ylo_q=-2.0,
    funo_u_b_logsm_ylo_burst=2.0,
    funo_u_b_logsm_yhi_x0=-1.5,
    funo_u_b_logsm_yhi_q=-2.0,
    funo_u_b_logsm_yhi_burst=2.0,
)

LGFUNO_PLAW_SLOPE = 3.0

LGFRACUNO_LGSM_K = 5.0
LGFRACUNO_LGFB_K = 3.0

BPOP_DEFAULT_PARAMS = jnp.array(list(BPOP_DEFAULT_PDICT.values()))


@jjit
def _get_funo_u_b_from_galpop(logsm, logfb, fracuno_pop_params):
    (
        funo_u_b_logsm_x0_x0,
        funo_u_b_logsm_x0_q,
        funo_u_b_logsm_x0_burst,
        funo_u_b_logsm_ylo_x0,
        funo_u_b_logsm_ylo_q,
        funo_u_b_logsm_ylo_burst,
        funo_u_b_logsm_yhi_x0,
        funo_u_b_logsm_yhi_q,
        funo_u_b_logsm_yhi_burst,
    ) = fracuno_pop_params

    funo_u_b_lgfb_x0 = _sigmoid(
        logsm,
        funo_u_b_logsm_x0_x0,
        LGFRACUNO_LGSM_K,
        funo_u_b_logsm_ylo_x0,
        funo_u_b_logsm_yhi_x0,
    )
    funo_u_b_lgfb_q = _sigmoid(
        logsm,
        funo_u_b_logsm_x0_q,
        LGFRACUNO_LGSM_K,
        funo_u_b_logsm_ylo_q,
        funo_u_b_logsm_yhi_q,
    )
    funo_u_b_lgfb_burst = _sigmoid(
        logsm,
        funo_u_b_logsm_x0_burst,
        LGFRACUNO_LGSM_K,
        funo_u_b_logsm_ylo_burst,
        funo_u_b_logsm_yhi_burst,
    )

    funo_u_b = _sigmoid(
        logfb,
        funo_u_b_lgfb_x0,
        LGFRACUNO_LGFB_K,
        funo_u_b_lgfb_q,
        funo_u_b_lgfb_burst,
    )
    return funo_u_b


@jjit
def _get_b_from_u_b(u_b):
    lo, hi = LGFB_B_BOUNDS
    mid = DEFAULT_LGFB_B
    b = _sigmoid(u_b, mid, BOUNDING_K, lo, hi)
    return b


@jjit
def _get_u_b_from_b(b):
    lo, hi = LGFB_B_BOUNDS
    mid = DEFAULT_LGFB_B
    u_b = _inverse_sigmoid(b, mid, BOUNDING_K, lo, hi)
    return u_b


DEFAULT_U_B = _get_u_b_from_b(DEFAULT_LGFB_B)


@jjit
def _get_b_from_u_params(logsm, logfb, lgfuno_pop_params):
    funo_u_b = _get_funo_u_b_from_galpop(logsm, logfb, lgfuno_pop_params)
    b = _get_b_from_u_b(funo_u_b)
    return b


@jjit
def mc_funobs(ran_key, logsm, logfb, lgfuno_pop_params):
    n_gals = logsm.shape[0]
    a = jnp.zeros(n_gals)
    g = jnp.zeros(n_gals) + LGFUNO_PLAW_SLOPE
    b = _get_b_from_u_params(logsm, logfb, lgfuno_pop_params)
    frac_unobs = b - powerlaw_rvs(ran_key, a, b, g)
    return frac_unobs
