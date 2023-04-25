""""""
from collections import OrderedDict
import typing
from jax import jit as jjit
from jax import numpy as jnp
from jax import vmap
from dsps.utils import _sigmoid, _inverse_sigmoid, powerlaw_rvs


class FunoBBounds(typing.NamedTuple):
    b_lo: jnp.float32
    b_hi: jnp.float32


FUNO_B_BOUNDS = FunoBBounds(0.05, 1.0)

DEFAULT_LGFB_B = 0.3
BOUNDING_K = 0.1

BPOP_DEFAULT_PDICT = OrderedDict(
    funo_b_logsm_x0_x0=10.0,
    funo_b_logsm_x0_q=10.0,
    funo_b_logsm_x0_burst=10.0,
    funo_b_logsm_ylo_x0=-1.5,
    funo_b_logsm_ylo_q=0.1,
    funo_b_logsm_ylo_burst=0.4,
    funo_b_logsm_yhi_x0=-1.5,
    funo_b_logsm_yhi_q=0.1,
    funo_b_logsm_yhi_burst=0.4,
)
BPOP_BOUNDS_PDICT = OrderedDict(
    funo_b_logsm_x0_x0=(8.0, 12.0),
    funo_b_logsm_x0_q=(8.0, 12.0),
    funo_b_logsm_x0_burst=(8.0, 12.0),
    funo_b_logsm_ylo_x0=(-3.0, 0.0),
    funo_b_logsm_ylo_q=FUNO_B_BOUNDS,
    funo_b_logsm_ylo_burst=FUNO_B_BOUNDS,
    funo_b_logsm_yhi_x0=(-3.0, 0.0),
    funo_b_logsm_yhi_q=FUNO_B_BOUNDS,
    funo_b_logsm_yhi_burst=FUNO_B_BOUNDS,
)

LGFUNO_PLAW_SLOPE = 3.0

LGFRACUNO_LGSM_K = 5.0
LGFRACUNO_LGFB_K = 3.0

BPOP_DEFAULT_PARAMS = jnp.array(list(BPOP_DEFAULT_PDICT.values()))
BPOP_BOUNDS = jnp.array(list(BPOP_BOUNDS_PDICT.values()))


@jjit
def _get_funo_b_from_galpop(logsm, logfb, fracuno_pop_params):
    (
        funo_b_logsm_x0_x0,
        funo_b_logsm_x0_q,
        funo_b_logsm_x0_burst,
        funo_b_logsm_ylo_x0,
        funo_b_logsm_ylo_q,
        funo_b_logsm_ylo_burst,
        funo_b_logsm_yhi_x0,
        funo_b_logsm_yhi_q,
        funo_b_logsm_yhi_burst,
    ) = fracuno_pop_params

    funo_b_lgfb_x0 = _sigmoid(
        logsm,
        funo_b_logsm_x0_x0,
        LGFRACUNO_LGSM_K,
        funo_b_logsm_ylo_x0,
        funo_b_logsm_yhi_x0,
    )
    funo_u_b_lgfb_q = _sigmoid(
        logsm,
        funo_b_logsm_x0_q,
        LGFRACUNO_LGSM_K,
        funo_b_logsm_ylo_q,
        funo_b_logsm_yhi_q,
    )
    funo_u_b_lgfb_burst = _sigmoid(
        logsm,
        funo_b_logsm_x0_burst,
        LGFRACUNO_LGSM_K,
        funo_b_logsm_ylo_burst,
        funo_b_logsm_yhi_burst,
    )

    funo_b = _sigmoid(
        logfb,
        funo_b_lgfb_x0,
        LGFRACUNO_LGFB_K,
        funo_u_b_lgfb_q,
        funo_u_b_lgfb_burst,
    )
    return funo_b


@jjit
def _get_p_from_u_p_scalar(u_p, bounds):
    lo, hi = bounds
    p0 = 0.5 * (lo + hi)
    p = _sigmoid(u_p, p0, BOUNDING_K, lo, hi)
    return p


@jjit
def _get_u_p_from_p_scalar(p, bounds):
    lo, hi = bounds
    p0 = 0.5 * (lo + hi)
    u_p = _inverse_sigmoid(p, p0, BOUNDING_K, lo, hi)
    return u_p


_get_p_from_u_p_vmap = jjit(vmap(_get_p_from_u_p_scalar, in_axes=(0, 0)))


@jjit
def get_params_from_u_params(u_params):
    return _get_p_from_u_p_vmap(u_params, BPOP_BOUNDS)


@jjit
def _get_b_from_u_params(logsm, logfb, lgfuno_pop_u_params):
    """This function seems to return a very shallow sigmoid between 0.515<funo_b<0.522.

    The following two evaluations show the buggy behavior:
        1. _get_b_from_u_params(-18.0, lgfbarr, params)
        2. _get_b_from_u_params(18.0, lgfbarr, params)

    See dev_obscurepop_burst.ipynb

    """
    lgfuno_pop_params = get_params_from_u_params(lgfuno_pop_u_params)
    funo_b = _get_funo_b_from_galpop(logsm, logfb, lgfuno_pop_params)
    return funo_b


@jjit
def mc_funobs(ran_key, logsm, logfb, lgfuno_pop_params):
    """This function does not seem to produce distributions
    that actually vary with logsm. See dev_obscurepop_burst.ipynb

    """
    n_gals = logsm.shape[0]
    a = jnp.zeros(n_gals)
    g = jnp.zeros(n_gals) + LGFUNO_PLAW_SLOPE
    b = _get_b_from_u_params(logsm, logfb, lgfuno_pop_params)
    frac_unobs = b - powerlaw_rvs(ran_key, a, b, g)
    return frac_unobs
