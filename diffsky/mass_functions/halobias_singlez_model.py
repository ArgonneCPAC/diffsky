""""""

from collections import OrderedDict, namedtuple

from jax import jit as jjit
from jax import numpy as jnp
from jax.nn import softplus

from ..utils import _inverse_sigmoid, _sigmoid
from ..utils import tw_utils as twu

XTP = 12.0
LGM_TABLE = jnp.array((10.5, 12.0, 14.0, 14.5, 15.0, 15.5))

HALOBIAS_PDICT = OrderedDict(
    hb_ytp=0.5, hb_s0=0.15, hb_s1=0.2, hb_s2=0.3, hb_s3=0.45, hb_s4=0.5, hb_s5=0.6
)
HaloBiasParams = namedtuple("HaloBiasParams", HALOBIAS_PDICT.keys())
HALOBIAS_PARAMS = HaloBiasParams(**HALOBIAS_PDICT)

_HALOBIAS_UPNAMES = ["u_" + key for key in HALOBIAS_PDICT.keys()]
HaloBiasUParams = namedtuple("HaloBiasUParams", _HALOBIAS_UPNAMES)

HB_YTP_PBOUNDS = (-2, 2)
HB_YTP_X0 = 0.5
BOUNDING_K = 0.1


@jjit
def get_bounded_halobias_params(u_hb_params):
    ytp = _get_bounded_ytp_params_kern(u_hb_params.u_hb_ytp)

    u_slope_params = (
        u_hb_params.u_hb_s0,
        u_hb_params.u_hb_s1,
        u_hb_params.u_hb_s2,
        u_hb_params.u_hb_s3,
        u_hb_params.u_hb_s4,
        u_hb_params.u_hb_s5,
    )
    slope_params = _get_bounded_slope_params_kern(u_slope_params)

    halobias_params = HaloBiasParams(ytp, *slope_params)

    return halobias_params


@jjit
def get_unbounded_halobias_params(hb_params):
    u_ytp = _get_unbounded_ytp_params_kern(hb_params.hb_ytp)

    slope_params = (
        hb_params.hb_s0,
        hb_params.hb_s1,
        hb_params.hb_s2,
        hb_params.hb_s3,
        hb_params.hb_s4,
        hb_params.hb_s5,
    )
    u_slope_params = _get_unbounded_slope_params_kern(slope_params)

    u_halobias_params = HaloBiasUParams(u_ytp, *u_slope_params)

    return u_halobias_params


@jjit
def _get_bounded_ytp_params_kern(u_hb_ytp):
    hb_ytp = _sigmoid(u_hb_ytp, HB_YTP_X0, BOUNDING_K, *HB_YTP_PBOUNDS)
    return hb_ytp


@jjit
def _get_unbounded_ytp_params_kern(hb_ytp):
    u_hb_ytp = _inverse_sigmoid(hb_ytp, HB_YTP_X0, BOUNDING_K, *HB_YTP_PBOUNDS)
    return u_hb_ytp


@jjit
def _get_bounded_slope_params_kern(u_slope_params):
    u_s0, u_s1, u_s2, u_s3, u_s4, u_s5 = u_slope_params

    s0 = softplus(u_s0)
    s1 = s0 + softplus(u_s1)

    s2 = s1 + softplus(u_s2)
    s3 = s2 + softplus(u_s3)
    s4 = s3 + softplus(u_s4)
    s5 = s4 + softplus(u_s5)

    slope_params = s0, s1, s2, s3, s4, s5
    return slope_params


@jjit
def _get_unbounded_slope_params_kern(slope_params):
    s0, s1, s2, s3, s4, s5 = slope_params

    u_s0 = _inverse_softplus(s0)
    u_s1 = _inverse_softplus(s1 - s0)
    u_s2 = _inverse_softplus(s2 - s1)
    u_s3 = _inverse_softplus(s3 - s2)
    u_s4 = _inverse_softplus(s4 - s3)
    u_s5 = _inverse_softplus(s5 - s4)

    u_slope_params = u_s0, u_s1, u_s2, u_s3, u_s4, u_s5
    return u_slope_params


@jjit
def _inverse_softplus(s):
    return jnp.log(jnp.exp(s) - 1.0)


@jjit
def predict_lgbias_kern(params, lgm):
    lgbias = _tw_quintuple_sigmoid_kern(
        lgm,
        params.hb_ytp,
        params.hb_s0,
        params.hb_s1,
        params.hb_s2,
        params.hb_s3,
        params.hb_s4,
        params.hb_s5,
    )
    return lgbias


@jjit
def _tw_quintuple_sigmoid_kern(x, ytp, s0, s1, s2, s3, s4, s5):
    slope_params = s0, s1, s2, s3, s4, s5
    slope = _tw_quintuple_slope(x, slope_params)
    return ytp + slope * (x - XTP)


@jjit
def _tw_quintuple_slope(x, y_table, x_table=LGM_TABLE):
    x0, x1, x2, x3, x4, x5 = x_table
    y0, y1, y2, y3, y4, y5 = y_table

    w02 = twu._tw_interp_kern(x, x0, x1, x2, y0, y1, y2)
    w24 = twu._tw_interp_kern(x, x2, x3, x4, y2, y3, y4)

    dx13 = (x3 - x1) / 3
    w04 = twu._tw_sigmoid(x, x2, dx13, w02, w24)

    dx45 = (x5 - x4) / 3
    x45 = 0.5 * (x4 + x5)
    w05 = twu._tw_sigmoid(x, x45, dx45, w04, y5)

    return w05


HALOBIAS_U_PARAMS = get_unbounded_halobias_params(HALOBIAS_PARAMS)
