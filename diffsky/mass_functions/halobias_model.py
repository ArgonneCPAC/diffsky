""""""

from collections import OrderedDict, namedtuple

from jax import jit as jjit

from ..utils import tw_utils as twu

XTP = 12.0
LGM_TABLE = (10.5, 13.0, 14.0, 14.5, 15.0, 15.5)

HALOBIAS_PDICT = OrderedDict(
    hb_ytp=0.5, hb_s0=0.15, hb_s1=0.2, hb_s2=0.3, hb_s3=0.5, hb_s4=0.5, hb_s5=0.6
)
HaloBiasParams = namedtuple("HaloBiasParams", HALOBIAS_PDICT.keys())
HALOBIAS_PARAMS = HaloBiasParams(**HALOBIAS_PDICT)


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
def _tw_quintuple_slope(wave, y_table, x_table=LGM_TABLE):
    x0, x1, x2, x3, x4, x5 = x_table
    y0, y1, y2, y3, y4, y5 = y_table

    w02 = twu._tw_interp_kern(wave, x0, x1, x2, y0, y1, y2)
    w24 = twu._tw_interp_kern(wave, x2, x3, x4, y2, y3, y4)

    dx13 = (x3 - x1) / 3
    w04 = twu._tw_sigmoid(wave, x2, dx13, w02, w24)

    dx45 = (x5 - x4) / 3
    x45 = 0.5 * (x4 + x5)
    w05 = twu._tw_sigmoid(wave, x45, dx45, w04, y5)

    return w05
