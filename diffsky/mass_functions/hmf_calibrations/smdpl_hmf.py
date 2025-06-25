"""Calibration of the HMF for SMDPL simulation"""

from ..hmf_model import DEFAULT_HMF_PARAMS

ytp_params = DEFAULT_HMF_PARAMS.ytp_params._replace(
    ytp_ytp=-4.824,
    ytp_x0=1.235,
    ytp_k=0.579,
    ytp_ylo=-0.191,
    ytp_yhi=-1.300,
)

x0_params = DEFAULT_HMF_PARAMS.x0_params._replace(
    x0_ytp=13.048,
    x0_x0=1.462,
    x0_k=2.220,
    x0_ylo=-0.810,
    x0_yhi=-0.556,
)

lo_params = DEFAULT_HMF_PARAMS.lo_params._replace(
    lo_x0=3.638,
    lo_k=0.688,
    lo_ylo=-0.790,
    lo_yhi=-2.459,
)

hi_params = DEFAULT_HMF_PARAMS.hi_params._replace(
    hi_ytp=-3.733,
    hi_x0=4.079,
    hi_k=1.629,
    hi_ylo=-0.410,
    hi_yhi=-0.841,
)

HMF_PARAMS = DEFAULT_HMF_PARAMS._replace(
    ytp_params=ytp_params,
    x0_params=x0_params,
    lo_params=lo_params,
    hi_params=hi_params,
)
