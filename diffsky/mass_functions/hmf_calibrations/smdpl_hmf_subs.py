"""Calibration of the HMF for host halos in the SMDPL simulation"""

from ..hmf_model import DEFAULT_HMF_PARAMS

ytp_params = DEFAULT_HMF_PARAMS.ytp_params._replace(
    ytp_ytp=-4.813,
    ytp_x0=1.201,
    ytp_k=0.518,
    ytp_ylo=-0.203,
    ytp_yhi=-1.334,
)

x0_params = DEFAULT_HMF_PARAMS.x0_params._replace(
    x0_ytp=13.048,
    x0_x0=1.458,
    x0_k=1.409,
    x0_ylo=-0.872,
    x0_yhi=-0.570,
)

lo_params = DEFAULT_HMF_PARAMS.lo_params._replace(
    lo_x0=3.607,
    lo_k=0.658,
    lo_ylo=-0.847,
    lo_yhi=-2.446,
)

hi_params = DEFAULT_HMF_PARAMS.hi_params._replace(
    hi_ytp=-3.705,
    hi_x0=4.491,
    hi_k=1.193,
    hi_ylo=-0.346,
    hi_yhi=-0.542,
)

HMF_PARAMS = DEFAULT_HMF_PARAMS._replace(
    ytp_params=ytp_params,
    x0_params=x0_params,
    lo_params=lo_params,
    hi_params=hi_params,
)
