"""
DiffHMF fit to the core mass function including cens and all cores.
Analogous to the unevolved subhalo function.

These best-fitting parameters were derived with the fit_subhalo_hmf_to_discovery.ipynb
notebook"""

from diffsky.mass_functions.hmf_model import DEFAULT_HMF_PARAMS

ytp_params = DEFAULT_HMF_PARAMS.ytp_params._replace(
    ytp_ytp=-5.568,
    ytp_x0=0.322,
    ytp_k=0.739,
    ytp_ylo=-0.305,
    ytp_yhi=-1.309,
)

x0_params = DEFAULT_HMF_PARAMS.x0_params._replace(
    x0_ytp=12.904,
    x0_x0=1.937,
    x0_k=3.865,
    x0_ylo=-0.994,
    x0_yhi=-0.375,
)

lo_params = DEFAULT_HMF_PARAMS.lo_params._replace(
    lo_x0=3.400,
    lo_k=0.734,
    lo_ylo=-0.921,
    lo_yhi=-2.616,
)

hi_params = DEFAULT_HMF_PARAMS.hi_params._replace(
    hi_ytp=-4.373,
    hi_x0=4.253,
    hi_k=1.209,
    hi_ylo=-0.352,
    hi_yhi=-0.540,
)

HMF_PARAMS = DEFAULT_HMF_PARAMS._make((ytp_params, x0_params, lo_params, hi_params))
