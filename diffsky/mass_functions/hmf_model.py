"""The predict_cuml_hmf and predict_differential_hmf functions
give differentiable implementations for the cumulative and differential mass functions,
respectively, for simulated host halos. These are both functions of mp,
the peak historical mass of the main progenitor halo.

"""

from collections import OrderedDict, namedtuple

from jax import grad
from jax import jit as jjit
from jax import vmap

from .kernels.hmf_kernels import lg_hmf_kern
from .utils import _sig_slope, _sigmoid

YTP_XTP = 3.0
X0_XTP = 3.0
HI_XTP = 3.0

DEFAULT_YTP_PDICT = OrderedDict(
    ytp_ytp=-5.27, ytp_x0=1.15, ytp_k=0.59, ytp_ylo=-0.24, ytp_yhi=-1.44
)
DEFAULT_X0_PDICT = OrderedDict(
    x0_ytp=12.95, x0_x0=1.41, x0_k=2.31, x0_ylo=-0.75, x0_yhi=-0.52
)
DEFAULT_LO_PDICT = OrderedDict(lo_x0=3.54, lo_k=0.69, lo_ylo=-0.77, lo_yhi=-2.61)
DEFAULT_HI_PDICT = OrderedDict(
    hi_ytp=-4.00, hi_x0=4.20, hi_k=1.29, hi_ylo=-0.51, hi_yhi=-0.20
)

Ytp_Params = namedtuple("Ytp_Params", DEFAULT_YTP_PDICT.keys())
X0_Params = namedtuple("X0_Params", DEFAULT_X0_PDICT.keys())
Lo_Params = namedtuple("Lo_Params", DEFAULT_LO_PDICT.keys())
Hi_Params = namedtuple("HI_Params", DEFAULT_HI_PDICT.keys())

DEFAULT_YTP_PARAMS = Ytp_Params(**DEFAULT_YTP_PDICT)
DEFAULT_X0_PARAMS = X0_Params(**DEFAULT_X0_PDICT)
DEFAULT_LO_PARAMS = Lo_Params(**DEFAULT_LO_PDICT)
DEFAULT_HI_PARAMS = Hi_Params(**DEFAULT_HI_PDICT)

DEFAULT_HMF_PDICT = OrderedDict(
    ytp_params=DEFAULT_YTP_PARAMS,
    x0_params=DEFAULT_X0_PARAMS,
    lo_params=DEFAULT_LO_PARAMS,
    hi_params=DEFAULT_HI_PARAMS,
)
HMF_Params = namedtuple("HMF_Params", DEFAULT_HMF_PDICT.keys())
DEFAULT_HMF_PARAMS = HMF_Params(**DEFAULT_HMF_PDICT)


@jjit
def predict_cuml_hmf(params, logmp, redshift):
    """Predict the cumulative comoving number density of host halos

    Parameters
    ----------
    params : namedtuple
        Fitting function parameters.
        Use DEFAULT_HMF_PARAMS for SMDPL-calibrated behavior.

    logmp : array, shape (n_halos, )
        Base-10 log of halo mass in units of Msun/h

    redshift : float

    Returns
    -------
    lg_cuml_hmf : array, shape (n_halos, )
        Base-10 log of cumulative comoving number density n(>logmp)
        in units of comoving (h/Mpc)**3

    """
    hmf_params = _get_singlez_cuml_hmf_params(params, redshift)
    return lg_hmf_kern(hmf_params, logmp)


@jjit
def _get_singlez_cuml_hmf_params(params, redshift):
    ytp = _ytp_vs_redshift(params.ytp_params, redshift)
    x0 = _x0_vs_redshift(params.x0_params, redshift)
    lo = _lo_vs_redshift(params.lo_params, redshift)
    hi = _hi_vs_redshift(params.hi_params, redshift)
    hmf_params = HMF_Params(ytp, x0, lo, hi)
    return hmf_params


@jjit
def _ytp_vs_redshift(params, redshift):
    p = (params.ytp_ytp, params.ytp_x0, params.ytp_k, params.ytp_ylo, params.ytp_yhi)
    return _sig_slope(redshift, YTP_XTP, *p)


@jjit
def _x0_vs_redshift(params, redshift):
    p = (params.x0_ytp, params.x0_x0, params.x0_k, params.x0_ylo, params.x0_yhi)
    return _sig_slope(redshift, X0_XTP, *p)


@jjit
def _lo_vs_redshift(params, redshift):
    p = (params.lo_x0, params.lo_k, params.lo_ylo, params.lo_yhi)
    return _sigmoid(redshift, *p)


@jjit
def _hi_vs_redshift(params, redshift):
    p = (params.hi_ytp, params.hi_x0, params.hi_k, params.hi_ylo, params.hi_yhi)
    return _sig_slope(redshift, HI_XTP, *p)


@jjit
def _diff_hmf_grad_kern(params, logmp, redshift):
    lgcuml_nd_pred = predict_cuml_hmf(params, logmp, redshift)
    cuml_nd_pred = 10**lgcuml_nd_pred
    return -cuml_nd_pred


_A = (None, 0, None)
_predict_differential_hmf = jjit(vmap(grad(_diff_hmf_grad_kern, argnums=1), in_axes=_A))


@jjit
def predict_differential_hmf(params, logmp, redshift):
    """Predict the differential comoving number density of host halos

    Parameters
    ----------
    params : namedtuple
        Fitting function parameters.
        Use DEFAULT_HMF_PARAMS for SMDPL-calibrated behavior.

    logmp : array, shape (n_halos, )
        Base-10 log of halo mass in units of Msun/h

    redshift : float

    Returns
    -------
    lg_hmf : array, shape (n_halos, )
        Base-10 log of differential comoving number density dn(logmp)/dlogmp
        in units of comoving (h/Mpc)**3 / dex

    """
    lg_hmf = _predict_differential_hmf(params, logmp, redshift)
    return lg_hmf
