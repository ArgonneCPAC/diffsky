"""The predict_cuml_hmf and predict_differential_hmf functions
give differentiable implementations for the cumulative and differential mass functions,
respectively, for simulated host halos. These are both functions of mp,
the peak historical mass of the main progenitor halo.

"""

from collections import OrderedDict, namedtuple

from jax import grad
from jax import jit as jjit
from jax import vmap

from .hmf_calibrations import DEFAULT_HMF_PARAMS, HMF_Params  # noqa
from .kernels.hmf_kernels import lg_hmf_kern
from .utils import _sig_slope, _sigmoid

YTP_XTP = 3.0
X0_XTP = 3.0
HI_XTP = 3.0


FLAT_HMF_PDICT = OrderedDict()
FLAT_HMF_PDICT.update(**DEFAULT_HMF_PARAMS.ytp_params._asdict())
FLAT_HMF_PDICT.update(**DEFAULT_HMF_PARAMS.x0_params._asdict())
FLAT_HMF_PDICT.update(**DEFAULT_HMF_PARAMS.lo_params._asdict())
FLAT_HMF_PDICT.update(**DEFAULT_HMF_PARAMS.hi_params._asdict())

FlatHMFParams = namedtuple("FlatHMFParams", FLAT_HMF_PDICT.keys())
FLAT_HMF_PARAMS = FlatHMFParams(**FLAT_HMF_PDICT)


@jjit
def predict_cuml_hmf(params, logmp, redshift):
    """Predict the cumulative comoving number density of host halos

    Parameters
    ----------
    params : namedtuple
        Fitting function parameters.
        Use DEFAULT_HMF_PARAMS for SMDPL-calibrated behavior.

    logmp : array, shape (n_halos, )
        Base-10 log of halo mass in units of Msun (not Msun/h)

    redshift : float

    Returns
    -------
    lg_cuml_hmf : array, shape (n_halos, )
        Base-10 log of cumulative comoving number density n(>logmp)
        in units of comoving (1/Mpc)**3

        Note that both number density and halo mass are defined in
        physical units (not h=1 units)

    """
    hmf_params = _get_singlez_cuml_hmf_params(params, redshift)
    return lg_hmf_kern(hmf_params, logmp)


@jjit
def _get_singlez_cuml_hmf_params(params, redshift):
    ytp = _ytp_vs_redshift(params, redshift)
    x0 = _x0_vs_redshift(params, redshift)
    lo = _lo_vs_redshift(params, redshift)
    hi = _hi_vs_redshift(params, redshift)
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
        Base-10 log of halo mass in units of Msun

    redshift : float

    Returns
    -------
    hmf : array, shape (n_halos, )
        Differential comoving number density dn(logmp)/dlogmp
        in units of comoving (1/Mpc)**3 / dex

        Note that both number density and halo mass are defined in
        physical units (not h=1 units)

    """
    hmf = _predict_differential_hmf(params, logmp, redshift)
    return hmf
