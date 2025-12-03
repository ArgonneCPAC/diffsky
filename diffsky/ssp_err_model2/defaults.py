""""""

from collections import OrderedDict, namedtuple

from jax import jit as jjit
from jax import numpy as jnp
from jax import vmap

from ..utils import _inverse_sigmoid, _sigmoid

ZERO_SSPERR_PDICT = dict(
    z0p0_dr_logsm0=10.0,
    z0p0_dr_ylo=0.0,
    z0p0_dr_yhi=0.0,
    z0p0_dfr_logsm0=10.0,
    z0p0_dfr_ylo=0.0,
    z0p0_dfr_yhi=0.0,
    z0p0_dnr_logsm0=10.0,
    z0p0_dnr_ylo=0.0,
    z0p0_dnr_yhi=0.0,
    z0p0_dur_logsm0=10.0,
    z0p0_dur_ylo=0.0,
    z0p0_dur_yhi=0.0,
    z0p0_dgr_logsm0=10.0,
    z0p0_dgr_ylo=0.0,
    z0p0_dgr_yhi=0.0,
    z0p0_dir_logsm0=10.0,
    z0p0_dir_ylo=0.0,
    z0p0_dir_yhi=0.0,
    z0p5_dr_logsm0=10.0,
    z0p5_dr_ylo=0.0,
    z0p5_dr_yhi=0.0,
    z0p5_dfr_logsm0=10.0,
    z0p5_dfr_ylo=0.0,
    z0p5_dfr_yhi=0.0,
    z0p5_dnr_logsm0=10.0,
    z0p5_dnr_ylo=0.0,
    z0p5_dnr_yhi=0.0,
    z0p5_dur_logsm0=10.0,
    z0p5_dur_ylo=0.0,
    z0p5_dur_yhi=0.0,
    z0p5_dgr_logsm0=10.0,
    z0p5_dgr_ylo=0.0,
    z0p5_dgr_yhi=0.0,
    z0p5_dir_logsm0=10.0,
    z0p5_dir_ylo=0.0,
    z0p5_dir_yhi=0.0,
    z1p1_dr_logsm0=10.0,
    z1p1_dr_ylo=0.0,
    z1p1_dr_yhi=0.0,
    z1p1_dfr_logsm0=10.0,
    z1p1_dfr_ylo=0.0,
    z1p1_dfr_yhi=0.0,
    z1p1_dnr_logsm0=10.0,
    z1p1_dnr_ylo=0.0,
    z1p1_dnr_yhi=0.0,
    z1p1_dur_logsm0=10.0,
    z1p1_dur_ylo=0.0,
    z1p1_dur_yhi=0.0,
    z1p1_dgr_logsm0=10.0,
    z1p1_dgr_ylo=0.0,
    z1p1_dgr_yhi=0.0,
    z1p1_dir_logsm0=10.0,
    z1p1_dir_ylo=0.0,
    z1p1_dir_yhi=0.0,
)

DEFAULT_SSPERR_PDICT = dict(
    z0p0_dr_logsm0=9.37,
    z0p0_dr_ylo=-0.71,
    z0p0_dr_yhi=0.05,
    z0p0_dfr_logsm0=10.0,
    z0p0_dfr_ylo=0.0,
    z0p0_dfr_yhi=0.0,
    z0p0_dnr_logsm0=8.98,
    z0p0_dnr_ylo=-0.31,
    z0p0_dnr_yhi=-0.28,
    z0p0_dur_logsm0=8.77,
    z0p0_dur_ylo=-0.37,
    z0p0_dur_yhi=-0.46,
    z0p0_dgr_logsm0=10.3,
    z0p0_dgr_ylo=-0.07,
    z0p0_dgr_yhi=0.22,
    z0p0_dir_logsm0=10.19,
    z0p0_dir_ylo=-0.13,
    z0p0_dir_yhi=0.27,
    z0p5_dr_logsm0=11.21,
    z0p5_dr_ylo=-0.15,
    z0p5_dr_yhi=0.08,
    z0p5_dfr_logsm0=10.0,
    z0p5_dfr_ylo=0.0,
    z0p5_dfr_yhi=0.0,
    z0p5_dnr_logsm0=9.63,
    z0p5_dnr_ylo=0.56,
    z0p5_dnr_yhi=-0.45,
    z0p5_dur_logsm0=10.25,
    z0p5_dur_ylo=0.04,
    z0p5_dur_yhi=0.20,
    z0p5_dgr_logsm0=10.99,
    z0p5_dgr_ylo=0.08,
    z0p5_dgr_yhi=-0.16,
    z0p5_dir_logsm0=11.12,
    z0p5_dir_ylo=-0.15,
    z0p5_dir_yhi=0.38,
    z1p1_dr_logsm0=10.61,
    z1p1_dr_ylo=0.16,
    z1p1_dr_yhi=-0.20,
    z1p1_dfr_logsm0=9.97,
    z1p1_dfr_ylo=-0.77,
    z1p1_dfr_yhi=0.07,
    z1p1_dnr_logsm0=10.91,
    z1p1_dnr_ylo=0.075,
    z1p1_dnr_yhi=-0.085,
    z1p1_dur_logsm0=9.95,
    z1p1_dur_ylo=-0.73,
    z1p1_dur_yhi=0.063,
    z1p1_dgr_logsm0=9.82,
    z1p1_dgr_ylo=-0.68,
    z1p1_dgr_yhi=0.34,
    z1p1_dir_logsm0=10.0,
    z1p1_dir_ylo=0.0,
    z1p1_dir_yhi=0.0,
)

LOGSM0_BOUNDS = (8.0, 15.0)
DMAG_BOUNDS = (-0.8, 0.8)
DCOLOR_BOUNDS = (-0.8, 0.8)

SSPERR_PBOUNDS_PDICT = OrderedDict(
    z0p0_dr_logsm0=LOGSM0_BOUNDS,
    z0p0_dr_ylo=DMAG_BOUNDS,
    z0p0_dr_yhi=DMAG_BOUNDS,
    z0p0_dfr_logsm0=LOGSM0_BOUNDS,
    z0p0_dfr_ylo=DCOLOR_BOUNDS,
    z0p0_dfr_yhi=DCOLOR_BOUNDS,
    z0p0_dnr_logsm0=LOGSM0_BOUNDS,
    z0p0_dnr_ylo=DCOLOR_BOUNDS,
    z0p0_dnr_yhi=DCOLOR_BOUNDS,
    z0p0_dur_logsm0=LOGSM0_BOUNDS,
    z0p0_dur_ylo=DCOLOR_BOUNDS,
    z0p0_dur_yhi=DCOLOR_BOUNDS,
    z0p0_dgr_logsm0=LOGSM0_BOUNDS,
    z0p0_dgr_ylo=DCOLOR_BOUNDS,
    z0p0_dgr_yhi=DCOLOR_BOUNDS,
    z0p0_dir_logsm0=LOGSM0_BOUNDS,
    z0p0_dir_ylo=DCOLOR_BOUNDS,
    z0p0_dir_yhi=DCOLOR_BOUNDS,
    z0p5_dr_logsm0=LOGSM0_BOUNDS,
    z0p5_dr_ylo=DMAG_BOUNDS,
    z0p5_dr_yhi=DMAG_BOUNDS,
    z0p5_dfr_logsm0=LOGSM0_BOUNDS,
    z0p5_dfr_ylo=DCOLOR_BOUNDS,
    z0p5_dfr_yhi=DCOLOR_BOUNDS,
    z0p5_dnr_logsm0=LOGSM0_BOUNDS,
    z0p5_dnr_ylo=DCOLOR_BOUNDS,
    z0p5_dnr_yhi=DCOLOR_BOUNDS,
    z0p5_dur_logsm0=LOGSM0_BOUNDS,
    z0p5_dur_ylo=DCOLOR_BOUNDS,
    z0p5_dur_yhi=DCOLOR_BOUNDS,
    z0p5_dgr_logsm0=LOGSM0_BOUNDS,
    z0p5_dgr_ylo=DCOLOR_BOUNDS,
    z0p5_dgr_yhi=DCOLOR_BOUNDS,
    z0p5_dir_logsm0=LOGSM0_BOUNDS,
    z0p5_dir_ylo=DCOLOR_BOUNDS,
    z0p5_dir_yhi=DCOLOR_BOUNDS,
    z1p1_dr_logsm0=LOGSM0_BOUNDS,
    z1p1_dr_ylo=DMAG_BOUNDS,
    z1p1_dr_yhi=DMAG_BOUNDS,
    z1p1_dfr_logsm0=LOGSM0_BOUNDS,
    z1p1_dfr_ylo=DCOLOR_BOUNDS,
    z1p1_dfr_yhi=DCOLOR_BOUNDS,
    z1p1_dnr_logsm0=LOGSM0_BOUNDS,
    z1p1_dnr_ylo=DCOLOR_BOUNDS,
    z1p1_dnr_yhi=DCOLOR_BOUNDS,
    z1p1_dur_logsm0=LOGSM0_BOUNDS,
    z1p1_dur_ylo=DCOLOR_BOUNDS,
    z1p1_dur_yhi=DCOLOR_BOUNDS,
    z1p1_dgr_logsm0=LOGSM0_BOUNDS,
    z1p1_dgr_ylo=DCOLOR_BOUNDS,
    z1p1_dgr_yhi=DCOLOR_BOUNDS,
    z1p1_dir_logsm0=LOGSM0_BOUNDS,
    z1p1_dir_ylo=DCOLOR_BOUNDS,
    z1p1_dir_yhi=DCOLOR_BOUNDS,
)


@jjit
def _get_bounded_ssperr_param(u_param, bound):
    lo, hi = bound
    mid = 0.5 * (lo + hi)
    return _sigmoid(u_param, mid, 0.1, lo, hi)


@jjit
def _get_unbounded_ssperr_param(param, bound):
    lo, hi = bound
    mid = 0.5 * (lo + hi)
    return _inverse_sigmoid(param, mid, 0.1, lo, hi)


_C = (0, 0)
_get_ssperr_params_kern = jjit(vmap(_get_bounded_ssperr_param, in_axes=_C))
_get_ssperr_u_params_kern = jjit(vmap(_get_unbounded_ssperr_param, in_axes=_C))


@jjit
def get_bounded_ssperr_params(u_params):
    u_params = jnp.array([getattr(u_params, u_pname) for u_pname in _SSPERR_UPNAMES])
    ssperr_params = _get_ssperr_params_kern(
        jnp.array(u_params), jnp.array(SSPERR_PBOUNDS)
    )
    return DEFAULT_SSPERR_PARAMS._make(ssperr_params)


@jjit
def get_unbounded_ssperr_params(params):
    params = jnp.array(
        [getattr(params, pname) for pname in DEFAULT_SSPERR_PARAMS._fields]
    )
    u_params = _get_ssperr_u_params_kern(jnp.array(params), jnp.array(SSPERR_PBOUNDS))
    return SSPErrUParams(*u_params)


SSPErrParams = namedtuple("SSPErrParams", DEFAULT_SSPERR_PDICT.keys())

_SSPERR_UPNAMES = ["u_" + key for key in SSPERR_PBOUNDS_PDICT.keys()]
SSPErrUParams = namedtuple("SSPErrUParams", _SSPERR_UPNAMES)

ZERO_SSPERR_PARAMS = SSPErrParams(**ZERO_SSPERR_PDICT)
DEFAULT_SSPERR_PARAMS = SSPErrParams(**DEFAULT_SSPERR_PDICT)
SSPERR_PBOUNDS = SSPErrParams(**SSPERR_PBOUNDS_PDICT)


ZERO_SSPERR_U_PARAMS = SSPErrUParams(*get_unbounded_ssperr_params(ZERO_SSPERR_PARAMS))
DEFAULT_SSPERR_U_PARAMS = SSPErrUParams(
    *get_unbounded_ssperr_params(DEFAULT_SSPERR_PARAMS)
)
