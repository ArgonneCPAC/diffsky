"""Calibration of the HMF for subhalos in the SMDPL simulation.

The target data were all subhalos in the DR1 UniverseMachine data release,
created by the measure_smdpl_hmf_script.py script and the sfr_catalog_*.bin files

"""

from collections import OrderedDict, namedtuple

DEFAULT_YTP_PDICT = OrderedDict(
    ytp_ytp=-4.813, ytp_x0=1.201, ytp_k=0.518, ytp_ylo=-0.203, ytp_yhi=-1.334
)
DEFAULT_X0_PDICT = OrderedDict(
    x0_ytp=13.048, x0_x0=1.458, x0_k=1.409, x0_ylo=-0.872, x0_yhi=-0.570
)
DEFAULT_LO_PDICT = OrderedDict(lo_x0=3.607, lo_k=0.658, lo_ylo=-0.847, lo_yhi=-2.446)
DEFAULT_HI_PDICT = OrderedDict(
    hi_ytp=-3.705, hi_x0=4.491, hi_k=1.193, hi_ylo=-0.346, hi_yhi=-0.542
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

HMF_PARAMS = HMF_Params(**DEFAULT_HMF_PDICT)
