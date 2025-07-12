"""Calibration of the HMF for subhalos in the SMDPL simulation.

The target data were all subhalos in the DR1 UniverseMachine data release,
created by the measure_smdpl_hmf_script.py script and the sfr_catalog_*.bin files

These calibrated parameters pertain to halo mass in units of Msun (not Msun/h)

"""

from collections import OrderedDict, namedtuple

DEFAULT_YTP_PDICT = OrderedDict(
    ytp_ytp=-5.318,
    ytp_x0=1.276,
    ytp_k=0.518,
    ytp_ylo=-0.214,
    ytp_yhi=-1.342,
)
DEFAULT_X0_PDICT = OrderedDict(
    x0_ytp=13.079, x0_x0=1.536, x0_k=1.312, x0_ylo=-0.829, x0_yhi=-0.537
)
DEFAULT_LO_PDICT = OrderedDict(lo_x0=3.642, lo_k=0.672, lo_ylo=-0.855, lo_yhi=-2.472)
DEFAULT_HI_PDICT = OrderedDict(
    hi_ytp=-3.763, hi_x0=4.385, hi_k=1.384, hi_ylo=-0.407, hi_yhi=-0.209
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
