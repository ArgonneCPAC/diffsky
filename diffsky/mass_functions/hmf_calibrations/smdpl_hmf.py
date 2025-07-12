"""Calibration of the HMF for host halos in the SMDPL simulation.

The target data were all centrals in the DR1 UniverseMachine data release,
created by the measure_smdpl_hmf_script.py script and the sfr_catalog_*.bin files

These calibrated parameters pertain to halo mass in units of Msun (not Msun/h)

"""

from collections import OrderedDict, namedtuple

DEFAULT_YTP_PDICT = OrderedDict(
    ytp_ytp=-5.330, ytp_x0=1.318, ytp_k=0.543, ytp_ylo=-0.171, ytp_yhi=-1.344
)
DEFAULT_X0_PDICT = OrderedDict(
    x0_ytp=13.030, x0_x0=1.781, x0_k=1.559, x0_ylo=-0.811, x0_yhi=-0.510
)
DEFAULT_LO_PDICT = OrderedDict(lo_x0=3.620, lo_k=0.693, lo_ylo=-0.792, lo_yhi=-2.442)
DEFAULT_HI_PDICT = OrderedDict(
    hi_ytp=-3.727, hi_x0=4.312, hi_k=1.113, hi_ylo=-0.426, hi_yhi=-0.265
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
