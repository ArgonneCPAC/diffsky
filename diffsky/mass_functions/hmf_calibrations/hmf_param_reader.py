""" """

from collections import namedtuple

ytp_pnames = ("ytp_ytp", "ytp_x0", "ytp_k", "ytp_ylo", "ytp_yhi")
x0_pnames = ("x0_ytp", "x0_x0", "x0_k", "x0_ylo", "x0_yhi")
lo_pnames = ("lo_x0", "lo_k", "lo_ylo", "lo_yhi")
hi_pnames = ("hi_ytp", "hi_x0", "hi_k", "hi_ylo", "hi_yhi")

Ytp_Params = namedtuple("Ytp_Params", ytp_pnames)
X0_Params = namedtuple("X0_Params", x0_pnames)
Lo_Params = namedtuple("Lo_Params", lo_pnames)
Hi_Params = namedtuple("Hi_Params", hi_pnames)


def read_hmf_model_params_from_txt(fname):
    with open(fname, "r") as fobj:
        all_params = dict()
        for raw_line in fobj:
            line = raw_line.strip()
            key, val = line.split("=")
            all_params[key] = float(val)
    return all_params


def load_hmf_model_params_from_txt(fname):

    all_params = read_hmf_model_params_from_txt(fname)

    ytp_params = Ytp_Params(*[all_params[key] for key in ytp_pnames])
    x0_params = X0_Params(*[all_params[key] for key in x0_pnames])
    lo_params = Lo_Params(*[all_params[key] for key in lo_pnames])
    hi_params = Hi_Params(*[all_params[key] for key in hi_pnames])

    hmf_pnames = ("ytp_params", "x0_params", "lo_params", "hi_params")
    HMF_Params = namedtuple("HMF_Params", hmf_pnames)
    hmf_params = HMF_Params(*(ytp_params, x0_params, lo_params, hi_params))

    return hmf_params
