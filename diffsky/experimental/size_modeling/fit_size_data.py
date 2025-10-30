"""
Module to assemble and fit trends in data vectors for selected data samples
"""
import os
import pickle
from collections import namedtuple
from importlib.resources import files
import numpy as np


FIT_DRN = files("diffaux").joinpath("size_modeling/FitResults")


def _sigmoid(x, x0, k, ymin, ymax):
    height_diff = ymax - ymin
    return ymin + height_diff / (1 + np.exp(-k * (x - x0)))


def _linear(x, ymin, slope):
    return ymin + slope * x


SigmoidParameters = namedtuple("SigmoidParameters", ("x0", "k", "ymin", "ymax"))
Samples_zFit = ["Starforming", "Quiescent"]
Parameters_zFit = ["B_{}", "beta_{}"]
Names_zFit = [p.format(s) for s in Samples_zFit for p in Parameters_zFit]
zFitParameters = namedtuple("zFitParameters", Names_zFit)

B_SF_i = SigmoidParameters(11.5, 2.7, 3.0, 40.0)
beta_SF_i = SigmoidParameters(11.3, 2.5, 0.15, 2.5)
B_Q_i = SigmoidParameters(11.0, 3.6, 1.5, 16.0)
beta_Q_i = SigmoidParameters(10.0, 8.0, 0.4, 1.2)

zFitParams_initial = zFitParameters(B_SF_i, beta_SF_i, B_Q_i, beta_Q_i)

Samples_MFit = ["Starforming", "Quiescent"]
Parameters_MFit = ["A_{}", "alpha_{}"]
Names_MFit = [p.format(s) for s in Samples_MFit for p in Parameters_MFit]
MFitParameters = namedtuple("MFitParameters", Names_MFit)

A_SF_i = SigmoidParameters(0.1, 2.0, 10.0, 3.5)
alpha_SF_i = SigmoidParameters(0.1, 1.0, 0.3, 0.15)
A_Q_i = SigmoidParameters(0.5, 2.0, 6.0, 1.0)
alpha_Q_i = SigmoidParameters(1.0, 1.0, 0.6, 0.5)

MFitParams_initial = MFitParameters(A_SF_i, alpha_SF_i, A_Q_i, alpha_Q_i)

LinearParameters = namedtuple("LinearParameters", ("ymin", "slope"))

Samples_MFit2 = ["Starforming", "Quiescent"]
Parameters_MFit2 = ["rp_{}", "alpha_{}", "beta_{}", "logMp_{}"]
Names_MFit2 = [p.format(s) for s in Samples_MFit2 for p in Parameters_MFit2]
MFit2Parameters = namedtuple("MFit2Parameters", Names_MFit2)

rp_SF_Sig_i = SigmoidParameters(0.5, 10.0, 5.4, 4.95)
alpha_SF_Sig_i = SigmoidParameters(0.9, 10.0, 0.49, 0.17)
beta_SF_Sig_i = SigmoidParameters(0.5, 10.0, 0.62, 0.27)
logMp_SF_Sig_i = SigmoidParameters(0.6, 10.0, 10.6, 11.0)
rp_Q_Sig_i = SigmoidParameters(0.5, 97.0, 2.5, 1.9)
alpha_Q_Sig_i = SigmoidParameters(0.9, 10.0, 0.15, 0.10)
beta_Q_Sig_i = SigmoidParameters(0.6, 20.0, 0.63, 0.68)
logMp_Q_Sig_i = SigmoidParameters(0.6, 10.0, 10.3, 10.6)

MFit2Params_Sig_initial = MFit2Parameters(
    rp_SF_Sig_i,
    alpha_SF_Sig_i,
    beta_SF_Sig_i,
    logMp_SF_Sig_i,
    rp_Q_Sig_i,
    alpha_Q_Sig_i,
    beta_Q_Sig_i,
    logMp_Q_Sig_i,
)

rp_SF_Lin_i = LinearParameters(5.6, -0.8)
alpha_SF_Lin_i = LinearParameters(0.16, 0.02)
beta_SF_Lin_i = LinearParameters(0.7, -0.5)
logMp_SF_Lin_i = LinearParameters(10.5, 0.3)
rp_Q_Lin_i = LinearParameters(2.0, -0.03)
alpha_Q_Lin_i = LinearParameters(0.13, -0.02)
beta_Q_Lin_i = LinearParameters(0.6, 0.09)
logMp_Q_Lin_i = LinearParameters(10.1, 0.5)

MFit2Params_Lin_initial = MFit2Parameters(
    rp_SF_Lin_i,
    alpha_SF_Lin_i,
    beta_SF_Lin_i,
    logMp_SF_Lin_i,
    rp_Q_Lin_i,
    alpha_Q_Lin_i,
    beta_Q_Lin_i,
    logMp_Q_Lin_i,
)


def write_fit_parameters(fits, fn="{}_fit_parameters.pkl", fit_type="Re_vs_z", fitdir=FIT_DRN):
    """
    write results of fitting validation data to pickle file for later use
    fits: dict of fit results which is output by the function fit_parameters
    fn: filename for pkl file
    fitdir: directory name for plk file
    """
    with open(os.path.join(fitdir, fn.format(fit_type)), "wb") as fh:
        pickle.dump(fits, fh)

    return


def read_fit_parameters(
    namedtuple, fn="{}_fit_parameters.pkl", fit_type="Re_vs_z", fitdir=FIT_DRN, fitval_key="popt"
):
    """
    read fit parameters from pickle file
    namedtuple: named tuple for fit parameters
    fn: filename of pkl file
    fit_type: fit type of fit parameters (key in fits dict); Re_vs_z is the only option so far
    fitdir: directory name for plk file
    fitval_key: key in fits dict containing fit parameters

    returns:
    fit_pars: named tuple of fit parameters resulting from fitting validation data
    fits: dict of fit results which is output by the function fit_parameters
    """
    with open(os.path.join(fitdir, fn.format(fit_type)), "rb") as fh:
        fits = pickle.load(fh)

    # convert dict values to named tuple
    fit_keys = [k for k in namedtuple._fields]
    print("Assembling {} from fit values for parameters {}".format(namedtuple.__name__, ", ".join(fit_keys)))
    fit_values = [fits[fit_type][k][fitval_key] for k in fit_keys]
    fit_pars = namedtuple(*fit_values)

    return fit_pars, fits


DEFAULT_MIXED_FIT_NAMEDTUPLES = {
    "Starforming": MFitParameters,
    "Quiescent": MFit2Parameters,
}
DEFAULT_MIXED_FIT_FITTYPES = {
    "Starforming": "Re_vs_Mstar",
    "Quiescent": "Re_vs_Mstar2",
}
DEFAULT_MIXED_FIT_FILENAMES = {
    "Starforming": "{}_fit_parameters.pkl",
    "Quiescent": "{}_betaQ_adjusted_fit_parameters.pkl",
}


def assemble_mixed_fit_parameters(
    samples,
    namedtuples=DEFAULT_MIXED_FIT_NAMEDTUPLES,
    fit_types=DEFAULT_MIXED_FIT_FITTYPES,
    plknames=DEFAULT_MIXED_FIT_FILENAMES,
):
    """
    namedtuples: dictionary of named tuples for mixed fits
    fit_types: dictionary of fit-types for mixed fits
    pklnames: dictionary of filename templates for mixed fits

    returns:
    fit_pars: dictionary of possible fit parameters for supplied fit_type
    """

    fit_pars = {}
    for sample in samples:
        fit_pars[sample], _ = read_fit_parameters(
            namedtuples[sample], fit_type=fit_types[sample], fn=plknames[sample]
        )

    return fit_pars


def median_size_vs_z(z, B, beta):
    Re_med = B * np.power(1 + z, -beta)
    return Re_med


def median_size_vs_mstar(M, A, alpha, M0=5e10):
    Re_med = A * np.power(M / M0, alpha)
    return Re_med


def median_size_vs_mstar2(M, rp, alpha, beta, logMp, delta=6):
    Mp = np.power(10, logMp)
    Re_med = (
        rp * np.power(M / Mp, alpha) * 0.5 * np.power((1 + np.power(M / Mp, delta)), (beta - alpha) / delta)
    )
    return Re_med


def get_color_mask(color, sample, UVJcolor_cut=1.5, UVJ=True):
    mask = np.ones(len(color), dtype=bool)
    if sample == "Starforming":
        if UVJ:
            mask = color < UVJcolor_cut
        else:
            print("Unknown color option")
    else:
        if UVJ:
            mask = color >= UVJcolor_cut
        else:
            print("Unknown color option")
    return mask


DEFAULT_FIT_FUNCTIONS = {"Starforming": _sigmoid, "Quiescent": _linear}
DEFAULT_SIZE_FUNCTIONS = {"Starforming": median_size_vs_mstar, "Quiescent": median_size_vs_mstar2}


def get_median_sizes(
    fit_parameters,
    log_Mstar,
    redshift,
    color,
    Ngals,
    samples,
    fit_types=DEFAULT_MIXED_FIT_FITTYPES,
    UVJcolor_cut=1.5,
    fit_funcs=DEFAULT_FIT_FUNCTIONS,
    size_funcs=DEFAULT_SIZE_FUNCTIONS,
):
    """
    fit_parameters: dictionary of named ntuple of fit parameters; read in using read_fit_parameters
    log_Mstar: array of length (Ngals), log10(stellar masses) of galaxies in units of Msun
    redshift: array of length Ngals, redshift of galaxies
    color: array length Ngals, color of galaxies
    Ngals: length of arrays stroing galaxy information
    samples: galaxy samples to process; sizes are generated for each sample
    fit_types: dictionary of fit_types for each sample
    UVJcolor_cut: color cut that selects between galaxy samples
    fit_funcs: dictionary of functions used by fit_parameters for each galaxy sample
    size_funcs: dictionary of functions used to generate medisn size for each galaxy sample

    returns
    sizes: array length (Ngals), size in kpc
    """

    R_med = np.zeros(Ngals)
    # determine parameter values from fit_parameters

    for sample in samples:
        mask = get_color_mask(color, sample, UVJcolor_cut=UVJcolor_cut)
        parameters = [par for par in fit_parameters[sample]._fields if sample in par]
        func_pars = [getattr(fit_parameters[sample], par) for par in parameters]
        fit_func = fit_funcs[sample]
        size_func = size_funcs[sample]
        if fit_types[sample] == "Re_vs_z":
            func_values = [fit_func(log_Mstar[mask], *fpar) for fpar in func_pars]
            R_med[mask] = size_func(redshift[mask], *func_values)
        elif "Re_vs_Mstar" in fit_types[sample]:
            func_values = [fit_func(redshift[mask], *fpar) for fpar in func_pars]
            Mstar = np.power(10, log_Mstar[mask])
            R_med[mask] = size_func(Mstar, *func_values)
    return R_med


MIN_SIZE = 0.5
MAX_SIZE = 40.0


def generate_sizes(
    fit_parameters,
    log_Mstar,
    redshift,
    color,
    fit_types=DEFAULT_MIXED_FIT_FITTYPES,
    samples=("Starforming", "Quiescent"),
    UVJcolor_cut=1.5,
    scatter=0.2,
    fit_funcs=DEFAULT_FIT_FUNCTIONS,
    size_funcs=DEFAULT_SIZE_FUNCTIONS,
    min_size=MIN_SIZE,
    max_size=MAX_SIZE,
):
    """
    fit_parameters: dictionary of named ntuple of fit parameters; read in using read_fit_parameters
    log_Mstar: array of length (Ngals), log10(stellar masses) of galaxies in units of Msun
    redshift: array of length Ngals, redshift of galaxies
    color: array length Ngals, color of galaxies
    fit_types: dictionary of fit_types for each sample
    samples: galaxy samples to process; sizes are generated for each sample
    UVJcolor_cut: color cut that selects between galaxy samples
    scatter:  scatter in dex to use for log-normal distribution
    fit_funcs: dictionary of functions used by fit_parameters for each galaxy sample
    size_funcs: dictionary of functions used to generate medisn size for each galaxy sample

    returns
    sizes: array length (Ngals), size in kpc

    """
    Ngals = len(log_Mstar)
    assert len(redshift) == Ngals, "Supplied redshifts don't match length of M* array"
    assert len(color) == Ngals, "Supplied colors don't match length of M* array"

    R_med = get_median_sizes(
        fit_parameters,
        log_Mstar,
        redshift,
        color,
        Ngals,
        samples,
        fit_types=fit_types,
        UVJcolor_cut=UVJcolor_cut,
        fit_funcs=fit_funcs,
        size_funcs=size_funcs,
    )

    logRe = np.random.normal(loc=np.log10(R_med), scale=scatter, size=Ngals)
    # clip sizes
    logRe = np.where(logRe > np.log10(max_size), np.log10(max_size), logRe)
    logRe = np.where(logRe < np.log10(min_size), np.log10(min_size), logRe)

    return np.power(10, logRe), R_med
