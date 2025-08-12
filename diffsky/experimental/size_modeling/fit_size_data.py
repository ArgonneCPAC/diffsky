"""OA
Module to assemble and fit trends in data vectors for selected data samples
"""
import os
import pickle
from collections import namedtuple
from importlib.resources import files

import numpy as np
from scipy.optimize import curve_fit

from ..validation.read_size_validation_data import validation_info

FIT_DRN = files("diffaux").joinpath("size_modeling/FitResults")


def get_data_vector(
    data,
    val_info,
    sample="Starforming",
    lambda_min=0.5,
    lambda_max=1.0,
    X="x-values",
    Y="y-values",
    dYp="y-errors+",
    dYn="y-errors-",
):
    # initialize
    xvec = np.asarray([])
    yvec = [np.asarray([]) for yname in val_info[Y]]
    dyvec = [np.asarray([]) for yname in val_info[Y]]

    for k, v in data.items():
        wave = val_info[k]["wavelength"]
        if wave >= lambda_min and wave <= lambda_max:
            print(f"Processing {k} {wave}")
            add_xvec = True
            for n, (y, dy, yn, dynp, dynn) in enumerate(
                zip(yvec, dyvec, val_info[Y], val_info[dYp], val_info[dYn])
            ):
                mask = np.abs(v[yn.format(sample)]) > 0.0
                if add_xvec:
                    xvec = np.concatenate((xvec, v[val_info[X]][mask]))
                    add_xvec = False
                y = np.concatenate((y, v[yn.format(sample)][mask]))
                yerr = np.fmax(v[dynp.format(sample)], v[dynn.format(sample)])
                dy = np.concatenate((dy, yerr[mask]))
                yvec[n] = y
                dyvec[n] = dy
        else:
            print(f"Skipping {k} {wave}")

    assert all(
        [len(xvec) == len(y) for y in yvec]
    ), "Mismatch in assembled data vectors"

    # convert any lists to dicts
    values = {}
    values[val_info[X] + f"_{sample}"] = xvec
    for y, dy, yname, dyname in zip(yvec, dyvec, val_info[Y], val_info[dYp]):
        values[yname.format(sample)] = y
        values[dyname.format(sample)] = dy

    return values


def _sigmoid(x, x0, k, ymin, ymax):
    height_diff = ymax - ymin
    return ymin + height_diff / (1 + np.exp(-k * (x - x0)))


# The variable xtp is not a free parameter,
# it is simply the abscissa value at which the normalization free
# parameter ytp is defined
def _sig_slope(x, xtp, ytp, x0, slope_k, lo, hi):
    slope = _sigmoid(x, x0, slope_k, lo, hi)
    return ytp + slope * (x - xtp)


SigmoidParameters = namedtuple("SigmoidParameters", ("x0", "k", "ymin", "ymax"))
Samples_zFit = ["Starforming", "Quiescent"]
Parameters_zFit = validation_info["Re_vs_z"]["y-values"]
Xvalue_zFit = validation_info["Re_vs_z"]["x-values"]
Names_zFit = [p.format(s) for s in Samples_zFit for p in Parameters_zFit]
zFitParameters = namedtuple("zFitParameters", Names_zFit)

B_SF_i = SigmoidParameters(11.5, 2.7, 3.0, 40.0)
beta_SF_i = SigmoidParameters(11.3, 2.5, 0.15, 2.5)
B_Q_i = SigmoidParameters(11.0, 3.6, 1.5, 16.0)
beta_Q_i = SigmoidParameters(10.0, 8.0, 0.4, 1.2)

zFitParams_initial = zFitParameters(B_SF_i, beta_SF_i, B_Q_i, beta_Q_i)


def collect_data_vectors(
    data, samples, validation_info, fit_type="Re_vs_z", lambda_min=0.5, lambda_max=1.0
):
    data_vectors = {}
    data_vectors[fit_type] = {}
    for sample in samples:
        values = get_data_vector(
            data[fit_type],
            validation_info[fit_type],
            sample=sample,
            lambda_min=lambda_min,
            lambda_max=lambda_max,
        )
        for k, v in values.items():
            data_vectors[fit_type][k] = v

    return data_vectors


def fit_parameters(
    data_vectors, Xvalue, p0_values, func=_sigmoid, error_prefix="d", error_suffix="+"
):
    fits = {}
    for name in p0_values._fields:
        fits[name] = {}
        p0 = getattr(p0_values, name)
        sample = name.split("_")[-1]
        X = data_vectors["_".join([Xvalue, sample])]
        y = data_vectors[name]
        dy = data_vectors[error_prefix + name + error_suffix]
        popt, pcov = curve_fit(func, X, y, sigma=dy, p0=p0, absolute_sigma=True)
        perr = np.sqrt(np.diag(pcov))
        fits[name]["popt"] = popt
        fits[name]["perr"] = perr
        fits[name]["pcov"] = pcov
        print("Fit parameters: ", name, popt)
        print("Errors: ", name, perr)

    return fits


def write_fit_parameters(
    fits, fn="{}_fit_parameters.pkl", fit_type="Re_vs_z", fitdir=FIT_DRN
):
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
    namedtuple,
    fn="{}_fit_parameters.pkl",
    fit_type="Re_vs_z",
    fitdir=FIT_DRN,
    fitval_key="popt",
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
    print(
        "Assembling {} from fit values for parameters {}".format(
            namedtuple.__name__, ", ".join(fit_keys)
        )
    )
    fit_values = [fits[fit_type][k][fitval_key] for k in fit_keys]
    fit_pars = namedtuple(*fit_values)

    return fit_pars, fits


def median_size_vs_z(z, B, beta):
    Re_med = B * np.power(1 + z, -beta)
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


def get_median_sizes(
    fit_parameters,
    log_Mstar,
    redshift,
    color,
    Ngals,
    samples,
    UVJcolor_cut=1.5,
    fit_func=_sigmoid,
    size_func=median_size_vs_z,
):
    """
    fit_parameters: named ntuple of fit parameters, which is read in using read_fit_parameters
    log_Mstar: array of length (Ngals), log10(stellar masses) of galaxies in units of Msun
    redshift: array of length Ngals, redshift of galaxies
    color: array length Ngals, color of galaxies
    Ngals: length of arrays stroing galaxy information
    samples: galaxy samples to process; sizes are generated for each sample
    UVJcolor_cut: color cut that selects between galaxy samples
    fit_func: function used by fit_parameters
    size_func: function used to generate medisn size

    returns
    sizes: array length (Ngals), size in kpc
    """

    R_med = np.zeros(Ngals)
    # determine parameter values from fit_parameters
    for sample in samples:
        mask = get_color_mask(color, sample, UVJcolor_cut=UVJcolor_cut)
        parameters = [par for par in fit_parameters._fields if sample in par]
        func_pars = [getattr(fit_parameters, par) for par in parameters]
        func_values = [fit_func(log_Mstar[mask], *fpar) for fpar in func_pars]
        R_med[mask] = size_func(redshift[mask], *func_values)

    return R_med


def get_scatter(R_med, scatter_hi, scatter_lo):
    scatter_up = R_med * (np.power(10, scatter_hi) - 1)
    scatter_down = R_med * (1 - np.power(10, -scatter_lo))
    return scatter_up, scatter_down


def generate_sizes(
    fit_parameters,
    log_Mstar,
    redshift,
    color,
    samples=("Starforming", "Quiescent"),
    UVJcolor_cut=1.5,
    scatter_hi=0.2,
    scatter_lo=0.2,
    fit_func=_sigmoid,
    size_func=median_size_vs_z,
):
    """
    fit_parameters: named ntuple of fit parameters, which is read in using read_fit_parameters
    log_Mstar: array of length (Ngals), log10(stellar masses) of galaxies in units of Msun
    redshift: array of length Ngals, redshift of galaxies
    color: array length Ngals, color of galaxies
    samples: galaxy samples to process; sizes are generated for each sample
    UVJcolor_cut: color cut that selects between galaxy samples
    scatter_lo:  scatter in dex below median size
    scatter_hi:  scatter in dex above median size
    fit_func: function used by fit_parameters
    size_func: function used to generate medisn size


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
        UVJcolor_cut=UVJcolor_cut,
        fit_func=fit_func,
        size_func=size_func,
    )
    scatter_up, scatter_down = get_scatter(
        R_med, scatter_hi=scatter_hi, scatter_lo=scatter_lo
    )

    sizes_hi = np.random.normal(loc=R_med, scale=scatter_hi, size=Ngals)
    sizes_lo = np.random.normal(loc=R_med, scale=scatter_lo, size=Ngals)

    return (
        np.where(sizes_lo < R_med, sizes_lo, sizes_hi),
        R_med,
        scatter_up,
        scatter_down,
    )


def assign_p0_values_to_fits(p0_values, fit_type="Re_vs_z"):
    """
    initialize fits dict with p0_values
    p0_values: named tuple of values of fit parameters

    returns
    fits: dictionary of fit parameters
    """
    fits = {}
    fits[fit_type] = {}
    for name in p0_values._fields:
        fits[fit_type][name] = {}
        fits[fit_type][name]["popt"] = getattr(p0_values, name)

    return fits
