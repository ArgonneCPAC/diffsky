"""
Test user facing function
"""
import os
from importlib.resources import files

import numpy as np

from diffaux.size_modeling.fit_size_data import (
    DEFAULT_FIT_FUNCTIONS,
    DEFAULT_MIXED_FIT_FITTYPES,
    DEFAULT_SIZE_FUNCTIONS,
    Samples_MFit,
    assemble_mixed_fit_parameters,
    generate_sizes,
)

TESTDATA_DRN = files("diffaux").joinpath("size_modeling/tests/testing_data")


def test_generate_sizes(
    lM_lo=7.5,
    lM_hi=12.0,
    Nm=10,
    z_lo=0.0,
    z_hi=3.0,
    Nz=4,
    color_SF=0.0,
    color_Q=2,
    testdata_dir=TESTDATA_DRN,
    UVJcolor_cut=1.5,
    scatter=0.2,
    read=True,
    fn="generate_sizes_test_data.txt",
    rtol=1e-4,
    variables=("logM", "z"),
    types=("SF", "Q"),
    var="R_med_{}",
):
    vals = [var.format(t) for t in list(types)]
    header = list(variables) + vals
    filename = os.path.join(testdata_dir, fn)
    if read:
        usecols = [header.index(header[i]) for i in range(len(header))]
        data = np.loadtxt(filename, unpack=True, usecols=usecols, skiprows=1)
        test_data = dict(zip(header, data))
        logM = test_data["logM"]
        z = test_data["z"]
    else:
        logM_values = np.linspace(lM_lo, lM_hi, Nm)
        z_values = np.linspace(z_lo, z_hi, Nz)
        z = np.repeat(z_values, Nm)
        logM = np.tile(logM_values, Nz)

    # assemble fit parameters and generate new size data
    results = {"logM": logM, "z": z}
    fit_pars = assemble_mixed_fit_parameters(Samples_MFit)
    for color, t in zip([color_SF, color_Q], types):
        _res = generate_sizes(
            fit_pars,
            logM,
            z,
            np.repeat(color, Nm * Nz),
            fit_types=DEFAULT_MIXED_FIT_FITTYPES,
            samples=Samples_MFit,
            size_funcs=DEFAULT_SIZE_FUNCTIONS,
            fit_funcs=DEFAULT_FIT_FUNCTIONS,
            UVJcolor_cut=UVJcolor_cut,
            scatter=scatter,
        )
        results[var.format(t)] = _res[1]

    # test or save median values
    if read:
        for k in header[len(variables) :]:
            test = np.isclose(results[k], test_data[k], rtol=rtol)
            assert np.all(test)
    else:
        np.savetxt(
            filename, np.vstack(list(results[k] for k in results)).T, fmt="%11.4e", header="    ".join(header)
        )
        print(f"Writing test data in {filename}")

    return
