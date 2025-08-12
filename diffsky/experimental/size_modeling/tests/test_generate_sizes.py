"""
Test user facing function
"""
import os
from importlib.resources import files

import numpy as np

from diffaux.size_modeling.fit_size_data import (
    Samples_zFit,
    generate_sizes,
    read_fit_parameters,
    zFitParameters,
)

TESTDATA_DRN = files("diffaux").joinpath("size_modeling/tests/testing_data")


def test_generate_sizes(
    lM_lo=9.0,
    lM_hi=12.0,
    Nm=4,
    z_lo=0.0,
    z_hi=3.0,
    Nz=4,
    color_SF=0.0,
    color_Q=2,
    testdata_dir=TESTDATA_DRN,
    UVJcolor_cut=1.5,
    scatter_hi=0.2,
    scatter_lo=0.2,
    read=True,
    fn="generate_sizes_test_data.txt",
    rtol=1e-4,
    variables=("logM", "z"),
    types=("SF", "Q"),
    values=("R_med_{}", "scatter_up_{}", "scatter_down_{}"),
):
    vals = [v.format(t) for t in list(types) for v in list(values)]
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

    # read in fit parameters and generate new size data
    results = {"logM": logM, "z": z}
    fit_pars, _ = read_fit_parameters(zFitParameters)
    for color, t in zip([color_SF, color_Q], types):
        _res = generate_sizes(
            fit_pars,
            logM,
            z,
            np.repeat(color, Nm * Nz),
            samples=Samples_zFit,
            UVJcolor_cut=UVJcolor_cut,
            scatter_hi=scatter_hi,
            scatter_lo=scatter_lo,
        )
        for n, v in enumerate(values):
            results[v.format(t)] = _res[n + 1]

    # test or save median values
    if read:
        for k in header[len(variables) :]:
            test = np.isclose(results[k], test_data[k], rtol=rtol)
            assert np.all(test)
    else:
        np.savetxt(
            filename,
            np.vstack(list(results[k] for k in results)).T,
            fmt="%11.4e",
            header="    ".join(header),
        )
        print(f"Writing test data in {filename}")

    return
