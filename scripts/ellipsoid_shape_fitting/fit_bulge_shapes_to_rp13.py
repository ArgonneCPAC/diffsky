"""Script refits the bulge shape PDF model to SDSS measurements of axis ratios of spirals"""

import os
from collections import namedtuple
from importlib.resources import files

import numpy as np
from jax import random as jran

from diffsky.data_loaders import io_utils
from diffsky.ellipsoidal_shapes import bulge_opt, bulge_shapes
from diffsky.ellipsoidal_shapes.diagnostics import plot_b_over_a_rp13 as pbarp13
from diffsky.mass_functions.fitting_utils.fitting_helpers import jax_adam_wrapper

_SUBDRN_RP13_TDATA = os.path.join("ellipsoidal_shapes", "tests", "testing_data")
BNAME_TDATA = "ellipsoid_b_over_a_pdf_rodriguez_padilla_2013.txt"

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-seed", help="Integer input to jax.random.key(seed)", default=0, type=int
    )
    parser.add_argument(
        "-ngals", help="Number of galaxies to estimate pdf", default=50_000, type=int
    )
    parser.add_argument(
        "-nsteps", help="Number of steps of gradient descent", default=1_000, type=int
    )
    parser.add_argument(
        "-drn_out", help="Output directory to store best-fit parameters", default=""
    )

    args = parser.parse_args()

    VariedParams = namedtuple(
        "VariedParams", ("c_min", "ba_min", "ba_peak", "ba_sigma")
    )
    varied_params = VariedParams._make(
        [
            getattr(bulge_shapes.DEFAULT_BULGE_PARAMS, key)
            for key in VariedParams._fields
        ]
    )

    ran_key = jran.key(args.seed)
    ran_key, loss_key = jran.split(ran_key, 2)

    DRN_RP13_TDATA = files("diffsky").joinpath(_SUBDRN_RP13_TDATA)
    fn_rp13_tdata = os.path.join(DRN_RP13_TDATA, BNAME_TDATA)
    target_data = np.loadtxt(fn_rp13_tdata, delimiter=",")
    X = target_data[:, 0]
    Y = target_data[:, 1]

    ba_bins = np.linspace(0.01, 0.99, 50)

    ran_key, mu_key, phi_key = jran.split(ran_key, 3)
    mu_ran = jran.uniform(mu_key, minval=-1, maxval=1, shape=(args.ngals,))
    phi_ran = jran.uniform(phi_key, minval=0, maxval=2 * np.pi, shape=(args.ngals,))

    default_params = bulge_shapes.DEFAULT_BULGE_PARAMS
    loss_data = X, Y, loss_key, mu_ran, phi_ran, ba_bins, default_params
    fit_args = bulge_opt.loss_and_grad_kern, varied_params, loss_data, args.nsteps
    _res = jax_adam_wrapper(*fit_args)
    p_best, loss, loss_hist = _res[:3]
    params_best = default_params._replace(**p_best._asdict())
    print(f"\nInitial parameters:\n{default_params}")

    print(f"\nBest-fitting parameters:\n{params_best}\n")

    fn_out = os.path.join(args.drn_out, "BulgeAxisRatioParams_rp13_bestfit.hdf5")
    io_utils.write_namedtuple_to_hdf5(params_best, fn_out)

    pbarp13.make_bulge_rp13_comparison_plot(
        bulge_params=params_best, fname="bulge_axis_ratio_rp13_comparison.png"
    )
