"""Functions and script for optimizing parameters of disk_shapes model"""

from collections import namedtuple

import numpy as np
from jax import jit as jjit
from jax import numpy as jnp
from jax import random as jran
from jax import value_and_grad

from .. import diffndhist
from . import disk_shapes as dshape
from . import ellipse_proj_kernels as eproj


@jjit
def _mae(pred, target):
    diff = pred - target
    return jnp.mean(jnp.abs(diff))


@jjit
def get_all_params_from_varied(varied_params, default_params):
    all_params = default_params._replace(**varied_params._asdict())
    return all_params


@jjit
def loss_kern(varied_params, loss_data):
    x_target, ba_pdf_target, ran_key, mu_ran, phi_ran, ba_bins, default_params = (
        loss_data
    )
    disk_params = get_all_params_from_varied(varied_params, default_params)
    loss = _loss_kern(
        disk_params, x_target, ba_pdf_target, ran_key, mu_ran, phi_ran, ba_bins
    )
    return loss


@jjit
def _loss_kern(disk_params, x_target, ba_pdf_target, ran_key, mu_ran, phi_ran, ba_bins):
    pred_data = x_target, ran_key, mu_ran, phi_ran, ba_bins

    ba_pdf_pred = _pred_kern(disk_params, pred_data)
    loss = _mae(ba_pdf_pred, ba_pdf_target)
    return loss


@jjit
def _pred_kern(disk_params, pred_data):
    x_target, ran_key, mu_ran, phi_ran, ba_bins = pred_data

    ran_key, disk_key = jran.split(ran_key, 2)
    ellipse2d = _pred_ellipse_samples_kern(disk_params, disk_key, mu_ran, phi_ran)

    nddata = jnp.array(ellipse2d.alpha / ellipse2d.beta).reshape((-1, 1))
    ndsig = jnp.zeros_like(nddata) + 0.05

    ndbins_lo = ba_bins[:-1].reshape((-1, 1))
    ndbins_hi = ba_bins[1:].reshape((-1, 1))

    wcounts = diffndhist.tw_ndhist(nddata, ndsig, ndbins_lo, ndbins_hi)
    ba_pdf_pred_table = (wcounts / wcounts.sum()) / jnp.diff(ba_bins)

    ba_binmids = 0.5 * (ba_bins[:-1] + ba_bins[1:])
    ba_pdf_pred = jnp.interp(x_target, ba_binmids, ba_pdf_pred_table)

    return ba_pdf_pred


loss_and_grad_kern = jjit(value_and_grad(loss_kern, argnums=0))


@jjit
def _pred_ellipse_samples_kern(disk_params, disk_key, mu_ran, phi_ran):
    ngals = mu_ran.shape[0]
    axis_ratios = dshape.sample_disk_axis_ratios(disk_key, ngals, disk_params)
    a = jnp.ones_like(mu_ran)
    b = a * axis_ratios.b_over_a
    c = a * axis_ratios.c_over_a
    ellipse2d = eproj.compute_ellipse2d(a, b, c, mu_ran, phi_ran)
    return ellipse2d


if __name__ == "__main__":
    import argparse

    from diffsky.mass_functions.fitting_utils.fitting_helpers import jax_adam_wrapper

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-seed", help="Integer input to jax.random.key(seed)", default=0, type=int
    )
    parser.add_argument(
        "-ngals", help="Number of galaxies to estimate pdf", default=50_000, type=int
    )
    parser.add_argument(
        "-tdata_fname",
        help="Number of galaxies to estimate pdf",
        default="spiral_b_over_a_pdf_rodriguez_padilla_2013.txt",
    )
    args = parser.parse_args()

    VariedParams = namedtuple("VariedParams", ("c_min", "c_max"))
    varied_params = VariedParams._make(
        [getattr(dshape.DEFAULT_DISK_PARAMS, key) for key in VariedParams._fields]
    )

    ran_key = jran.key(args.seed)
    ran_key, loss_key = jran.split(ran_key, 2)

    target_data = np.loadtxt(args.tdata_fname, delimiter=",")
    X = target_data[:, 0]
    Y = target_data[:, 1]

    ba_bins = np.linspace(0.01, 0.99, 50)

    ran_key, mu_key, phi_key = jran.split(ran_key, 3)
    mu_ran = jran.uniform(mu_key, minval=-1, maxval=1, shape=(args.ngals,))
    phi_ran = jran.uniform(phi_key, minval=0, maxval=2 * np.pi, shape=(args.ngals,))

    default_params = dshape.DEFAULT_DISK_PARAMS
    loss_data = X, Y, loss_key, mu_ran, phi_ran, ba_bins, default_params
    args = loss_and_grad_kern, varied_params, loss_data, 100
    _res = jax_adam_wrapper(*args)
    p_best, loss, loss_hist = _res[:3]
    disk_params_best = default_params._replace(**p_best._asdict())
    print(f"\nInitial parameters:\n{default_params}")

    print(f"\nBest-fitting parameters:\n{disk_params_best}")
