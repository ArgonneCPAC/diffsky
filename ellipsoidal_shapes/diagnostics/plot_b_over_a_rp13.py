""" """

import os

import numpy as np
from jax import random as jran

from .. import bulge_shapes, disk_shapes
from .. import ellipse_proj_kernels as eproj

try:
    from matplotlib import pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
MATPLOTLIB_MSG = "Must have matplotlib installed to use this function"

_THIS_DRNAME = os.path.dirname(os.path.abspath(__file__))
DRN_ESHAPES = os.path.dirname(_THIS_DRNAME)
DRN_RP13_TDATA = os.path.join(DRN_ESHAPES, "tests", "testing_data")

BNAME_TDATA_ELLIPSOID = "ellipsoid_b_over_a_pdf_rodriguez_padilla_2013.txt"
BNAME_TDATA_SPIRAL = "spiral_b_over_a_pdf_rodriguez_padilla_2013.txt"


def _mae(pred, target):
    diff = pred - target
    return np.mean(np.abs(diff))


def make_bulge_rp13_comparison_plot(
    ngals=50_000,
    drn_tdata=DRN_RP13_TDATA,
    bulge_params=bulge_shapes.DEFAULT_BULGE_PARAMS,
    bulge_params2=bulge_shapes.DEFAULT_BULGE_PARAMS,
    fname=None,
    enforce_tol=float("inf"),
):
    """"""
    assert HAS_MATPLOTLIB, MATPLOTLIB_MSG

    ran_key = jran.key(0)

    ran_key, mu_key, phi_key = jran.split(ran_key, 3)
    mu_ran = jran.uniform(mu_key, minval=-1, maxval=1, shape=(ngals,))
    phi_ran = jran.uniform(phi_key, minval=0, maxval=2 * np.pi, shape=(ngals,))

    fn_rp13_tdata = os.path.join(drn_tdata, BNAME_TDATA_ELLIPSOID)

    target_data = np.loadtxt(fn_rp13_tdata, delimiter=",")
    ba_pdf_abscissa_target = target_data[:, 0]
    ba_pdf_target = target_data[:, 1]

    ba_bins = np.linspace(0.01, 0.99, 50)
    ba_binmids = 0.5 * (ba_bins[:-1] + ba_bins[1:])

    ran_key, bulge_key, bulge_key2 = jran.split(ran_key, 3)

    a = np.ones(ngals)

    axis_ratios = bulge_shapes.sample_bulge_axis_ratios(bulge_key, ngals, bulge_params)
    bulge_ellipse2d = eproj.compute_ellipse2d(
        a, a * axis_ratios.b_over_a, a * axis_ratios.c_over_a, mu_ran, phi_ran
    )
    ba_pdf_model, __ = np.histogram(
        bulge_ellipse2d.beta / bulge_ellipse2d.alpha, ba_bins, density=True
    )

    axis_ratios2 = bulge_shapes.sample_bulge_axis_ratios(
        bulge_key2, ngals, bulge_params2
    )
    bulge2_ellipse2d = eproj.compute_ellipse2d(
        a, a * axis_ratios2.b_over_a, a * axis_ratios2.c_over_a, mu_ran, phi_ran
    )
    ba_pdf_model2, __ = np.histogram(
        bulge2_ellipse2d.beta / bulge2_ellipse2d.alpha, ba_bins, density=True
    )

    fig, ax = plt.subplots(1, 1)
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(0.001, 3.5)
    xlabel = ax.set_xlabel(r"$\beta / \alpha$")
    ylabel = ax.set_ylabel(r"${\rm PDF}$")

    rp13_label = "sdss ellipticals, Rodriguez Padilla (2013)"
    ax.fill_between(
        ba_pdf_abscissa_target, ba_pdf_target, alpha=0.5, color="gray", label=rp13_label
    )
    ax.plot(ba_binmids, ba_pdf_model, color="k", label="best-fit model bulges")
    ax.plot(ba_binmids, ba_pdf_model2, "--", color="k", label="default model bulges")
    ax.legend()

    ba_pdf_pred = np.interp(ba_pdf_abscissa_target, ba_binmids, ba_pdf_model)
    mae_loss = _mae(ba_pdf_pred, ba_pdf_target)
    assert mae_loss < enforce_tol, mae_loss

    if fname is not None:
        fig.savefig(
            fname, bbox_extra_artists=[xlabel, ylabel], bbox_inches="tight", dpi=200
        )

    return fig


def make_disk_rp13_comparison_plot(
    ngals=50_000,
    drn_tdata=DRN_RP13_TDATA,
    disk_params=disk_shapes.DEFAULT_DISK_PARAMS,
    disk_params2=disk_shapes.DEFAULT_DISK_PARAMS,
    fname=None,
    enforce_tol=float("inf"),
):
    """"""
    assert HAS_MATPLOTLIB, MATPLOTLIB_MSG

    ran_key = jran.key(0)

    ran_key, mu_key, phi_key = jran.split(ran_key, 3)
    mu_ran = jran.uniform(mu_key, minval=-1, maxval=1, shape=(ngals,))
    phi_ran = jran.uniform(phi_key, minval=0, maxval=2 * np.pi, shape=(ngals,))

    fn_rp13_tdata = os.path.join(drn_tdata, BNAME_TDATA_SPIRAL)

    target_data = np.loadtxt(fn_rp13_tdata, delimiter=",")
    ba_pdf_abscissa_target = target_data[:, 0]
    ba_pdf_target = target_data[:, 1]

    ba_bins = np.linspace(0.01, 0.99, 50)
    ba_binmids = 0.5 * (ba_bins[:-1] + ba_bins[1:])

    ran_key, disk_key, disk_key2 = jran.split(ran_key, 3)

    a = np.ones(ngals)

    axis_ratios = disk_shapes.sample_disk_axis_ratios(disk_key, ngals, disk_params)
    gal_ellipse2d = eproj.compute_ellipse2d(
        a, a * axis_ratios.b_over_a, a * axis_ratios.c_over_a, mu_ran, phi_ran
    )
    ba_pdf_model, __ = np.histogram(
        gal_ellipse2d.beta / gal_ellipse2d.alpha, ba_bins, density=True
    )

    axis_ratios2 = disk_shapes.sample_disk_axis_ratios(disk_key2, ngals, disk_params2)
    gal_ellipse2d2 = eproj.compute_ellipse2d(
        a, a * axis_ratios2.b_over_a, a * axis_ratios2.c_over_a, mu_ran, phi_ran
    )
    ba_pdf_model2, __ = np.histogram(
        gal_ellipse2d2.beta / gal_ellipse2d2.alpha, ba_bins, density=True
    )

    fig, ax = plt.subplots(1, 1)
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(0.001, 3.5)
    xlabel = ax.set_xlabel(r"$\beta / \alpha$")
    ylabel = ax.set_ylabel(r"${\rm PDF}$")

    rp13_label = "SDSS spirals, Rodriguez Padilla (2013)"
    ax.fill_between(
        ba_pdf_abscissa_target, ba_pdf_target, alpha=0.5, color="gray", label=rp13_label
    )
    ax.plot(ba_binmids, ba_pdf_model, color="k", label="best-fit model disks")
    ax.plot(ba_binmids, ba_pdf_model2, "--", color="k", label="default model disks")
    ax.legend()

    ba_pdf_pred = np.interp(ba_pdf_abscissa_target, ba_binmids, ba_pdf_model)
    mae_loss = _mae(ba_pdf_pred, ba_pdf_target)
    assert mae_loss < enforce_tol, mae_loss

    if fname is not None:
        fig.savefig(
            fname, bbox_extra_artists=[xlabel, ylabel], bbox_inches="tight", dpi=200
        )

    return fig
