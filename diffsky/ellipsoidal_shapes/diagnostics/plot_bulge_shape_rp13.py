""" """

import os

import numpy as np
from jax import random as jran
from matplotlib import pyplot as plt

from .. import bulge_shapes as shape_model
from .. import ellipse_proj_kernels as eproj

_THIS_DRNAME = os.path.dirname(os.path.abspath(__file__))
DRN_ESHAPES = os.path.dirname(_THIS_DRNAME)
DRN_RP13_TDATA = os.path.join(DRN_ESHAPES, "tests", "testing_data")

BNAME_TDATA = "ellipsoid_b_over_a_pdf_rodriguez_padilla_2013.txt"


def make_rp13_comparison_plot(
    ngals=50_000,
    drn_tdata=DRN_RP13_TDATA,
    bulge_params=shape_model.DEFAULT_BULGE_PARAMS,
    fname=None,
):
    ran_key = jran.key(0)

    ran_key, mu_key, phi_key = jran.split(ran_key, 3)
    mu_ran = jran.uniform(mu_key, minval=-1, maxval=1, shape=(ngals,))
    phi_ran = jran.uniform(phi_key, minval=0, maxval=2 * np.pi, shape=(ngals,))

    fn_rp13_tdata = os.path.join(drn_tdata, BNAME_TDATA)

    target_data = np.loadtxt(fn_rp13_tdata, delimiter=",")
    ba_pdf_abscissa_target = target_data[:, 0]
    ba_pdf_target = target_data[:, 1]

    ba_bins = np.linspace(0.01, 0.99, 50)
    ba_binmids = 0.5 * (ba_bins[:-1] + ba_bins[1:])

    ran_key, bulge_key = jran.split(ran_key, 2)

    axis_ratios = shape_model.sample_bulge_axis_ratios(bulge_key, ngals, bulge_params)
    a = np.ones(ngals)
    b = a * axis_ratios.b_over_a
    c = a * axis_ratios.c_over_a
    bulge_ellipse2d = eproj.compute_ellipse2d(a, b, c, mu_ran, phi_ran)

    ba = bulge_ellipse2d.beta / bulge_ellipse2d.alpha
    ba_pdf_model, __ = np.histogram(ba, ba_bins, density=True)

    fig, ax = plt.subplots(1, 1)
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(0.001, 3.5)
    xlabel = ax.set_xlabel(r"$\beta / \alpha$")
    ylabel = ax.set_ylabel(r"${\rm PDF}$")

    rp13_label = "Rodriguez Padilla (2013)"
    ax.fill_between(
        ba_pdf_abscissa_target, ba_pdf_target, alpha=0.5, color="gray", label=rp13_label
    )
    ax.plot(ba_binmids, ba_pdf_model, color="k", label="model")
    ax.legend()

    if fname is not None:
        fig.savefig(
            fname, bbox_extra_artists=[xlabel, ylabel], bbox_inches="tight", dpi=200
        )

    return fig
