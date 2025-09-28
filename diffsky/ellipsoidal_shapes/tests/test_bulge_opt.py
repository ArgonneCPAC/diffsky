""""""

import numpy as np
from jax import random as jran

from .. import bulge_opt, bulge_shapes


def test_loss_kern():
    ran_key = jran.key(0)
    x_target = np.linspace(0.05, 0.95, 20)
    ba_pdf_target = np.ones_like(x_target)
    ba_pdf_target = ba_pdf_target / ba_pdf_target.sum() / np.diff(x_target)[0]
    ba_bins = np.linspace(0.01, 0.99, 50)
    n_gals = 200
    ran_key, mu_key, phi_key, loss_key = jran.split(ran_key, 4)
    mu_ran = jran.uniform(mu_key, minval=-1, maxval=1, shape=(n_gals,))
    phi_ran = jran.uniform(phi_key, minval=0, maxval=2 * np.pi, shape=(n_gals,))

    default_params = bulge_shapes.DEFAULT_BULGE_PARAMS
    loss_data = (
        x_target,
        ba_pdf_target,
        loss_key,
        mu_ran,
        phi_ran,
        ba_bins,
        default_params,
    )
    loss, grads = bulge_opt.loss_and_grad_kern(default_params, loss_data)
    assert np.all(np.isfinite(loss))
    assert np.all(loss > 0)
    assert np.all(np.abs(grads) > 0)
