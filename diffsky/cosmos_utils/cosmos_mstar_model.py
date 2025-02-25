"""
"""

from collections import OrderedDict, namedtuple

import numpy as np
from jax import jit as jjit
from jax import numpy as jnp
from jax import value_and_grad
from jax.example_libraries import optimizers as jax_opt

from ..data_loaders.cosmos20_loader import load_cosmos20

DEFAULT_PDICT = OrderedDict(
    b0=18.01,
    i=-0.45,
    gr=0.37,
    ri=0.59,
    iz=1.00,
    redshift=1.25,
)
ModelParams = namedtuple("ModelParams", DEFAULT_PDICT.keys())
DEFAULT_PARAMS = ModelParams(*list(DEFAULT_PDICT.values()))

data_keys = list(DEFAULT_PDICT.keys())[1:]
PhotData = namedtuple("PhotData", data_keys + ["logsm"])


COSMOS_PHOT_KEYS = (
    "HSC_g_MAG",
    "HSC_r_MAG",
    "HSC_i_MAG",
    "HSC_z_MAG",
    "HSC_y_MAG",
)


def load_cosmos20_tdata(
    phot_keys=COSMOS_PHOT_KEYS, zlo=0.5, zhi=2.5, sm_key="lp_mass_best", **kwargs
):
    """Load the COSMOS-20 dataset used to define training data for the scaling relation.

    Parameters
    ----------
    phot_keys : list of strings

    zlo, zhi : floats

    sm_key : string

    Returns
    -------
    photdata : namedtuple
        Fields defined by PhotData at module top
        Fields contain only the minimum information to train the approximate model

    """

    cosmos = load_cosmos20(**kwargs)
    msk_goodphot = np.ones(len(cosmos)).astype(bool)
    for key in phot_keys:
        msk_goodphot = msk_goodphot & np.isfinite(cosmos[key])
    msk_z = (cosmos["photoz"] >= zlo) & (cosmos["photoz"] <= zhi)
    msk_sm = np.isfinite(cosmos[sm_key])

    msk = msk_goodphot & msk_z & msk_sm
    cat = dict()
    cat["photoz"] = np.copy(cosmos[msk]["photoz"])
    cat["logsm"] = np.copy(cosmos[msk][sm_key])

    for key in phot_keys:
        newkey = key.split("_")[1]
        cat[newkey] = np.copy(cosmos[msk][key])

    gr = cat["g"] - cat["r"]
    ri = cat["r"] - cat["i"]
    iz = cat["i"] - cat["z"]

    photdata = PhotData(cat["i"], gr, ri, iz, cat["photoz"], cat["logsm"])
    return photdata


@jjit
def predict_logsm(params, photdata):
    """Approximate model of LePhare stellar mass from COSMOS griz photometry

    Parameters
    ----------
    params : namedtuple
        Fields defined by DEFAULT_PARAMS at top of module

    photdata: namedtuple
        Fields defined by PhotData at top of module

    Returns
    -------
    logsm : ndarray, shape (n, )

    """
    logsm = (
        params.b0
        + params.i * photdata.i
        + params.gr * photdata.gr
        + params.ri * photdata.ri
        + params.iz * photdata.iz
        + params.redshift * photdata.redshift
    )
    return logsm


@jjit
def _mse(x, y):
    diff = y - x
    return jnp.mean(diff * diff)


@jjit
def _mae(x, y):
    diff = y - x
    return jnp.mean(jnp.abs(diff))


@jjit
def _loss_kern(params, photdata):
    pred = predict_logsm(params, photdata)
    target = photdata.logsm
    return _mae(pred, target)


loss_and_grad_func = jjit(value_and_grad(_loss_kern))


def fit_model(n_steps, photdata, params_init=DEFAULT_PARAMS, step_size=0.001):
    """Find best-fitting parameters for the approximate stellar mass model

    Parameters
    ----------
    n_steps : int

    photdata: namedtuple
        Fields defined by PhotData at top of module

    Returns
    -------
    best_fit_params : namedtuple
        Fields defined by DEFAULT_PARAMS at top of module

    loss_arr : ndarray, shape (n_steps, )

    """
    loss_collector = []
    opt_init, opt_update, get_params = jax_opt.adam(step_size)
    opt_state = opt_init(params_init)
    params = get_params(opt_state)

    loss_init, grads = loss_and_grad_func(params_init, photdata)

    for istep in range(n_steps):
        params = get_params(opt_state)
        loss, grads = loss_and_grad_func(params, photdata)
        opt_state = opt_update(istep, grads, opt_state)
        loss_collector.append(loss)

    loss_arr = np.array(loss_collector)
    best_fit_params = get_params(opt_state)

    return best_fit_params, loss_arr


if __name__ == "__main__":
    photdata = load_cosmos20_tdata()

    p_best, loss_arr_init = fit_model(100, photdata, step_size=0.01)
    p_best, loss_arr_init2 = fit_model(
        100, photdata, params_init=p_best, step_size=0.01
    )
    p_best, loss_arr0 = fit_model(1_000, photdata, params_init=p_best, step_size=0.1)
    p_best, loss_arr1 = fit_model(1_000, photdata, params_init=p_best, step_size=0.01)
    loss_arr = np.concatenate((loss_arr_init, loss_arr_init2, loss_arr0, loss_arr1))

    print("Best fit model:\n")
    for p, key in zip(p_best, p_best._fields):
        print(f"{key}={p:.2f},")

    np.save("loss_history", loss_arr)
