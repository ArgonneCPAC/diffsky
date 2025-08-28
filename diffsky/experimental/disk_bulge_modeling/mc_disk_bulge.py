"""
Generate disk-bulge decomposition
"""
from collections import OrderedDict, namedtuple

import jax.numpy as jnp
import numpy as np
from diffstar.utils import cumulative_mstar_formed_galpop
from dsps.constants import SFR_MIN

from .disk_bulge_kernels import (
    _bulge_sfh_vmap,
    _sigmoid_2d,
    calc_tform_pop,
)

DEFAULT_FBULGE_PDICT = OrderedDict(
    early_logsm0_x0=10.0,
    early_logssfr0_x0=-10.0,
    early_logssfr0_k=0.2,
    early_logsm0_k=0.2,
    early_zmin=1.0,
    early_zmax=0.2,
    late_logsm0_x0=10.0,
    late_logssfr0_x0=-10.0,
    late_logssfr0_k=0.2,
    late_logsm0_k=0.2,
    late_zmax=0.1,
    tcrit_logsm0_x0=10.0,
    tcrit_logssfr0_x0=-10.0,
    tcrit_logssfr0_k=0.5,
    tcrit_logsm0_k=0.8,
)
Fbulge2dParams = namedtuple("Fbulge2dParams", DEFAULT_FBULGE_PDICT.keys())
DEFAULT_FBULGE_2dSIGMOID_PARAMS = Fbulge2dParams(**DEFAULT_FBULGE_PDICT)


def mc_disk_bulge(
    ran_key, tarr, sfh_pop, fbulge_2d_params=DEFAULT_FBULGE_2dSIGMOID_PARAMS,
):
    """Decompose input SFHs into disk and bulge contributions

    Parameters
    ----------
    ran_key : jax.random.PRNGKey

    tarr : ndarray, shape (n_t, )

    sfh_pop : ndarray, shape (n_gals, n_t)

    fbulge_2d_params : named tuple of parameters for 2d-sigmoid

    new_model : boolean flag to switch between new and old model

    Returns
    -------
    fbulge_params : ndarray, shape (n_gals, 3)
        tcrit_bulge = fbulge_params[:, 0]
        fbulge_early = fbulge_params[:, 1]
        fbulge_late = fbulge_params[:, 2]

    smh : ndarray, shape (n_gals, n_t)
        Stellar mass history of galaxy in units of Msun

    eff_bulge : ndarray, shape (n_gals, n_t)
        History of in-situ bulge growth efficiency for every galaxy

    sfh_bulge : ndarray, shape (n_gals, n_t)
        Star formation history of bulge in units of Msun/yr

    smh_bulge : ndarray, shape (n_gals, n_t)
        Stellar mass history of bulge in units of Msun

    bth : ndarray, shape (n_gals, n_t)
        History of bulge-to-total mass ratio of every galaxy

    """
    sfh_pop = np.where(sfh_pop < SFR_MIN, SFR_MIN, sfh_pop)
    smh_pop = cumulative_mstar_formed_galpop(tarr, sfh_pop)
    t10 = calc_tform_pop(tarr, smh_pop, 0.1)
    t90 = calc_tform_pop(tarr, smh_pop, 0.9)
    logsm0 = jnp.log10(smh_pop[:, -1])

    ssfr = jnp.divide(sfh_pop, smh_pop)
    logssfr0 = jnp.log10(ssfr[:, -1])
    fbulge_params = generate_fbulge_parameters_2d_sigmoid(
        ran_key, logsm0, logssfr0, t10, t90, fbulge_2d_params
    )

    _res = _bulge_sfh_vmap(tarr, sfh_pop, fbulge_params)
    smh, eff_bulge, sfh_bulge, smh_bulge, bth = _res
    return fbulge_params, smh, eff_bulge, sfh_bulge, smh_bulge, bth


def generate_fbulge_parameters_2d_sigmoid(
    ran_key, logsm0, logssfr0, t10, t90, f_bulge_params
):
    fbulge_early = _sigmoid_2d(
        logssfr0,
        f_bulge_params.early_logssfr0_x0,
        logsm0,
        f_bulge_params.early_logsm0_x0,
        f_bulge_params.early_logssfr0_k,
        f_bulge_params.early_logsm0_k,
        f_bulge_params.early_zmin,
        f_bulge_params.early_zmax,
    )

    fbulge_late = _sigmoid_2d(
        logssfr0,
        f_bulge_params.late_logssfr0_x0,
        logsm0,
        f_bulge_params.late_logsm0_x0,
        f_bulge_params.late_logssfr0_k,
        f_bulge_params.late_logsm0_k,
        fbulge_early,
        f_bulge_params.late_zmax,
    )

    fbulge_tcrit = _sigmoid_2d(
        logssfr0,
        f_bulge_params.tcrit_logssfr0_x0,
        logsm0,
        f_bulge_params.tcrit_logsm0_x0,
        f_bulge_params.tcrit_logssfr0_k,
        f_bulge_params.tcrit_logsm0_k,
        t90,
        t10,
    )

    fbulge_param_arr = np.asarray((fbulge_tcrit, fbulge_early, fbulge_late)).T

    return fbulge_param_arr
