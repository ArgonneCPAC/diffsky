"""
Generate disk-bulge decomposition
"""
from collections import OrderedDict, namedtuple

import jax.numpy as jnp
import numpy as np
from diffstar.utils import cumulative_mstar_formed_galpop
from dsps.constants import SFR_MIN
from jax import random as jran

from .disk_bulge_kernels import (
    _bulge_sfh_vmap,
    _get_params_from_u_params_vmap,
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
FbulgeParams = namedtuple("FbulgeParams", DEFAULT_FBULGE_PDICT.keys())
DEFAULT_FBULGEPARAMS = FbulgeParams(**DEFAULT_FBULGE_PDICT)


def mc_disk_bulge(
    ran_key, tarr, sfh_pop, FbulgeFixedParams=DEFAULT_FBULGEPARAMS, new_model=True
):
    """Decompose input SFHs into disk and bulge contributions

    Parameters
    ----------
    ran_key : jax.random.PRNGKey

    tarr : ndarray, shape (n_t, )

    sfh_pop : ndarray, shape (n_gals, n_t)

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

    if new_model:
        ssfr = jnp.divide(sfh_pop, smh_pop)
        logssfr0 = jnp.log10(ssfr[:, -1])
        fbulge_params = generate_fbulge_parameters_2d_sigmoid(
            ran_key, logsm0, logssfr0, t10, t90, FbulgeFixedParams
        )
    else:
        fbulge_params = generate_fbulge_params(ran_key, t10, t90, logsm0)

    _res = _bulge_sfh_vmap(tarr, sfh_pop, fbulge_params)
    smh, eff_bulge, sfh_bulge, smh_bulge, bth = _res
    return fbulge_params, smh, eff_bulge, sfh_bulge, smh_bulge, bth


def generate_fbulge_params(
    ran_key,
    t10,
    t90,
    logsm0,
    mu_u_tcrit=2,
    delta_mu_u_tcrit=3,
    mu_u_early=5,
    delta_mu_u_early=0.1,
    mu_u_late=5,
    delta_mu_u_late=3,
    scale_u_early=10,
    scale_u_late=8,
    scale_u_tcrit=20,
):
    n = t10.size
    tcrit_key, early_key, late_key = jran.split(ran_key, 3)
    u_tcrit_table = [
        mu_u_tcrit - delta_mu_u_tcrit * scale_u_tcrit,
        mu_u_tcrit + delta_mu_u_tcrit * scale_u_tcrit,
    ]
    logsm_table = 8, 11.5
    mu_u_tcrit_pop = np.interp(logsm0, logsm_table, u_tcrit_table)
    mc_u_tcrit = jran.normal(tcrit_key, shape=(n,)) * scale_u_tcrit + mu_u_tcrit_pop

    u_early_table = [
        mu_u_early - delta_mu_u_early * scale_u_early,
        mu_u_early + delta_mu_u_early * scale_u_early,
    ]
    mu_u_early_pop = np.interp(logsm0, logsm_table, u_early_table)
    mc_u_early = jran.normal(early_key, shape=(n,)) * scale_u_early + mu_u_early_pop

    u_late_table = [
        mu_u_late + delta_mu_u_late * scale_u_late,
        mu_u_late - delta_mu_u_late * scale_u_late,
    ]
    mu_u_late_pop = np.interp(logsm0, logsm_table, u_late_table)
    mc_u_late = jran.normal(late_key, shape=(n,)) * scale_u_late + mu_u_late_pop

    u_params = np.array((mc_u_tcrit, mc_u_early, mc_u_late)).T
    fbulge_tcrit, fbulge_early, fbulge_late = _get_params_from_u_params_vmap(
        u_params, t10, t90
    )
    fbulge_params = np.array((fbulge_tcrit, fbulge_early, fbulge_late)).T
    return fbulge_params


def generate_fbulge_parameters_2d_sigmoid(
    ran_key, logsm0, logssfr0, t10, t90, FbulgeParams
):
    fbulge_early = _sigmoid_2d(
        logssfr0,
        FbulgeParams.early_logssfr0_x0,
        logsm0,
        FbulgeParams.early_logsm0_x0,
        FbulgeParams.early_logssfr0_k,
        FbulgeParams.early_logsm0_k,
        FbulgeParams.early_zmin,
        FbulgeParams.early_zmax,
    )

    fbulge_late = _sigmoid_2d(
        logssfr0,
        FbulgeParams.late_logssfr0_x0,
        logsm0,
        FbulgeParams.late_logsm0_x0,
        FbulgeParams.late_logssfr0_k,
        FbulgeParams.late_logsm0_k,
        fbulge_early,
        FbulgeParams.late_zmax,
    )

    fbulge_tcrit = _sigmoid_2d(
        logssfr0,
        FbulgeParams.tcrit_logssfr0_x0,
        logsm0,
        FbulgeParams.tcrit_logsm0_x0,
        FbulgeParams.tcrit_logssfr0_k,
        FbulgeParams.tcrit_logsm0_k,
        t90,
        t10,
    )

    fbulge_params = np.asarray((fbulge_tcrit, fbulge_early, fbulge_late)).T

    return fbulge_params
