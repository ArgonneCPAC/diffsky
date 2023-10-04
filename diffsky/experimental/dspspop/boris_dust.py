"""
"""
from collections import OrderedDict

import numpy as np
from dsps.utils import _inverse_sigmoid, _sigmoid, powerlaw_rvs
from jax import jit as jjit
from jax import numpy as jnp
from jax import vmap

powerlaw_rvs_vmap = jjit(vmap(powerlaw_rvs, in_axes=(None, 0, 0, 0)))

DEFAULT_PDICT = OrderedDict(
    funo_logfburst_x0_logsm_x0_young=10.0,
    funo_logfburst_x0_logsm_ylo_young=-3.0,
    funo_logfburst_x0_logsm_yhi_young=-2.0,
    funo_logfburst_ylo_logsm_x0_young=10.25,
    funo_logfburst_ylo_logsm_ylo_young=0.1,
    funo_logfburst_ylo_logsm_yhi_young=0.05,
    funo_logfburst_yhi_logsm_x0_young=10.75,
    funo_logfburst_yhi_logsm_ylo_young=0.3,
    funo_logfburst_yhi_logsm_yhi_young=0.15,
    funo_logssfr_x0_logsm_x0_old=10.25,
    funo_logssfr_x0_logsm_ylo_old=-10.5,
    funo_logssfr_x0_logsm_yhi_old=-11.25,
    funo_logssfr_ylo_logsm_x0_old=9.5,
    funo_logssfr_ylo_logsm_ylo_old=0.15,
    funo_logssfr_ylo_logsm_yhi_old=0.05,
    funo_logssfr_yhi_logsm_x0_old=10.75,
    funo_logssfr_yhi_logsm_ylo_old=0.1,
    funo_logssfr_yhi_logsm_yhi_old=0.25,
    funo_vs_ssp_age_x0_logsm_x0=10.0,
    funo_vs_ssp_age_x0_logsm_ylo=0.3,
    funo_vs_ssp_age_x0_logsm_yhi=0.05,
)

LOGSM_X0_BOUNDS = 9.0, 12.0
LOGSSFR_X0_BOUNDS = -12.5, -7.0
LOGFB_X0_BOUNDS = -5.0, -1.0
FUNO_BOUNDS = 0.0, 1.0

BOUNDS_DICT = OrderedDict(
    funo_logfburst_x0_logsm_x0_young=LOGSM_X0_BOUNDS,
    funo_logfburst_x0_logsm_ylo_young=LOGFB_X0_BOUNDS,
    funo_logfburst_x0_logsm_yhi_young=LOGFB_X0_BOUNDS,
    funo_logfburst_ylo_logsm_x0_young=LOGSM_X0_BOUNDS,
    funo_logfburst_ylo_logsm_ylo_young=FUNO_BOUNDS,
    funo_logfburst_ylo_logsm_yhi_young=FUNO_BOUNDS,
    funo_logfburst_yhi_logsm_x0_young=LOGSM_X0_BOUNDS,
    funo_logfburst_yhi_logsm_ylo_young=FUNO_BOUNDS,
    funo_logfburst_yhi_logsm_yhi_young=FUNO_BOUNDS,
    funo_logssfr_x0_logsm_x0_old=LOGSM_X0_BOUNDS,
    funo_logssfr_x0_logsm_ylo_old=LOGSSFR_X0_BOUNDS,
    funo_logssfr_x0_logsm_yhi_old=LOGSSFR_X0_BOUNDS,
    funo_logssfr_ylo_logsm_x0_old=LOGSM_X0_BOUNDS,
    funo_logssfr_ylo_logsm_ylo_old=FUNO_BOUNDS,
    funo_logssfr_ylo_logsm_yhi_old=FUNO_BOUNDS,
    funo_logssfr_yhi_logsm_x0_old=LOGSM_X0_BOUNDS,
    funo_logssfr_yhi_logsm_ylo_old=FUNO_BOUNDS,
    funo_logssfr_yhi_logsm_yhi_old=FUNO_BOUNDS,
    funo_vs_ssp_age_x0_logsm_x0=LOGSM_X0_BOUNDS,
    funo_vs_ssp_age_x0_logsm_ylo=FUNO_BOUNDS,
    funo_vs_ssp_age_x0_logsm_yhi=FUNO_BOUNDS,
)

LGFRACUNO_LGSM_K = 5.0
LGFRACUNO_LGFB_K = 3.0
LGFRACUNO_LGSSFR_K = 5.0
LGFRACUNO_LGAGE_K = 5.0
BOUNDING_K = 0.1

LGFUNO_PLAW_SLOPE = 2.0

DEFAULT_PARAMS = np.array(list(DEFAULT_PDICT.values()))
PARAM_BOUNDS = np.array(list(BOUNDS_DICT.values()))


@jjit
def _age_dep_funo_from_params_kern(
    gal_logsm,
    gal_logfburst,
    gal_logssfr,
    ssp_lg_age_gyr,
    funo_logfburst_x0_logsm_x0_young,
    funo_logfburst_x0_logsm_ylo_young,
    funo_logfburst_x0_logsm_yhi_young,
    funo_logfburst_ylo_logsm_x0_young,
    funo_logfburst_ylo_logsm_ylo_young,
    funo_logfburst_ylo_logsm_yhi_young,
    funo_logfburst_yhi_logsm_x0_young,
    funo_logfburst_yhi_logsm_ylo_young,
    funo_logfburst_yhi_logsm_yhi_young,
    funo_logssfr_x0_logsm_x0_old,
    funo_logssfr_x0_logsm_ylo_old,
    funo_logssfr_x0_logsm_yhi_old,
    funo_logssfr_ylo_logsm_x0_old,
    funo_logssfr_ylo_logsm_ylo_old,
    funo_logssfr_ylo_logsm_yhi_old,
    funo_logssfr_yhi_logsm_x0_old,
    funo_logssfr_yhi_logsm_ylo_old,
    funo_logssfr_yhi_logsm_yhi_old,
    funo_vs_ssp_age_x0_logsm_x0,
    funo_vs_ssp_age_x0_logsm_ylo,
    funo_vs_ssp_age_x0_logsm_yhi,
):
    #  compute params for young stars

    funo_logsm_x0_young = _sigmoid(
        gal_logsm,
        funo_logfburst_x0_logsm_x0_young,
        LGFRACUNO_LGSM_K,
        funo_logfburst_x0_logsm_ylo_young,
        funo_logfburst_x0_logsm_yhi_young,
    )

    funo_logsm_ylo_young = _sigmoid(
        gal_logsm,
        funo_logfburst_ylo_logsm_x0_young,
        LGFRACUNO_LGSM_K,
        funo_logfburst_ylo_logsm_ylo_young,
        funo_logfburst_ylo_logsm_yhi_young,
    )

    funo_logsm_yhi_young = _sigmoid(
        gal_logsm,
        funo_logfburst_yhi_logsm_x0_young,
        LGFRACUNO_LGSM_K,
        funo_logfburst_yhi_logsm_ylo_young,
        funo_logfburst_yhi_logsm_yhi_young,
    )

    funo_vs_ssp_age_young = _sigmoid(
        gal_logfburst,
        funo_logsm_x0_young,
        LGFRACUNO_LGFB_K,
        funo_logsm_ylo_young,
        funo_logsm_yhi_young,
    )

    #  compute params for old stars
    funo_logsm_x0_old = _sigmoid(
        gal_logsm,
        funo_logssfr_x0_logsm_x0_old,
        LGFRACUNO_LGSM_K,
        funo_logssfr_x0_logsm_ylo_old,
        funo_logssfr_x0_logsm_yhi_old,
    )

    funo_logsm_ylo_old = _sigmoid(
        gal_logsm,
        funo_logssfr_ylo_logsm_x0_old,
        LGFRACUNO_LGSM_K,
        funo_logssfr_ylo_logsm_ylo_old,
        funo_logssfr_ylo_logsm_yhi_old,
    )

    funo_logsm_yhi_old = _sigmoid(
        gal_logsm,
        funo_logssfr_yhi_logsm_x0_old,
        LGFRACUNO_LGSM_K,
        funo_logssfr_yhi_logsm_ylo_old,
        funo_logssfr_yhi_logsm_yhi_old,
    )

    funo_vs_ssp_age_old = _sigmoid(
        gal_logssfr,
        funo_logsm_x0_old,
        LGFRACUNO_LGSSFR_K,
        funo_logsm_ylo_old,
        funo_logsm_yhi_old,
    )

    #  compute the lg_age transition between young and old stars

    funo_vs_ssp_age_x0 = _sigmoid(
        gal_logsm,
        funo_vs_ssp_age_x0_logsm_x0,
        LGFRACUNO_LGSM_K,
        funo_vs_ssp_age_x0_logsm_ylo,
        funo_vs_ssp_age_x0_logsm_yhi,
    )

    #  compute parameter b, the high-end cutoff in the power-law PDF P(F_uno)

    funo = _sigmoid(
        ssp_lg_age_gyr,
        funo_vs_ssp_age_x0,
        LGFRACUNO_LGAGE_K,
        funo_vs_ssp_age_young,
        funo_vs_ssp_age_old,
    )

    return funo


N_PARAMS = len(DEFAULT_PARAMS)
_a = [0, 0, 0, None, *[None] * N_PARAMS]
_b = [None, None, None, 0, *[None] * N_PARAMS]
_age_dep_funo_from_params_singlegal = jjit(
    vmap(_age_dep_funo_from_params_kern, in_axes=_b)
)
_age_dep_funo_from_params_vmap = jjit(
    vmap(_age_dep_funo_from_params_singlegal, in_axes=_a)
)


@jjit
def _get_funo_from_params_singlegal(
    gal_logsm, gal_logfburst, gal_logssfr, ssp_lg_age_gyr, funo_params
):
    return _age_dep_funo_from_params_singlegal(
        gal_logsm, gal_logfburst, gal_logssfr, ssp_lg_age_gyr, *funo_params
    )


@jjit
def _get_funo_from_u_params_singlegal(
    gal_logsm, gal_logfburst, gal_logssfr, ssp_lg_age_gyr, funo_u_params
):
    funo_params = get_params_from_u_params(funo_u_params)
    return _age_dep_funo_from_params_singlegal(
        gal_logsm, gal_logfburst, gal_logssfr, ssp_lg_age_gyr, *funo_params
    )


@jjit
def _get_funo_from_params_galpop(
    gal_logsm, gal_logfburst, gal_logssfr, ssp_lg_age_gyr, funo_params
):
    return _age_dep_funo_from_params_vmap(
        gal_logsm, gal_logfburst, gal_logssfr, ssp_lg_age_gyr, *funo_params
    )


@jjit
def _get_funo_from_u_params_galpop(
    gal_logsm, gal_logfburst, gal_logssfr, ssp_lg_age_gyr, funo_u_params
):
    funo_params = get_params_from_u_params(funo_u_params)
    return _age_dep_funo_from_params_vmap(
        gal_logsm, gal_logfburst, gal_logssfr, ssp_lg_age_gyr, *funo_params
    )


@jjit
def _get_p_from_u_p_scalar(u_p, bounds):
    lo, hi = bounds
    p0 = 0.5 * (lo + hi)
    p = _sigmoid(u_p, p0, BOUNDING_K, lo, hi)
    return p


@jjit
def _get_u_p_from_p_scalar(p, bounds):
    lo, hi = bounds
    p0 = 0.5 * (lo + hi)
    u_p = _inverse_sigmoid(p, p0, BOUNDING_K, lo, hi)
    return u_p


_get_p_from_u_p_vmap = jjit(vmap(_get_p_from_u_p_scalar, in_axes=(0, 0)))
_get_u_p_from_p_vmap = jjit(vmap(_get_u_p_from_p_scalar, in_axes=(0, 0)))


@jjit
def get_params_from_u_params(u_params):
    return _get_p_from_u_p_vmap(u_params, PARAM_BOUNDS)


@jjit
def get_u_params_from_params(params):
    return _get_u_p_from_p_vmap(params, PARAM_BOUNDS)


DEFAULT_U_PARAMS = get_u_params_from_params(DEFAULT_PARAMS)


@jjit
def mc_funobs_from_u_params(
    ran_key, gal_logsm, gal_logfburst, gal_logssfr, ssp_lg_age_gyr, funo_u_params
):
    n_gals = gal_logsm.shape[0]
    n_age = ssp_lg_age_gyr.shape[0]
    a = jnp.zeros((n_gals, n_age))
    g = jnp.zeros((n_gals, n_age)) + LGFUNO_PLAW_SLOPE
    b = _get_funo_from_u_params_galpop(
        gal_logsm, gal_logfburst, gal_logssfr, ssp_lg_age_gyr, funo_u_params
    )
    frac_unobs = b - powerlaw_rvs_vmap(ran_key, a, b, g)
    return frac_unobs
