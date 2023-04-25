"""
"""
import numpy as np
from collections import OrderedDict
from jax import jit as jjit
from jax import vmap
from jax import numpy as jnp
from dsps.sed.stellar_age_weights import _calc_age_weights_from_logsm_table
from dsps.experimental.diffburst import _burst_age_weights_pop


_A = (None, 0, None, None)
_get_age_weights_from_tables_pop = jjit(
    vmap(_calc_age_weights_from_logsm_table, in_axes=_A)
)

DEFAULT_LGFBURST_PDICT = OrderedDict(
    lgf_ssfr_x0_ylo=-9.5,
    lgf_ssfr_x0_yhi=-9.5,
    lgf_ssfr_ylo_ylo=-4.0,
    lgf_ssfr_ylo_yhi=-4.0,
    lgf_ssfr_yhi_ylo=-4.0,
    lgf_ssfr_yhi_yhi=-4.0,
)
LGFBURST_BOUNDS_PDICT = OrderedDict(
    lgf_ssfr_x0_ylo=(-11.5, -8.0),
    lgf_ssfr_x0_yhi=(-11.5, -8.0),
    lgf_ssfr_ylo_ylo=(-6.0, -1.5),
    lgf_ssfr_ylo_yhi=(-6.0, -1.5),
    lgf_ssfr_yhi_ylo=(-6.0, -1.5),
    lgf_ssfr_yhi_yhi=(-6.0, -1.5),
)
DEFAULT_LGFBURST_PARAMS = np.array(list(DEFAULT_LGFBURST_PDICT.values()))
LGFBURST_BOUNDS = np.array(list(LGFBURST_BOUNDS_PDICT.values()))
LFG_LGSM_X0 = 10.0
LGF_SSFR_K = 2.0


@jjit
def _get_bursty_age_weights_pop(
    gal_lgt_table, gal_logsm_tables, dburst_pop, ssp_lg_age, lgfburst_pop, t_obs
):
    age_weights_smooth = _get_age_weights_from_tables_pop(
        gal_lgt_table, gal_logsm_tables, ssp_lg_age, t_obs
    )

    age_weights_burst = _burst_age_weights_pop(ssp_lg_age, dburst_pop)

    fburst_pop = 10 ** lgfburst_pop.reshape((-1, 1))
    age_weights = fburst_pop * age_weights_burst + (1 - fburst_pop) * age_weights_smooth

    return age_weights, age_weights_smooth, age_weights_burst


@jjit
def _sigmoid(x, x0, k, ymin, ymax):
    height_diff = ymax - ymin
    return ymin + height_diff / (1 + jnp.exp(-k * (x - x0)))


@jjit
def _inverse_sigmoid(y, x0, k, ylo, yhi):
    lnarg = (yhi - ylo) / (y - ylo) - 1
    return x0 - jnp.log(lnarg) / k


@jjit
def _get_bounded_lgfburst_param(u_param, bound):
    lo, hi = bound
    mid = 0.5 * (lo + hi)
    return _sigmoid(u_param, mid, 0.1, lo, hi)


@jjit
def _get_unbounded_lgfburst_param(param, bound):
    lo, hi = bound
    mid = 0.5 * (lo + hi)
    return _inverse_sigmoid(param, mid, 0.1, lo, hi)


_C = (0, 0)
_get_lgfburst_params_kern = jjit(vmap(_get_bounded_lgfburst_param, in_axes=_C))
_get_lgfburst_u_params_kern = jjit(vmap(_get_unbounded_lgfburst_param, in_axes=_C))


@jjit
def _get_bounded_lgfburst_params(u_params):
    params = _get_lgfburst_params_kern(u_params, LGFBURST_BOUNDS)
    return params


@jjit
def _get_unbounded_lgfburst_params(params):
    u_params = _get_lgfburst_u_params_kern(params, LGFBURST_BOUNDS)
    return u_params


@jjit
def _get_lgf_ssfr_x0_vs_logsm(logsm_obs, lgf_ssfr_x0_ylo, lgf_ssfr_x0_yhi):
    lgf_ssfr_x0 = _sigmoid(
        logsm_obs, LFG_LGSM_X0, LGF_SSFR_K, lgf_ssfr_x0_ylo, lgf_ssfr_x0_yhi
    )
    return lgf_ssfr_x0


@jjit
def _get_lgf_ssfr_ylo_vs_logsm(logsm_obs, lgf_ssfr_ylo_ylo, lgf_ssfr_ylo_yhi):
    lgf_ssfr_ylo = _sigmoid(
        logsm_obs, LFG_LGSM_X0, LGF_SSFR_K, lgf_ssfr_ylo_ylo, lgf_ssfr_ylo_yhi
    )
    return lgf_ssfr_ylo


@jjit
def _get_lgf_ssfr_yhi_vs_logsm(logsm_obs, lgf_ssfr_yhi_ylo, lgf_ssfr_yhi_yhi):
    lgf_ssfr_ylo = _sigmoid(
        logsm_obs, LFG_LGSM_X0, LGF_SSFR_K, lgf_ssfr_yhi_ylo, lgf_ssfr_yhi_yhi
    )
    return lgf_ssfr_ylo


@jjit
def _get_lgfburst(logsm_obs, logssfr_obs, fburst_params):
    lgf_ssfr_x0_ylo, lgf_ssfr_x0_yhi = fburst_params[0:2]
    lgf_ssfr_ylo_ylo, lgf_ssfr_ylo_yhi = fburst_params[2:4]
    lgf_ssfr_yhi_ylo, lgf_ssfr_yhi_yhi = fburst_params[4:]
    lgf_ssfr_x0 = _get_lgf_ssfr_x0_vs_logsm(logsm_obs, lgf_ssfr_x0_ylo, lgf_ssfr_x0_yhi)
    lgf_ssfr_ylo = _get_lgf_ssfr_ylo_vs_logsm(
        logsm_obs, lgf_ssfr_ylo_ylo, lgf_ssfr_ylo_yhi
    )

    lgf_ssfr_yhi = _get_lgf_ssfr_yhi_vs_logsm(
        logsm_obs, lgf_ssfr_yhi_ylo, lgf_ssfr_yhi_yhi
    )

    lgfburst = _sigmoid(
        logssfr_obs, lgf_ssfr_x0, LGF_SSFR_K, lgf_ssfr_ylo, lgf_ssfr_yhi
    )

    return lgfburst
