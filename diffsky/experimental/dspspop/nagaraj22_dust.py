"""
"""
from collections import OrderedDict

from dsps.utils import _sigmoid, _tw_sigmoid
from jax import jit as jjit
from jax import lax
from jax import numpy as jnp

TAU_PDICT = OrderedDict(
    taueff_ssfr_x0=-9.5,
    taueff_ssfr_k=1.5,
    taueff_ssfr_ylo=0.4,
    taueff_ssfr_yhi_x0=10,
    taueff_ssfr_yhi_k=2,
    taueff_ssfr_yhi_ylo=0.7,
    taueff_ssfr_yhi_yhi=1.3,
    zboost_x0=1.2,
    zboost_k=2.5,
    zboost_lgssfr_x0=-9.5,
    zboost_lgssfr_h=0.5,
    zboost_lgssfr_ylo_lgsm_x0=10,
    zboost_lgssfr_ylo_lgsm_h=0.5,
    zboost_lgssfr_ylo_lgsm_ylo=0.45,
    zboost_lgssfr_ylo_lgsm_yhi=0.0,
    zboost_lgssfr_yhi=0.1,
)
TAU_PARAMS = jnp.array(list(TAU_PDICT.values()))

TAU_BOUNDS_PDICT = OrderedDict(
    taueff_ssfr_x0=(-11.5, -8.5),
    taueff_ssfr_ylo=(0.1, 5.0),
    taueff_ssfr_yhi_ylo=(0.1, 100.0),
    taueff_ssfr_yhi_yhi=(0.1, 100.0),
    zboost_x0=(0.5, 1.5),
    zboost_lgssfr_ylo_lgsm_x0=(9.0, 12.0),
    zboost_lgssfr_yhi=(0.0, 5.0),
)

DELTA_PDICT = OrderedDict(delta_x0=-0.35, delta_k=4, delta_ylo=-0.6, delta_yhi=0.1)
DELTA_PARAMS = jnp.array(list(DELTA_PDICT.values()))


def get_median_dust_params(
    lgsm,
    lgssfr,
    z,
    tau_params=TAU_PARAMS,
    delta_params=DELTA_PARAMS,
):
    z, lgssfr, lgsm = get_1d_arrays(z, lgssfr, lgsm)

    return _get_median_dust_params_kern(lgsm, lgssfr, z, tau_params, delta_params)


@jjit
def _get_median_dust_params_kern(lgsm, lgssfr, z, tau_params, delta_params):
    median_tau = _get_taueff(z, lgssfr, lgsm, *tau_params)
    median_av = _av_from_tau(median_tau)

    median_delta = _taueff_to_delta(median_tau, *delta_params)
    median_eb = _eb_from_delta_kc13(median_delta)

    return median_eb, median_delta, median_av


@jjit
def _eb_from_delta_kc13(delta):
    return 0.85 - 1.9 * delta


@jjit
def _taueff_to_delta(taueff, delta_x0, delta_k, delta_ylo, delta_yhi):
    lgtau = jnp.log10(taueff)
    delta = _sigmoid(lgtau, delta_x0, delta_k, delta_ylo, delta_yhi)
    return delta


@jjit
def _av_from_tau(tau):
    logarg = (1 - lax.exp(-tau)) / tau
    Av = -2.5 * jnp.log10(logarg)
    return Av


@jjit
def _get_taueff(
    z,
    lgssfr,
    lgsm,
    taueff_ssfr_x0,
    taueff_ssfr_k,
    taueff_ssfr_ylo,
    taueff_ssfr_yhi_x0,
    taueff_ssfr_yhi_k,
    taueff_ssfr_yhi_ylo,
    taueff_ssfr_yhi_yhi,
    zboost_x0,
    zboost_k,
    zboost_lgssfr_x0,
    zboost_lgssfr_h,
    zboost_lgssfr_ylo_lgsm_x0,
    zboost_lgssfr_ylo_lgsm_h,
    zboost_lgssfr_ylo_lgsm_ylo,
    zboost_lgssfr_ylo_lgsm_yhi,
    zboost_lgssfr_yhi,
):
    taueff_z0 = _taueff_vs_lgssfr_lgsm_z0(
        lgssfr,
        lgsm,
        taueff_ssfr_x0,
        taueff_ssfr_k,
        taueff_ssfr_ylo,
        taueff_ssfr_yhi_x0,
        taueff_ssfr_yhi_k,
        taueff_ssfr_yhi_ylo,
        taueff_ssfr_yhi_yhi,
    )
    zboost = _get_redshift_boost(
        z,
        lgsm,
        lgssfr,
        zboost_x0,
        zboost_k,
        zboost_lgssfr_x0,
        zboost_lgssfr_h,
        zboost_lgssfr_ylo_lgsm_x0,
        zboost_lgssfr_ylo_lgsm_h,
        zboost_lgssfr_ylo_lgsm_ylo,
        zboost_lgssfr_ylo_lgsm_yhi,
        zboost_lgssfr_yhi,
    )
    taueff = taueff_z0 + zboost
    taueff = jnp.where(taueff < 0, 0.0, taueff)
    return taueff


@jjit
def _taueff_vs_lgssfr_lgsm_z0(
    lgssfr,
    lgsm,
    taueff_ssfr_x0,
    taueff_ssfr_k,
    taueff_ssfr_ylo,
    taueff_ssfr_yhi_x0,
    taueff_ssfr_yhi_k,
    taueff_ssfr_yhi_ylo,
    taueff_ssfr_yhi_yhi,
):
    taueff_ssfr_yhi = _taueff_vs_lgssfr_yhi_vs_lgsm(
        lgsm,
        taueff_ssfr_yhi_x0,
        taueff_ssfr_yhi_k,
        taueff_ssfr_yhi_ylo,
        taueff_ssfr_yhi_yhi,
    )
    return _sigmoid(
        lgssfr, taueff_ssfr_x0, taueff_ssfr_k, taueff_ssfr_ylo, taueff_ssfr_yhi
    )


@jjit
def _taueff_vs_lgssfr_yhi_vs_lgsm(
    lgsm,
    taueff_ssfr_yhi_x0,
    taueff_ssfr_yhi_k,
    taueff_ssfr_yhi_ylo,
    taueff_ssfr_yhi_yhi,
):
    return _sigmoid(
        lgsm,
        taueff_ssfr_yhi_x0,
        taueff_ssfr_yhi_k,
        taueff_ssfr_yhi_ylo,
        taueff_ssfr_yhi_yhi,
    )


@jjit
def _get_redshift_boost(
    z,
    lgsm,
    lgssfr,
    zboost_x0,
    zboost_k,
    zboost_lgssfr_x0,
    zboost_lgssfr_h,
    zboost_lgssfr_ylo_lgsm_x0,
    zboost_lgssfr_ylo_lgsm_h,
    zboost_lgssfr_ylo_lgsm_ylo,
    zboost_lgssfr_ylo_lgsm_yhi,
    zboost_lgssfr_yhi,
):
    #####
    # get boost to tau at high-redshift (zboost_yhi)

    # First compute the boost for quenched galaxies
    # large boost for low-mass quenched galaxies
    # zero boost for massive quenched galaxies
    zboost_lgssfr_ylo = _tw_sigmoid(
        lgsm,
        zboost_lgssfr_ylo_lgsm_x0,
        zboost_lgssfr_ylo_lgsm_h,
        zboost_lgssfr_ylo_lgsm_ylo,
        zboost_lgssfr_ylo_lgsm_yhi,
    )
    # for star-forming galaxies, boost to tau is small and mass-independent
    zboost_yhi = _tw_sigmoid(
        lgssfr, zboost_lgssfr_x0, zboost_lgssfr_h, zboost_lgssfr_ylo, zboost_lgssfr_yhi
    )

    zboost_ylo = 0.0  # No boost at z=0
    return _sigmoid(z, zboost_x0, zboost_k, zboost_ylo, zboost_yhi)


def get_1d_arrays(*args):
    """Return a list of ndarrays of the same length.

    Each arg must be either an ndarray of shape (npts, ), or a scalar.

    """
    results = [jnp.atleast_1d(arg) for arg in args]
    sizes = [arr.size for arr in results]
    npts = max(sizes)
    msg = "All input arguments should be either a float or ndarray of shape ({0}, )"
    assert set(sizes) <= set((1, npts)), msg.format(npts)

    result = [jnp.zeros(npts).astype(arr.dtype) + arr for arr in results]
    return result
