"""
Kernels for disk-bulge modeling
"""
from collections import OrderedDict

import numpy as np
from diffstar.utils import cumulative_mstar_formed
from dsps.constants import SFR_MIN
from dsps.sed.stellar_age_weights import calc_age_weights_from_sfh_table
from jax import jit as jjit
from jax import lax, vmap
from jax import numpy as jnp

from dsps.sfh.diffburst import (
    _pureburst_age_weights_from_params as _burst_age_weights_from_params,
)
from .disk_knots import _disk_knot_kern, _disk_knot_vmap
from ...utils.tw_utils import _tw_sigmoid

FBULGE_MIN = 0.05
FBULGE_MAX = 0.95

BOUNDING_K = 0.1

DEFAULT_FBULGE_EARLY = 0.75
DEFAULT_FBULGE_LATE = 0.15

DEFAULT_FBULGE_PDICT = OrderedDict(fbulge_tcrit=8.0, fbulge_early=0.5, fbulge_late=0.1)
DEFAULT_FBULGE_PARAMS = np.array(list(DEFAULT_FBULGE_PDICT.values()))


_linterp_vmap = jjit(vmap(jnp.interp, in_axes=(0, None, 0)))

_A = (None, 0, 0)
_burst_age_weights_from_params_vmap = jjit(
    vmap(_burst_age_weights_from_params, in_axes=_A)
)

_D = (None, 0, None, 0)
calc_age_weights_from_sfh_table_vmap = jjit(
    vmap(calc_age_weights_from_sfh_table, in_axes=_D)
)


@jjit
def _sigmoid_2d(x, x0, y, y0, kx, ky, zmin, zmax):
    height_diff = zmax - zmin
    return zmin + height_diff * lax.logistic(kx * (x - x0) - ky * (y - y0))


@jjit
def _bulge_fraction_kernel(t, thalf, frac_early, frac_late, dt):
    """typical values of 10.0, 0.7, 0.1
    frac_late < frac_late is needed to make bulges redder and so that
    bulge fractions increase with stellar mass
    """
    tw_h = dt / 6.0
    return _tw_sigmoid(t, thalf, tw_h, frac_early, frac_late)


@jjit
def calc_tform_kern(abscissa, xarr, tform_frac):
    fracarr = xarr / xarr[-1]
    return jnp.interp(tform_frac, fracarr, abscissa)


_calc_tform_pop_kern = jjit(vmap(calc_tform_kern, in_axes=[None, 0, None]))


@jjit
def calc_tform_pop(tarr, smh_pop, tform_frac):
    """Calculate the formation time of a population

    Parameters
    ----------
    tarr : ndarray, shape(nt, )

    smh_pop : ndarray, shape(npop, nt)

    tform_frac : float
        Fraction used in the formation time definition
        tform_frac=0.5 corresponds to the half-mass time, for example

    """
    return _calc_tform_pop_kern(tarr, smh_pop, tform_frac)


@jjit
def _bulge_fraction_vs_tform(t, t10, t90, params):
    fbulge_tcrit, fbulge_early, fbulge_late = params
    dt = t90 - t10
    fbulge = _bulge_fraction_kernel(t, fbulge_tcrit, fbulge_early, fbulge_late, dt)
    return fbulge


@jjit
def _bulge_sfh(tarr, sfh, fbulge_params):
    sfh = jnp.where(sfh < SFR_MIN, SFR_MIN, sfh)
    smh = cumulative_mstar_formed(tarr, sfh)
    fracmh = smh / smh[-1]
    t10 = jnp.interp(0.1, fracmh, tarr)
    t90 = jnp.interp(0.9, fracmh, tarr)
    eff_bulge = _bulge_fraction_vs_tform(tarr, t10, t90, fbulge_params)
    sfh_bulge = eff_bulge * sfh
    smh_bulge = cumulative_mstar_formed(tarr, sfh_bulge)
    bth = smh_bulge / smh
    return smh, eff_bulge, sfh_bulge, smh_bulge, bth


_B = (None, 0, 0)
_bulge_sfh_vmap = jjit(vmap(_bulge_sfh, in_axes=_B))


@jjit
def _get_observed_quantity(t_obs, tarr, quantity):
    q_obs = jnp.interp(t_obs, tarr, quantity)
    return q_obs


_C = (0, None, 0)  # map t_obs and sfh for each galaxy
_get_observed_quantity_vmap = jjit(vmap(_get_observed_quantity, in_axes=_C))


@jjit
def get_observed_quantity_pop(t_obs, tarr, quantity):
    """Calculate the observed quantity for population

    Parameters
    ----------
    t_obs : times of observation, ndarray, shape(npop, )
    tarr : times for histories, ndarray, shape(nt, )

    quantity : history for quantity, ndarray, shape(npop, nt)
    """
    return _get_observed_quantity_vmap(t_obs, tarr, quantity)


def decompose_sfhpop_into_bulge_disk_knots(
    gal_fbulge_params,
    gal_fknot,
    gal_t_obs,
    gal_t_table,
    gal_sfh,
    gal_fburst,
    gal_burstshape_params,
    ssp_lg_age_gyr,
):
    """Decompose the SFH of a Diffsky galaxy into three components:
    bulges, diffuse disk, and star-forming knots in the disks

    Parameters
    ----------
    gal_fbulge_params : ndarray, shape (n_gals, 3)
        Bulge efficiency parameters (tcrit, fbulge_early, fbulge_late) for each galaxy

    gal_fknot : ndarray, shape (n_gals, )
        Fraction of the disk mass in bursty star-forming knots for each galaxy

    gal_t_obs : ndarray, shape (n_gals, )
        Age of the universe in Gyr at the redshift of each galaxy

    gal_t_table : ndarray, shape (n_t, )
        Grid in cosmic time t in Gyr at which SFH of the galaxy population is tabulated
        gal_t_table should increase monotonically and it should span the
        full range of gal_t_obs, including some padding of a few million years

    gal_sfh : ndarray, shape (n_gals, n_t)
        Grid in SFR in Msun/yr for each galaxy tabulated at the input gal_t_table

    gal_fburst : ndarray, shape (n_gals, )
        Fraction of stellar mass in the burst population of each galaxy

    gal_burstshape_params : ndarray, shape (n_gals, 3)
        Parameters controlling P(τ) for burst population in each galaxy
        lgfburst = gal_burstshape_params[:, 0]
        lgyr_peak = gal_burstshape_params[:, 1]
        lgyr_max = gal_burstshape_params[:, 2]

    ssp_lg_age_gyr : ndarray, shape (n_age, )
        Grid in age τ at which the SSPs are computed, stored as log10(τ/Gyr)

    Returns
    -------
    mbulge : ndarray, shape (n_gals, )
        Total stellar mass in Msun formed in the bulge at time gal_t_obs

    mdd : ndarray, shape (n_gals, )
        Total stellar mass in Msun formed in the diffuse disk at time gal_t_obs

    mknot : ndarray, shape (n_gals, )
        Total stellar mass in Msun formed in star-forming knots at time gal_t_obs

    mburst : ndarray, shape (n_gals, )
        Total stellar mass in Msun in the burst population at time gal_t_obs

    bulge_age_weights : ndarray, shape (n_gals, n_age)
        Probability distribution P(τ_age) for bulge of each galaxy

    dd_age_weights : ndarray, shape (n_gals, n_age)
        Probability distribution P(τ_age) for diffuse disk of each galaxy

    knot_age_weights : ndarray, shape (n_gals, n_age)
        Probability distribution P(τ_age) for star-forming knots of each galaxy

    bulge_sfh : ndarray, shape (n_gals, n_t)
        Grid in SFR in Msun/yr for each galaxy bulge tabulated at the input gal_t_table

    gal_frac_bulge_t_obs : ndarray, shape (n_gals, )
        Bulge/total mass ratio at gal_t_obs for every galaxy

    """
    ssp_lg_age_yr = ssp_lg_age_gyr + 9.0
    lgyr_peak = gal_burstshape_params[:, 1]
    lgyr_max = gal_burstshape_params[:, 2]
    gal_burst_age_weights = _burst_age_weights_from_params_vmap(
        ssp_lg_age_yr, lgyr_peak, lgyr_max,
    )
    return _decompose_sfhpop_into_bulge_disk_knots(
        gal_fbulge_params,
        gal_fknot,
        gal_t_obs,
        gal_t_table,
        gal_sfh,
        gal_fburst,
        gal_burst_age_weights,
        ssp_lg_age_gyr,
    )


@jjit
def _decompose_sfh_singlegal_into_bulge_disk_knots(
    fbulge_params,
    fknot,
    t_obs,
    t_table,
    sfh_table,
    fburst,
    age_weights_burst,
    ssp_lg_age_gyr,
):
    _res = _bulge_sfh(t_table, sfh_table, fbulge_params)
    smh, eff_bulge, bulge_sfh, smh_bulge, bulge_to_total_history = _res

    bulge_sfh = jnp.where(bulge_sfh < SFR_MIN, SFR_MIN, bulge_sfh)
    frac_bulge_t_obs = jnp.interp(t_obs, t_table, bulge_to_total_history)

    bulge_age_weights = calc_age_weights_from_sfh_table(
        t_table, bulge_sfh, ssp_lg_age_gyr, t_obs
    )
    disk_sfh = sfh_table - bulge_sfh
    disk_sfh = jnp.where(disk_sfh < SFR_MIN, SFR_MIN, disk_sfh)

    args = (
        t_table,
        t_obs,
        sfh_table,
        disk_sfh,
        fburst,
        fknot,
        age_weights_burst,
        ssp_lg_age_gyr,
    )
    _knot_info = _disk_knot_kern(*args)
    mstar_tot, mburst, mdd, mknot, dd_age_weights, knot_age_weights = _knot_info

    mbulge = frac_bulge_t_obs * mstar_tot
    masses = mbulge, mdd, mknot, mburst
    age_weights = bulge_age_weights, dd_age_weights, knot_age_weights
    ret = (*masses, *age_weights, bulge_sfh, frac_bulge_t_obs)
    return ret


@jjit
def _decompose_sfhpop_into_bulge_disk_knots(
    gal_fbulge_params,
    gal_fknot,
    gal_t_obs,
    gal_t_table,
    gal_sfh,
    gal_fburst,
    age_weights_burst,
    ssp_lg_age_gyr,
):
    _res = _bulge_sfh_vmap(gal_t_table, gal_sfh, gal_fbulge_params)
    smh, eff_bulge, bulge_sfh, smh_bulge, bulge_to_total_history = _res

    bulge_sfh = jnp.where(bulge_sfh < SFR_MIN, SFR_MIN, bulge_sfh)
    gal_frac_bulge_t_obs = _linterp_vmap(gal_t_obs, gal_t_table, bulge_to_total_history)

    bulge_age_weights = calc_age_weights_from_sfh_table_vmap(
        gal_t_table, bulge_sfh, ssp_lg_age_gyr, gal_t_obs
    )

    disk_sfh = gal_sfh - bulge_sfh
    disk_sfh = jnp.where(disk_sfh < SFR_MIN, SFR_MIN, disk_sfh)

    args = (
        gal_t_table,
        gal_t_obs,
        gal_sfh,
        disk_sfh,
        gal_fburst,
        gal_fknot,
        age_weights_burst,
        ssp_lg_age_gyr,
    )
    _knot_info = _disk_knot_vmap(*args)
    mstar_tot, mburst, mdd, mknot, dd_age_weights, knot_age_weights = _knot_info

    mbulge = gal_frac_bulge_t_obs * mstar_tot
    masses = mbulge, mdd, mknot, mburst
    age_weights = bulge_age_weights, dd_age_weights, knot_age_weights
    ret = (*masses, *age_weights, bulge_sfh, gal_frac_bulge_t_obs)
    return ret
