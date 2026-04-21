""""""

from collections import namedtuple
from functools import partial

from jax import jit as jjit
from jax import random as jran

from ...ssp_err_model import ssp_err_model
from .. import mc_diffstarpop_wrappers as mcdw
from ..disk_bulge_modeling import disk_knots
from . import ssp_weight_kernels as sspwk

PHOT_RAN_KEYS = (
    "mc_is_q",
    "uran_av",
    "uran_delta",
    "uran_funo",
    "uran_pburst",
    "delta_mag_ssp_scatter",
)
PhotRandoms = namedtuple("PhotRandoms", PHOT_RAN_KEYS)

DBKRandoms = namedtuple("DBKRandoms", ("fknot", "uran_fbulge"))
DiffMergeRandoms = namedtuple("DiffMergeRandoms", ("uran_pmerge",))


@jjit
def get_mc_phot_randoms(ran_key, diffstarpop_params, mah_params, cosmo_params):
    n_gals = mah_params.logm0.size

    # Monte Carlo diffstar params
    ran_key, sfh_key = jran.split(ran_key, 2)
    sfh_params, mc_is_q = mcdw.mc_diffstarpop_cens_wrapper(
        diffstarpop_params, sfh_key, mah_params, cosmo_params
    )
    # Generate randoms for stochasticity in dust attenuation curves
    ran_key, dust_key = jran.split(ran_key, 2)
    dust_randoms = sspwk.get_dust_randoms(dust_key, n_gals)

    # Randoms for burstiness
    ran_key, burst_key = jran.split(ran_key, 2)
    uran_pburst = sspwk.get_burstiness_randoms(burst_key, n_gals)

    # Scatter for SSP errors
    ran_key, ssp_key = jran.split(ran_key, 2)
    delta_mag_ssp_scatter = ssp_err_model.get_delta_mag_ssp_scatter(ssp_key, n_gals)

    phot_randoms = PhotRandoms(
        mc_is_q,
        dust_randoms.uran_av,
        dust_randoms.uran_delta,
        dust_randoms.uran_funo,
        uran_pburst,
        delta_mag_ssp_scatter,
    )
    return phot_randoms, sfh_params


@partial(jjit, static_argnames=["n_gals"])
def get_mc_dbk_randoms(dbk_key, n_gals):
    fknot_key, fbulge_key = jran.split(dbk_key, 2)
    fknot = jran.uniform(
        fknot_key, minval=0, maxval=disk_knots.FKNOT_MAX, shape=(n_gals,)
    )
    uran_fbulge = jran.uniform(fbulge_key, shape=(n_gals,))

    return DBKRandoms(fknot=fknot, uran_fbulge=uran_fbulge)


@partial(jjit, static_argnames=["n_gals"])
def get_merging_randoms(pmerge_key, n_gals):
    uran_pmerge = jran.uniform(pmerge_key, shape=(n_gals,))
    return DiffMergeRandoms(uran_pmerge=uran_pmerge)


@jjit
def get_mc_dbk_phot_randoms(ran_key, diffstarpop_params, mah_params, cosmo_params):
    phot_key, dbk_key = jran.split(ran_key, 2)
    phot_randoms, sfh_params = get_mc_phot_randoms(
        phot_key, diffstarpop_params, mah_params, cosmo_params
    )
    n_gals = sfh_params[0].shape[0]
    dbk_randoms = get_mc_dbk_randoms(dbk_key, n_gals)
    return phot_randoms, sfh_params, dbk_randoms


@jjit
def get_mc_phot_merge_randoms(ran_key, diffstarpop_params, mah_params, cosmo_params):
    phot_key, merge_key = jran.split(ran_key, 2)
    phot_randoms, sfh_params = get_mc_phot_randoms(
        phot_key, diffstarpop_params, mah_params, cosmo_params
    )
    n_gals = sfh_params[0].shape[0]
    merging_randoms = get_merging_randoms(merge_key, n_gals)
    return phot_randoms, sfh_params, merging_randoms


@jjit
def get_mc_dbk_phot_merge_randoms(
    ran_key, diffstarpop_params, mah_params, cosmo_params
):
    phot_key, dbk_key, merge_key = jran.split(ran_key, 3)
    phot_randoms, sfh_params = get_mc_phot_randoms(
        phot_key, diffstarpop_params, mah_params, cosmo_params
    )

    n_gals = sfh_params[0].shape[0]
    dbk_randoms = get_mc_dbk_randoms(dbk_key, n_gals)
    merging_randoms = get_merging_randoms(merge_key, n_gals)

    return phot_randoms, sfh_params, dbk_randoms, merging_randoms
