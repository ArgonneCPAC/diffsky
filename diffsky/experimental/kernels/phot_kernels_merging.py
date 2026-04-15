""""""

from jax import random as jran

from . import mc_phot_kernels as mcpk


def _mc_dbk_specphot_kern_merging(
    ran_key, diffstarpop_params, mah_params, cosmo_params
):
    _res = _dbk_specphot_kern_merging(
        ran_key, diffstarpop_params, mah_params, cosmo_params
    )


def get_mc_dbk_specphot_merging_randoms(
    ran_key, diffstarpop_params, mah_params, cosmo_params
):
    dbk_merging_key, mc_phot_key = jran.split(ran_key, 2)
    phot_randoms, sfh_params = mcpk.get_mc_phot_randoms(
        mc_phot_key, diffstarpop_params, mah_params, cosmo_params
    )
    dbk_specphot_randoms = get_dbk_merging_randoms(dbk_merging_key, phot_randoms)
    return dbk_specphot_randoms, sfh_params


def get_dbk_merging_randoms(ran_key, phot_randoms):
    return dbk_specphot_randoms
