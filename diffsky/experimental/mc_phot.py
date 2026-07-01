# flake8: noqa: E402
""""""

from jax import config

config.update("jax_enable_x64", True)

from collections import namedtuple

from diffstar import DEFAULT_DIFFSTAR_PARAMS
from diffstar.defaults import FB
from dsps.cosmology import DEFAULT_COSMOLOGY
from jax import jit as jjit

from ..param_utils import diffsky_param_wrapper as dpw
from ..param_utils import diffsky_param_wrapper_merging as dpwm
from .kernels import (
    _dbk_sed_kern,
    _mc_dbk_photline_kern_merging,
    _mc_phot_kern_merging,
    _mc_photline_kern_merging,
    _sed_kern,
    mc_randoms,
)


@jjit
def mc_lc_phot(
    ran_key,
    lc_data,
    mc_merge,
    *,
    param_collection=dpwm.DEFAULT_PARAM_COLLECTION,
    cosmo_params=DEFAULT_COSMOLOGY,
    fb=FB,
):
    """Populate the input lightcone with galaxy photometry

    Parameters
    ----------
    ran_key : jax.random.key

    lc_data : namedtuple
        Contains info about the halo lightcone, SED inputs, and diffsky parameters

    mc_merge : int
        Integer specifying whether merging is stochastic.
        Use mc_merge=0 in gradient descent applications, and mc_merge=1 for mock-making

    Returns
    -------
    phot_kern_results : namedtuple
        Contains info about the galaxy SEDs

    phot_randoms : namedtuple
        Contains info about randoms used in photometry

    merging_randoms : namedtuple
        Contains info about randoms used in merging

    """

    args = (
        ran_key,
        lc_data.z_obs,
        lc_data.t_obs,
        lc_data.mah_params,
        lc_data.ssp_data,
        lc_data.precomputed_ssp_mag_table,
        lc_data.z_phot_table,
        lc_data.wave_eff_table,
        param_collection.diffstarpop_params,
        param_collection.mzr_params,
        param_collection.spspop_params,
        param_collection.scatter_params,
        param_collection.ssperr_params,
        param_collection.merging_params,
        cosmo_params,
        fb,
        lc_data.logmp_infall,
        lc_data.logmhost_infall,
        lc_data.t_infall,
        lc_data.is_central,
        lc_data.sat_weight,
        lc_data.halo_indx,
        mc_merge,
    )

    phot_kern_results, phot_randoms, merging_randoms = _mc_phot_kern_merging(*args)
    return phot_kern_results, phot_randoms, merging_randoms


@jjit
def mc_lc_photline(
    ran_key,
    lc_data,
    mc_merge,
    *,
    param_collection=dpwm.DEFAULT_PARAM_COLLECTION,
    cosmo_params=DEFAULT_COSMOLOGY,
    fb=FB,
):
    """Populate the input lightcone with galaxy photometry

    Parameters
    ----------
    ran_key : jax.random.key

    lc_data : namedtuple
        Contains info about the halo lightcone, SED inputs, and diffsky parameters

    Returns
    -------
    results : dict
        Contains info about the galaxy SEDs

    """
    args = (
        ran_key,
        lc_data.z_obs,
        lc_data.t_obs,
        lc_data.mah_params,
        lc_data.ssp_data,
        lc_data.precomputed_ssp_mag_table,
        lc_data.z_phot_table,
        lc_data.wave_eff_table,
        lc_data.line_wave_table,
        param_collection.diffstarpop_params,
        param_collection.mzr_params,
        param_collection.spspop_params,
        param_collection.scatter_params,
        param_collection.ssperr_params,
        param_collection.merging_params,
        cosmo_params,
        fb,
        lc_data.logmp_infall,
        lc_data.logmhost_infall,
        lc_data.t_infall,
        lc_data.is_central,
        lc_data.sat_weight,
        lc_data.halo_indx,
        mc_merge,
    )

    _res = _mc_photline_kern_merging(*args)
    phot_kern_results, phot_randoms, spec_kern_results = _res

    line_fields = list(lc_data.ssp_data.ssp_emline_wave._fields)
    fields = list(phot_kern_results._fields) + line_fields
    n_lines = len(line_fields)
    lines = [spec_kern_results.linelum_gal[:, i] for i in range(n_lines)]

    SpecPhotResults = namedtuple("SpecPhotResults", fields)
    spec_phot_results = SpecPhotResults(*phot_kern_results, *lines)

    return spec_phot_results


@jjit
def mc_lc_sed(
    ran_key,
    lc_data,
    mc_merge,
    *,
    param_collection=dpw.DEFAULT_PARAM_COLLECTION,
    cosmo_params=DEFAULT_COSMOLOGY,
    fb=FB,
):
    """Populate the input lightcone with galaxy SEDs

    Parameters
    ----------
    ran_key : jax.random.key

    lc_data : namedtuple
        Contains info about the halo lightcone, SED inputs, and diffsky parameters

    Returns
    -------
    results : dict
        Contains info about the galaxy SEDs

    """

    phot_kern_results, phot_randoms, merging_randoms = _mc_phot_kern_merging(
        ran_key,
        lc_data.z_obs,
        lc_data.t_obs,
        lc_data.mah_params,
        lc_data.ssp_data,
        lc_data.precomputed_ssp_mag_table,
        lc_data.z_phot_table,
        lc_data.wave_eff_table,
        *param_collection,
        DEFAULT_COSMOLOGY,
        fb,
        lc_data.logmp_infall,
        lc_data.logmhost_infall,
        lc_data.t_infall,
        lc_data.is_central,
        lc_data.sat_weight,
        lc_data.halo_indx,
        mc_merge,
    )
    sfh_params = DEFAULT_DIFFSTAR_PARAMS._make(
        [getattr(phot_kern_results, key) for key in DEFAULT_DIFFSTAR_PARAMS._fields]
    )

    args = (
        phot_randoms,
        merging_randoms,
        sfh_params,
        lc_data.z_obs,
        lc_data.t_obs,
        lc_data.mah_params,
        lc_data.ssp_data,
        param_collection.mzr_params,
        param_collection.spspop_params,
        param_collection.scatter_params,
        param_collection.ssperr_params,
        param_collection.merging_params,
        cosmo_params,
        fb,
        lc_data.logmp_infall,
        lc_data.logmhost_infall,
        lc_data.t_infall,
        lc_data.is_central,
        lc_data.sat_weight,
        lc_data.halo_indx,
        mc_merge,
    )
    sed_kern_results = _sed_kern(*args)
    return sed_kern_results


@jjit
def mc_lc_dbk_photline(
    ran_key,
    lc_data,
    mc_merge,
    *,
    param_collection=dpwm.DEFAULT_PARAM_COLLECTION,
    cosmo_params=DEFAULT_COSMOLOGY,
    fb=FB,
):
    """Populate the input lightcone with disk/bulge/knot photometry

    Parameters
    ----------
    ran_key : jax.random.key

    lc_data : namedtuple
        Contains info about the halo lightcone, SED inputs, and diffsky parameters

    Returns
    -------
    results : dict
        Contains info about disk/bulge/knot photometry

    """

    args = (
        ran_key,
        lc_data.z_obs,
        lc_data.t_obs,
        lc_data.mah_params,
        lc_data.ssp_data,
        lc_data.precomputed_ssp_mag_table,
        lc_data.z_phot_table,
        lc_data.wave_eff_table,
        lc_data.line_wave_table,
        param_collection.diffstarpop_params,
        param_collection.mzr_params,
        param_collection.spspop_params,
        param_collection.scatter_params,
        param_collection.ssperr_params,
        param_collection.merging_params,
        cosmo_params,
        fb,
        lc_data.logmp_infall,
        lc_data.logmhost_infall,
        lc_data.t_infall,
        lc_data.is_central,
        lc_data.sat_weight,
        lc_data.halo_indx,
        mc_merge,
    )
    dbk_phot_info, dbk_weights = _mc_dbk_photline_kern_merging(*args)
    return dbk_phot_info


@jjit
def mc_lc_dbk_sed(
    ran_key,
    lc_data,
    mc_merge,
    *,
    param_collection=dpwm.DEFAULT_PARAM_COLLECTION,
    cosmo_params=DEFAULT_COSMOLOGY,
    fb=FB,
):
    """Populate the input lightcone with disk/bulge/knot SEDs

    Parameters
    ----------
    ran_key : jax.random.key

    lc_data : namedtuple
        Contains info about the halo lightcone, SED inputs, and diffsky parameters

    Returns
    -------
    results : dict
        Contains info about disk/bulge/knot SEDs

    """

    _res = mc_randoms.get_mc_dbk_phot_merge_randoms(
        ran_key, param_collection.diffstarpop_params, lc_data.mah_params, cosmo_params
    )
    phot_randoms, sfh_params, dbk_randoms, merging_randoms = _res

    args = (
        phot_randoms,
        dbk_randoms,
        merging_randoms,
        sfh_params,
        lc_data.z_obs,
        lc_data.t_obs,
        lc_data.mah_params,
        lc_data.ssp_data,
        param_collection.mzr_params,
        param_collection.spspop_params,
        param_collection.scatter_params,
        param_collection.ssperr_params,
        param_collection.merging_params,
        cosmo_params,
        fb,
        lc_data.logmp_infall,
        lc_data.logmhost_infall,
        lc_data.t_infall,
        lc_data.is_central,
        lc_data.sat_weight,
        lc_data.halo_indx,
        mc_merge,
    )
    dbk_sed_info = _dbk_sed_kern(*args)
    return dbk_sed_info
