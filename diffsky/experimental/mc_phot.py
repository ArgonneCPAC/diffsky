# flake8: noqa: E402
""""""

from jax import config

config.update("jax_enable_x64", True)

from collections import namedtuple

from diffstar import DEFAULT_DIFFSTAR_PARAMS
from diffstar.defaults import FB
from dsps.cosmology import DEFAULT_COSMOLOGY
from jax import numpy as jnp

from ..param_utils import diffsky_param_wrapper as dpw
from ..param_utils import diffsky_param_wrapper_merging as dpwm
from .kernels import dbk_specphot_kernels as dbkspk
from .kernels import mc_phot_kernels as mcpk
from .kernels import mc_randoms


def mc_lc_phot(
    ran_key,
    lc_data,
    diffstarpop_params=dpw.DEFAULT_PARAM_COLLECTION.diffstarpop_params,
    mzr_params=dpw.DEFAULT_PARAM_COLLECTION.mzr_params,
    spspop_params=dpw.DEFAULT_PARAM_COLLECTION.spspop_params,
    scatter_params=dpw.DEFAULT_PARAM_COLLECTION.scatter_params,
    ssperr_params=dpw.DEFAULT_PARAM_COLLECTION.ssperr_params,
    cosmo_params=DEFAULT_COSMOLOGY,
    fb=FB,
    skip_param_check=False,
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
    param_collection = dpw.ParamCollection(
        diffstarpop_params, mzr_params, spspop_params, scatter_params, ssperr_params
    )
    if not skip_param_check:
        assert dpw.check_param_collection_is_ok(param_collection)

    phot_kern_results, phot_randoms = mcpk._mc_phot_kern(
        ran_key,
        lc_data.z_obs,
        lc_data.t_obs,
        lc_data.mah_params,
        lc_data.ssp_data,
        lc_data.precomputed_ssp_mag_table,
        lc_data.z_phot_table,
        lc_data.wave_eff_table,
        diffstarpop_params,
        mzr_params,
        spspop_params,
        scatter_params,
        ssperr_params,
        cosmo_params,
        fb,
    )
    phot_kern_results = phot_kern_results._asdict()
    for key, val in zip(lc_data.mah_params._fields, lc_data.mah_params):
        phot_kern_results[key] = val
    return phot_kern_results


def mc_lc_phot_merging(
    ran_key,
    lc_data,
    diffstarpop_params=dpwm.DEFAULT_PARAM_COLLECTION.diffstarpop_params,
    mzr_params=dpwm.DEFAULT_PARAM_COLLECTION.mzr_params,
    spspop_params=dpwm.DEFAULT_PARAM_COLLECTION.spspop_params,
    scatter_params=dpwm.DEFAULT_PARAM_COLLECTION.scatter_params,
    ssperr_params=dpwm.DEFAULT_PARAM_COLLECTION.ssperr_params,
    merging_params=dpwm.DEFAULT_PARAM_COLLECTION.merging_params,
    cosmo_params=DEFAULT_COSMOLOGY,
    fb=FB,
    skip_param_check=False,
    mc_merge=1,
):
    param_collection = dpwm.ParamCollection(
        diffstarpop_params,
        mzr_params,
        spspop_params,
        scatter_params,
        ssperr_params,
        merging_params,
    )
    if not skip_param_check:
        assert dpwm.check_param_collection_is_ok(param_collection)

    _res = mcpk._mc_phot_kern_merging(
        ran_key,
        lc_data.z_obs,
        lc_data.t_obs,
        lc_data.mah_params,
        lc_data.ssp_data,
        lc_data.precomputed_ssp_mag_table,
        lc_data.z_phot_table,
        lc_data.wave_eff_table,
        diffstarpop_params,
        mzr_params,
        spspop_params,
        scatter_params,
        ssperr_params,
        merging_params,
        cosmo_params,
        fb,
        lc_data.logmp_infall,
        lc_data.logmhost_infall,
        lc_data.t_infall,
        lc_data.is_central,
        lc_data.nhalos,
        lc_data.halo_indx,
        mc_merge,
    )
    phot_kern_results, phot_randoms = _res
    phot_kern_results = phot_kern_results._asdict()
    for key, val in zip(lc_data.mah_params._fields, lc_data.mah_params):
        phot_kern_results[key] = val

    return phot_kern_results


def mc_lc_specphot(
    ran_key,
    lc_data,
    line_wave_table,
    diffstarpop_params=dpw.DEFAULT_PARAM_COLLECTION.diffstarpop_params,
    mzr_params=dpw.DEFAULT_PARAM_COLLECTION.mzr_params,
    spspop_params=dpw.DEFAULT_PARAM_COLLECTION.spspop_params,
    scatter_params=dpw.DEFAULT_PARAM_COLLECTION.scatter_params,
    ssperr_params=dpw.DEFAULT_PARAM_COLLECTION.ssperr_params,
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
    phot_kern_results, phot_randoms, spec_kern_results = mcpk._mc_specphot_kern(
        ran_key,
        lc_data.z_obs,
        lc_data.t_obs,
        lc_data.mah_params,
        lc_data.ssp_data,
        lc_data.precomputed_ssp_mag_table,
        lc_data.z_phot_table,
        lc_data.wave_eff_table,
        line_wave_table,
        diffstarpop_params,
        mzr_params,
        spspop_params,
        scatter_params,
        ssperr_params,
        cosmo_params,
        fb,
    )
    phot_kern_results = phot_kern_results._asdict()
    for key, val in zip(lc_data.mah_params._fields, lc_data.mah_params):
        phot_kern_results[key] = val

    for i, emline_name in enumerate(lc_data.ssp_data.ssp_emline_wave._fields):
        phot_kern_results[emline_name] = spec_kern_results.linelum_gal[:, i]

    return phot_kern_results


def mc_lc_sed(
    ran_key,
    lc_data,
    diffstarpop_params=dpw.DEFAULT_PARAM_COLLECTION.diffstarpop_params,
    mzr_params=dpw.DEFAULT_PARAM_COLLECTION.mzr_params,
    spspop_params=dpw.DEFAULT_PARAM_COLLECTION.spspop_params,
    scatter_params=dpw.DEFAULT_PARAM_COLLECTION.scatter_params,
    ssperr_params=dpw.DEFAULT_PARAM_COLLECTION.ssperr_params,
    cosmo_params=DEFAULT_COSMOLOGY,
    fb=FB,
    skip_param_check=False,
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
    param_collection = dpw.ParamCollection(
        diffstarpop_params, mzr_params, spspop_params, scatter_params, ssperr_params
    )
    if not skip_param_check:
        assert dpw.check_param_collection_is_ok(param_collection)

    phot_kern_results, phot_randoms = mcpk._mc_phot_kern(
        ran_key,
        lc_data.z_obs,
        lc_data.t_obs,
        lc_data.mah_params,
        lc_data.ssp_data,
        lc_data.precomputed_ssp_mag_table,
        lc_data.z_phot_table,
        lc_data.wave_eff_table,
        diffstarpop_params,
        mzr_params,
        spspop_params,
        scatter_params,
        ssperr_params,
        cosmo_params,
        fb,
    )
    sfh_params = DEFAULT_DIFFSTAR_PARAMS._make(
        [getattr(phot_kern_results, key) for key in DEFAULT_DIFFSTAR_PARAMS._fields]
    )
    sed_kern_results = mcpk._sed_kern(
        phot_randoms,
        sfh_params,
        lc_data.z_obs,
        lc_data.t_obs,
        lc_data.mah_params,
        lc_data.ssp_data,
        mzr_params,
        spspop_params,
        scatter_params,
        ssperr_params,
        cosmo_params,
        fb,
    )
    rest_sed = sed_kern_results[0]
    phot_kern_results = phot_kern_results._asdict()
    phot_kern_results["rest_sed"] = rest_sed
    return phot_kern_results


def mc_lc_dbk_phot(
    ran_key,
    lc_data,
    diffstarpop_params=dpw.DEFAULT_PARAM_COLLECTION.diffstarpop_params,
    mzr_params=dpw.DEFAULT_PARAM_COLLECTION.mzr_params,
    spspop_params=dpw.DEFAULT_PARAM_COLLECTION.spspop_params,
    scatter_params=dpw.DEFAULT_PARAM_COLLECTION.scatter_params,
    ssperr_params=dpw.DEFAULT_PARAM_COLLECTION.ssperr_params,
    cosmo_params=DEFAULT_COSMOLOGY,
    fb=FB,
    return_dbk_weights=False,
    skip_param_check=False,
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
    param_collection = dpw.ParamCollection(
        diffstarpop_params, mzr_params, spspop_params, scatter_params, ssperr_params
    )
    if not skip_param_check:
        assert dpw.check_param_collection_is_ok(param_collection)

    dbk_phot_info, dbk_weights = mcpk._mc_dbk_phot_kern(
        ran_key,
        lc_data.z_obs,
        lc_data.t_obs,
        lc_data.mah_params,
        lc_data.ssp_data,
        lc_data.precomputed_ssp_mag_table,
        lc_data.z_phot_table,
        lc_data.wave_eff_table,
        diffstarpop_params,
        mzr_params,
        spspop_params,
        scatter_params,
        ssperr_params,
        cosmo_params,
        fb,
    )
    dbk_phot_info = dbk_phot_info._asdict()
    dbk_phot_info["mstar_bulge"] = dbk_weights.mstar_bulge.flatten()
    dbk_phot_info["mstar_disk"] = dbk_weights.mstar_disk.flatten()
    dbk_phot_info["mstar_knots"] = dbk_weights.mstar_knots.flatten()

    if return_dbk_weights:
        return dbk_phot_info, dbk_weights
    else:
        return dbk_phot_info


def mc_lc_dbk_sed(
    ran_key,
    lc_data,
    diffstarpop_params=dpw.DEFAULT_PARAM_COLLECTION.diffstarpop_params,
    mzr_params=dpw.DEFAULT_PARAM_COLLECTION.mzr_params,
    spspop_params=dpw.DEFAULT_PARAM_COLLECTION.spspop_params,
    scatter_params=dpw.DEFAULT_PARAM_COLLECTION.scatter_params,
    ssperr_params=dpw.DEFAULT_PARAM_COLLECTION.ssperr_params,
    cosmo_params=DEFAULT_COSMOLOGY,
    fb=FB,
    skip_param_check=False,
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
    param_collection = dpw.ParamCollection(
        diffstarpop_params, mzr_params, spspop_params, scatter_params, ssperr_params
    )
    if not skip_param_check:
        assert dpw.check_param_collection_is_ok(param_collection)

    phot_randoms, sfh_params, dbk_randoms = mc_randoms.get_mc_dbk_phot_randoms(
        ran_key, diffstarpop_params, lc_data.mah_params, cosmo_params
    )

    dbk_sed_info, dbk_weights = dbkspk._dbk_sed_kern(
        phot_randoms.mc_is_q,
        phot_randoms.uran_av,
        phot_randoms.uran_delta,
        phot_randoms.uran_funo,
        phot_randoms.uran_pburst,
        phot_randoms.delta_mag_ssp_scatter,
        dbk_randoms.uran_fbulge,
        dbk_randoms.fknot,
        sfh_params,
        lc_data.z_obs,
        lc_data.t_obs,
        lc_data.mah_params,
        lc_data.ssp_data,
        dpw.DEFAULT_PARAM_COLLECTION.mzr_params,
        dpw.DEFAULT_PARAM_COLLECTION.spspop_params,
        dpw.DEFAULT_PARAM_COLLECTION.scatter_params,
        dpw.DEFAULT_PARAM_COLLECTION.ssperr_params,
        DEFAULT_COSMOLOGY,
        fb,
    )

    dbk_sed_info = dbk_sed_info._asdict()
    return dbk_sed_info
