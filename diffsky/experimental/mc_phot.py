# flake8: noqa: E402
""""""

from jax import config

config.update("jax_enable_x64", True)

from collections import namedtuple

from diffstar import DEFAULT_DIFFSTAR_PARAMS
from diffstar.defaults import FB
from dsps.cosmology import DEFAULT_COSMOLOGY

from ..param_utils import diffsky_param_wrapper as dpw
from .kernels import mc_phot_kernels as mcpk


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


def mc_lc_specphot(
    ran_key,
    lc_data,
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
    phot_kern_results, phot_randoms, gal_linefluxes = mcpk._mc_specphot_kern(
        ran_key,
        lc_data.z_obs,
        lc_data.t_obs,
        lc_data.mah_params,
        lc_data.ssp_data,
        lc_data.precomputed_ssp_mag_table,
        lc_data.precomputed_ssp_lineflux_cgs_table,
        lc_data.z_phot_table,
        lc_data.wave_eff_table,
        lc_data.line_wave_table,
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

    for i, emline_name in enumerate(lc_data.ssp_data.emlines._fields):
        phot_kern_results[emline_name] = gal_linefluxes[:, i]

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
    assert dpw.check_param_collection_is_ok(param_collection)

    dbk_sed_info, dbk_weights = mc_lc_dbk_phot(
        ran_key,
        lc_data,
        diffstarpop_params=diffstarpop_params,
        mzr_params=mzr_params,
        spspop_params=spspop_params,
        scatter_params=scatter_params,
        ssperr_params=ssperr_params,
        cosmo_params=cosmo_params,
        fb=fb,
        return_dbk_weights=True,
    )
    SEDInfo = namedtuple("SEDInfo", list(dbk_sed_info.keys()))
    dbk_sed_info = SEDInfo(**dbk_sed_info)
    sed_bulge, sed_disk, sed_knots = mcpk._mc_lc_dbk_sed_kern(
        dbk_sed_info,
        dbk_weights,
        lc_data.z_obs,
        lc_data.ssp_data,
        spspop_params,
        scatter_params,
        ssperr_params,
    )
    dbk_sed_info = dbk_sed_info._asdict()
    dbk_sed_info["rest_sed_bulge"] = sed_bulge
    dbk_sed_info["rest_sed_disk"] = sed_disk
    dbk_sed_info["rest_sed_knots"] = sed_knots
    return dbk_sed_info
