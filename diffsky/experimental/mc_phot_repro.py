# flake8: noqa: E402
""""""

from jax import config

config.update("jax_enable_x64", True)


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
    phot_kern_results, phot_randoms = mcpk._mc_phot_kern(
        ran_key,
        diffstarpop_params,
        lc_data.z_obs,
        lc_data.t_obs,
        lc_data.mah_params,
        lc_data.ssp_data,
        lc_data.precomputed_ssp_mag_table,
        lc_data.z_phot_table,
        lc_data.wave_eff_table,
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
    phot_kern_results, phot_randoms = mcpk._mc_phot_kern(
        ran_key,
        diffstarpop_params,
        lc_data.z_obs,
        lc_data.t_obs,
        lc_data.mah_params,
        lc_data.ssp_data,
        lc_data.precomputed_ssp_mag_table,
        lc_data.z_phot_table,
        lc_data.wave_eff_table,
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
):
    dbk_phot_info, dbk_weights = mcpk._mc_lc_dbk_phot_kern(
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

    return dbk_phot_info
