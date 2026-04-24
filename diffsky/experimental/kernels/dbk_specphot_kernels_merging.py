""""""

from collections import namedtuple

from jax import jit as jjit
from jax import numpy as jnp

from . import dbk_specphot_kernels as dbkspk
from . import mc_randoms
from . import phot_kernels_merging as pkm
from . import specphot_kernels_merging as spkm


@jjit
def _mc_dbk_specphot_kern_merging(
    ran_key,
    z_obs,
    t_obs,
    mah_params,
    ssp_data,
    precomputed_ssp_mag_table,
    z_phot_table,
    wave_eff_table,
    line_wave_table,
    diffstarpop_params,
    mzr_params,
    spspop_params,
    scatter_params,
    ssp_err_pop_params,
    merge_params,
    cosmo_params,
    fb,
    logmp_infall,
    logmhost_infall,
    t_infall,
    is_central,
    nhalos_weights,
    halo_indx,
    mc_merge,
):
    _res = mc_randoms.get_mc_dbk_phot_merge_randoms(
        ran_key, diffstarpop_params, mah_params, cosmo_params
    )
    phot_randoms, sfh_params, dbk_randoms, merging_randoms = _res

    dbk_specphot_info, dbk_weights = _dbk_specphot_kern_merging(
        phot_randoms,
        sfh_params,
        dbk_randoms,
        merging_randoms,
        z_obs,
        t_obs,
        mah_params,
        ssp_data,
        precomputed_ssp_mag_table,
        z_phot_table,
        wave_eff_table,
        line_wave_table,
        mzr_params,
        spspop_params,
        scatter_params,
        ssp_err_pop_params,
        merge_params,
        cosmo_params,
        fb,
        logmp_infall,
        logmhost_infall,
        t_infall,
        is_central,
        nhalos_weights,
        halo_indx,
        mc_merge,
    )
    return dbk_specphot_info, dbk_weights


@jjit
def _dbk_specphot_kern_merging(
    phot_randoms,
    sfh_params,
    dbk_randoms,
    merging_randoms,
    z_obs,
    t_obs,
    mah_params,
    ssp_data,
    precomputed_ssp_mag_table,
    z_phot_table,
    wave_eff_table,
    line_wave_table,
    mzr_params,
    spspop_params,
    scatter_params,
    ssp_err_pop_params,
    merge_params,
    cosmo_params,
    fb,
    logmp_infall,
    logmhost_infall,
    t_infall,
    is_central,
    nhalos_weights,
    halo_indx,
    mc_merge,
):
    dbk_specphot_info, dbk_weights = dbkspk._dbk_specphot_kern(
        phot_randoms,
        sfh_params,
        dbk_randoms,
        z_obs,
        t_obs,
        mah_params,
        ssp_data,
        precomputed_ssp_mag_table,
        z_phot_table,
        wave_eff_table,
        line_wave_table,
        mzr_params,
        spspop_params,
        scatter_params,
        ssp_err_pop_params,
        cosmo_params,
        fb,
    )

    _res = pkm._get_phot_kern_merging_quantities(
        dbk_specphot_info,
        merging_randoms,
        t_obs,
        merge_params,
        logmp_infall,
        logmhost_infall,
        t_infall,
        is_central,
        nhalos_weights,
        halo_indx,
        mc_merge,
    )
    mstar_in_situ, mstar_obs, flux_in_situ, flux_obs, p_merge = _res

    args = (
        dbk_specphot_info,
        mstar_in_situ,
        mstar_obs,
        flux_in_situ,
        flux_obs,
        p_merge,
        merging_randoms.uran_pmerge,
    )
    dbk_specphot_info = pkm._update_phot_kern_results_with_merging(*args)

    args = dbk_specphot_info, dbk_specphot_info, nhalos_weights, halo_indx
    linelums_obs, linelum_in_situ = spkm._get_linelum_kern_merging_quantities(*args)

    dbk_specphot_info = spkm._get_linelum_results_with_merging(
        dbk_specphot_info, linelums_obs, linelum_in_situ
    )

    dbk_specphot_info, dbk_weights = _get_dbk_specphot_info_with_merging(
        dbk_specphot_info, dbk_weights, mstar_in_situ, mstar_obs, flux_in_situ, flux_obs
    )
    return dbk_specphot_info, dbk_weights


@jjit
def _get_dbk_specphot_info_with_merging(
    dbk_specphot_info, dbk_weights, mstar_in_situ, mstar_obs, flux_in_situ, flux_obs
):
    n_gals = dbk_specphot_info.logsm_obs.size

    frac_dm = mstar_obs / mstar_in_situ
    dmag = -2.5 * jnp.log10(flux_obs / flux_in_situ)

    ex_situ_dict = dict()
    in_situ_dict = dict()

    mstar_colnames = ("mstar_bulge", "mstar_disk", "mstar_knots")
    for name in mstar_colnames:
        outname = name.replace("mstar", "logsm")
        ex_situ_dict[outname] = jnp.log10(getattr(dbk_weights, name) * frac_dm)
        in_situ_dict[outname + "_in_situ"] = jnp.log10(getattr(dbk_weights, name))

    mag_colnames = ("obs_mags_bulge", "obs_mags_disk", "obs_mags_knots")
    for name in mag_colnames:
        ex_situ_dict[name] = getattr(dbk_specphot_info, name) + dmag
        in_situ_dict[name + "_in_situ"] = getattr(dbk_specphot_info, name)

    line_colnames = ("linelum_bulge", "linelum_disk", "linelum_knots")
    for name in line_colnames:
        _f = frac_dm.reshape((n_gals, 1))
        ex_situ_dict[name] = getattr(dbk_specphot_info, name) * _f
        in_situ_dict[name + "_in_situ"] = getattr(dbk_specphot_info, name)

    dbk_specphot_info = dbk_specphot_info._replace(**ex_situ_dict)
    dbk_weights = dbk_weights._replace(
        mstar_bulge=dbk_weights.mstar_bulge * frac_dm,
        mstar_disk=dbk_weights.mstar_disk * frac_dm,
        mstar_knots=dbk_weights.mstar_knots * frac_dm,
    )

    new_keys = list(in_situ_dict.keys())
    dbk_specphot_info_keys = list(dbk_specphot_info._fields) + new_keys
    MCDBKSpecPhotInfo = namedtuple("MCDBKSpecPhotInfo", dbk_specphot_info_keys)
    dbk_specphot_info = MCDBKSpecPhotInfo(**dbk_specphot_info._asdict(), **in_situ_dict)
    return dbk_specphot_info, dbk_weights
