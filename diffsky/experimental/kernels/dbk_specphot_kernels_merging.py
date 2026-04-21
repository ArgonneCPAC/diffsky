""""""

from collections import namedtuple

from jax import jit as jjit
from jax import numpy as jnp

from ...merging import compute_x_tot_from_x_in_situ, merging_model
from . import dbk_specphot_kernels as dbkspk


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
):
    dbk_specphot_info, dbk_weights = dbkspk._mc_dbk_specphot_kern(
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
        cosmo_params,
        fb,
    )

    upids = jnp.where(is_central == 1, -1.0, 0.0)
    merge_prob = merging_model.get_p_merge_from_merging_params(
        merge_params, logmp_infall, logmhost_infall, t_obs, t_infall, upids
    )
    mstar_in_situ = 10**dbk_specphot_info.logsm_obs
    mstar_obs = compute_x_tot_from_x_in_situ(
        mstar_in_situ, merge_prob, nhalos_weights, halo_indx
    )

    dbk_specphot_info, dbk_weights = _get_dbk_specphot_info_with_merging(
        dbk_specphot_info, dbk_weights, mstar_in_situ, mstar_obs
    )
    return dbk_specphot_info, dbk_weights


@jjit
def _get_dbk_specphot_info_with_merging(
    dbk_specphot_info, dbk_weights, mstar_in_situ, mstar_obs
):
    n_gals = dbk_specphot_info.logsm_obs.size

    frac_dm = mstar_obs / mstar_in_situ
    dmag = -2.5 * jnp.log10(frac_dm)

    ex_situ_dict = dict()
    in_situ_dict = dict()

    ex_situ_dict["logsm_obs"] = dbk_specphot_info.logsm_obs + jnp.log10(frac_dm)
    in_situ_dict["logsm_obs" + "_in_situ"] = dbk_specphot_info.logsm_obs

    mstar_colnames = ("mstar_bulge", "mstar_disk", "mstar_knots")
    for name in mstar_colnames:
        outname = name.replace("mstar", "logsm")
        ex_situ_dict[outname] = jnp.log10(getattr(dbk_weights, name) * frac_dm)
        in_situ_dict[outname + "_in_situ"] = jnp.log10(getattr(dbk_weights, name))

    mag_colnames = ("obs_mags", "obs_mags_bulge", "obs_mags_disk", "obs_mags_knots")
    for name in mag_colnames:
        _dmag = dmag.reshape((n_gals, 1))
        ex_situ_dict[name] = getattr(dbk_specphot_info, name) + _dmag
        in_situ_dict[name + "_in_situ"] = getattr(dbk_specphot_info, name)

    line_colnames = ("linelum_gal", "linelum_bulge", "linelum_disk", "linelum_knots")
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
