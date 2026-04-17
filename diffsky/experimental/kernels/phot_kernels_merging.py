""""""

from jax import numpy as jnp

from ...merging import compute_x_tot_from_x_in_situ, merging_model
from . import mc_phot_kernels as mcpk


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
    dbk_specphot_info, dbk_weights = mcpk._mc_dbk_specphot_kern(
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
    frac_dm = mstar_obs / mstar_in_situ
    dmag = -2.5 * jnp.log10(frac_dm)

    mstar_colnames = ("mstar_bulge", "mstar_disk", "mstar_knots")
    mstar_dict = dict()
    for name in mstar_colnames:
        mstar_dict[name] = getattr(dbk_specphot_info, name) * frac_dm

    mag_dict = dict()
    mag_colnames = ("obs_mags", "obs_mags_bulge", "obs_mags_disk", "obs_mags_knots")
    for name in mag_colnames:
        mag_dict[name] = getattr(dbk_specphot_info, name) + dmag

    linelum_dict = dict()
    for name in ssp_data.ssp_emline_wave._fields:
        linelum_dict[name] = getattr(dbk_specphot_info, name) * frac_dm
        for k in ("_bulge", "_disk", "_knots"):
            kname = name + k
            linelum_dict[kname] = getattr(dbk_specphot_info, kname) * frac_dm

    return mstar_dict, mag_dict, linelum_dict
