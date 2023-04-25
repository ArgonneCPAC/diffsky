"""
"""
from dsps.sed import calc_ssp_weights_sfh_table_lognormal_mdf
from jax import vmap, jit as jjit
from dsps.cosmology import age_at_z
from jax import numpy as jnp


_a = (None, 0, 0, None, None, None, 0)
calc_ssp_weights_sfh_table_lognormal_mdf_vmap = jjit(
    vmap(calc_ssp_weights_sfh_table_lognormal_mdf, in_axes=_a)
)


@jjit
def get_obs_photometry_singlez(
    ssp_obsmag_table,
    ssp_lgmet,
    ssp_lg_age,
    gal_t_table,
    gal_sfr_table,
    cosmo_params,
    z_obs,
    lgmet_scatter=0.2,
):
    pass
