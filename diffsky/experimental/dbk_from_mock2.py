""""""

from collections import namedtuple

from diffmah import logmh_at_t_obs
from diffstar import calc_sfh_galpop
from dsps.cosmology import age_at_z0
from dsps.metallicity import umzr
from dsps.sfh import diffburst
from dsps.sfh.diffburst import DEFAULT_BURST_PARAMS
from jax import jit as jjit
from jax import numpy as jnp
from jax import vmap

from ..dustpop.tw_dust import DEFAULT_DUST_PARAMS
from ..ssp_err_model2 import ssp_err_model
from . import mc_phot_repro
from . import photometry_interpolation as photerp
from .disk_bulge_modeling import disk_bulge_kernels as dbk
from .disk_bulge_modeling import disk_knots
from .disk_bulge_modeling import mc_disk_bulge as mcdb

_BPOP = (None, 0, 0)
_pureburst_age_weights_from_params_vmap = jjit(
    vmap(diffburst._pureburst_age_weights_from_params, in_axes=_BPOP)
)
DBK_PHOT_INFO_KEYS = (
    "logmp_obs",
    "logsm_obs",
    "logssfr_obs",
    "sfh_table",
    "obs_mags",
    "obs_mags_bulge",
    "obs_mags_disk",
    "obs_mags_knots",
    *dbk.FbulgeParams._fields,
    "eff_bulge_history",
    "sfh_bulge",
    "smh_bulge",
    "bulge_to_total_history",
    *DEFAULT_BURST_PARAMS._fields,
    *DEFAULT_DUST_PARAMS._fields,
    "ssp_weights",
)
DBK_PhotInfo = namedtuple("DBK_PhotInfo", DBK_PHOT_INFO_KEYS)


@jjit
def _reproduce_mock_phot_kern(
    mc_is_q,
    uran_av,
    uran_delta,
    uran_funo,
    uran_pburst,
    delta_mag_ssp_scatter,
    sfh_params,
    z_obs,
    t_obs,
    mah_params,
    ssp_data,
    precomputed_ssp_mag_table,
    z_phot_table,
    wave_eff_table,
    mzr_params,
    spspop_params,
    scatter_params,
    ssp_err_pop_params,
    cosmo_params,
    fb,
):

    phot_randoms = mc_phot_repro.PhotRandoms(
        mc_is_q,
        uran_av,
        uran_delta,
        uran_funo,
        uran_pburst,
        delta_mag_ssp_scatter,
    )

    phot_kern_results = mc_phot_repro._phot_kern(
        phot_randoms,
        sfh_params,
        z_obs,
        t_obs,
        mah_params,
        ssp_data,
        precomputed_ssp_mag_table,
        z_phot_table,
        wave_eff_table,
        mzr_params,
        spspop_params,
        scatter_params,
        ssp_err_pop_params,
        cosmo_params,
        fb,
    )
    return phot_kern_results, phot_randoms

    # return dbk_phot_info
