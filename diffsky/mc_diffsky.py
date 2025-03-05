"""
"""

from collections import OrderedDict, namedtuple

from diffmah.diffmahpop_kernels.bimod_censat_params import DEFAULT_DIFFMAHPOP_PARAMS
from diffstar.defaults import T_TABLE_MIN
from diffstar.utils import cumulative_mstar_formed_galpop
from diffstarpop import mc_diffstarpop_tpeak as mcdsp
from diffstarpop.defaults import DEFAULT_DIFFSTARPOP_PARAMS
from diffstarpop.param_utils import mc_select_diffstar_params
from dsps.cosmology import flat_wcdm
from dsps.cosmology.defaults import DEFAULT_COSMOLOGY
from dsps.metallicity import umzr
from dsps.photometry import photpop
from dsps.sed import metallicity_weights as zmetw
from dsps.sed import stellar_age_weights as saw
from jax import jit as jjit
from jax import nn
from jax import numpy as jnp
from jax import random as jran
from jax import vmap

from .burstpop import diffqburstpop
from .dustpop import avpop_mono, deltapop, funopop_ssfr, tw_dust, tw_dustpop_mono
from .mass_functions.mc_diffmah_tpeak import mc_host_halos, mc_subhalos
from .phot_utils import get_interpolated_lsst_tcurves, get_wave_eff_from_tcurves
from .utils import _inverse_sigmoid

N_T = 100

DEFAULT_SCATTER_PDICT = OrderedDict(
    delta_scatter=5.0,
    av_scatter=5.0,
    lgfburst_scatter=5.0,
    lgmet_scatter=5.0,
    funo_scatter=5.0,
)
ScatterParams = namedtuple("ScatterParams", list(DEFAULT_SCATTER_PDICT.keys()))
DEFAULT_SCATTER_PARAMS = ScatterParams(*DEFAULT_SCATTER_PDICT.values())

_interp_vmap_single_t_obs = jjit(vmap(jnp.interp, in_axes=(None, None, 0)))

# gal_t_table, gal_sfr_table, ssp_lg_age_gyr, t_obs, sfr_min
_A = (None, 0, None, None, None)
_calc_age_weights_galpop = jjit(vmap(saw.calc_age_weights_from_sfh_table, in_axes=_A))

# gal_lgmet, gal_lgmet_scatter, ssp_lgmet
_M = (0, None, None)
_calc_lgmet_weights_galpop = jjit(
    vmap(zmetw.calc_lgmet_weights_from_lognormal_mdf, in_axes=_M)
)

# diffburstpop_params, logsm, logssfr, ssp_lg_age_gyr, smooth_age_weights
_B = (None, 0, 0, None, 0)
_calc_bursty_age_weights_vmap = jjit(
    vmap(diffqburstpop.calc_bursty_age_weights_from_diffburstpop_params, in_axes=_B)
)


def mc_diffstar_galpop(
    ran_key,
    z_obs,
    lgmp_min,
    volume_com=None,
    hosts_logmh_at_z=None,
    cosmo_params=DEFAULT_COSMOLOGY,
    diffstarpop_params=DEFAULT_DIFFSTARPOP_PARAMS,
    n_t=N_T,
    return_internal_quantities=False,
):
    """Generate a population of galaxies with diffmah MAH and diffstar SFH

    Parameters
    ----------
    ran_key : jran.PRNGKey

    z_obs : float
        Redshift of the halo population

    lgmp_min : float
        Base-10 log of the halo mass competeness limit of the generated population
        Smaller values of lgmp_min produce more halos in the returned sample
        A small fraction of halos will have slightly smaller masses than lgmp_min

    volume_com : float, optional
        volume_com = Lbox**3 where Lbox is in comoving in units of Mpc/h
        Default is None, in which case argument hosts_logmh_at_z must be passed

        Larger values of volume_com produce more halos in the returned sample

    hosts_logmh_at_z : ndarray, optional
        Grid of host halo masses at the input redshift.
        Default is None, in which case volume_com argument must be passed
        and the host halo mass function will be randomly sampled.

    return_internal_quantities : bool, optional
        If True, returned data will include additional info such as
        the separate SFHs for the probabilistic main and quenched sequences.
        Default is False, in which case only a single SFH will be returned.

    Returns
    -------
    diffsky_data : dict
        Diffstar galaxy population

    """
    mah_key, sfh_key = jran.split(ran_key, 2)

    t0 = flat_wcdm.age_at_z0(*cosmo_params)
    t_table = jnp.linspace(T_TABLE_MIN, t0, n_t)

    subcat = mc_subhalos(
        mah_key,
        z_obs,
        lgmp_min,
        volume_com=volume_com,
        hosts_logmh_at_z=hosts_logmh_at_z,
        cosmo_params=cosmo_params,
        diffmahpop_params=DEFAULT_DIFFMAHPOP_PARAMS,
    )

    logmu_infall = subcat.logmp_ult_inf - subcat.logmhost_ult_inf
    t_obs = flat_wcdm._age_at_z_kern(z_obs, *cosmo_params)
    args = (
        diffstarpop_params,
        subcat.mah_params,
        subcat.logmp0,
        logmu_infall,
        subcat.logmhost_ult_inf,
        t_obs - subcat.t_ult_inf,
        sfh_key,
        t_table,
    )

    _res = mcdsp.mc_diffstar_sfh_galpop(*args)
    sfh_ms, sfh_q, frac_q, mc_is_q = _res[2:]
    sfh_table = jnp.where(mc_is_q.reshape((-1, 1)), sfh_q, sfh_ms)
    smh_table = cumulative_mstar_formed_galpop(t_table, sfh_table)
    diffstar_params_ms, diffstar_params_q = _res[0:2]
    sfh_params = mc_select_diffstar_params(
        diffstar_params_q, diffstar_params_ms, mc_is_q
    )

    diffstar_data = dict()
    diffstar_data["subcat"] = subcat
    diffstar_data["t_table"] = t_table
    diffstar_data["t_obs"] = t_obs
    diffstar_data["sfh"] = sfh_table
    diffstar_data["smh"] = smh_table
    diffstar_data["mc_quenched"] = mc_is_q
    diffstar_data["sfh_params"] = sfh_params

    diffstar_data["logsm_obs"] = _interp_vmap_single_t_obs(
        t_obs, t_table, jnp.log10(diffstar_data["smh"])
    )
    logsfh_obs = _interp_vmap_single_t_obs(
        t_obs, t_table, jnp.log10(diffstar_data["sfh"])
    )
    diffstar_data["logssfr_obs"] = logsfh_obs - diffstar_data["logsm_obs"]

    if return_internal_quantities:
        diffstar_data["sfh_ms"] = sfh_ms
        diffstar_data["sfh_q"] = sfh_q
        diffstar_data["frac_q"] = frac_q
        diffstar_data["sfh_params_ms"] = diffstar_params_ms
        diffstar_data["sfh_params_q"] = diffstar_params_q

    return diffstar_data


def mc_diffstar_cenpop(
    ran_key,
    z_obs,
    lgmp_min=None,
    volume_com=None,
    hosts_logmh_at_z=None,
    cosmo_params=DEFAULT_COSMOLOGY,
    diffstarpop_params=DEFAULT_DIFFSTARPOP_PARAMS,
    n_t=N_T,
    return_internal_quantities=False,
):
    """Generate a population of central galaxies with diffmah MAH and diffstar SFH

    Parameters
    ----------
    ran_key : jran.PRNGKey

    z_obs : float
        Redshift of the halo population

    lgmp_min : float
        Base-10 log of the halo mass competeness limit of the generated population
        Smaller values of lgmp_min produce more halos in the returned sample
        A small fraction of halos will have slightly smaller masses than lgmp_min

    volume_com : float, optional
        volume_com = Lbox**3 where Lbox is in comoving in units of Mpc/h
        Default is None, in which case argument hosts_logmh_at_z must be passed

        Larger values of volume_com produce more halos in the returned sample

    hosts_logmh_at_z : ndarray, optional
        Grid of host halo masses at the input redshift.
        Default is None, in which case volume_com argument must be passed
        and the host halo mass function will be randomly sampled.

    return_internal_quantities : bool, optional
        If True, returned data will include additional info such as
        the separate SFHs for the probabilistic main and quenched sequences.
        Default is False, in which case only a single SFH will be returned.

    Returns
    -------
    diffsky_data : dict
        Diffstar galaxy population

    """

    mah_key, sfh_key = jran.split(ran_key, 2)

    t0 = flat_wcdm.age_at_z0(*cosmo_params)
    t_table = jnp.linspace(T_TABLE_MIN, t0, n_t)

    subcat = mc_host_halos(
        mah_key,
        z_obs,
        lgmp_min=lgmp_min,
        volume_com=volume_com,
        hosts_logmh_at_z=hosts_logmh_at_z,
        cosmo_params=cosmo_params,
        diffmahpop_params=DEFAULT_DIFFMAHPOP_PARAMS,
    )

    logmu_infall = subcat.logmp_ult_inf - subcat.logmhost_ult_inf
    t_obs = flat_wcdm._age_at_z_kern(z_obs, *cosmo_params)
    args = (
        diffstarpop_params,
        subcat.mah_params,
        subcat.logmp0,
        logmu_infall,
        subcat.logmhost_ult_inf,
        t_obs - subcat.t_ult_inf,
        sfh_key,
        t_table,
    )

    _res = mcdsp.mc_diffstar_sfh_galpop(*args)
    sfh_ms, sfh_q, frac_q, mc_is_q = _res[2:]
    sfh_table = jnp.where(mc_is_q.reshape((-1, 1)), sfh_q, sfh_ms)
    smh_table = cumulative_mstar_formed_galpop(t_table, sfh_table)
    diffstar_params_ms, diffstar_params_q = _res[0:2]
    sfh_params = mc_select_diffstar_params(
        diffstar_params_q, diffstar_params_ms, mc_is_q
    )

    diffstar_data = dict()
    diffstar_data["subcat"] = subcat
    diffstar_data["t_table"] = t_table
    diffstar_data["t_obs"] = t_obs
    diffstar_data["sfh"] = sfh_table
    diffstar_data["smh"] = smh_table
    diffstar_data["mc_quenched"] = mc_is_q
    diffstar_data["sfh_params"] = sfh_params

    diffstar_data["logsm_obs"] = _interp_vmap_single_t_obs(
        t_obs, t_table, jnp.log10(diffstar_data["smh"])
    )
    logsfh_obs = _interp_vmap_single_t_obs(
        t_obs, t_table, jnp.log10(diffstar_data["sfh"])
    )
    diffstar_data["logssfr_obs"] = logsfh_obs - diffstar_data["logsm_obs"]

    if return_internal_quantities:
        diffstar_data["sfh_ms"] = sfh_ms
        diffstar_data["sfh_q"] = sfh_q
        diffstar_data["frac_q"] = frac_q
        diffstar_data["sfh_params_ms"] = diffstar_params_ms
        diffstar_data["sfh_params_q"] = diffstar_params_q

    return diffstar_data


def mc_diffsky_galhalo_pop(
    ran_key,
    z_obs,
    lgmp_min,
    ssp_data,
    volume_com=None,
    hosts_logmh_at_z=None,
    cosmo_params=DEFAULT_COSMOLOGY,
    diffstarpop_params=DEFAULT_DIFFSTARPOP_PARAMS,
    dustpop_params=tw_dustpop_mono.DEFAULT_DUSTPOP_PARAMS,
    mzr_params=umzr.DEFAULT_MZR_PARAMS,
    lgmet_scatter=umzr.MZR_SCATTER,
    diffburstpop_params=diffqburstpop.DEFAULT_DIFFBURSTPOP_PARAMS,
    scatter_params=DEFAULT_SCATTER_PARAMS,
    n_t=N_T,
    return_internal_quantities=False,
):
    n_met, n_age, n_ssp_wave = ssp_data.ssp_flux.shape

    lgmet_key, diffstar_key = jran.split(ran_key, 2)
    diffsky_data = mc_diffstar_galpop(
        diffstar_key,
        z_obs,
        lgmp_min,
        volume_com=volume_com,
        hosts_logmh_at_z=hosts_logmh_at_z,
        cosmo_params=cosmo_params,
        diffstarpop_params=diffstarpop_params,
        n_t=n_t,
        return_internal_quantities=return_internal_quantities,
    )
    n_gals = diffsky_data["sfh"].shape[0]

    lgmet_med = umzr.mzr_model(
        diffsky_data["logsm_obs"], diffsky_data["t_obs"], *mzr_params
    )
    unorm = jran.normal(lgmet_key, shape=lgmet_med.shape) * lgmet_scatter
    diffsky_data["lgmet_med"] = lgmet_med + unorm

    diffsky_data["smooth_age_weights"] = _calc_age_weights_galpop(
        diffsky_data["t_table"],
        diffsky_data["sfh"],
        ssp_data.ssp_lg_age_gyr,
        diffsky_data["t_obs"],
        saw.SFR_MIN,
    )

    diffsky_data["lgmet_weights"] = _calc_lgmet_weights_galpop(
        diffsky_data["lgmet_med"], lgmet_scatter, ssp_data.ssp_lgmet
    )

    _args = (
        diffburstpop_params,
        diffsky_data["logsm_obs"],
        diffsky_data["logssfr_obs"],
        ssp_data.ssp_lg_age_gyr,
        diffsky_data["smooth_age_weights"],
    )
    bursty_age_weights, burst_params = _calc_bursty_age_weights_vmap(*_args)
    diffsky_data["bursty_age_weights"] = bursty_age_weights

    _wmet = diffsky_data["lgmet_weights"].reshape((n_gals, n_met, 1))
    _amet = diffsky_data["smooth_age_weights"].reshape((n_gals, 1, n_age))
    smooth_weights = _wmet * _amet
    smooth_weights = smooth_weights / smooth_weights.sum()

    _bmet = diffsky_data["bursty_age_weights"].reshape((n_gals, 1, n_age))
    bursty_weights = _wmet * _bmet
    bursty_weights = bursty_weights / bursty_weights.sum()

    diffsky_data["smooth_ssp_weights"] = smooth_weights
    diffsky_data["bursty_ssp_weights"] = bursty_weights

    return diffsky_data


@jjit
def calc_dust_ftrans(
    dustpop_params,
    wave_aa,
    logsm,
    logssfr,
    redshift,
    ssp_lg_age_gyr,
    random_draw,
    scatter_params,
):
    av = avpop_mono.get_av_from_avpop_params_singlegal(
        dustpop_params.avpop_params, logsm, logssfr, redshift, ssp_lg_age_gyr
    )
    delta = deltapop.get_delta_from_deltapop_params(
        dustpop_params.deltapop_params, logsm, logssfr
    )
    funo = funopop_ssfr.get_funo_from_funopop_params(
        dustpop_params.funopop_params, logssfr
    )

    suav = jnp.log(jnp.exp(av) - 1)
    noisy_suav = _inverse_sigmoid(
        random_draw, suav, scatter_params.av_scatter, 0.0, 1.0
    )
    noisy_av = nn.softplus(noisy_suav)

    udelta = deltapop._get_unbounded_deltapop_param(delta, deltapop.DELTAPOP_BOUNDS)
    noisy_udelta = _inverse_sigmoid(
        random_draw, udelta, scatter_params.delta_scatter, 0.0, 1.0
    )
    noisy_delta = deltapop._get_bounded_deltapop_param(
        noisy_udelta, deltapop.DELTAPOP_BOUNDS
    )

    ufuno = funopop_ssfr._get_u_p_from_p_scalar(funo, funopop_ssfr.FUNO_BOUNDS)
    noisy_ufuno = _inverse_sigmoid(
        random_draw, ufuno, scatter_params.funo_scatter, 0.0, 1.0
    )
    noisy_funo = funopop_ssfr._get_p_from_u_p_scalar(
        noisy_ufuno, funopop_ssfr.FUNO_BOUNDS
    )

    dust_params = tw_dust.DustParams(noisy_av, noisy_delta, noisy_funo)
    ftrans = tw_dust.calc_dust_frac_trans(wave_aa, dust_params)

    return ftrans


_A = (None, 0, None, None, None, None, None, None)
_B = [None, None, 0, 0, None, None, 0, None]

_f = jjit(vmap(calc_dust_ftrans, in_axes=_A))
calc_dust_ftrans_vmap = jjit(vmap(_f, in_axes=_B))
