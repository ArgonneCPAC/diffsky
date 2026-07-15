"""
Wrappers around diffhalos.lightcone_generators that computes data
to generate lightcones with star formation histories.

The functions get as input:
- halo lightcone settings: redshift range, mass range, etc.
"""

from collections import namedtuple

from diffhalos.lightcone_generators import mc_lightcone as mcl
from diffhalos.lightcone_generators import mc_lightcone_halos as mclh
from diffmah import logmh_at_t_obs
from dsps.constants import T_TABLE_MIN
from dsps.cosmology import flat_wcdm
from jax import numpy as jnp

N_SFH_TABLE = 100

_lc_data_sfh_centrals_keys = (
    "cen_weight",
    "z_obs",
    "t_obs",
    "logmp_obs",
    "mah_params",
    "logmp0",
    "t_table",
    "lgt0",
)
LCDataSFHCentrals = namedtuple("LCDataSFHCentrals", _lc_data_sfh_centrals_keys)

lc_data_sfh_keys = (
    *_lc_data_sfh_centrals_keys,
    "sat_weight",
    "t_infall",
    "logmp_infall",
    "logmhost_infall",
    "is_central",
    "halo_indx",
    "halo_weight",
)
LCDataSFH = namedtuple("LCDataSFH", lc_data_sfh_keys)


def mc_lc_data_sfh(
    ran_key,
    z_min,
    z_max,
    lgmp_min,
    lgmsub_min,
    sky_area_degsq,
    *,
    cosmo_params=flat_wcdm.PLANCK15,
    logmp_cutoff=11.0,
):
    """
    Generate a monte carlo lightcone of host halos and subhalos.

    Parameters
    ----------
    ran_key : jax.random.key()
    z_min, z_max : floats
    lgmp_min : float
    lgmsub_min : float
    sky_area_degsq : float

    Returns
    -------

    """
    halopop = mcl.mc_lc(
        ran_key=ran_key,
        lgmp_min=lgmp_min,
        lgmsub_min=lgmsub_min,
        z_min=z_min,
        z_max=z_max,
        sky_area_degsq=sky_area_degsq,
        cosmo_params=cosmo_params,
        logmp_cutoff=logmp_cutoff,
    )

    logt0 = halopop.logt0
    t0 = 10**logt0
    t_table = jnp.linspace(T_TABLE_MIN, t0, N_SFH_TABLE)

    is_central = halopop.central.astype(int)
    n_tot = len(is_central)

    t_infall = jnp.where(is_central, t0 + jnp.zeros(n_tot), halopop.mah_params.t_peak)

    logmp_infall = halopop.logmp_obs

    mah_params_host = halopop.mah_params._make(
        [x[halopop.halo_indx] for x in halopop.mah_params]
    )
    logmhost_infall = logmh_at_t_obs(mah_params_host, halopop.t_obs, logt0)

    lc_data = LCDataSFH(
        cen_weight=halopop.cen_weight,
        z_obs=halopop.z_obs,
        t_obs=halopop.t_obs,
        logmp_obs=halopop.logmp_obs,
        mah_params=halopop.mah_params,
        logmp0=halopop.logmp0,
        #
        t_table=t_table,
        lgt0=logt0,
        #
        sat_weight=halopop.sat_weight,
        t_infall=t_infall,
        logmp_infall=logmp_infall,
        logmhost_infall=logmhost_infall,
        is_central=is_central,
        halo_indx=halopop.halo_indx,
        halo_weight=halopop.halo_weight,
    )

    return lc_data


def weighted_lc_data_sfh(
    ran_key,
    n_host_halos,
    z_min,
    z_max,
    lgmp_min,
    lgmp_max,
    sky_area_degsq,
    *,
    cosmo_params=flat_wcdm.PLANCK15,
    logmp_cutoff=11.0,
):
    """Weighted lightcone of halos with diffstar SFH

    Parameters
    ----------
    ran_key : jax.random.key()
    n_host_halos : int
    z_min, z_max : floats
    lgmp_min, lgmp_max : floats
    sky_area_degsq : float

    Returns
    -------
    sfh_lightcone : namedtuple

    """
    halopop = mcl.weighted_lc(
        ran_key=ran_key,
        n_host_halos=n_host_halos,
        z_min=z_min,
        z_max=z_max,
        lgmp_min=lgmp_min,
        lgmp_max=lgmp_max,
        sky_area_degsq=sky_area_degsq,
        cosmo_params=cosmo_params,
        logmp_cutoff=logmp_cutoff,
    )

    logt0 = halopop.logt0
    t0 = 10**logt0
    t_table = jnp.linspace(T_TABLE_MIN, t0, N_SFH_TABLE)

    is_central = halopop.central.astype(int)
    n_tot = len(is_central)

    t_infall = jnp.where(is_central, t0 + jnp.zeros(n_tot), halopop.mah_params.t_peak)

    logmp_infall = halopop.logmp_obs

    mah_params_host = halopop.mah_params._make(
        [x[halopop.halo_indx] for x in halopop.mah_params]
    )
    logmhost_infall = logmh_at_t_obs(mah_params_host, halopop.t_obs, logt0)

    lc_data = LCDataSFH(
        cen_weight=halopop.cen_weight,
        z_obs=halopop.z_obs,
        t_obs=halopop.t_obs,
        logmp_obs=halopop.logmp_obs,
        mah_params=halopop.mah_params,
        logmp0=halopop.logmp0,
        #
        t_table=t_table,
        lgt0=logt0,
        #
        sat_weight=halopop.sat_weight,
        t_infall=t_infall,
        logmp_infall=logmp_infall,
        logmhost_infall=logmhost_infall,
        is_central=is_central,
        halo_indx=halopop.halo_indx,
        halo_weight=halopop.halo_weight,
    )
    return lc_data


# --- Host halo lightcones functions ---


def mc_lc_data_sfh_centrals(
    ran_key,
    z_min,
    z_max,
    lgmp_min,
    sky_area_degsq,
    *,
    cosmo_params=flat_wcdm.PLANCK15,
    logmp_cutoff=11.0,
):
    """Weighted lightcone of host halos with diffstar SFH

    Parameters
    ----------
    ran_key : jax.random.key()
    z_min, z_max : floats
    lgmp_min : float
    sky_area_degsq : float

    Returns
    -------

    """
    cenpop = mclh.mc_lc_halos(
        ran_key=ran_key,
        lgmp_min=lgmp_min,
        z_min=z_min,
        z_max=z_max,
        sky_area_degsq=sky_area_degsq,
        cosmo_params=cosmo_params,
        logmp_cutoff=logmp_cutoff,
    )

    logt0 = cenpop.logt0
    t0 = 10**logt0
    t_table = jnp.linspace(T_TABLE_MIN, t0, N_SFH_TABLE)

    lc_data = LCDataSFHCentrals(
        cen_weight=cenpop.cen_weight,
        z_obs=cenpop.z_obs,
        t_obs=cenpop.t_obs,
        logmp_obs=cenpop.logmp_obs,
        mah_params=cenpop.mah_params,
        logmp0=cenpop.logmp0,
        #
        t_table=t_table,
        lgt0=logt0,
    )

    return lc_data


def weighted_lc_data_sfh_centrals(
    ran_key,
    num_halos,
    z_min,
    z_max,
    lgmp_min,
    lgmp_max,
    sky_area_degsq,
    *,
    cosmo_params=flat_wcdm.PLANCK15,
    logmp_cutoff=11.0,
):
    """Weighted lightcone of host halos with diffstar SFH

    Parameters
    ----------
    ran_key : jax.random.key()
    n_host_halos : int
    z_min, z_max : floats
    lgmp_min, lgmp_max : floats
    sky_area_degsq : float

    Returns
    -------

    """
    cenpop = mclh.weighted_lc_halos(
        ran_key=ran_key,
        n_halos=num_halos,
        z_min=z_min,
        z_max=z_max,
        lgmp_min=lgmp_min,
        lgmp_max=lgmp_max,
        sky_area_degsq=sky_area_degsq,
        cosmo_params=cosmo_params,
        logmp_cutoff=logmp_cutoff,
    )

    logt0 = cenpop.logt0
    t0 = 10**logt0
    t_table = jnp.linspace(T_TABLE_MIN, t0, N_SFH_TABLE)

    lc_data = LCDataSFHCentrals(
        cen_weight=cenpop.cen_weight,
        z_obs=cenpop.z_obs,
        t_obs=cenpop.t_obs,
        logmp_obs=cenpop.logmp_obs,
        mah_params=cenpop.mah_params,
        logmp0=cenpop.logmp0,
        #
        t_table=t_table,
        lgt0=logt0,
    )

    return lc_data
