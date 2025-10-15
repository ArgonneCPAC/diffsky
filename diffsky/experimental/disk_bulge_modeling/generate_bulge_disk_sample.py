import jax.numpy as jnp
import numpy as np
from dsps.cosmology.defaults import DEFAULT_COSMOLOGY
from dsps.cosmology.flat_wcdm import age_at_z
from jax import random as jran

from ...mc_diffsky import mc_diffstar_cenpop, mc_diffstar_galpop
from .mc_disk_bulge import DEFAULT_FBULGE_2dSIGMOID_PARAMS, mc_disk_bulge

ran_key = jran.key(0)
halo_key, ran_key = jran.split(ran_key, 2)

# Invert t_table for redshifts


def get_redshifts_from_times(
    t_table,
    cosmo_params,
    zmin=0.001,
    zmax=50,
    Ngrid=200,
):
    zgrid = np.logspace(np.log10(zmax), np.log10(zmin), Ngrid)
    age_grid = age_at_z(zgrid, *cosmo_params)
    redshifts = np.interp(t_table, age_grid, zgrid)

    return redshifts


# Generate subcat and SFH catalog


def get_bulge_disk_test_sample(
    ran_key,
    lgmp_min=11.0,
    redshift=0.05,
    Lbox=100.0,
    centrals=True,
    cosmology=DEFAULT_COSMOLOGY,
):
    volume_com = Lbox**3
    args = (ran_key, redshift, lgmp_min, volume_com)
    if centrals:
        diffstar = mc_diffstar_cenpop(*args, return_internal_quantities=True)
    else:
        diffstar = mc_diffstar_galpop(*args, return_internal_quantities=True)

    print("Generated data shape = ", diffstar["sfh"].shape)

    diffstar["sSFR"] = jnp.divide(diffstar["sfh"], diffstar["smh"])
    diffstar["z_table"] = get_redshifts_from_times(diffstar["t_table"], cosmology)

    return diffstar


def get_bulge_disk_decomposition(
    diffstar, fbulge_2d_params=DEFAULT_FBULGE_2dSIGMOID_PARAMS
):
    _res = mc_disk_bulge(
        diffstar["t_table"],
        diffstar["sfh"],
        fbulge_2d_params=fbulge_2d_params,
    )
    fbulge_params, smh, eff_bulge, sfh_bulge, smh_bulge, bth = _res

    diffstar["fbulge_tcrit"] = fbulge_params.fbulge_tcrit
    diffstar["fbulge_early"] = fbulge_params.fbulge_early
    diffstar["fbulge_late"] = fbulge_params.fbulge_late
    diffstar["bth"] = bth
    diffstar["eff_bulge"] = eff_bulge

    # Save or compute bulge and disk quantities
    diffstar["sfh_bulge"] = sfh_bulge
    diffstar["smh_bulge"] = smh_bulge
    diffstar["sSFR_bulge"] = jnp.divide(sfh_bulge, smh_bulge)
    diffstar["smh_disk"] = diffstar["smh"] - smh_bulge
    diffstar["sfh_disk"] = diffstar["sfh"] - sfh_bulge
    diffstar["sSFR_disk"] = jnp.divide(diffstar["sfh_disk"], diffstar["smh_disk"])

    return diffstar
