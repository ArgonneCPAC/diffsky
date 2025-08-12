import jax.numpy as jnp
import numpy as np
from diffsky.mc_diffsky import mc_diffstar_cenpop, mc_diffstar_galpop
from dsps.cosmology.defaults import DEFAULT_COSMOLOGY
from dsps.cosmology.flat_wcdm import age_at_z
from jax import random as jran

from .mc_disk_bulge import DEFAULT_FBULGEPARAMS, mc_disk_bulge

ran_key = jran.key(0)
halo_key, ran_key = jran.split(ran_key, 2)

# Invert t_table for redshifts


def get_redshifts_from_times(
    t_table, cosmo_params, zmin=0.001, zmax=50, Ngrid=200, zcheck=3
):
    zgrid = np.logspace(np.log10(zmax), np.log10(zmin), Ngrid)
    age_grid = age_at_z(zgrid, *cosmo_params)
    redshifts = np.interp(t_table, age_grid, zgrid)
    mask = redshifts <= zcheck
    t_interp = age_at_z(redshifts, *cosmo_params)
    check = np.isclose(t_interp, t_table, atol=1e-3, rtol=1e-3)
    print(
        f"Check times calculated from redshifts are within 1e-3 for z< {zcheck}: {np.all(check[mask])}"
    )
    return redshifts


def get_zindexes(zvalues, redshifts):
    zindexes = [int(np.abs(redshifts - z).argmin()) for z in zvalues]
    zs = [float(redshifts[i]) for i in zindexes]
    return zindexes, zs


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
    ran_key, diffstar, FbulgeFixedParams=DEFAULT_FBULGEPARAMS, new_model=True
):
    _res = mc_disk_bulge(
        ran_key,
        diffstar["t_table"],
        diffstar["sfh"],
        FbulgeFixedParams=FbulgeFixedParams,
        new_model=new_model,
    )
    fbulge_params, smh, eff_bulge, sfh_bulge, smh_bulge, bth = _res

    diffstar["tcrit_bulge"] = fbulge_params[:, 0]
    diffstar["fbulge_early"] = fbulge_params[:, 1]
    diffstar["fbulge_late"] = fbulge_params[:, 2]
    diffstar["bth"] = bth
    diffstar["eff_bulge"] = eff_bulge

    # Save or compute bulge and disk quantities
    diffstar["sfh_bulge"] = sfh_bulge
    diffstar["smh_bulge"] = smh_bulge
    diffstar["sSFR_bulge"] = jnp.divide(sfh_bulge, smh_bulge)
    diffstar["smh_disk"] = diffstar["smh"] - smh_bulge
    diffstar["sfh_disk"] = diffstar["sfh"] - sfh_bulge
    diffstar["sSFR_disk"] = jnp.divide(diffstar["sfh_disk"], diffstar["smh_disk"])

    # Check that returned smh agrees with value in diffstar
    msg = "Returned smh does not match values in test sample"
    assert jnp.isclose(diffstar["smh"] / smh, smh / smh).all(), msg
    bmask = smh_bulge > diffstar["smh"]
    assert np.count_nonzero(bmask) == 0, "Some bulge masses exceed total stellar masses"

    return diffstar
