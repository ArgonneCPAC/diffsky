import jax.numpy as jnp
import numpy as np
from jax import random as jran
from dsps.cosmology.defaults import DEFAULT_COSMOLOGY
from dsps.cosmology.flat_wcdm import age_at_z

from ..generate_bulge_disk_sample import (
    get_bulge_disk_decomposition,
    get_bulge_disk_test_sample,
    get_redshifts_from_times,
)
from ..mc_disk_bulge import mc_disk_bulge


def test_bulge_disk_test_sample():
    ran_key = jran.key(0)
    halo_key, ran_key = jran.split(ran_key, 2)
    lgmp_min = 11.0
    redshift = 0.05
    Lbox = 50.0
    diffstar_cens = get_bulge_disk_test_sample(
        ran_key, lgmp_min=lgmp_min, redshift=redshift, Lbox=Lbox
    )
    keys = list(diffstar_cens.keys())
    assert "t_table" in keys, "t_table not returned"
    n_t = len(diffstar_cens["t_table"])
    assert "sfh" in keys, "sfh not returned"
    n_gals = len(diffstar_cens["sfh"])
    assert "smh" in keys, "smh not returned"
    assert diffstar_cens["smh"].shape == (n_gals, n_t)


def test_get_redshifts_from_times():
    nt = 100
    t_table = np.linspace(0.1, 13.8, nt)

    redshifts = get_redshifts_from_times(t_table, DEFAULT_COSMOLOGY)

    zcheck = 3
    mask = redshifts <= zcheck
    t_interp = age_at_z(redshifts, *DEFAULT_COSMOLOGY)
    check = np.isclose(t_interp, t_table, atol=1e-3, rtol=1e-3)
    msg = "times calculated from redshifts not within 1e-3 for z<{}".format(zcheck)
    assert np.all(check[mask]), msg


def test_get_bulge_disk_decomposition():
    ran_key = jran.key(0)
    halo_key, ran_key = jran.split(ran_key, 2)
    lgmp_min = 11.0
    redshift = 0.05
    Lbox = 50.0
    diffstar_cens = get_bulge_disk_test_sample(
        ran_key, lgmp_min=lgmp_min, redshift=redshift, Lbox=Lbox
    )

    disk_bulge_key, ran_key = jran.split(ran_key, 2)
    _res = mc_disk_bulge(ran_key, diffstar_cens["t_table"], diffstar_cens["sfh"])
    fbulge_params, smh, eff_bulge, sfh_bulge, smh_bulge, bth = _res

    diffstar_cens = get_bulge_disk_decomposition(ran_key, diffstar_cens)

    # Check that returned smh agrees with value in diffstar
    msg = "Returned smh does not match values in test sample"
    assert jnp.isclose(diffstar_cens["smh"] / smh, smh / smh).all(), msg
    bmask = smh_bulge > diffstar_cens["smh"]
    assert np.count_nonzero(bmask) == 0, "Some bulge masses exceed total stellar masses"
