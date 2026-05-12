""" """

import numpy as np
from diffmah import mah_halopop
from jax import jit as jjit
from jax import numpy as jnp
from jax import random as jran
from jax import vmap

from .. import mc_lightcone_halos as mclh

interp_vmap = jjit(vmap(jnp.interp, in_axes=(0, None, 0)))


def test_sobol_lightcone_diffstar_cens():
    """Enforce mc_lightcone_diffstar_cens returns reasonable results"""
    ran_key = jran.key(0)
    lgmp_min, lgmp_max = 10.0, 16.5
    sky_area_degsq = 1.0
    num_halos = 5_000

    z_min, z_max = 0.01, 3.0
    args = (ran_key, num_halos, z_min, z_max, lgmp_min, lgmp_max, sky_area_degsq)
    cenpop = mclh.sobol_lightcone_diffstar_cens(*args)
    assert np.all(np.isfinite(cenpop["logmp_obs"]))
    for mah_param in cenpop["mah_params"]:
        assert np.all(np.isfinite(mah_param))

    # logmp_obs should agree when recomputed
    t0 = 13.8
    tarr = np.linspace(0.1, t0, 100)
    dmhdt, log_mah = mah_halopop(cenpop["mah_params"], tarr, np.log10(t0))

    logmp_obs = interp_vmap(cenpop["t_obs"], tarr, log_mah)
    assert np.allclose(logmp_obs, cenpop["logmp_obs"], rtol=1e-3)

    # logmp_obs should be within the range and also span it
    EPS = 0.01
    assert np.all(logmp_obs > lgmp_min - EPS), logmp_obs.min()
    assert np.all(logmp_obs < lgmp_max + EPS), logmp_obs.max()

    DELTA = 0.25
    assert np.any(logmp_obs < lgmp_min + DELTA), logmp_obs.max()
    assert np.any(logmp_obs > lgmp_max - DELTA), logmp_obs.max()

    # z_obs should be within the range and also span it
    EPS = 0.01
    assert np.all(cenpop["z_obs"] > z_min - EPS), cenpop["z_obs"].min()
    assert np.all(cenpop["z_obs"] < z_max + EPS), cenpop["z_obs"].max()

    DELTA = 0.25
    assert np.any(cenpop["z_obs"] < z_min + DELTA), cenpop["z_obs"].max()
    assert np.any(cenpop["z_obs"] > z_max - DELTA), cenpop["z_obs"].max()

    assert np.all(np.isfinite(cenpop["logsm_obs"]))
    assert np.all(np.isfinite(cenpop["logssfr_obs"]))
    assert cenpop["logsm_obs"].min() > 2

    assert np.all(np.isfinite(cenpop["diffstarpop_data"]["sfh_params_ms"]))
    assert np.all(np.isfinite(cenpop["diffstarpop_data"]["sfh_params_q"]))

    assert np.all(np.isfinite(cenpop["diffstarpop_data"]["sfh_ms"]))
    assert np.all(np.isfinite(cenpop["diffstarpop_data"]["sfh_q"]))
    assert np.all(cenpop["diffstarpop_data"]["frac_q"] >= 0)
    assert np.all(cenpop["diffstarpop_data"]["frac_q"] <= 1)
