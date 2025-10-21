import jax.numpy as jnp
import numpy as np
from diffstar.utils import cumulative_mstar_formed_galpop
from dsps.constants import SFR_MIN
from jax import random as jran

from ..disk_bulge_kernels import _bulge_sfh_vmap, calc_tform_pop
from ..mc_disk_bulge import (
    DEFAULT_FBULGE_2dSIGMOID_PARAMS,
    generate_fbulge_parameters_2d_sigmoid,
    mc_disk_bulge,
)


def test_mc_disk_bulge_component_functions_work_together():
    ran_key = jran.PRNGKey(0)

    n_t = 200
    tarr = np.linspace(0.01, 13.8, n_t)

    n_gals = 5_000
    ran_key, ran_key_sfh = jran.split(ran_key, 2)

    ran_sfh_pop = jran.uniform(ran_key_sfh, minval=0, maxval=100, shape=(n_gals, n_t))

    ran_sfh_pop = np.where(ran_sfh_pop < SFR_MIN, SFR_MIN, ran_sfh_pop)
    smh_pop = cumulative_mstar_formed_galpop(tarr, ran_sfh_pop)
    t10 = calc_tform_pop(tarr, smh_pop, 0.1)
    t90 = calc_tform_pop(tarr, smh_pop, 0.9)
    logsm0 = jnp.log10(smh_pop[:, -1])

    ssfr = jnp.divide(ran_sfh_pop, smh_pop)
    logssfr0 = jnp.log10(ssfr[:, -1])
    fbulge_params = generate_fbulge_parameters_2d_sigmoid(
        logsm0,
        logssfr0,
        t10,
        t90,
        DEFAULT_FBULGE_2dSIGMOID_PARAMS,
    )

    assert fbulge_params.fbulge_tcrit.shape == (n_gals,)
    assert fbulge_params.fbulge_early.shape == (n_gals,)
    assert fbulge_params.fbulge_late.shape == (n_gals,)
    for x in fbulge_params:
        assert np.all(np.isfinite(x))

    _res = _bulge_sfh_vmap(tarr, ran_sfh_pop, fbulge_params)
    for x in _res:
        assert np.all(np.isfinite(x))

    smh, fbulge, sfh_bulge, smh_bulge, bth = _res
    assert smh.shape == (n_gals, n_t)
    assert fbulge.shape == (n_gals, n_t)

    assert np.all(sfh_bulge <= ran_sfh_pop)
    assert np.all(smh_bulge <= smh_pop)

    assert np.all(bth > 0)
    assert np.all(bth < 1)


def test_mc_disk_bulge():
    ran_key = jran.PRNGKey(0)

    n_t = 200
    tarr = np.linspace(0.01, 13.8, n_t)

    n_gals = 5_000
    ran_key, ran_key_sfh = jran.split(ran_key, 2)

    ran_sfh_pop = jran.uniform(ran_key_sfh, minval=0, maxval=100, shape=(n_gals, n_t))

    disk_bulge_history = mc_disk_bulge(
        tarr,
        ran_sfh_pop,
        fbulge_2d_params=DEFAULT_FBULGE_2dSIGMOID_PARAMS,
    )
    assert disk_bulge_history.mstar_history.shape == (n_gals, n_t)
    assert disk_bulge_history.eff_bulge_history.shape == (n_gals, n_t)
    assert disk_bulge_history.fbulge_params.fbulge_tcrit.shape == (n_gals,)
    assert disk_bulge_history.fbulge_params.fbulge_early.shape == (n_gals,)
    assert disk_bulge_history.fbulge_params.fbulge_late.shape == (n_gals,)

    assert np.all(disk_bulge_history.sfh_bulge <= ran_sfh_pop)
    assert np.all(disk_bulge_history.smh_bulge <= disk_bulge_history.mstar_history)

    assert np.all(disk_bulge_history.bulge_to_total_history > 0)
    assert np.all(disk_bulge_history.bulge_to_total_history < 1)
