import numpy as np
from diffstar.utils import cumulative_mstar_formed_galpop
from dsps.constants import SFR_MIN
from jax import random as jran
import jax.numpy as jnp

from ..disk_bulge_kernels import calc_tform_pop
from ..mc_disk_bulge import (
    _bulge_sfh_vmap,
    generate_fbulge_parameters_2d_sigmoid,
    mc_disk_bulge,
    DEFAULT_FBULGE_2dSIGMOID_PARAMS,
)


def test_mc_disk_bulge_component_functions_work_together():
    ran_key = jran.PRNGKey(0)

    n_t = 200
    tarr = np.linspace(0.01, 13.8, n_t)

    n_gals = 5_000
    ran_key_sfh, ran_key_fbulge = jran.split(ran_key, 2)

    ran_sfh_pop = jran.uniform(ran_key_sfh, minval=0, maxval=100, shape=(n_gals, n_t))

    ran_sfh_pop = np.where(ran_sfh_pop < SFR_MIN, SFR_MIN, ran_sfh_pop)
    smh_pop = cumulative_mstar_formed_galpop(tarr, ran_sfh_pop)
    t10 = calc_tform_pop(tarr, smh_pop, 0.1)
    t90 = calc_tform_pop(tarr, smh_pop, 0.9)
    logsm0 = jnp.log10(smh_pop[:, -1])

    ssfr = jnp.divide(ran_sfh_pop, smh_pop)
    logssfr0 = jnp.log10(ssfr[:, -1])
    fbulge_params = generate_fbulge_parameters_2d_sigmoid(
        ran_key_fbulge, logsm0, logssfr0, t10, t90,
        DEFAULT_FBULGE_2dSIGMOID_PARAMS,
    )

    assert fbulge_params.shape == (n_gals, 3)
    assert np.all(np.isfinite(fbulge_params))

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
    ran_key_sfh, ran_key_fbulge = jran.split(ran_key, 2)

    ran_sfh_pop = jran.uniform(ran_key_sfh, minval=0, maxval=100, shape=(n_gals, n_t))

    _res = mc_disk_bulge(ran_key, tarr, ran_sfh_pop,
                         fbulge_2d_params=DEFAULT_FBULGE_2dSIGMOID_PARAMS,
                         )
    fbulge_params, smh_pop, effbulge, sfh_bulge, smh_bulge, bth = _res
    assert smh_pop.shape == (n_gals, n_t)
    assert effbulge.shape == (n_gals, n_t)
    assert fbulge_params.shape == (n_gals, 3)

    assert np.all(sfh_bulge <= ran_sfh_pop)
    assert np.all(smh_bulge <= smh_pop)

    assert np.all(bth > 0)
    assert np.all(bth < 1)
