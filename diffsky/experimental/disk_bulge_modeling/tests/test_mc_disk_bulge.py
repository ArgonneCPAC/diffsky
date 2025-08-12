import numpy as np
from diffstar.fitting_helpers.fitting_kernels import _integrate_sfr
from dsps.constants import SFR_MIN
from dsps.utils import _jax_get_dt_array
from jax import jit as jjit
from jax import random as jran
from jax import vmap

from ..disk_bulge_kernels import FBULGE_MAX, FBULGE_MIN, calc_tform_pop
from ..mc_disk_bulge import _bulge_sfh_vmap, generate_fbulge_params, mc_disk_bulge

_B = (0, None)
_integrate_sfr_vmap = jjit(vmap(_integrate_sfr, in_axes=_B))


def test_mc_disk_bulge_component_functions_work_together():
    ran_key = jran.PRNGKey(0)

    n_t = 200
    tarr = np.linspace(0.01, 13.8, n_t)

    n_gals = 5_000
    ran_key_sfh, ran_key_fbulge = jran.split(ran_key, 2)

    ran_sfh_pop = jran.uniform(ran_key_sfh, minval=0, maxval=100, shape=(n_gals, n_t))

    dtarr = _jax_get_dt_array(tarr)
    ran_sfh_pop = np.where(ran_sfh_pop < SFR_MIN, SFR_MIN, ran_sfh_pop)
    smh_pop = _integrate_sfr_vmap(ran_sfh_pop, dtarr)
    t10 = calc_tform_pop(tarr, smh_pop, 0.1)
    t90 = calc_tform_pop(tarr, smh_pop, 0.9)
    logsm0 = smh_pop[:, -1]

    fbulge_params = generate_fbulge_params(ran_key_fbulge, t10, t90, logsm0)
    assert fbulge_params.shape == (n_gals, 3)
    assert np.all(np.isfinite(fbulge_params))

    _res = _bulge_sfh_vmap(tarr, ran_sfh_pop, fbulge_params)
    for x in _res:
        assert np.all(np.isfinite(x))

    smh, fbulge, sfh_bulge, smh_bulge, bth = _res
    assert smh.shape == (n_gals, n_t)
    assert fbulge.shape == (n_gals, n_t)
    assert np.all(fbulge > FBULGE_MIN)
    assert np.all(fbulge < FBULGE_MAX)

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

    _res = mc_disk_bulge(ran_key, tarr, ran_sfh_pop)
    fbulge_params, smh_pop, fbulge, sfh_bulge, smh_bulge, bth = _res
    assert smh_pop.shape == (n_gals, n_t)
    assert fbulge.shape == (n_gals, n_t)
    assert fbulge_params.shape == (n_gals, 3)

    assert np.all(fbulge > FBULGE_MIN)
    assert np.all(fbulge < FBULGE_MAX)

    assert np.all(sfh_bulge <= ran_sfh_pop)
    assert np.all(smh_bulge <= smh_pop)

    assert np.all(bth > 0)
    assert np.all(bth < 1)
