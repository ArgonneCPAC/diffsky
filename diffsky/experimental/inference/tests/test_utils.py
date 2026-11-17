""" """

from collections import namedtuple

import jax.numpy as jnp
import numpy as np
from jax import random as jran
from jax.flatten_util import ravel_pytree

from ....param_utils import diffsky_param_wrapper_merging as dpwm
from .. import utils

MockParams = namedtuple("MockParams", ("a", "b"))


def _get_default_uparam_flat():
    uparam_flat = dpwm.DiffskyUParamsFlat(*jnp.arange(len(dpwm.U_PNAMES_FLAT)))
    return uparam_flat


def _get_two_varied_param_names():
    names = [n for n in dpwm.U_PNAMES_FLAT if n.startswith("u_frac_quench_cen_")]
    return names[:2]


def test_get_var_param_flat_from_param_flat_returns_requested_fields():
    """Enforce that get_var_param_flat_from_param_flat returns a namedtuple
    containing only the requested subset of parameters"""
    uparam_flat = _get_default_uparam_flat()
    varied_params_list = _get_two_varied_param_names()

    var_param_flat = utils.get_var_param_flat_from_param_flat(
        uparam_flat, varied_params_list
    )

    assert var_param_flat._fields == tuple(varied_params_list)
    expected = [dpwm.U_PNAMES_FLAT.index(name) for name in varied_params_list]
    assert np.allclose(var_param_flat, expected)


def test_get_var_param_flat_from_param_flat_fails_when_passing_bogus_name():
    """Enforce that get_var_param_flat_from_param_flat does not accept
    parameter names that are not fields of param_flat"""
    uparam_flat = _get_default_uparam_flat()
    varied_params_list = _get_two_varied_param_names() + ["u_bogus_param_name"]
    try:
        utils.get_var_param_flat_from_param_flat(uparam_flat, varied_params_list)
        raise NameError("get_var_param_flat_from_param_flat should not accept bogus")
    except TypeError:
        pass


def test_get_uparam_coll_from_var_uparam_flat_only_changes_varied_params():
    """Enforce that get_uparam_coll_from_var_uparam_flat updates the varied
    parameters while leaving the fixed parameters unchanged"""
    uparam_flat = _get_default_uparam_flat()
    varied_params_list = _get_two_varied_param_names()
    var_uparam_flat = utils.get_var_param_flat_from_param_flat(
        uparam_flat, varied_params_list
    )
    var_uparam_flat = var_uparam_flat._make([jnp.array(7.0), jnp.array(-3.0)])

    uparam_coll = utils.get_uparam_coll_from_var_uparam_flat(
        var_uparam_flat, uparam_flat
    )
    uparam_flat_updated, _ = ravel_pytree(uparam_coll)
    uparam_flat_orig, _ = ravel_pytree(uparam_flat)

    var_flat_idx = utils.compute_varied_params_indices(var_uparam_flat, uparam_flat)
    n_params = len(dpwm.U_PNAMES_FLAT)
    fixed_idx = np.setdiff1d(np.arange(n_params), np.array(var_flat_idx))
    assert np.allclose(uparam_flat_updated[fixed_idx], uparam_flat_orig[fixed_idx])
    assert np.allclose(uparam_flat_updated[var_flat_idx], [7.0, -3.0])


def test_compute_varied_params_indices_are_correct():
    uparam_flat = _get_default_uparam_flat()
    varied_params_list = _get_two_varied_param_names()
    var_uparam_flat = utils.get_var_param_flat_from_param_flat(
        uparam_flat, varied_params_list
    )

    indices = utils.compute_varied_params_indices(var_uparam_flat, uparam_flat)

    expected = [uparam_flat._fields.index(name) for name in varied_params_list]
    assert np.allclose(indices, expected)


def test_bounded_name_strips_u_prefix():
    assert utils.bounded_name("u_fstar_tdelay") == "fstar_tdelay"
    assert utils.bounded_name("frac_quench_cen_k") == "frac_quench_cen_k"


def test_unpack_nested_samples_returns_n_single_sample_collections():
    """Enforce that unpack_nested_samples splits a ParamCollection with a
    leading sample axis into n single-sample ParamCollections"""
    ran_key = jran.key(0)
    n_samples = 4
    n_params = len(dpwm.PNAMES_FLAT)
    params_batch = jran.normal(ran_key, (n_samples, n_params))

    batched_param_flat = dpwm.DiffskyParamsFlat(*params_batch.T)
    param_coll_batch = dpwm.get_param_collection_from_flat_array(batched_param_flat)

    param_coll_list = utils.unpack_nested_samples(param_coll_batch)

    assert len(param_coll_list) == n_samples
    for i, param_coll in enumerate(param_coll_list):
        param_flat, _ = ravel_pytree(param_coll)
        assert param_flat.shape == (n_params,)
        assert np.allclose(param_flat, params_batch[i])


def test_get_flat_params_all_same_shape_broadcasts_leaves():
    params_flat = MockParams(a=jnp.zeros(5) + 3.0, b=jnp.array(7.0))

    params_flat_same_shape = utils.get_flat_params_all_same_shape(params_flat)

    assert params_flat_same_shape.a.shape == (5,)
    assert params_flat_same_shape.b.shape == (5,)
    assert np.allclose(params_flat_same_shape.a, 3.0)
    assert np.allclose(params_flat_same_shape.b, 7.0)
