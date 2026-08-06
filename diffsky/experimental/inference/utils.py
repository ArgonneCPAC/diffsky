import jax
import jax.numpy as jnp
from collections import namedtuple

from ...param_utils import diffsky_param_wrapper_merging as dpwm


def get_var_param_flat_from_param_flat(param_flat, varied_params_list=None):
    """
    param_flat or uparam_flat (``DiffskyParamsFlat`` or ``DiffskyUParamsFlat``).
    Select parameters from param_flat based on their names listed at varied_params_list.

    Returns a namedtuple ``VariedParams`` with similar structure as ``DiffskyParamsFlat`` or ``DiffskyUParamsFlat``, but containing only the subset of varied parameters.
    """

    if varied_params_list is None:
        # vary all
        varied_params_list = param_flat._fields

    VariedParams = namedtuple("VariedParams", varied_params_list)
    varied_params_dict = {
        var: fix
        for var, fix in param_flat._asdict().items()
        if var in varied_params_list
    }

    return VariedParams(**varied_params_dict)


@jax.jit
def get_uparam_coll_from_var_uparam_flat(var_uparam_flat, uparam_flat):
    """
    Functions to join varied and fixed u parameters into a single variable.
    """
    uparam_flat_updated = uparam_flat._replace(**var_uparam_flat._asdict())
    uparam_coll_updated = dpwm.get_u_param_collection_from_u_param_array(
        uparam_flat_updated
    )
    return uparam_coll_updated


def get_param_coll_from_var_param_flat(var_param_flat, param_flat):
    """
    Functions to join varied and fixed bounded parameters into a single variable.
    """
    param_flat_updated = param_flat._replace(**var_param_flat._asdict())
    param_coll_updated = dpwm.get_param_collection_from_flat_array(param_flat_updated)
    return param_coll_updated


def compute_varied_params_indices(var_uparam_flat, uparam_flat):
    return jnp.array(
        [uparam_flat._fields.index(field) for field in var_uparam_flat._fields]
    )


def bounded_name(name):
    """Unbounded (``u_``-prefixed) -> bounded diffsky parameter name."""
    # TODO: this may not work for all diffsky parameters.
    return name[2:] if name.startswith("u_") else name


def unpack_nested_samples(data):
    """
    Transforms ParamCollection with multiple samples into multiple ParamCollection, each with a single sample.

    This is useful to unpack HMC ParamCollection with multiple samples into multiple ParamCollections.
    """

    # Check if current item is a namedtuple
    if isinstance(data, tuple) and hasattr(data, "_fields"):
        # Pega os campos e os valores processados recursivamente
        fields = data._fields
        # Zip the values of each field
        unpacked_fields = zip(
            *[unpack_nested_samples(getattr(data, f)) for f in fields]
        )

        # Rebuild namedtuple for each sample
        return [type(data)(*v) for v in unpacked_fields]

    # If data is a list or array (the samples per se), return as it is
    return data


def get_flat_params_all_same_shape(params_flat_all_multishape):
    """
    Given a namedtuple with multiple depths, return this namedtuple with the same depth in all leaves.
    """
    sizes = [x.size for x in params_flat_all_multishape]
    sample_size = max(sizes)
    ZZ = jnp.zeros(sample_size)
    seq = [ZZ + x for x in params_flat_all_multishape]
    flat_u_params_all_same_shape = params_flat_all_multishape._make(seq)

    return flat_u_params_all_same_shape
