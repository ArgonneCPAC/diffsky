import jax.numpy as jnp

from .upweighting import downsample_and_upweight


def downsample_and_upweight_cat(catalog, *args, **kwargs):
    in_sample, upweights = downsample_and_upweight(
        catalog.logmp_t_obs, catalog.ult_host_indx, *args, **kwargs
    )

    in_sample = in_sample & in_sample[catalog.ult_host_indx]
    downsampled_cat = recursive_namedtuple_cut_and_reindex(catalog, in_sample)
    upweights = upweights[in_sample]

    return downsampled_cat, upweights


def recursive_namedtuple_cut_and_reindex(catalog, selection, indx_str="indx"):
    """Perform selection on namedtuple of catalog columns"""
    catdict = catalog._asdict()
    indmap = jnp.cumsum(selection) - 1
    indmap = jnp.concatenate([indmap, jnp.array([indmap[-1] + 1])])
    failure_index = len(selection)
    for key, val in catdict.items():
        if hasattr(val, "shape") and val.shape:
            if val.shape[0] == selection.shape[0]:
                if indx_str in key:
                    val = jnp.where(selection[val], val, failure_index)
                    val = val[selection]
                    catdict[key] = indmap[val]
                else:
                    catdict[key] = val[selection]

        elif hasattr(val, "_asdict") and hasattr(val, "_replace"):
            val = recursive_namedtuple_cut_and_reindex(val, selection, indx_str)
            catdict[key] = val
        else:
            catdict[key] = val

    return catalog._replace(**catdict)
