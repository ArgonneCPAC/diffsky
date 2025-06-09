# flake8: noqa: E402
"""Functions related to calculated the merging
probability and applying the merging model.
"""

from jax import config

config.update("jax_enable_x64", True)

from collections import OrderedDict, namedtuple

from dsps.utils import _inverse_sigmoid, _sigmoid
from jax import jit as jjit
from jax import numpy as jnp
from jax import random, vmap

DEFAULT_MERGE_PDICT = OrderedDict(
    t_delay_min=12.018,  # 2.563,
    t_delay_max=2.884,  # 6.411,
    k_t_delay=16.126,  # 15.686,
    host_crit=12.286,  # 13.268,
    t_crit_min=12.341,  # 12.355,
    t_crit_max=1.361,  # 1.592,
    k_t=12.376,  # 12.707,
    sub_crit=11.768,  # 10.976,
    k_min=18.150,  # 18.181,
    k_max=0.170,  # 0.428,
    k_k=1.496,  # 0.749,
    p_max=0.999,
)

T_BOUNDS = (0.0, 14.0)
K_BOUNDS = (0.0, 20.0)
LGMP_BOUNDS = (6.0, 16.0)
P_BOUNDS = (0.0, 1.0)

MERGE_PBOUNDS_PDICT = OrderedDict(
    t_delay_min=T_BOUNDS,
    t_delay_max=T_BOUNDS,
    k_t_delay=K_BOUNDS,
    host_crit=LGMP_BOUNDS,
    t_crit_min=T_BOUNDS,
    t_crit_max=T_BOUNDS,
    k_t=K_BOUNDS,
    sub_crit=LGMP_BOUNDS,
    k_min=K_BOUNDS,
    k_max=K_BOUNDS,
    k_k=K_BOUNDS,
    p_max=P_BOUNDS,
)

MergeParams = namedtuple("MergeParams", DEFAULT_MERGE_PDICT.keys())

_MERGE_UPNAMES = ["u_" + key for key in MERGE_PBOUNDS_PDICT.keys()]
MergeUParams = namedtuple("MergeUParams", _MERGE_UPNAMES)

DEFAULT_MERGE_PARAMS = MergeParams(**DEFAULT_MERGE_PDICT)
MERGE_PBOUNDS = MergeParams(**MERGE_PBOUNDS_PDICT)


@jjit
def _get_bounded_merge_param(u_param, bound):
    lo, hi = bound
    mid = 0.5 * (lo + hi)
    return _sigmoid(u_param, mid, 0.1, lo, hi)


@jjit
def _get_unbounded_merge_param(param, bound):
    lo, hi = bound
    mid = 0.5 * (lo + hi)
    return _inverse_sigmoid(param, mid, 0.1, lo, hi)


_C = (0, 0)
_get_merge_params_kern = jjit(vmap(_get_bounded_merge_param, in_axes=_C))
_get_merge_u_params_kern = jjit(vmap(_get_unbounded_merge_param, in_axes=_C))


@jjit
def get_bounded_merge_params(u_params):
    u_params = jnp.array([getattr(u_params, u_pname) for u_pname in _MERGE_UPNAMES])
    merge_params = _get_merge_params_kern(jnp.array(u_params), jnp.array(MERGE_PBOUNDS))
    return MergeParams(*merge_params)


@jjit
def get_unbounded_merge_params(params):
    params = jnp.array(
        [getattr(params, pname) for pname in DEFAULT_MERGE_PARAMS._fields]
    )
    u_params = _get_merge_u_params_kern(jnp.array(params), jnp.array(MERGE_PBOUNDS))
    return MergeUParams(*u_params)


DEFAULT_MERGE_U_PARAMS = MergeUParams(*get_unbounded_merge_params(DEFAULT_MERGE_PARAMS))


@jjit
def p_infall(t_interest, k_infall, t_infall, t_delay, p_max):
    """Return merging probability as a function of infall time
       Merging probability is the softmax function (between 0 and 1)
       of a sigmoid (between -1 and p_max), so that merging probability
       is bound between 0 and 1, and is 0 at t_start.

    Parameters
    ----------
    t_interest : float
        cosmic time of interest (in Gyr)

    k_infall : float
        slope of merging probability sigmoid

    t_infall : float
        infall time (in Gyr)

    t_delay : float
        delay time (in Gyr)

    p_max : float
        maximum merging probability

    Returns
    -------

    ss : float
        merging probability

    """
    t_start = t_infall + t_delay
    s = _sigmoid(t_interest, t_start, k_infall, -0.999999, p_max)
    ss = jnp.sqrt(s * s) * _sigmoid(s, 0.05, 20.0, 0.0, 1.0)
    return ss


@jjit
def get_p_merge_from_merging_params(
    merging_params, log_mpeak_infall, log_mhost_infall, t_interest, t_infall, upids
):
    """Return the merging probability of a galaxy

    Parameters
    ----------
    merging_u_params : named tuple
        unbound merging parameters

    log_mpeak_infall : float
        subhalo peak mass at infall time

    log_mhost_infall : float
        host halo peak mass at subhalo's infall time

    t_interest : float
        cosmic time of interest (in Gyr)

    t_infall : float
        subhalo infall time (in Gyr)

    upids : int
        uber parent id of galaxy

    Returns
    -------

    ss : float
        merging probability

    """
    # Smaller subs have a longer period of rapid merging
    # Larger subs only experience rapid merging at very early times
    t_crit = _sigmoid(
        log_mpeak_infall,
        merging_params.sub_crit,
        merging_params.k_t,
        merging_params.t_crit_min,
        merging_params.t_crit_max,
    )

    # There is a period of rapid merging where things reach p_max more quickly
    k_infall = _sigmoid(
        t_infall, t_crit, merging_params.k_k, merging_params.k_min, merging_params.k_max
    )

    # Galaxies in larger hosts experience a longer delay time
    # between infall and merging
    t_delay = _sigmoid(
        log_mhost_infall,
        merging_params.host_crit,
        merging_params.k_t_delay,
        merging_params.t_delay_min,
        merging_params.t_delay_max,
    )

    # Merging probability is a function of t_infall,
    # log_mpeak_infall, and log_mhost_infall
    P = p_infall(t_interest, k_infall, t_infall, t_delay, merging_params.p_max)

    # Central galaxies never disrupt, but satellites might!
    p = jnp.where(upids == -1, 0.0, 1.0)

    return P * p


@jjit
def get_p_merge_from_merging_u_params(
    merging_u_params, log_mpeak_infall, log_mhost_infall, t_interest, t_infall, upids
):
    merging_params = get_bounded_merge_params(merging_u_params)
    p_merge = get_p_merge_from_merging_params(
        merging_params, log_mpeak_infall, log_mhost_infall, t_interest, t_infall, upids
    )

    return p_merge


@jjit
def merge(
    merging_u_params,
    log_mpeak_infall,
    log_mhost_infall,
    t_interest,
    t_infall,
    upids,
    sfr,
    indx_to_deposit,
    do_merging,
    MC,
):
    """Apply merging model to galaxies

    Parameters
    ----------
    merging_u_params : named_tuple
        unbound merging parameters

    log_mpeak_infall : float
        subhalo peak mass at infall time

    log_mhost_infall : float
        host halo peak mass at subhalo's infall time

    t_interest : float
        cosmic time of interest (in Gyr)

    t_infall : float
        subhalo infall time (in Gyr)

    upids : int
        uber parent id of galaxy

    sfr : array (len > 1)
        quantity to merge (star formation history, flux, etc)

    indx_to_deposit : int
        index of central galaxy on which to deposit stellar mass

    do_merging : int
        0 = ignore (e.g. this object only does one round of merging)
        1 = calculate merging probability

    MC : int
        0 = probabilistic
        1 = monte carlo

    Returns
    -------

    total_sfr : array (len > 1)
        merged quantity

    """
    merge_prob = do_merging * get_p_merge_from_merging_u_params(
        merging_u_params,
        log_mpeak_infall,
        log_mhost_infall,
        t_interest,
        t_infall,
        upids,
    )

    N = len(merge_prob)
    key = random.PRNGKey(0)
    prn = random.uniform(key, shape=(N,))
    mc_merge = jnp.where(merge_prob <= prn, 0, 1)

    merge_prob = jnp.where(MC < 1, merge_prob, mc_merge)

    ngals = sfr.shape[0]
    indx_to_keep = jnp.arange(ngals).astype("i8")

    sfr_to_deposit = sfr * merge_prob[:, jnp.newaxis]
    sfr_to_keep = sfr * (1 - merge_prob)[:, jnp.newaxis]
    total_sfr = jnp.zeros_like(sfr)
    total_sfr = total_sfr.at[indx_to_deposit].add(sfr_to_deposit)
    total_sfr = total_sfr.at[indx_to_keep].add(sfr_to_keep)
    return total_sfr


@jjit
def merge_with_MC_draws(
    merging_u_params,
    log_mpeak_infall,
    log_mhost_infall,
    t_interest,
    t_infall,
    upids,
    sfr,
    indx_to_deposit,
    do_merging,
    MC,
    prn,
):
    """Apply merging model to galaxies (with predetermined array
    of random numbers for monte carlo draws)

    Parameters
    ----------
    merging_u_params : named_tuple
        unbound merging parameters

    log_mpeak_infall : float
        subhalo peak mass at infall time

    log_mhost_infall : float
        host halo peak mass at subhalo's infall time

    t_interest : float
        cosmic time of interest (in Gyr)

    t_infall : float
        subhalo infall time (in Gyr)

    upids : int
        uber parent id of galaxy

    sfr : array (len > 1)
        quantity to merge (star formation history, flux, etc)

    indx_to_deposit : int
        index of central galaxy on which to deposit stellar mass

    do_merging : int
        0 = ignore (e.g. this object only does one round of merging)
        1 = calculate merging probability

    MC : int
        0 = probabilistic
        1 = monte carlo

    prn : float
        pseudo random number for monte carlo draws

    Returns
    -------

    total_sfr : array (len > 1)
        merged quantity

    """
    merge_prob = do_merging * get_p_merge_from_merging_u_params(
        merging_u_params,
        log_mpeak_infall,
        log_mhost_infall,
        t_interest,
        t_infall,
        upids,
    )

    mc_merge = jnp.where(merge_prob <= prn, 0.0, 1.0)
    merge_prob = jnp.where(MC < 1, merge_prob, mc_merge)

    ngals = sfr.shape[0]
    indx_to_keep = jnp.arange(ngals).astype("i8")

    sfr_to_deposit = sfr * merge_prob[:, jnp.newaxis]
    sfr_to_keep = sfr * (1 - merge_prob)[:, jnp.newaxis]
    total_sfr = jnp.zeros_like(sfr)
    total_sfr = total_sfr.at[indx_to_deposit].add(sfr_to_deposit)
    total_sfr = total_sfr.at[indx_to_keep].add(sfr_to_keep)
    return total_sfr


@jjit
def merge_model_with_preprocessing(
    merging_u_params,
    log_mpeak_penultimate_infall,
    log_mpeak_ultimate_infall,
    log_mhost_penultimate_infall,
    log_mhost_ultimate_infall,
    t_interest,
    t_penultimate_infall,
    t_ultimate_infall,
    upids,
    sfr,
    penultimate_dump,
    ultimate_dump,
    MC,
):
    """Apply two rounds of merging to galaxies
    (including satellite preprocessing)

    Parameters
    ----------
    merging_u_params : named tuple
        unbound merging parameters

    log_mpeak_penultimate_infall : float
        subhalo peak mass at penultimate infall time

    log_mpeak_ultimate_infall : float
        subhalo peak mass at ultimate infall time

    log_mhost_penultimate_infall : float
        penultimate host halo peak mass at subhalo's infall time

    log_mhost_ultimate_infall : float
        ultimate host halo peak mass at subhalo's infall time

    t_interest : float
        cosmic time of interest (in Gyr)

    t_penultimate_infall : float
        subhalo's penultimate infall time (in Gyr)

    t_ultimate_infall : float
        subhalo's ultimate infall time (in Gyr)

    upids : int
        uber parent id of galaxy

    sfr : array (len > 1)
        quantity to merge (star formation history, flux, etc)

    penultimate_dump : int
        index of penultimate central galaxy on which to deposit stellar mass

    ultimate_dump : int
        index of ultimate central galaxy on which to deposit stellar mass

    MC : int
        0 = probabilistic
        1 = monte carlo

    Returns
    -------

    total_sfr_2 : array (len > 1)
        merged quantity

    """
    p1 = jnp.ones(len(t_penultimate_infall))

    total_sfr_1 = merge(
        merging_u_params,
        log_mpeak_penultimate_infall,
        log_mhost_penultimate_infall,
        t_interest,
        t_penultimate_infall,
        upids,
        sfr,
        penultimate_dump,
        p1,
        MC,
    )

    p2 = jnp.where(t_penultimate_infall == t_ultimate_infall, 0.0, 1.0)

    total_sfr_2 = merge(
        merging_u_params,
        log_mpeak_ultimate_infall,
        log_mhost_ultimate_infall,
        t_interest,
        t_ultimate_infall,
        upids,
        total_sfr_1,
        ultimate_dump,
        p2,
        MC,
    )

    return total_sfr_2


@jjit
def merge_model_with_preprocessing_mc_draws(
    merging_u_params,
    log_mpeak_penultimate_infall,
    log_mpeak_ultimate_infall,
    log_mhost_penultimate_infall,
    log_mhost_ultimate_infall,
    t_interest,
    t_penultimate_infall,
    t_ultimate_infall,
    upids,
    sfr,
    penultimate_dump,
    ultimate_dump,
    MC,
    mc_draws,
):
    """Apply two rounds of merging to galaxies
    (including satellite preprocessing)
    (with predetermined array of random numbers for monte carlo draws)

    Parameters
    ----------
    merging_u_params : named tuple
        unbound merging parameters

    log_mpeak_penultimate_infall : float
        subhalo peak mass at penultimate infall time

    log_mpeak_ultimate_infall : float
        subhalo peak mass at ultimate infall time

    log_mhost_penultimate_infall : float
        penultimate host halo peak mass at subhalo's infall time

    log_mhost_ultimate_infall : float
        ultimate host halo peak mass at subhalo's infall time

    t_interest : float
        cosmic time of interest (in Gyr)

    t_penultimate_infall : float
        subhalo's penultimate infall time (in Gyr)

    t_ultimate_infall : float
        subhalo's ultimate infall time (in Gyr)

    upids : int
        uber parent id of galaxy

    sfr : array (len > 1)
        quantity to merge (star formation history, flux, etc)

    penultimate_dump : int
        index of penultimate central galaxy on which to deposit stellar mass

    ultimate_dump : int
        index of ultimate central galaxy on which to deposit stellar mass

    MC : int
        0 = probabilistic
        1 = monte carlo

    mc_draws : float
        pseudo random number for monte carlo draws


    Returns
    -------

    total_sfr_2 : array (len > 1)
        merged quantity

    """
    p1 = jnp.ones(len(t_penultimate_infall))

    total_sfr_1 = merge_with_MC_draws(
        merging_u_params,
        log_mpeak_penultimate_infall,
        log_mhost_penultimate_infall,
        t_interest,
        t_penultimate_infall,
        upids,
        sfr,
        penultimate_dump,
        p1,
        MC,
        mc_draws,
    )

    p2 = jnp.where(t_penultimate_infall == t_ultimate_infall, 0.0, 1.0)

    total_sfr_2 = merge_with_MC_draws(
        merging_u_params,
        log_mpeak_ultimate_infall,
        log_mhost_ultimate_infall,
        t_interest,
        t_ultimate_infall,
        upids,
        total_sfr_1,
        ultimate_dump,
        p2,
        MC,
        mc_draws,
    )

    return total_sfr_2
