"""
"""

import numpy as np

from ..merging_model import (
    DEFAULT_MERGE_PDICT,
    MERGE_PBOUNDS_PDICT,
    DEFAULT_MERGE_U_PARAMS,
    DEFAULT_MERGE_PARAMS,
    get_bounded_merge_params,
    get_unbounded_merge_params,
    p_infall,
    get_p_merge_from_merging_params,
    get_p_merge_from_merging_u_params,
    merge,
    merge_model_with_preprocessing,
    merge_model_with_preprocessing_mc_draws,
    merge_with_MC_draws
)


TOL = 1e-2


def test_default_params_are_in_bounds():
    for key in DEFAULT_MERGE_PDICT.keys():
        bounds = MERGE_PBOUNDS_PDICT[key]
        val = DEFAULT_MERGE_PDICT[key]
        assert bounds[0] < val < bounds[1], key


def test_param_u_param_names_propagate_properly():
    gen = zip(DEFAULT_MERGE_U_PARAMS._fields, DEFAULT_MERGE_PARAMS._fields)
    for u_key, key in gen:
        assert u_key[:2] == "u_"
        assert u_key[2:] == key

    inferred_default_params = get_bounded_merge_params(DEFAULT_MERGE_U_PARAMS)
    assert set(inferred_default_params._fields) == set(DEFAULT_MERGE_PARAMS._fields)

    inferred_default_u_params = get_unbounded_merge_params(DEFAULT_MERGE_PARAMS)
    assert set(inferred_default_u_params._fields) == set(
        DEFAULT_MERGE_U_PARAMS._fields
    )


def test_get_bounded_merging_params_fails_when_passing_params():
    try:
        get_bounded_merge_params(DEFAULT_MERGE_PARAMS)
        raise NameError("get_bounded_merge_params should not accept u_params")
    except AttributeError:
        pass


def test_get_unbounded_merge_params_fails_when_passing_u_params():
    try:
        get_unbounded_merge_params(DEFAULT_MERGE_U_PARAMS)
        raise NameError("get_unbounded_merge_params should not accept u_params")
    except AttributeError:
        pass


def test_get_p_merge_from_merging_u_params_fails_when_passing_params():

    log_mpeak_infall = np.array([9.0, 10.0, 11.0, 12.0])
    log_mhost_infall = np.array([12.0, 13.0, 14.0, 15.0])
    t_interest = 13.7
    t_infall = np.array([1.0, 4.0, 7.0, 10.0])
    upids = np.array([1, 1, 1, -1])

    try:
        get_p_merge_from_merging_u_params(
            DEFAULT_MERGE_PARAMS,
            log_mpeak_infall,
            log_mhost_infall,
            t_interest,
            t_infall,
            upids
        )
        raise NameError(
            "get_p_merge_from_merging_u_params should not accept params"
        )
    except AttributeError:
        pass


def test_get_p_merge_from_merging_params_fails_when_passing_u_params():

    log_mpeak_infall = np.array([9.0, 10.0, 11.0, 12.0])
    log_mhost_infall = np.array([12.0, 13.0, 14.0, 15.0])
    t_interest = 13.7
    t_infall = np.array([1.0, 4.0, 7.0, 10.0])
    upids = np.array([1, 1, 1, -1])

    try:
        get_p_merge_from_merging_params(
            DEFAULT_MERGE_U_PARAMS,
            log_mpeak_infall,
            log_mhost_infall,
            t_interest,
            t_infall,
            upids
        )
        raise NameError(
            "get_p_merge_from_merging_params should not accept u_params"
        )
    except AttributeError:
        pass


def test_pinfall_evaluates():
    t_interest = np.linspace(0.1, 13.7, 10)
    k_infall = 1.0
    t_infall = 1.0
    t_delay = 1.0
    p_max = 0.99
    p = p_infall(t_interest, k_infall, t_infall, t_delay, p_max)
    assert np.all(~np.isnan(p))
    assert np.all(p >= 0.0)
    assert np.all(p <= 1.0)


def test_get_p_merge_from_merging_u_params_evaluates():
    log_mpeak_infall = np.array([9.0, 10.0, 11.0, 12.0])
    log_mhost_infall = np.array([12.0, 13.0, 14.0, 15.0])
    t_interest = 13.7
    t_infall = np.array([1.0, 4.0, 7.0, 10.0])
    upids = np.array([1, 1, 1, -1])
    p = get_p_merge_from_merging_u_params(
        DEFAULT_MERGE_U_PARAMS,
        log_mpeak_infall,
        log_mhost_infall,
        t_interest,
        t_infall,
        upids,
    )
    assert np.all(~np.isnan(p))
    assert np.all(p >= 0.0)
    assert np.all(p <= 1.0)
    assert p[-1] == 0.0


def test_get_p_merge_from_merging_params_evaluates():
    log_mpeak_infall = np.array([9.0, 10.0, 11.0, 12.0])
    log_mhost_infall = np.array([12.0, 13.0, 14.0, 15.0])
    t_interest = 13.7
    t_infall = np.array([1.0, 4.0, 7.0, 10.0])
    upids = np.array([1, 1, 1, -1])
    p = get_p_merge_from_merging_params(
        DEFAULT_MERGE_PARAMS,
        log_mpeak_infall,
        log_mhost_infall,
        t_interest,
        t_infall,
        upids,
    )
    assert np.all(~np.isnan(p))
    assert np.all(p >= 0.0)
    assert np.all(p <= 1.0)
    assert p[-1] == 0.0


def test_merge_evaluates():
    log_mpeak_infall = np.array([9.0, 10.0, 11.0, 12.0])
    log_mhost_infall = np.array([12.0, 13.0, 14.0, 15.0])
    t_interest = 13.7
    t_infall = np.array([1.0, 4.0, 7.0, 10.0])
    upids = np.array([1, 1, 1, -1])
    sfr = np.array(([1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]))
    sfr = sfr.astype("f8")
    indx_to_deposit = np.array([3, 3, 3, 3])
    do_merging = np.array([1, 1, 1, 1])
    MC = 0
    merged_things = merge(
        DEFAULT_MERGE_U_PARAMS,
        log_mpeak_infall,
        log_mhost_infall,
        t_interest,
        t_infall,
        upids,
        sfr,
        indx_to_deposit,
        do_merging,
        MC,
    )
    assert np.all(~np.isnan(merged_things))
    assert np.all(merged_things >= 0.0)
    assert np.any(merged_things[3] > sfr[3])
    assert np.any(merged_things[0] < sfr[0])


def test_merge_with_MC_draws_evaluates():
    log_mpeak_infall = np.array([9.0, 10.0, 11.0, 12.0])
    log_mhost_infall = np.array([12.0, 13.0, 14.0, 15.0])
    t_interest = 13.7
    t_infall = np.array([1.0, 4.0, 7.0, 10.0])
    upids = np.array([1, 1, 1, -1])
    sfr = np.array(([1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]))
    sfr = sfr.astype("f8")
    indx_to_deposit = np.array([3, 3, 3, 3])
    do_merging = np.array([1, 1, 1, 1])
    MC = 1
    prn = np.array([0.1, 0.1, 0.99, 0.99])
    merged_things = merge_with_MC_draws(
        DEFAULT_MERGE_U_PARAMS,
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
    )
    assert np.all(~np.isnan(merged_things))
    assert np.all(merged_things >= 0.0)
    assert np.any(merged_things[3] > sfr[3])
    assert np.all(merged_things[0] == 0.0)


def test_merge_model_with_preprocessing_evaluates():
    log_mpeak_penultimate_infall = np.array([8.0, 9.0, 10.0, 12.0])
    log_mpeak_ultimate_infall = np.array([9.0, 10.0, 11.0, 12.0])
    log_mhost_penultimate_infall = np.array([11.0, 12.0, 13.0, 15.0])
    log_mhost_ultimate_infall = np.array([12.0, 13.0, 14.0, 15.0])
    t_interest = 13.7
    t_penultimate_infall = np.array([1.0, 4.0, 7.0, 10.0])
    t_ultimate_infall = np.array([2.0, 5.0, 8.0, 10.0])
    upids = np.array([1, 1, 1, -1])
    sfr = np.array(([1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]))
    sfr = sfr.astype("f8")
    penultimate_dump = np.array([2, 2, 2, 3])
    ultimate_dump = np.array([3, 3, 3, 3])
    MC = 0
    merged_things = merge_model_with_preprocessing(
        DEFAULT_MERGE_U_PARAMS,
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
    )
    assert np.all(~np.isnan(merged_things))
    assert np.all(merged_things >= 0.0)
    assert np.any(merged_things[3] > sfr[3])
    assert np.any(merged_things[0] < sfr[0])


def test_merge_model_with_preprocessing_mc_draws_evaluates():
    log_mpeak_penultimate_infall = np.array([8.0, 9.0, 10.0, 12.0])
    log_mpeak_ultimate_infall = np.array([9.0, 10.0, 11.0, 12.0])
    log_mhost_penultimate_infall = np.array([11.0, 12.0, 13.0, 15.0])
    log_mhost_ultimate_infall = np.array([12.0, 13.0, 14.0, 15.0])
    t_interest = 13.7
    t_penultimate_infall = np.array([1.0, 4.0, 7.0, 10.0])
    t_ultimate_infall = np.array([2.0, 5.0, 8.0, 10.0])
    upids = np.array([1, 1, 1, -1])
    sfr = np.array(([1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]))
    sfr = sfr.astype("f8")
    penultimate_dump = np.array([2, 2, 2, 3])
    ultimate_dump = np.array([3, 3, 3, 3])
    MC = 0
    mc_draws = np.array([0.1, 0.1, 0.99, 0.99])
    merged_things = merge_model_with_preprocessing_mc_draws(
        DEFAULT_MERGE_U_PARAMS,
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
    )
    assert np.all(~np.isnan(merged_things))
    assert np.all(merged_things >= 0.0)
    assert np.any(merged_things[3] > sfr[3])
    assert np.any(merged_things[0] < sfr[0])
