"""
"""

import numpy as np

from ..merging_model import DEFAULT_MERGE_U_PARAMS

from ..fitmerge_multi_redshift import (
    scalar_smf,
    csmf_sats_cens_all,
    condition,
    csmf,
    model_histogram,
    conditional_model_histogram,
    mapped_model_csmf,
    model_mass,
    sat_frac
)


def test_scalar_smf():

    logM = 12.0
    lobins = np.array([11.0, 11.5, 12.0, 12.5, 13.0])
    a_interest = 1.0
    sigma = 0.05
    bin_width = 0.5
    comoving_volume = 20.0**3

    hist = scalar_smf(
        logM, lobins, a_interest, sigma, bin_width, comoving_volume)

    assert ~np.isnan(hist)


def test_csmf_sats_cens_all():

    all_sm_no_zero = np.array([10.0, 11.0, 10.5, 11.4, 12.2])
    host_lgmp = np.array([11.5, 12.0, 11.0, 12.5, 13.0])
    upid = np.array([1, -1, 1, -1, -1])
    a = 1.0
    lobins = np.array([11.0, 11.5, 12.0, 12.5, 13.0])
    sigma = 0.05
    bin_width = 0.5
    comoving_volume = 20.0**3

    smf = csmf_sats_cens_all(
        all_sm_no_zero, host_lgmp, upid, a, lobins,
        sigma, bin_width, comoving_volume)

    assert np.all(~np.isnan(smf))


def test_condition():

    host_lgmp = np.array([12.1, 12.5, 13.1, 14.1, 12.8, 13.5, 14.6])

    condition12, condition13, condition14 = condition(host_lgmp)

    assert len(condition12) == 3
    assert len(condition13) == 2
    assert len(condition14) == 2


def test_csmf():

    all_sm_no_zero = np.array([10.0, 11.0, 10.5, 11.4, 12.2])
    host_lgmp = np.array([11.5, 12.0, 11.0, 12.5, 13.0])
    a = 1.0
    lobins = np.array([11.0, 11.5, 12.0, 12.5, 13.0])
    sigma = 0.05
    bin_width = 0.5
    comoving_volume = 20.0**3

    smf = csmf(
        all_sm_no_zero, host_lgmp, a, lobins,
        sigma, bin_width, comoving_volume)

    assert np.all(~np.isnan(smf))


def test_model_histogram():

    logMpeak_penultimate_infall = np.array([8.0, 9.0, 10.0, 12.0])
    logMpeak_ultimate_infall = np.array([9.0, 10.0, 11.0, 12.0])
    logMhost_penultimate_infall = np.array([11.0, 12.0, 13.0, 15.0])
    logMhost_ultimate_infall = np.array([12.0, 13.0, 14.0, 15.0])
    tinterest = 13.7
    t_penultimate_infall = np.array([1.0, 4.0, 7.0, 10.0])
    t_ultimate_infall = np.array([2.0, 5.0, 8.0, 10.0])
    upids = np.array([1, 1, 1, -1])
    sfr = np.array(([1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]))
    sfr = sfr.astype("f8")
    penultimate_dump = np.array([2, 2, 2, 3])
    ultimate_dump = np.array([3, 3, 3, 3])
    MC = 0
    dT = np.array([1, 1, 1, 1])
    i_interest = 3
    a_interest = 1.0
    sigma = 0.05
    bin_width = 0.5
    comoving_volume = 20.0**3
    lobins = np.array([11.0, 11.5, 12.0, 12.5])

    hist = model_histogram(
        DEFAULT_MERGE_U_PARAMS,
        logMpeak_penultimate_infall, logMpeak_ultimate_infall,
        logMhost_penultimate_infall, logMhost_ultimate_infall,
        tinterest, t_penultimate_infall, t_ultimate_infall, upids, sfr,
        penultimate_dump, ultimate_dump, MC, dT, i_interest, a_interest,
        sigma, bin_width, comoving_volume, lobins)

    assert ~np.isnan(hist)


def test_conditional_model_histogram():

    logMpeak_penultimate_infall = np.array([8.0, 9.0, 10.0, 12.0])
    logMpeak_ultimate_infall = np.array([9.0, 10.0, 11.0, 12.0])
    logMhost_penultimate_infall = np.array([11.0, 12.0, 13.0, 15.0])
    logMhost_ultimate_infall = np.array([12.0, 13.0, 14.0, 15.0])
    tinterest = 13.7
    t_penultimate_infall = np.array([1.0, 4.0, 7.0, 10.0])
    t_ultimate_infall = np.array([2.0, 5.0, 8.0, 10.0])
    upids = np.array([1, 1, 1, -1])
    sfr = np.array(([1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]))
    sfr = sfr.astype("f8")
    penultimate_dump = np.array([2, 2, 2, 3])
    ultimate_dump = np.array([3, 3, 3, 3])
    MC = 0
    dT = np.array([1, 1, 1, 1])
    i_interest = 3
    a_interest = 1.0
    sigma = 0.05
    bin_width = 0.5
    comoving_volume = 20.0**3
    lobins = np.array([11.0, 11.5, 12.0, 12.5])
    keep = np.array([1, 0, 1, 0])

    hist = conditional_model_histogram(
        DEFAULT_MERGE_U_PARAMS,
        logMpeak_penultimate_infall, logMpeak_ultimate_infall,
        logMhost_penultimate_infall, logMhost_ultimate_infall,
        tinterest, t_penultimate_infall, t_ultimate_infall,
        upids, sfr, penultimate_dump, ultimate_dump, MC, dT,
        i_interest, a_interest, sigma, bin_width, comoving_volume,
        lobins, keep)

    assert ~np.isnan(hist)


def test_mapped_model_csmf():

    logMpeak_penultimate_infall = np.array([8.0, 9.0, 10.0, 12.0])
    logMpeak_ultimate_infall = np.array([9.0, 10.0, 11.0, 12.0])
    logMhost_penultimate_infall = np.array([11.0, 12.0, 13.0, 15.0])
    logMhost_ultimate_infall = np.array([12.0, 13.0, 14.0, 15.0])
    tinterest = 13.7
    t_penultimate_infall = np.array([1.0, 4.0, 7.0, 10.0])
    t_ultimate_infall = np.array([2.0, 5.0, 8.0, 10.0])
    upids = np.array([1, 1, 1, -1])
    sfr = np.array(([1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]))
    sfr = sfr.astype("f8")
    penultimate_dump = np.array([2, 2, 2, 3])
    ultimate_dump = np.array([3, 3, 3, 3])
    MC = 0
    dT = np.array([1, 1, 1, 1])
    i_interest = 3
    a_interest = 1.0
    sigma = 0.05
    bin_width = 0.5
    comoving_volume = 20.0**3
    lobins = np.array([11.0, 11.5, 12.0, 12.5, 13.0])
    keep = np.array([1, 0, 1, 0])

    hist = mapped_model_csmf(
        DEFAULT_MERGE_U_PARAMS,
        logMpeak_penultimate_infall, logMpeak_ultimate_infall,
        logMhost_penultimate_infall, logMhost_ultimate_infall,
        tinterest, t_penultimate_infall, t_ultimate_infall,
        upids, sfr, penultimate_dump, ultimate_dump, MC, dT,
        i_interest, a_interest, sigma, bin_width, comoving_volume,
        lobins, keep)

    assert np.all(~np.isnan(hist))


def test_model_mass():

    logMpeak_penultimate_infall = np.array([8.0, 9.0, 10.0, 12.0])
    logMpeak_ultimate_infall = np.array([9.0, 10.0, 11.0, 12.0])
    logMhost_penultimate_infall = np.array([11.0, 12.0, 13.0, 15.0])
    logMhost_ultimate_infall = np.array([12.0, 13.0, 14.0, 15.0])
    tinterest = 13.7
    t_penultimate_infall = np.array([1.0, 4.0, 7.0, 10.0])
    t_ultimate_infall = np.array([2.0, 5.0, 8.0, 10.0])
    upids = np.array([1, 1, 1, -1])
    sfr = np.array(([1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]))
    sfr = sfr.astype("f8")
    penultimate_dump = np.array([2, 2, 2, 3])
    ultimate_dump = np.array([3, 3, 3, 3])
    MC = 0
    dT = np.array([1, 1, 1, 1])
    i_interest = 3

    M_interest = model_mass(
        DEFAULT_MERGE_U_PARAMS,
        logMpeak_penultimate_infall, logMpeak_ultimate_infall,
        logMhost_penultimate_infall, logMhost_ultimate_infall,
        tinterest, t_penultimate_infall, t_ultimate_infall,
        upids, sfr, penultimate_dump, ultimate_dump,
        MC, dT, i_interest)

    assert np.all(~np.isnan(M_interest))


def test_sat_frac():

    logMpeak_penultimate_infall = np.array([8.0, 9.0, 10.0, 12.0])
    logMpeak_ultimate_infall = np.array([9.0, 10.0, 11.0, 12.0])
    logMhost_penultimate_infall = np.array([11.0, 12.0, 13.0, 15.0])
    logMhost_ultimate_infall = np.array([12.0, 13.0, 14.0, 15.0])
    tinterest = 13.7
    t_penultimate_infall = np.array([1.0, 4.0, 7.0, 10.0])
    t_ultimate_infall = np.array([2.0, 5.0, 8.0, 10.0])
    upids = np.array([1, 1, 1, -1])
    sfr = np.array(([1., 2., 3., 4.], [1., 2., 3., 4.], [1., 2., 3., 4.], [1., 2., 3., 4.]))
    sfr = sfr.astype("f8")
    penultimate_dump = np.array([2, 2, 2, 3])
    ultimate_dump = np.array([3, 3, 3, 3])
    MC = 0
    dT = np.array([1, 1, 1, 1])
    i_interest = 3
    bin_width = 1.0
    bins = np.array([0.0, 1.0])

    satfrac = sat_frac(
        DEFAULT_MERGE_U_PARAMS,
        logMpeak_penultimate_infall, logMpeak_ultimate_infall,
        logMhost_penultimate_infall, logMhost_ultimate_infall,
        tinterest, t_penultimate_infall, t_ultimate_infall,
        upids, sfr, penultimate_dump, ultimate_dump, MC, dT,
        i_interest, bins, bin_width)

    print(satfrac)

    assert np.all(~np.isnan(satfrac))
