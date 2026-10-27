""""""

import numpy as np

from .. import merging_kernels as mk


def test_compute_x_tot_from_x_in_situ_hard_coded_example_0():
    x_cen0, x_cen1, x_cen2 = 4.0, 3.0, 2.0
    x_sat00, x_sat10, x_sat11 = 0.2, 0.4, 0.6

    x_in_situ = np.array((x_cen0, x_sat00, x_cen1, x_sat10, x_sat11, x_cen2))
    p_merge = np.array((0, 1, 0, 1, 1, 0)).astype("float")
    nsat_weights = np.ones_like(x_in_situ)
    halo_indx = np.array((0, 0, 2, 2, 2, 5)).astype("int")

    args = x_in_situ, p_merge, nsat_weights, halo_indx
    x_tot = mk.compute_x_tot_from_x_in_situ(*args)

    x_tot_correct = np.array((4.2, 0.0, 4.0, 0, 0, 2))
    assert np.allclose(x_tot, x_tot_correct, rtol=1e-4)

    _F = 0.2
    x_tot = mk.compute_x_tot_from_x_in_situ(*args, frac_receive=_F)
    x_tot_correct = np.array(
        (x_cen0 + _F * x_sat00, 0.0, x_cen1 + _F * (x_sat10 + x_sat11), 0, 0, x_cen2)
    )
    assert np.allclose(x_tot, x_tot_correct, rtol=1e-4)


def test_compute_x_tot_from_x_in_situ_hard_coded_example_1():
    """Same as test_compute_x_tot_from_x_in_situ_hard_coded_example_0 but different array indexing"""
    x_cen0, x_cen1, x_cen2 = 4.0, 3.0, 2.0
    x_sat00, x_sat10, x_sat11 = 0.2, 0.4, 0.6

    x_in_situ = np.array((x_cen0, x_cen1, x_cen2, x_sat00, x_sat10, x_sat11))
    p_merge = np.array((0, 0, 0, 1, 1, 1)).astype("float")
    nsat_weights = np.ones_like(x_in_situ)
    halo_indx = np.array((0, 1, 2, 0, 1, 1)).astype("int")

    args = x_in_situ, p_merge, nsat_weights, halo_indx
    x_tot = mk.compute_x_tot_from_x_in_situ(*args)

    x_tot_correct = np.array((4.2, 4.0, 2.0, 0, 0, 0))
    assert np.allclose(x_tot, x_tot_correct, rtol=1e-4)

    _F = 0.2
    x_tot = mk.compute_x_tot_from_x_in_situ(*args, frac_receive=_F)
    x_tot_correct = np.array(
        (x_cen0 + _F * x_sat00, x_cen1 + _F * (x_sat10 + x_sat11), x_cen2, 0, 0, 0)
    )
    assert np.allclose(x_tot, x_tot_correct, rtol=1e-4)
