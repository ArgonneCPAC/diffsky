""" """

from ....param_utils import diffsky_param_wrapper_merging as dpwm
from .. import dictionaries


def test_param_subsets_names_are_valid_diffsky_u_params():
    """Enforce that every parameter name in PARAM_SUBSETS is a valid
    flat diffsky unbounded parameter name"""
    u_pnames = set(dpwm.U_PNAMES_FLAT)
    for subset_name, param_names in dictionaries.PARAM_SUBSETS.items():
        for name in param_names:
            assert name in u_pnames, f"{subset_name}: {name} is not a valid u_param"
