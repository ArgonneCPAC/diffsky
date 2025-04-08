""" """

from .. import hacc_sims


def test_global_variables_are_stable():
    assert hacc_sims.DIFFMAH_MASS_COLNAME == "infall_tree_node_mass"
