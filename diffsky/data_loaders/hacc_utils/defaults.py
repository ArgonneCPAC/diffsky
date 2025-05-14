""""""

# flake8: noqa

__all__ = ("N_MIN_MAH_PTS", "DIFFMAH_MASS_COLNAME", "N_PTCL_COREFOREST")

# Simulated MAHs with fewer points than N_MIN_MAH_PTS will get a synthetic MAH
N_MIN_MAH_PTS = 4

# DIFFMAH_MASS_COLNAME is the column used in the diffmah fits of the coreforest files
DIFFMAH_MASS_COLNAME = "infall_tree_node_mass"

# Minimum number of particles to use in coreforest analyses for each simulation
N_PTCL_COREFOREST = 100
