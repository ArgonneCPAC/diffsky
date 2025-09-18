# flake8: noqa: F401
"""Stabilize a few imports from diffsky.experimental"""


def test_mc_weighted_lightcone_data_imports():
    from ..lc_phot_kern import mc_weighted_lightcone_data


def test_mc_weighted_diffsky_lightcone_imports():
    from ..mc_diffsky_seds import mc_weighted_diffsky_lightcone


def test_mc_weighted_halo_lightcone_imports():
    from ..mc_lightcone_halos import mc_weighted_halo_lightcone
