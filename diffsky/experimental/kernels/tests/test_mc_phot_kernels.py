""""""


def test_mc_phot_kernels_imports():
    from .. import mc_phot_kernels as mcpk

    assert hasattr(mcpk, "LGMET_SCATTER")
