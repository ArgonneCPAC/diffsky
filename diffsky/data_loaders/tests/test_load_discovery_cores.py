"""
"""

from ...mass_functions.mc_diffmah_tpeak import SubhaloCatalog as MCSubhaloCatalog
from .. import load_discovery_cores as ldc


def test_subhalo_catalog_exists_and_has_some_attributes():
    assert hasattr(ldc.SubhaloCatalog, "mah_params")
    assert hasattr(ldc.SubhaloCatalog, "logmp0")

    assert hasattr(ldc.SubhaloCatalog, "logmp0")

    excluded_keys = (
        "halo_ids",
        "host_mah_params",
        "logmhost_pen_inf",
        "logmhost_ult_inf",
    )
    for key in MCSubhaloCatalog._fields:
        if key not in excluded_keys:
            msg = f"`{key}` missing from diffdesi data loader"
            assert key in ldc.SubhaloCatalog._fields, msg
