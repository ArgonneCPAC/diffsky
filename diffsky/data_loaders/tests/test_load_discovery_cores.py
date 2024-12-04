"""
"""

from .. import load_discovery_cores as ldc


def test_something():
    assert hasattr(ldc.SubhaloCatalog, "mah_params")
    assert hasattr(ldc.SubhaloCatalog, "logmp0")
