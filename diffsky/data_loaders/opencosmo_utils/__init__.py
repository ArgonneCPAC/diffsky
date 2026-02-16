try:
    import opencosmo
except ImportError:
    raise ImportError(
        "The opencosmo_utils module requires opencosmo to be installed in your "
        "environment. You can install it with pip or conda-forge"
    )

from .compute import (
    compute_dbk_phot_from_diffsky_mock,
    compute_dbk_seds_from_diffsky_mock,
    compute_phot_from_diffsky_mock,
    compute_seds_from_diffsky_mock,
)
from .load import add_transmission_curves, load_diffsky_mock

__all__ = [
    "load_diffsky_mock",
    "compute_dbk_phot_from_diffsky_mock",
    "compute_dbk_seds_from_diffsky_mock",
    "compute_phot_from_diffsky_mock",
    "compute_seds_from_diffsky_mock",
    "add_transmission_curves",
]
