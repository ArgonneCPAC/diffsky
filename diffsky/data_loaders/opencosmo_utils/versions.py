import warnings
from importlib import import_module
from typing import Optional

import opencosmo as oc

# Set to "strict" to require exact version matches, "minor" to require matching
# major.minor only (e.g. 0.3), or None to disable version checking entirely.

# Package names listed here are skipped during version checking even if
# mode is set. Add entries when a known version mismatch is
# acceptable for a particular library.
VERSION_CHECK_IGNORE: set[str] = set()

_ALLOWED_MODES = ("strict", "minor")

_VERSION_CHECK_DISABLED_WARNING = (
    "Version checking is disabled (mode = None in versions.py). "
    "Photometry and SED calculations may not work quite right if your versions of the "
    "diffsky libraries do not match those used to generate this catalog."
)


def check_versions(ds: oc.Lightcone, mode: Optional[str] = None):
    """
    Check the versions of diffsky packages recorded in the catalog header against
    the versions installed in the user's environment. Behaviour is controlled by
    the module-level constants mode and VERSION_CHECK_IGNORE.

    Raises ValueError on a version mismatch, or if the catalog does not contain
    diffsky header information.
    """
    if mode is None:
        warnings.warn(_VERSION_CHECK_DISABLED_WARNING, UserWarning, stacklevel=2)
        return

    if mode not in _ALLOWED_MODES:
        raise ValueError(f"mode must be one of {_ALLOWED_MODES} or None, got {mode!r}")

    try:
        diffsky_versions = ds.header.diffsky_versions
    except AttributeError:
        raise ValueError("This data does not appear to be a diffsky catalog!")

    for package_name, expected_version in diffsky_versions.items():
        if expected_version is None:  # diffstarpop compatability
            continue
        if package_name in VERSION_CHECK_IGNORE:
            continue
        package = import_module(package_name)
        installed_version = getattr(package, "__version__")
        __verify_version(expected_version, installed_version, package_name, mode)


def _parse_minor(version_str: str) -> str:
    """Extract 'major.minor' from a version string.

    Handles PEP 440 local version identifiers (e.g. ``0.3.2.dev0+g1234abc``)
    and simple dash-separated pre-release suffixes (e.g. ``0.3.2-dev``).
    """
    # Strip local version identifier (everything after '+')
    version_str = version_str.split("+")[0]
    parts = version_str.split(".")
    if len(parts) >= 2:
        return f"{parts[0]}.{parts[1]}"
    return version_str


def __verify_version(expected_version, installed_version, package_name, mode):
    if mode == "strict" and expected_version != installed_version:
        raise ValueError(
            f"Installed version of {package_name} ({installed_version}) does not "
            "exactly match the version used to generate this catalog "
            "You can set `version_check = None` to disable version checking, but"
            "this may cause issues with phomtetry and SED calucations"
        )
    elif mode == "minor":
        expected_minor = _parse_minor(expected_version)
        installed_minor = _parse_minor(installed_version)
        if expected_minor != installed_minor:
            raise ValueError(
                f"Installed version of {package_name} ({installed_version}) does not "
                f"match the minor version used to generate this catalog "
                f"({expected_version}). "
                f"Expected {expected_minor}, got {installed_minor}. "
                "You can set `version_check = None` to disable version checking, but"
                "this may cause issues with phomtetry and SED calucations"
            )
