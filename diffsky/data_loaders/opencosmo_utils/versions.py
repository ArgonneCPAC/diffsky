from importlib import import_module

import opencosmo as oc

ALLOWED_MAX = {"numpy": None}


def check_versions(ds: oc.Lightcone):
    """
    Check the versions of diff packages in the header of the lightcone, and compare
    to the versions installed in the user's environment. A mismatch raises a warning,
    rather than an error. A user may be using the diffsky interface to load the catalog,
    with no intention of actually passing the data to any diffsky routines.

    DOES raise an error if the diffsky header information is not found, as this indicates
    the user *thinks* they are working with a diffsky catalog when they are not.
    """
    try:
        diffsky_versions = ds.header.diffsky_versions
    except AttributeError:
        raise ValueError("This data does not appear to be a diffsky catalog!")

    for package_name, expected_version in diffsky_versions.items():
        if expected_version is None:  # diffstarpop compatability
            continue
        package = import_module(package_name)
        installed_version = getattr(package, "__version__")
        __verify_version(expected_version, installed_version, package_name)


def __verify_version(expected_version, installed_version, package_name):
    pass
