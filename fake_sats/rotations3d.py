"""Utilities for rotating 3d vectors"""

import numpy as np

from . import vector_utilities as vectu

__all__ = [
    "rotation_matrices_from_vectors",
]


def rotation_matrices_from_angles(angles, directions):
    """
    Calculate a collection of rotation matrices defined by
    an input collection of rotation angles and rotation axes.

    Parameters
    ----------
    angles : ndarray
        Numpy array of shape (npts, ) storing a collection of rotation angles

    directions : ndarray
        Numpy array of shape (npts, 3) storing a collection of rotation axes in 3d

    Returns
    -------
    matrices : ndarray
        Numpy array of shape (npts, 3, 3) storing a collection of rotation matrices

    """
    directions = vectu.normalized_vectors(directions)
    angles = np.atleast_1d(angles)
    npts = directions.shape[0]

    sina = np.sin(angles)
    cosa = np.cos(angles)

    R1 = np.zeros((npts, 3, 3))
    R1[:, 0, 0] = cosa
    R1[:, 1, 1] = cosa
    R1[:, 2, 2] = cosa

    R2 = directions[..., None] * directions[:, None, :]
    R2 = R2 * np.repeat(1.0 - cosa, 9).reshape((npts, 3, 3))

    directions *= sina.reshape((npts, 1))
    R3 = np.zeros((npts, 3, 3))
    R3[:, [1, 2, 0], [2, 0, 1]] -= directions
    R3[:, [2, 0, 1], [1, 2, 0]] += directions

    return R1 + R2 + R3


def rotation_matrices_from_vectors(v0, v1):
    """
    Calculate a collection of rotation matrices defined by two sets of vectors,
    v1 into v2, such that the resulting matrices rotate v1 into v2 about
    the mutually perpendicular axis.

    Parameters
    ----------
    v0 : ndarray
        Numpy array of shape (npts, 3) storing a collection
        of initial vector orientations.

        Note that the normalization of `v0` will be ignored.

    v1 : ndarray
        Numpy array of shape (npts, 3) storing a collection of final vectors.

        Note that the normalization of `v1` will be ignored.

    Returns
    -------
    matrices : ndarray
        Numpy array of shape (npts, 3, 3) rotating each v0 into the corresponding v1

    """
    v0 = vectu.normalized_vectors(v0)
    v1 = vectu.normalized_vectors(v1)
    directions = vectu.vectors_normal_to_planes(v0, v1)
    angles = vectu.angles_between_list_of_vectors(v0, v1)

    # Edge case: where angles are 0.0, replace directions with v0
    mask_a = (
        np.isnan(directions[:, 0])
        | np.isnan(directions[:, 1])
        | np.isnan(directions[:, 2])
    )
    mask_b = angles == 0.0
    mask = mask_a | mask_b
    directions[mask] = v0[mask]

    return rotation_matrices_from_angles(angles, directions)
