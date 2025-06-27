"""
A set of vector calculations to aid in rotation calculations
"""

import numpy as np

__all__ = [
    "elementwise_dot",
    "elementwise_norm",
    "normalized_vectors",
    "angles_between_list_of_vectors",
    "vectors_normal_to_planes",
    "project_onto_plane",
    "rotate_vector_collection",
]


def normalized_vectors(vectors):
    """
    Return a unit vector for each n-dimensional vector in the input list
    of n-dimensional points.

    Parameters
    ----------
    x : ndarray
        Numpy array of shape (npts, ndim) storing a collection of n-dimensional points

    Returns
    -------
    normed_x : ndarray
        Numpy array of shape (npts, ndim)

    """

    vectors = np.atleast_2d(vectors)
    npts = vectors.shape[0]

    with np.errstate(divide="ignore", invalid="ignore"):
        return vectors / elementwise_norm(vectors).reshape((npts, -1))


def elementwise_norm(x):
    """
    Calculate the normalization of each element in a list of n-dimensional points.

    Parameters
    ----------
    x : ndarray
        Numpy array of shape (npts, ndim) storing a collection of n-dimensional points

    Returns
    -------
    result : ndarray
        Numpy array of shape (npts, ) storing the norm of each n-dimensional point in x.

    """

    x = np.atleast_2d(x)
    return np.sqrt(np.sum(x**2, axis=1))


def elementwise_dot(x, y):
    """
    Calculate the dot product between
    each pair of elements in two input lists of n-dimensional points.

    Parameters
    ----------
    x : ndarray
        Numpy array of shape (npts, ndim) storing a collection of n-dimensional vectors

    y : ndarray
        Numpy array of shape (npts, ndim) storing a collection of n-dimensional vectors

    Returns
    -------
    result : ndarray
        Numpy array of shape (npts, ) storing the dot product between each
        pair of corresponding vectors in x and y.

    """

    x = np.atleast_2d(x)
    y = np.atleast_2d(y)
    return np.sum(x * y, axis=1)


def angles_between_list_of_vectors(v0, v1, tol=1e-3, vn=None):
    """Calculate the angle between a collection of n-dimensional vectors

    Parameters
    ----------
    v0 : ndarray
        Numpy array of shape (npts, ndim) storing a collection of ndim-D vectors
        Note that the normalization of `v0` will be ignored.

    v1 : ndarray
        Numpy array of shape (npts, ndim) storing a collection of ndim-D vectors
        Note that the normalization of `v1` will be ignored.

    tol : float, optional
        Acceptable numerical error for errors in angle.
        This variable is only used to round off numerical noise that otherwise
        causes exceptions to be raised by the inverse cosine function.
        Default is 0.001.

    n1 : ndarray
        normal vector

    Returns
    -------
    angles : ndarray
        Numpy array of shape (npts, ) storing the angles between each pair of
        corresponding points in v0 and v1.

        Returned values are in units of radians spanning [0, pi].

    """

    dot = elementwise_dot(normalized_vectors(v0), normalized_vectors(v1))

    if vn is None:
        #  Protect against tiny numerical excesses beyond the range [-1 ,1]
        mask1 = (dot > 1) & (dot < 1 + tol)
        dot = np.where(mask1, 1.0, dot)
        mask2 = (dot < -1) & (dot > -1 - tol)
        dot = np.where(mask2, -1.0, dot)
        a = np.arccos(dot)
    else:
        cross = np.cross(v0, v1)
        a = np.arctan2(elementwise_dot(cross, vn), dot)

    return a


def vectors_normal_to_planes(x, y):
    """
    Given a collection of 3d vectors x and y, return a collection of
    3d unit-vectors that are orthogonal to x and y.

    Parameters
    ----------
    x : ndarray
        Numpy array of shape (npts, 3) storing a collection of 3d vectors

        Note that the normalization of `x` will be ignored.

    y : ndarray
        Numpy array of shape (npts, 3) storing a collection of 3d vectors

        Note that the normalization of `y` will be ignored.

    Returns
    -------
    z : ndarray
        Numpy array of shape (npts, 3). Each 3d vector in z will be orthogonal
        to the corresponding vector in x and y.

    """
    return normalized_vectors(np.cross(x, y))


def project_onto_plane(x1, x2):
    """
    Given a collection of vectors, x1 and x2, project each vector
    in x1 onto the plane normal to the corresponding vector x2.

    Parameters
    ----------
    x1 : ndarray
        Numpy array of shape (npts, 3) storing a collection of 3d points

    x2 : ndarray
        Numpy array of shape (npts, 3) storing a collection of 3d points

    Returns
    -------
    result : ndarray
        Numpy array of shape (npts, 3) storing a collection of 3d points

    """

    n = normalized_vectors(x2)
    d = elementwise_dot(x1, n)

    return x1 - d[:, np.newaxis] * n


def rotate_vector_collection(rotation_matrices, vectors, optimize=False):
    """
    Given a collection of rotation matrices and a collection of n-dimensional vectors,
    apply an asscoiated matrix to rotate corresponding vector(s).

    Parameters
    ----------
    rotation_matrices : ndarray
        The options are:
        1.) array of shape (npts, ndim, ndim) storing a collection of rotation matrices.
        2.) array of shape (ndim, ndim) storing a single rotation matrix

    vectors : ndarray
        The corresponding options for above are:
        1.) array of shape (npts, ndim) storing a collection of ndim-dimensional vectors
        2.) array of shape (npts, ndim) storing a collection of ndim-dimensional vectors

    Returns
    -------
    rotated_vectors : ndarray
        Numpy array of shape (npts, ndim) storing
        a collection of ndim-dimensional vectors

    Notes
    -----
    This function is set up to preform either rotation operations on a single collection
    of vectors, either applying a single rotation matrix to all vectors
    in the collection, or applying a unique rotation matrix to each vector in the set.

    """

    ndim_rotm = np.shape(rotation_matrices)[-1]
    ndim_vec = np.shape(vectors)[-1]
    assert ndim_rotm == ndim_vec

    if len(np.shape(vectors)) == 2:
        ntps, ndim = np.shape(vectors)
        nsets = 0
    elif len(np.shape(vectors)) == 3:
        nsets, ntps, ndim = np.shape(vectors)

    # apply same rotation matrix to all vectors
    if len(np.shape(rotation_matrices)) == 2:
        if nsets == 1:
            vectors = vectors[0]
        return np.dot(rotation_matrices, vectors.T).T
    # rotate each vector by associated rotation matrix
    else:
        ein_string = "ijk,ik->ij"
        n1, ndim = np.shape(vectors)

        try:
            return np.einsum(ein_string, rotation_matrices, vectors, optimize=optimize)
        except TypeError:
            return np.einsum(ein_string, rotation_matrices, vectors)
