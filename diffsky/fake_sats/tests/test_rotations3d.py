""" """

import numpy as np
from jax import random as jran

from .. import rotations3d
from .. import vector_utilities as vectu


def test_rotation_matrices_from_vectors():
    """Enforce that the rotation_matrices_from_vectors function returns a set of
    matrices that correctly rotate the input v0 into the input v1"""
    ran_key = jran.key(0)
    n_vectors = 25

    v0_key, v1_key = jran.split(ran_key, 2)
    v0_collection = jran.uniform(v0_key, minval=-1, maxval=1, shape=(n_vectors, 3))
    v1_collection = jran.uniform(v1_key, minval=-1, maxval=1, shape=(n_vectors, 3))

    v0_collection = vectu.normalized_vectors(v0_collection)
    v1_collection = vectu.normalized_vectors(v1_collection)

    rot_matrices = rotations3d.rotation_matrices_from_vectors(
        v0_collection, v1_collection
    )

    v_collection = vectu.rotate_vector_collection(rot_matrices, v0_collection)
    assert np.allclose(v_collection, v1_collection, rtol=1e-3)
