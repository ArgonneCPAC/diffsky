"""
"""
import numpy as np

um_dtype = np.dtype(
    dtype=[
        ("id", "i8"),
        ("descid", "i8"),
        ("upid", "i8"),
        ("flags", "i4"),
        ("uparent_dist", "f4"),
        ("pos", "f4", (6)),
        ("vmp", "f4"),
        ("lvmp", "f4"),
        ("mp", "f4"),
        ("m", "f4"),
        ("v", "f4"),
        ("r", "f4"),
        ("rank1", "f4"),
        ("rank2", "f4"),
        ("ra", "f4"),
        ("rarank", "f4"),
        ("A_UV", "f4"),
        ("sm", "f4"),
        ("icl", "f4"),
        ("sfr", "f4"),
        ("obs_sm", "f4"),
        ("obs_sfr", "f4"),
        ("obs_uv", "f4"),
        ("empty", "f4"),
    ],
    align=True,
)
