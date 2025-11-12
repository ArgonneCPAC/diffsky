""" """

import os

import h5py
import numpy as np

from .. import load_flat_hdf5

REQUIRED_METADATA_ATTRS = ("creation_date", "header", "mock_version_name")

BNPAT_LC_MOCK = "data-{0}.{1}.diffsky_gals.hdf5"

HLINE = "----------"


def get_lc_mock_data_report(fn_lc_mock):
    report = dict()
    data = load_flat_hdf5(fn_lc_mock, dataset="data")

    msg = check_all_columns_are_finite(fn_lc_mock, data=data)
    if len(msg) > 0:
        report["finite_colums"] = msg

    msg = check_host_pos_is_near_galaxy_pos(fn_lc_mock, data=data)
    if len(msg) > 0:
        report["nfw_host_distance"] = msg

    msg = check_metadata(fn_lc_mock)
    if len(msg) > 0:
        report["metadata"] = msg

    msg = check_all_data_columns_have_metadata(fn_lc_mock)
    if len(msg) > 0:
        report["column_metadata"] = msg

    return report


def write_lc_mock_report_to_disk(report, fn_lc_mock, drn_report):
    if len(report) > 0:
        bn_report = os.path.basename(fn_lc_mock).replace(".hdf5", ".report.txt")
        fn_report = os.path.join(drn_report, bn_report)
        with open(fn_report, "w") as fn_out:
            for test_key, test_result in report.items():
                fn_out.write(test_key + "\n")
                for line in test_result:
                    fn_out.write(line + "\n")
                fn_out.write(HLINE + "\n")


def check_all_columns_are_finite(fn_lc_mock, data=None):
    bn = os.path.basename(fn_lc_mock)

    if data is None:
        data = load_flat_hdf5(fn_lc_mock, dataset="data")

    msg = []
    for key, arr in data.items():
        if not np.all(np.isfinite(arr)):
            s = f"Column {key} in {bn} has either NaN or inf"
            msg.append(s)

    return msg


def check_all_data_columns_have_metadata(fn_lc_mock):

    msg = []
    with h5py.File(fn_lc_mock, "r") as hdf:
        for key in hdf["data"].keys():
            try:
                unit = hdf["data/" + key].attrs["unit"]
                assert len(unit) > 0
                description = hdf["data/" + key].attrs["description"]
                assert len(description) > 0
            except (KeyError, AssertionError):
                s = f"{key} is missing metadata"
                msg.append(s)
    return msg


def check_metadata(fn_lc_mock):

    msg = []
    with h5py.File(fn_lc_mock, "r") as hdf:
        try:
            # Check all scalar metadata
            avail_medata_attrs = list(hdf["metadata"].attrs.keys())
            assert set(avail_medata_attrs) >= set(REQUIRED_METADATA_ATTRS)

            creation_date = hdf["metadata"].attrs["creation_date"]
            assert len(creation_date) > 0
            mock_version_name = hdf["metadata"].attrs["mock_version_name"]
            assert len(mock_version_name.split("_")) > 1
            header = hdf["metadata"].attrs["header"]
            assert "This file contains diffsky" in header

            # Check cosmology metadata
            Om0 = hdf["metadata/cosmology"].attrs["Om0"]
            assert 0 < Om0 < 1
            all_hacc_cosmo_params = ("Ob0", "Om0", "h", "ns", "sigma8", "w0", "wa")
            assert set(all_hacc_cosmo_params) == set(
                hdf["metadata/cosmology"].attrs.keys()
            )

            # Check nbody_info metadata
            expected_info = ("Lbox", "n_particles", "particle_mass", "sim_name")
            assert set(expected_info) == set(hdf["metadata/nbody_info"].attrs.keys())
            Lbox = hdf["metadata/nbody_info"].attrs["Lbox"]
            assert Lbox > 0
            n_particles = hdf["metadata/nbody_info"].attrs["n_particles"]
            assert n_particles > 0
            particle_mass = hdf["metadata/nbody_info"].attrs["particle_mass"]
            assert particle_mass > 0
            sim_name = hdf["metadata/nbody_info"].attrs["sim_name"]
            assert len(sim_name) > 0

        except:  # noqa
            s = "metadata is incorrect"
            msg.append(s)

    return msg


def check_host_pos_is_near_galaxy_pos(fn_lc_mock, data=None):
    """host position should be reasonably close to galaxy position"""
    if data is None:
        data = load_flat_hdf5(fn_lc_mock)

    bn = os.path.basename(fn_lc_mock)

    host_x = data["x"][data["top_host_idx"]]
    host_y = data["y"][data["top_host_idx"]]
    host_z = data["z"][data["top_host_idx"]]
    dx = data["x_nfw"] - host_x
    dy = data["y_nfw"] - host_y
    dz = data["z_nfw"] - host_z
    host_dist = np.sqrt(dx**2 + dy**2 + dz**2)

    msg = []
    n_very_far = np.sum(host_dist > 5)
    if n_very_far > 10:
        s = f"{n_very_far} galaxies in {bn} with "
        s += "unexpectedly large xyz distance from top_host_idx"
        msg.append(s)

    msk_cen = data["central"]
    mean_sat_dist = np.abs(np.mean(host_dist[~msk_cen]))
    std_sat_dist = np.std(host_dist[~msk_cen])
    if mean_sat_dist > 1.0:
        s = f"<dist_sat>={mean_sat_dist:.2f} Mpc/h is unexpectedly large"
        msg.append(s)
    if std_sat_dist > 1.0:
        s = f"std(dist_sat)={std_sat_dist:.2f} Mpc/h is unexpectedly large"
        msg.append(s)

    return msg
