"""This script optimizes the hmf_model.py parameters according to Last Journey.

The script writes the HMF target data to the following drn:
diffsky/mass_functions/hmf_calibrations/tests/testing_data
Take care: there is existing data in this drn that may be overwritten by this script

The best-fitting parameters are written to the following ASCII file:
lj_hmf_params.txt
This ASCII file is programmatically read by hmf_param_reader.py,
and so running this script updates the last journey hmf_model parameters

"""

import argparse
import os
from glob import glob

import matplotlib.cm as cm
import numpy as np
from matplotlib import pyplot as plt

from diffsky.data_loaders.hacc_utils import get_diffsky_info_from_hacc_sim
from diffsky.mass_functions import hmf_model
from diffsky.mass_functions.fitting_utils import fit_hmf_model

DEFAULT_NCHUNKS = 10


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("drn_target_data", help="Location of target data")
    parser.add_argument(
        "drn_testing_data",
        help="Location to store collated target data for downstream unit testing",
    )
    parser.add_argument("fit_type", help="Halo population", choices=["cens", "all"])
    parser.add_argument(
        "fname_params_out", help="Output fname of best-fitting parameters"
    )
    parser.add_argument(
        "-nchunks",
        help="Number of chunks per subvolume. Take care that this is consistent with "
        "the settings used to produce the target data with measure_hmf_target_data_hacc.py",
        type=int,
        default=DEFAULT_NCHUNKS,
    )

    args = parser.parse_args()
    drn_target_data = args.drn_target_data
    drn_testing_data = args.drn_testing_data
    fit_type = args.fit_type
    fname_params_out = args.fname_params_out
    nchunks = args.nchunks

    Z_TABLE = np.load(os.path.join(drn_target_data, "redshift_bins.npy"))
    LOGMP_BINS = np.load(os.path.join(drn_target_data, "logmp_bins.npy"))

    fn_list = glob(os.path.join(drn_target_data, f"*_{fit_type}.npy"))
    n_chunkdata_tot = len(fn_list)
    print(f"Number of chunks of target data = {n_chunkdata_tot}")
    cuml_counts = np.load(fn_list[0])
    cuml_counts = np.zeros_like(cuml_counts)

    sim_info = get_diffsky_info_from_hacc_sim("LastJourney")

    for fn in fn_list:
        cuml_counts_fn = np.load(fn)
        cuml_counts = cuml_counts + cuml_counts_fn

    Lbox_mpc = sim_info.sim.rl / sim_info.cosmo_params.h  # Mpc
    Vbox_mpc = Lbox_mpc**3

    vol_chunk = Vbox_mpc / sim_info.num_subvols / nchunks
    vol_target_data = vol_chunk * n_chunkdata_tot

    cuml_density_target_data = cuml_counts / vol_target_data

    # save target data to testing_data
    fn_lj_hmf_redshift_bins = os.path.join(drn_testing_data, "lj_hmf_redshift_bins.txt")
    np.savetxt(fn_lj_hmf_redshift_bins, Z_TABLE)

    fn_lj_hmf_logmp_bins = os.path.join(drn_testing_data, "lj_hmf_logmp_bins.txt")
    np.savetxt(fn_lj_hmf_logmp_bins, LOGMP_BINS)

    fn_lj_hmf_cuml_density = os.path.join(
        drn_testing_data, f"lj_hmf_cuml_density_{fit_type}.txt"
    )
    np.savetxt(fn_lj_hmf_cuml_density, cuml_density_target_data)

    loss_data_collector = []
    for iz in range(len(Z_TABLE)):
        z_target = Z_TABLE[iz]
        cuml_density_target = cuml_density_target_data[:, iz]
        msk_nonzero = cuml_density_target > 2e-9
        lgmp_target = LOGMP_BINS[msk_nonzero]
        lg_cuml_density_target = np.log10(cuml_density_target[msk_nonzero])
        loss_data = (z_target, lgmp_target, lg_cuml_density_target)
        loss_data_collector.append(loss_data)

    n_steps = 500
    print(f"...running fitter with {n_steps} steps of gradient descent")
    _res = fit_hmf_model.hmf_fitter(
        loss_data_collector, n_warmup=0, n_steps=n_steps, step_size=0.05
    )
    p_best, loss, loss_hist, params_hist, fit_terminates = _res
    print(f"Best-fit log10(loss) = {np.log10(loss):.2f}\n")

    print("Writing optimizer diagnostic plots to disk")

    # Plot loss curve
    fig, ax = plt.subplots(1, 1)
    __ = ax.plot(np.log10(loss_hist))
    xlabel = ax.set_xlabel(r"${\rm step}$")
    ylabel = ax.set_ylabel(r"${\rm log_{10}loss}$")
    fig.savefig(
        f"loss_curve_{fit_type}.png",
        bbox_extra_artists=[xlabel, ylabel],
        bbox_inches="tight",
        dpi=200,
    )

    colors = cm.coolwarm(np.linspace(0, 1, Z_TABLE.size))  # blue first
    # Plot upshot diagnostic
    fig, ax = plt.subplots(1, 1)
    yscale = ax.set_yscale("log")

    for iz, loss_data in enumerate(loss_data_collector):
        redshift, logmp_target, log_cuml_density_target = loss_data
        pred_hmf = hmf_model.predict_cuml_hmf(p_best, logmp_target, redshift)

        __ = ax.plot(logmp_target, 10**log_cuml_density_target, color=colors[iz])
        __ = ax.plot(logmp_target, 10**pred_hmf, "--", color=colors[iz])
    xlabel = ax.set_xlabel(r"${\rm log10 M_h}$")
    ylabel = ax.set_ylabel(r"${\rm cumulative\ density\ [1/Mpc^3]}$")
    fig.savefig(
        f"hmf_model_best_fit_{fit_type}.png",
        bbox_extra_artists=[xlabel, ylabel],
        bbox_inches="tight",
        dpi=200,
    )

    print(f"\nWriting best-fit parameters to {fname_params_out}\n")
    with open(fname_params_out, "w") as fout:

        for pname, pval in zip(p_best.ytp_params._fields, p_best.ytp_params):
            fout.write(f"{pname}={pval:.2f}\n")
        for pname, pval in zip(p_best.x0_params._fields, p_best.x0_params):
            fout.write(f"{pname}={pval:.2f}\n")
        for pname, pval in zip(p_best.lo_params._fields, p_best.lo_params):
            fout.write(f"{pname}={pval:.2f}\n")
        for pname, pval in zip(p_best.hi_params._fields, p_best.hi_params):
            fout.write(f"{pname}={pval:.2f}\n")
