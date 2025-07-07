""" """

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
NUM_SUBVOLS_LJ = 192


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("drn_target_data", help="Location of target data")
    parser.add_argument("fit_type", help="Halo population", choices=["cens", "all"])

    args = parser.parse_args()
    drn_target_data = args.drn_target_data
    fit_type = args.fit_type

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

    vol_chunk = Vbox_mpc / NUM_SUBVOLS_LJ / DEFAULT_NCHUNKS
    vol_target_data = vol_chunk * n_chunkdata_tot

    cuml_density_target_data = cuml_counts / vol_target_data

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
    print(f"Best-fit log10(loss) = {np.log10(loss):.2f}")

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

    print("\nPrinting best-fit parameters:\n")

    fnout = f"best_fit_params_{fit_type}.txt"
    with open(fnout, "w") as fout:

        for pname, pval in zip(p_best.ytp_params._fields, p_best.ytp_params):
            fout.write(f"{pname}={pval:.2f},\n")
        for pname, pval in zip(p_best.x0_params._fields, p_best.x0_params):
            fout.write(f"{pname}={pval:.2f},\n")
        for pname, pval in zip(p_best.lo_params._fields, p_best.lo_params):
            fout.write(f"{pname}={pval:.2f},\n")
        for pname, pval in zip(p_best.hi_params._fields, p_best.hi_params):
            fout.write(f"{pname}={pval:.2f},\n")
