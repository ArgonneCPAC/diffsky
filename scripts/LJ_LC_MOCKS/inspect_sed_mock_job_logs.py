"""Inspect the job logs of make_sed_lc_mock_lj.py"""

import argparse
import os
from glob import glob

BNPAT = "run_sed_mocks_{0}_to_{1}.sh"


def _get_bad_example(bad_job_collector):
    if len(bad_job_collector) == 0:
        bad_example = []
    else:
        bad_example = []
        with open(bad_job_collector[0], "r") as f:
            for line in f:
                bad_example.append(line.strip())
    return bad_example


def check_o_logs(drn):
    fn_list = glob(os.path.join(drn, "run_*.o*"))
    bad_jobs = []
    incompl_jobs = []

    for fn in fn_list:
        line_collector = []
        with open(fn, "r") as f:
            for line in f:
                line_collector.append(line.strip())

        if len(line_collector) < 2:
            incompl_jobs.append(fn)
        elif "Total runtime" not in line_collector[-1]:
            bad_jobs.append(fn)

    bad_example = _get_bad_example(bad_jobs)

    return incompl_jobs, bad_jobs, bad_example


def check_e_logs(drn):
    fn_list = glob(os.path.join(drn, "run_*.sh.e*"))
    bad_job_collector = []
    for fn in fn_list:
        line_counter = 0
        with open(fn, "r") as f:
            for line_counter, line in enumerate(f):
                pass
        if line_counter > 0:
            bad_job_collector.append(fn)

    bad_example = _get_bad_example(bad_job_collector)

    return bad_job_collector, bad_example


def check_for_absent_jobs(drn, istart=0, bn_pat=BNPAT):
    n_batch = infer_n_batch(drn, bn_pat=bn_pat)
    iend = infer_iend(drn, bn_pat=bn_pat)
    nchar = len(str(iend))
    fn_list = sorted(glob(os.path.join(drn, bn_pat.format("*", "*"))))
    bn_list = [os.path.basename(fn) for fn in fn_list]
    absent_job_list = []
    for i in range(istart, iend + n_batch, n_batch):
        ichar = f"{i:0{nchar}d}"
        j = i + n_batch - 1
        jchar = f"{j:0{nchar}d}"
        bn_i = bn_pat.format(ichar, jchar)
        if bn_i not in bn_list:
            absent_job_list.append(bn_i)

    return bn_list, absent_job_list


def check_for_absent_logs(drn, istart=0, bn_pat=BNPAT):
    bn_list = check_for_absent_jobs(drn, istart=istart, bn_pat=bn_pat)[0]
    fn_list = [os.path.join(drn, bn) for bn in bn_list]
    absent_e_logs = []
    absent_o_logs = []
    for fn in fn_list:
        e_fn_pat = fn + ".e*"
        e_fn_list = glob(e_fn_pat)
        if len(e_fn_list) != 1:
            absent_e_logs.append(fn)

        o_fn_pat = fn + ".o*"
        o_fn_list = glob(o_fn_pat)
        if len(o_fn_list) != 1:
            absent_o_logs.append(fn)

    return absent_e_logs, absent_o_logs


def infer_n_batch(drn, bn_pat=BNPAT):
    fn_list = sorted(glob(os.path.join(drn, bn_pat.format("*", "*"))))
    bn0 = os.path.basename(fn_list[0])
    n_batch = int(bn0.split("_")[6].replace(".sh", "")) - int(bn0.split("_")[4]) + 1
    return n_batch


def infer_iend(drn, bn_pat=BNPAT):
    fn_list = sorted(glob(os.path.join(drn, bn_pat.format("*", "*"))))
    bn_last = os.path.basename(fn_list[-1])
    iend = int(bn_last.split("_")[4])
    return iend


def record_uncollated_rank_data(drn, fnout="uncollated_rank_data.txt"):
    pat = os.path.join(drn, "lc_cores-*_rank_0.hdf5")
    fn_list = glob(pat)
    with open(fnout, "w") as fout:
        for fn in fn_list:
            fout.write(fn + "\n")


def record_diffsky_gals_fnames(
    drn, bnpat="lc_cores-*.*.diffsky_gals.hdf5", fnout="diffsky_gals_fnames.txt"
):
    pat = os.path.join(drn, bnpat)
    fn_list = glob(pat)
    with open(fnout, "w") as fout:
        for fn in fn_list:
            fout.write(fn + "\n")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("drn", help="Directory storing job logs")

    args = parser.parse_args()
    drn = args.drn

    incompl_jobs, bad_jobs, bad_example = check_o_logs(drn)
    bad_job_collector, bad_example = check_e_logs(drn)

    all_pass = (
        (len(incompl_jobs) == 0) & (len(bad_jobs) == 0) & (len(bad_job_collector) == 0)
    )
    if all_pass:
        print("Every job terminated without raising an exception")
    else:
        print("Some scripts failed")
