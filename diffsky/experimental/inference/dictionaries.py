"""
This is just something that helps me selecting subsets parameters to infer jointly.
"""

diffstarpop_params = []

diffstarpop_means = [
    # "u_mean_ulgm_mseq_xtp",
    "u_mean_ulgm_mseq_ytp",
    # "vmean_ulgm_mseq_lo",
    # "u_mean_ulgm_mseq_hi",
    # "u_mean_ulgy_mseq_xtp",
    "u_mean_ulgy_mseq_ytp",
    # "u_mean_ulgy_mseq_lo",
    # "u_mean_ulgy_mseq_hi",
    "u_mean_ul_mseq_int",
    # "u_mean_ul_mseq_slp",
    "u_mean_uh_mseq_int",
    # "u_mean_uh_mseq_slp",
    # "u_mean_ulgm_qseq_xtp",
    "u_mean_ulgm_qseq_ytp",
    # "u_mean_ulgm_qseq_lo",
    # "u_mean_ulgm_qseq_hi",
    # "u_mean_ulgy_qseq_xtp",
    "u_mean_ulgy_qseq_ytp",
    # "u_mean_ulgy_qseq_lo",
    # "u_mean_ulgy_qseq_hi",
    "u_mean_ul_qseq_int",
    # "u_mean_ul_qseq_slp",
    "u_mean_uh_qseq_int",
    # "u_mean_uh_qseq_slp",
    # "u_mean_uqt_xtp",
    "u_mean_uqt_ytp",
    # "u_mean_uqt_lo",
    # "u_mean_uqt_hi",
    "u_mean_uqs_int",
    # "u_mean_uqs_slp",
    "u_mean_udrop_int",
    # "u_mean_udrop_slp",
    "u_mean_urej_int",
    # "u_mean_urej_slp",
]


PARAM_SUBSETS = {
    "diffstarpop_means": diffstarpop_means,
}
