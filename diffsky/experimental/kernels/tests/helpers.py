""""""

import numpy as np

TOL = 1e-8


def check_phot_kern_merging_results(phot_kern_results, lc_data):
    n_gals = lc_data.z_obs.size
    n_z_table, n_bands, n_met, n_age = lc_data.precomputed_ssp_mag_table.shape

    skip_fields = (
        "diffstar_info_ms",
        "diffstar_info_q",
        "burstiness_info_ms",
        "burstiness_info_q",
    )
    for key in phot_kern_results._fields:
        if key not in skip_fields:
            arr = getattr(phot_kern_results, key)
            try:
                assert np.all(np.isfinite(arr))
            except ValueError:
                raise ValueError(f"{key} is not a flat array")

    assert np.all(phot_kern_results.p_merge >= 0)
    assert np.all(phot_kern_results.p_merge <= 1)
    assert np.any(phot_kern_results.p_merge > 0)
    assert np.any(phot_kern_results.p_merge < 1)

    # Enforce consistent array shapes
    assert phot_kern_results.p_merge.shape == (n_gals,)
    assert phot_kern_results.logsm_obs.shape == (n_gals,)
    assert phot_kern_results.logsm_obs_in_situ.shape == (n_gals,)
    assert phot_kern_results.obs_mags.shape == (n_gals, n_bands)
    assert phot_kern_results.obs_mags_in_situ.shape == (n_gals, n_bands)

    msk_cen = lc_data.is_central == 1
    msk_sat = ~msk_cen

    # Enforce centrals can only get brighter and satellites can only get dimmer
    name = "obs_mags"
    x = getattr(phot_kern_results, name)
    y = getattr(phot_kern_results, name + "_in_situ")
    assert np.all(x[msk_cen] <= y[msk_cen] + TOL)
    assert np.all(x[msk_sat] >= y[msk_sat] - TOL)

    # Enforce merging is nontrivial
    assert np.any(x[msk_cen] < y[msk_cen])
    assert np.any(x[msk_sat] > y[msk_sat])

    # Enforce centrals can only get more massive and satellites less massive
    name = "logsm_obs"
    x = getattr(phot_kern_results, name)
    y = getattr(phot_kern_results, name + "_in_situ")
    assert np.all(x[msk_cen] >= y[msk_cen] - TOL)
    assert np.all(x[msk_sat] <= y[msk_sat] + TOL)
    # Enforce merging is nontrivial
    assert np.any(x[msk_cen] > y[msk_cen])
    assert np.any(x[msk_sat] < y[msk_sat])


def check_spec_kern_merging_results(spec_kern_results, lc_data):
    n_gals = lc_data.z_obs.size
    n_lines = len(lc_data.ssp_data.ssp_emline_wave)

    assert spec_kern_results.linelum_gal.shape == (n_gals, n_lines)
    assert spec_kern_results.linelum_gal_in_situ.shape == (n_gals, n_lines)

    skip_fields = (
        "diffstar_info_ms",
        "diffstar_info_q",
        "burstiness_info_ms",
        "burstiness_info_q",
    )
    for key in spec_kern_results._fields:
        if key not in skip_fields:
            arr = getattr(spec_kern_results, key)
            try:
                assert np.all(np.isfinite(arr))
            except ValueError:
                raise ValueError(f"{key} is not a flat array")

    msk_cen = lc_data.is_central == 1
    msk_sat = ~msk_cen

    # Enforce centrals can only get more massive and satellites less massive
    name = "linelum_gal"
    x = np.log10(getattr(spec_kern_results, name))
    y = np.log10(getattr(spec_kern_results, name + "_in_situ"))
    assert np.all(x[msk_cen] >= y[msk_cen] - TOL)
    assert np.all(x[msk_sat] <= y[msk_sat] + TOL)
    # Enforce merging is nontrivial
    assert np.any(x[msk_cen] > y[msk_cen])
    assert np.any(x[msk_sat] < y[msk_sat])
