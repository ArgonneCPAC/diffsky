""""""

DBKMassKeys = ("mstar_diffuse_disk", "mstar_bulge", "mstar_knots")


def get_disk_bulge_knot_masses(mock):
    """Calculate the stellar mass of the disk/bulge/knot decomposition from a mock

    Parameters
    ----------
    mock : dict
        Dictionary of mock data.
        Should have keys `logsm_obs`, `bulge_to_total`, `fknot`

    Returns
    --------
    dbk_masses : dict
        Dictionary with mass decomposition in units of Msun

    """
    dbk_masses = _get_disk_bulge_knot_masses(
        mock["logsm_obs"], mock["bulge_to_total"], mock["fknot"]
    )
    return dbk_masses


def _get_disk_bulge_knot_masses(logsm_obs, bulge_to_total, fknot):
    mstar_tot = 10**logsm_obs
    mstar_bulge = bulge_to_total * mstar_tot
    mstar_disk_tot = mstar_tot - mstar_bulge
    mstar_knots = fknot * mstar_disk_tot
    mstar_diffuse_disk = mstar_disk_tot - mstar_knots

    dbk_masses = dict(
        mstar_diffuse_disk=mstar_diffuse_disk,
        mstar_bulge=mstar_bulge,
        mstar_knots=mstar_knots,
    )
    return dbk_masses
