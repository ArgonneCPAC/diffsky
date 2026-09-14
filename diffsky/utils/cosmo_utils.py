""""""

from dsps.cosmology.flat_wcdm import C_SPEED


def get_redshift_obs_from_redshift_true(z_true, vpec_kms):
    """Calculate effect of peculiar velocity on observed redshift

    1 + z_obs = (1+z_true) * (1+v/c)

    Parameters
    ----------
    z_true : array, shape (n, )

    vpec_kms : array, shape (n, )
        Peculiar velocity along the line of sight in km/s units

    Returns
    -------
    redshift_obs : array, shape (n, )
        Observed redshift

    """
    c_kms = C_SPEED / 1000.0
    one_plus_zobs = (1 + z_true) * (1 + vpec_kms / c_kms)
    redshift_obs = one_plus_zobs - 1.0
    return redshift_obs
