"""Utility functions for making mocks"""

import datetime


def get_mock_version_name(mock_nickname, date_string=None):
    """Get version name of output mock

    Parameters
    ----------
    mock_nickname : string
        e.g., 'tng' or 'umachine'

    date_string : string, optional
        e.g., '11_05_2025'
        Default is current date

    Returns
    -------
    mock_version_name : string
        e.g., 'tng_11_05_2025'

    """
    if date_string is None:
        day = datetime.date.today()
        y, m, d = str(day).split("-")
        date_string = "_".join((m, d, y))

    mock_version_name = mock_nickname + "_" + date_string

    return mock_version_name
