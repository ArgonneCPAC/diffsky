""""""

import datetime

from .. import mock_utils


def test_get_mock_version_name_today():
    mock_nickname = "Daryl"
    mock_version_name = mock_utils.get_mock_version_name(mock_nickname)

    nick = mock_version_name.split("_")[0]

    assert nick == mock_nickname

    m, d, y = mock_version_name.split("_")[1:]
    day = datetime.date.today()
    y_correct, m_correct, d_correct = str(day).split("-")
    assert y == y_correct
    assert m == m_correct
    assert d == d_correct


def test_get_mock_version_name_manual_date():
    mock_nickname = "Daryl"
    date_string_in = "10_05_2024"
    mock_version_name = mock_utils.get_mock_version_name(mock_nickname, date_string_in)

    nick = mock_version_name.split("_")[0]
    date_string_correct = "_".join(mock_version_name.split("_")[1:])

    assert nick == mock_nickname
    assert date_string_in == date_string_correct
