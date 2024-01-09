"""
"""


def test_diffsky_has_main_branch_dsps():
    try:
        from dsps.sfh import diffburst  # noqa

        HAS_DSPS_MAIN = True
    except ImportError:
        HAS_DSPS_MAIN = False
    assert HAS_DSPS_MAIN, "Failed to import dsps.experimental.diffburst"
