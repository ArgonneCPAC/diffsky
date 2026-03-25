""""""

from ...utils import _sigmoid


def frac_disk_dominated(logsm):
    return _sigmoid(logsm, 10.75, 1.5, 0.9, 0.1)
