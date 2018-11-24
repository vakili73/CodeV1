import numpy as np


def _round(x, decimals=0):
    multiplier = 10**decimals
    return np.round(x * multiplier) / multiplier


a = np.array([0.15, 2.5, 0.51])

print(_round(a, 1))