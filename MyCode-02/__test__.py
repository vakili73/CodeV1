import numpy as np

from Database import Utils
from Database import CONFIG

from matplotlib import pyplot as plt

shape = CONFIG['nist']['shape']
X_train, X_test, y_train, y_test = Utils.laod_data('nist')

X_train = Utils.reshape(X_train / 255.0, shape)
X_test = Utils.reshape(X_test / 255.0, shape)
