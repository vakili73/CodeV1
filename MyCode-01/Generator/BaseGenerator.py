
import numpy as np


class BaseGenerator(object):
    def __init__(self, augment=False, allowable=[],
                 X_train: np.ndarray = np.array([]),
                 y_train: np.ndarray = np.array([]),
                 X_test: np.ndarray = np.array([]),
                 y_test: np.ndarray = np.array([])):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        pass

    def get_batch(self, *args, **kwargs):
        raise NotImplementedError
    pass
