
import numpy as np


class BaseGenerator(object):
    def __init__(self, augment=False, allowable=[],
                 X_train: np.ndarray = np.array([]),
                 y_train: np.ndarray = np.array([]),
                 X_test: np.ndarray = np.array([]),
                 y_test: np.ndarray = np.array([])):
        self.augment = augment
        self.allowable = allowable
        self.X_train = X_train.copy()
        self.y_train = y_train.copy()
        self.X_test = X_test.copy()
        self.y_test = y_test.copy()
        pass

    def get_batch(self, *args, **kwargs):
        raise NotImplementedError
    pass
