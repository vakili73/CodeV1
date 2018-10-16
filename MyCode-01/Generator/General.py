from .BaseGenerator import BaseGenerator

import numpy as np

class General(BaseGenerator):
    def __init__(self, augment=False, allowable=[],
                 X_train: np.ndarray = np.array([]),
                 y_train: np.ndarray = np.array([]),
                 X_test: np.ndarray = np.array([]),
                 y_test: np.ndarray = np.array([])):
        super().__init__(augment, allowable,
                         X_train, y_train, X_test, y_test)
        pass

    def get_batch(self):
        pass