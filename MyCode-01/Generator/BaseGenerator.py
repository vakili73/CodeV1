
import numpy as np


class BaseGenerator(object):
    def __init__(self, augment=False, allowable=[],
                 batch_size: int = 128,
                 X_train: np.ndarray = np.array([]),
                 y_train: np.ndarray = np.array([])):
        self.augment = augment
        self.allowable = allowable
        self.batch_size = batch_size

        self.X_train = X_train.copy()
        self.y_train = y_train.copy()

        self.classes = np.unique(y_train)
        self.length = np.size(y_train)
        
        self.state = None
        pass

    def get_batch(self, *args, **kwargs):
        raise NotImplementedError
    pass
